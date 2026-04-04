"""
train.py
========
Script training untuk IndoBERT Extractive Summarizer.

Strategi Training 2 Tahap:
  Tahap 1: Bekukan lapisan bawah BERT → latih head lebih cepat (5 epoch)
  Tahap 2: Cairkan semua BERT → fine-tune end-to-end (3 epoch)

Fungsi Loss: Binary Cross-Entropy (BCE) dengan class weight
  (karena jumlah kalimat positif << negatif)

Optimizer: AdamW dengan linear warmup + cosine decay
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# Tambah root ke sys.path agar bisa import dari src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import IndoBERTSumExtractor, save_model
from src.dataset import create_dataloader, load_processed_data
from src.evaluate import evaluate_rouge_batch, compute_classification_metrics


# ─────────────────────────────────────────────────────────────────
#  KONFIGURASI TRAINING
# ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    'bert_model_name': 'indobenchmark/indobert-base-p1',
    'freeze_bert_layers': 8,          # Bekukan N layer awal untuk tahap 1
    'num_transformer_layers': 2,
    'num_heads': 8,
    'dropout': 0.1,

    # Data
    'data_dir': 'data/processed',
    'dataset_name': 'combined',       # 'combined', 'indosum', atau 'ormas'
    'max_sent_len': 128,
    'max_sentences': 40,

    # Training
    'batch_size': 4,
    'grad_accumulation': 4,           # Efektif batch = 4*4 = 16
    'num_epochs_stage1': 5,           # Training dengan BERT terbekukan
    'num_epochs_stage2': 3,           # Fine-tuning penuh
    'learning_rate_stage1': 5e-4,
    'learning_rate_stage2': 2e-5,
    'bert_lr_stage2': 1e-5,           # LR lebih kecil untuk BERT
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,

    # Loss
    'pos_weight': 3.0,                # Bobot kelas positif (kalimat relevan)

    # Lainnya
    'save_dir': 'models/indobert_summarizer',
    'log_dir': 'logs',
    'seed': 42,
    'num_workers': 0,
    'eval_every': 500,                # Evaluasi setiap N step
    'save_best': True,
    'device': 'auto',
}


# ─────────────────────────────────────────────────────────────────
#  UTILITAS
# ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set seed untuk reprodusibilitas."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = 'auto') -> torch.device:
    """Deteksi dan return device terbaik."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[INFO] Menggunakan GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            device = torch.device('cpu')
            print("[INFO] GPU tidak tersedia, menggunakan CPU (proses lebih lambat)")
    else:
        device = torch.device(device_str)
    return device


class TrainingLogger:
    """Logger sederhana untuk menyimpan metrics training."""

    def __init__(self, log_dir: str, run_name: str = 'run'):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = Path(log_dir) / f"{run_name}_log.jsonl"
        self.history: List[Dict] = []

    def log(self, metrics: Dict):
        """Simpan metrics ke file."""
        self.history.append(metrics)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')

    def print_metrics(self, metrics: Dict, prefix: str = ''):
        """Tampilkan metrics di console."""
        msg = prefix + ' | '.join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                   for k, v in metrics.items())
        print(msg)


# ─────────────────────────────────────────────────────────────────
#  FUNGSI TRAINING SATU EPOCH
# ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: IndoBERTSumExtractor,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    grad_accumulation: int = 1,
    max_grad_norm: float = 1.0,
    epoch: int = 1
) -> Dict:
    """
    Satu epoch training.

    Returns:
        Dict berisi loss rata-rata dan metrics lainnya
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)        # (B, S, T)
        attention_mask = batch['attention_mask'].to(device)  # (B, S, T)
        labels = batch['labels'].to(device)              # (B, S)
        sent_mask = batch['sent_mask'].to(device)        # (B, S)

        # Forward pass
        scores = model(input_ids, attention_mask, sent_mask)  # (B, S)

        # Hitung loss hanya pada kalimat nyata (bukan padding)
        valid_mask = ~sent_mask  # True = kalimat nyata
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask]

        loss = criterion(valid_scores, valid_labels)
        loss = loss / grad_accumulation

        # Backward pass
        loss.backward()

        total_loss += loss.item() * grad_accumulation

        # Kumpulkan prediksi
        preds = (valid_scores.detach() > 0.5).float().cpu().numpy()
        lbls = valid_labels.detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(lbls.tolist())

        # Update optimizer setiap grad_accumulation step
        if (step + 1) % grad_accumulation == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        # Update progress bar
        pbar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})

    # Hitung metrics akhir
    avg_loss = total_loss / len(dataloader)
    metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds)
    )
    metrics['loss'] = avg_loss
    return metrics


# ─────────────────────────────────────────────────────────────────
#  FUNGSI EVALUASI
# ─────────────────────────────────────────────────────────────────

def evaluate_epoch(
    model: IndoBERTSumExtractor,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 1
) -> Dict:
    """
    Evaluasi model pada dataset validasi.

    Returns:
        Dict berisi loss dan metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            sent_mask = batch['sent_mask'].to(device)

            scores = model(input_ids, attention_mask, sent_mask)

            valid_mask = ~sent_mask
            valid_scores = scores[valid_mask]
            valid_labels = labels[valid_mask]

            loss = criterion(valid_scores, valid_labels)
            total_loss += loss.item()

            preds = (valid_scores > 0.5).float().cpu().numpy()
            lbls = valid_labels.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.tolist())

            pbar.set_postfix({'loss': f"{total_loss / (len(all_preds) or 1):.4f}"})

    avg_loss = total_loss / len(dataloader)
    metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds)
    )
    metrics['loss'] = avg_loss
    return metrics


# ─────────────────────────────────────────────────────────────────
#  PIPELINE TRAINING UTAMA
# ─────────────────────────────────────────────────────────────────

def train(config: Optional[Dict[str, Any]] = None):
    """
    Pipeline training lengkap dengan 2 tahap.

    Args:
        config: Konfigurasi training (default = DEFAULT_CONFIG)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg['seed'])
    device = get_device(cfg['device'])

    print("\n" + "=" * 65)
    print("  TRAINING IndoBERT Extractive Summarizer")
    print("=" * 65)
    print(f"  Dataset   : {cfg['dataset_name']}")
    print(f"  Device    : {device}")
    print(f"  Batch size: {cfg['batch_size']} (×{cfg['grad_accumulation']} accum)")
    print("=" * 65 + "\n")

    # ── Muat data ───────────────────────────────────────────────
    print("[1/6] Memuat data ...")
    train_data, val_data, test_data = load_processed_data(
        cfg['data_dir'], cfg['dataset_name']
    )
    if not train_data:
        print("[ERROR] Data training kosong! Jalankan src/preprocess.py terlebih dahulu.")
        return

    # ── Muat tokenizer ──────────────────────────────────────────
    print(f"\n[2/6] Memuat tokenizer: {cfg['bert_model_name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['bert_model_name'])

    # ── Buat DataLoader ─────────────────────────────────────────
    print("\n[3/6] Membuat DataLoader ...")
    train_loader = create_dataloader(
        train_data, tokenizer,
        batch_size=cfg['batch_size'],
        max_sent_len=cfg['max_sent_len'],
        max_sentences=cfg['max_sentences'],
        shuffle=True,
        num_workers=cfg['num_workers']
    )
    val_loader = create_dataloader(
        val_data, tokenizer,
        batch_size=cfg['batch_size'],
        max_sent_len=cfg['max_sent_len'],
        max_sentences=cfg['max_sentences'],
        shuffle=False,
        num_workers=cfg['num_workers']
    )
    print(f"  Train: {len(train_loader)} batch | Val: {len(val_loader)} batch")

    # ── Inisialisasi model ───────────────────────────────────────
    print(f"\n[4/6] Inisialisasi model ...")
    model = IndoBERTSumExtractor(
        bert_model_name=cfg['bert_model_name'],
        num_transformer_layers=cfg['num_transformer_layers'],
        num_heads=cfg['num_heads'],
        dropout=cfg['dropout'],
        freeze_bert_layers=cfg['freeze_bert_layers']
    ).to(device)
    print(f"  Parameter bisa dilatih: {model.get_trainable_params():,}")
    print(f"  Total parameter: {model.get_total_params():,}")

    # ── Loss function ────────────────────────────────────────────
    # Model mengeluarkan probabilitas sigmoid, jadi gunakan weighted BCE custom.
    criterion_weighted = WeightedBCELoss(pos_weight=cfg['pos_weight'])

    logger = TrainingLogger(cfg['log_dir'], run_name=f"indobert_{int(time.time())}")
    best_val_f1 = 0.0
    best_model_path = None

    # ════════════════════════════════════════════════════════════
    #  TAHAP 1: Training hanya head (BERT terbekukan)
    # ════════════════════════════════════════════════════════════
    print(f"\n[5/6] TAHAP 1: Training head ({cfg['num_epochs_stage1']} epoch) ...")
    print("      (Layer BERT atas dibekukan, hanya melatih inter-sent transformer)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer1 = torch.optim.AdamW(
        trainable_params,
        lr=cfg['learning_rate_stage1'],
        weight_decay=cfg['weight_decay']
    )
    total_steps1 = len(train_loader) * cfg['num_epochs_stage1'] // cfg['grad_accumulation']
    warmup_steps1 = int(total_steps1 * cfg['warmup_ratio'])
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, warmup_steps1, total_steps1)

    for epoch in range(1, cfg['num_epochs_stage1'] + 1):
        t_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer1, scheduler1,
            criterion_weighted, device,
            cfg['grad_accumulation'], cfg['max_grad_norm'], epoch
        )
        val_metrics = evaluate_epoch(
            model, val_loader, criterion_weighted, device, epoch
        )
        elapsed = time.time() - t_start

        # Cetak progress
        print(
            f"\n  Epoch {epoch}/{cfg['num_epochs_stage1']} "
            f"({elapsed:.0f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"F1: {train_metrics.get('f1', 0):.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"F1: {val_metrics.get('f1', 0):.4f}"
        )
        logger.log({'stage': 1, 'epoch': epoch, 'train': train_metrics, 'val': val_metrics})

        # Simpan best model
        val_f1 = val_metrics.get('f1', 0)
        if val_f1 > best_val_f1 and cfg['save_best']:
            best_val_f1 = val_f1
            best_model_path = save_model(model, cfg['save_dir'])
            print(f"  ✅ Best model disimpan (Val F1: {best_val_f1:.4f})")

        # Simpan checkpoint epoch
        save_model(model, cfg['save_dir'], epoch=epoch)

    # ════════════════════════════════════════════════════════════
    #  TAHAP 2: Fine-tuning penuh (BERT dicairkan)
    # ════════════════════════════════════════════════════════════
    print(f"\n[6/6] TAHAP 2: Fine-tuning penuh ({cfg['num_epochs_stage2']} epoch) ...")
    model.unfreeze_bert()
    print(f"  Parameter bisa dilatih: {model.get_trainable_params():,}")

    # Gunakan learning rate berbeda untuk BERT vs head
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.inter_sent_transformer.parameters()) +
        list(model.classifier.parameters()) +
        list(model.pos_encoding.parameters())
    )
    optimizer2 = torch.optim.AdamW([
        {'params': bert_params, 'lr': cfg['bert_lr_stage2']},
        {'params': head_params, 'lr': cfg['learning_rate_stage2']},
    ], weight_decay=cfg['weight_decay'])

    total_steps2 = len(train_loader) * cfg['num_epochs_stage2'] // cfg['grad_accumulation']
    warmup_steps2 = int(total_steps2 * cfg['warmup_ratio'])
    scheduler2 = get_cosine_schedule_with_warmup(optimizer2, warmup_steps2, total_steps2)

    for epoch in range(1, cfg['num_epochs_stage2'] + 1):
        t_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer2, scheduler2,
            criterion_weighted, device,
            cfg['grad_accumulation'], cfg['max_grad_norm'],
            epoch + cfg['num_epochs_stage1']
        )
        val_metrics = evaluate_epoch(
            model, val_loader, criterion_weighted, device,
            epoch + cfg['num_epochs_stage1']
        )
        elapsed = time.time() - t_start

        print(
            f"\n  Epoch {epoch}/{cfg['num_epochs_stage2']} "
            f"({elapsed:.0f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"F1: {train_metrics.get('f1', 0):.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"F1: {val_metrics.get('f1', 0):.4f}"
        )
        logger.log({'stage': 2, 'epoch': epoch, 'train': train_metrics, 'val': val_metrics})

        val_f1 = val_metrics.get('f1', 0)
        if val_f1 > best_val_f1 and cfg['save_best']:
            best_val_f1 = val_f1
            best_model_path = save_model(model, cfg['save_dir'])
            print(f"  ✅ Best model diperbarui (Val F1: {best_val_f1:.4f})")

    # Simpan model final
    final_path = save_model(model, cfg['save_dir'])
    print(f"\n{'=' * 65}")
    print(f"  ✅ Training selesai!")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Model tersimpan di: {cfg['save_dir']}")
    print(f"{'=' * 65}")

    # Simpan tokenizer juga
    tokenizer.save_pretrained(cfg['save_dir'])
    print(f"  Tokenizer tersimpan di: {cfg['save_dir']}")

    return model, tokenizer, best_model_path


# ─────────────────────────────────────────────────────────────────
#  WEIGHTED BCE LOSS
# ─────────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy dengan bobot kelas positif.
    Mengatasi imbalance antara kalimat positif (ringkasan) dan negatif.
    """

    def __init__(self, pos_weight: float = 3.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        loss = -(
            self.pos_weight * target * torch.log(pred) +
            (1 - target) * torch.log(1 - pred)
        )
        return loss.mean()


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training IndoBERT Summarizer')
    parser.add_argument('--dataset', default='combined',
                        choices=['combined', 'indosum', 'ormas'],
                        help='Dataset yang digunakan')
    parser.add_argument('--epochs1', type=int, default=5,
                        help='Jumlah epoch tahap 1')
    parser.add_argument('--epochs2', type=int, default=3,
                        help='Jumlah epoch tahap 2')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Ukuran batch')
    parser.add_argument('--save_dir', default='models/indobert_summarizer',
                        help='Direktori menyimpan model')
    args = parser.parse_args()

    config = {
        **DEFAULT_CONFIG,
        'dataset_name': args.dataset,
        'num_epochs_stage1': args.epochs1,
        'num_epochs_stage2': args.epochs2,
        'batch_size': args.batch_size,
        'save_dir': args.save_dir,
    }

    train(config)
