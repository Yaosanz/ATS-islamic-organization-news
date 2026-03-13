"""
train_light.py
==============
Training RINGAN - menggunakan embedding yang sudah di-pre-komputasi.

Tidak memerlukan BERT saat training → SANGAT lebih cepat di CPU!
Melatih hanya: Inter-Sentence Transformer (2 layer) + Linear Classifier

Waktu estimasi CPU: ~30-60 menit untuk 5 epoch (71K dokumen)
GPU: ~5-10 menit

Penggunaan:
    python train_light.py                          # default: indosum, 5 epoch
    python train_light.py --dataset combined       # dataset gabungan
    python train_light.py --epochs 10 --lr 1e-3   # kustom
    python train_light.py --quick                  # demo cepat 1 epoch
"""

import sys
import os
import time
import pickle
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent

EMB_DIR   = ROOT / 'data' / 'embeddings'
MODEL_DIR = ROOT / 'models' / 'indobert_summarizer'
LOG_DIR   = ROOT / 'logs'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_SIZE = 768  # IndoBERT hidden dim
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────
#  DATASET UNTUK CACHED EMBEDDINGS
# ─────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """Dataset yang memuat embedding yang sudah di-pre-komputasi."""

    def __init__(self, data: list, max_sentences: int = 40):
        self.data = data
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        emb = item['embeddings']   # (N, 768)
        lbl = np.array(item['labels'], dtype=np.float32)

        n_sents = min(emb.shape[0], self.max_sentences)
        emb = emb[:n_sents]
        lbl = lbl[:n_sents]

        return {
            'embeddings': torch.tensor(emb, dtype=torch.float32),  # (N, 768)
            'labels': torch.tensor(lbl, dtype=torch.float32),       # (N,)
            'n_sents': n_sents,
            'sentences': item.get('sentences', [])[:n_sents]
        }


def collate_emb(batch):
    """Collate: pad dokumen ke max kalimat dalam batch."""
    max_sents = max(b['n_sents'] for b in batch)

    emb_list, lbl_list, mask_list = [], [], []
    for b in batch:
        n = b['n_sents']
        pad = max_sents - n
        emb = b['embeddings']   # (N, 768)
        lbl = b['labels']       # (N,)

        if pad > 0:
            emb = torch.cat([emb, torch.zeros(pad, HIDDEN_SIZE)], dim=0)
            lbl = torch.cat([lbl, torch.zeros(pad)], dim=0)

        mask = torch.zeros(max_sents, dtype=torch.bool)
        mask[n:] = True

        emb_list.append(emb)
        lbl_list.append(lbl)
        mask_list.append(mask)

    return {
        'embeddings': torch.stack(emb_list),  # (B, S, 768)
        'labels': torch.stack(lbl_list),       # (B, S)
        'sent_mask': torch.stack(mask_list),   # (B, S) bool
    }


# ─────────────────────────────────────────────────────────────────
#  MODEL RINGAN (tanpa BERT)
# ─────────────────────────────────────────────────────────────────

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SummarizationHead(nn.Module):
    """
    Head untuk extractive summarization.
    Input: pre-computed sentence embeddings (B, S, 768)
    Output: sentence scores (B, S)
    """

    def __init__(self, hidden=768, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(hidden, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=2048, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, embs, sent_mask=None):
        """
        Args:
            embs: (B, S, 768)
            sent_mask: (B, S) bool - True = padding
        Returns:
            scores: (B, S)
        """
        x = self.pos_enc(self.dropout(embs))
        x = self.transformer(x, src_key_padding_mask=sent_mask)
        logits = self.classifier(x).squeeze(-1)  # (B, S)
        return torch.sigmoid(logits)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────
#  WEIGHTED BCE LOSS
# ─────────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=3.0):
        super().__init__()
        self.w = pos_weight

    def forward(self, pred, target):
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        return (-(self.w * target * torch.log(pred) +
                  (1 - target) * torch.log(1 - pred))).mean()


# ─────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, scheduler=None,
              device=DEVICE, desc="Train"):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch in pbar:
            embs  = batch['embeddings'].to(device)
            lbls  = batch['labels'].to(device)
            mask  = batch['sent_mask'].to(device)

            scores = model(embs, mask)

            valid = ~mask
            loss = criterion(scores[valid], lbls[valid])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler: scheduler.step()

            total_loss += loss.item()
            preds = (scores[valid].detach() > 0.5).float().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(lbls[valid].cpu().numpy().tolist())
            pbar.set_postfix(loss=f"{total_loss/(len(all_preds)+1):.4f}")

    avg_loss = total_loss / max(len(loader), 1)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    return {
        'loss': avg_loss,
        'f1':  f1_score(all_labels, all_preds, zero_division=0),
        'prec': precision_score(all_labels, all_preds, zero_division=0),
        'rec':  recall_score(all_labels, all_preds, zero_division=0),
        'acc':  accuracy_score(all_labels, all_preds),
    }


# ─────────────────────────────────────────────────────────────────
#  EVALUASI ROUGE
# ─────────────────────────────────────────────────────────────────

def evaluate_rouge_sample(model, data, n_sample=200, ratio=0.3, device=DEVICE):
    """Evaluasi ROUGE pada sampel kecil dataset."""
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    model.eval()
    r1, r2, rl = [], [], []
    sample = data[:n_sample]

    with torch.no_grad():
        for item in tqdm(sample, desc="ROUGE eval", leave=False):
            sents = item.get('sentences', [])
            if not sents:
                continue
            emb = torch.tensor(item['embeddings'], dtype=torch.float32).unsqueeze(0).to(device)
            scores = model(emb).squeeze(0).cpu().numpy()

            n_select = max(1, int(len(sents) * ratio))
            top_idx = sorted(np.argsort(scores)[-n_select:])
            predicted = ' '.join(sents[i] for i in top_idx if i < len(sents))

            # Buat referensi dari label positif
            ref_idx = [i for i, l in enumerate(item['labels'][:len(sents)]) if l == 1]
            if not ref_idx:
                continue
            reference = ' '.join(sents[i] for i in ref_idx if i < len(sents))

            sc = scorer.score(reference, predicted)
            r1.append(sc['rouge1'].fmeasure)
            r2.append(sc['rouge2'].fmeasure)
            rl.append(sc['rougeL'].fmeasure)

    return {
        'rouge1': np.mean(r1) if r1 else 0,
        'rouge2': np.mean(r2) if r2 else 0,
        'rougeL': np.mean(rl) if rl else 0,
    }


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 65)
    print("  TRAINING RINGAN - IndoBERT Summarizer (Cached Embeddings)")
    print("=" * 65)
    print(f"  Dataset : {args.dataset}")
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print("=" * 65)

    # ── Muat data embedding ──────────────────────────────────────
    print("\n[1/5] Memuat data embedding...")
    splits = {}
    for split in ['train', 'val', 'test']:
        path = EMB_DIR / f"{args.dataset}_{split}_emb.pkl"
        if not path.exists():
            print(f"  [ERROR] File tidak ditemukan: {path}")
            print(f"  Jalankan dulu: python precompute_embeddings.py --dataset {args.dataset}")
            sys.exit(1)
        with open(path, 'rb') as f:
            splits[split] = pickle.load(f)
        print(f"  {split:5s}: {len(splits[split]):,} dokumen")

    if args.quick:
        # Mode demo: gunakan subset kecil
        n = 500
        splits['train'] = splits['train'][:n]
        splits['val']   = splits['val'][:50]
        print(f"\n  [DEMO MODE] Menggunakan {n} sampel training")

    # ── Buat DataLoader ──────────────────────────────────────────
    print("\n[2/5] Membuat DataLoader...")
    train_ds = EmbeddingDataset(splits['train'], max_sentences=args.max_sents)
    val_ds   = EmbeddingDataset(splits['val'],   max_sentences=args.max_sents)
    test_ds  = EmbeddingDataset(splits['test'],  max_sentences=args.max_sents)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_emb, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_emb, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_emb, num_workers=0)
    print(f"  Train: {len(train_loader)} batch | Val: {len(val_loader)} batch")

    # ── Inisialisasi model ───────────────────────────────────────
    print("\n[3/5] Inisialisasi model (Transformer Head saja)...")
    model = SummarizationHead(
        hidden=HIDDEN_SIZE, n_layers=2, n_heads=8, dropout=0.1
    ).to(DEVICE)
    print(f"  Parameter: {model.count_params():,}")

    criterion = WeightedBCELoss(pos_weight=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # ── Training loop ────────────────────────────────────────────
    print(f"\n[4/5] Training {args.epochs} epoch...")
    history = []
    best_f1 = 0.0
    best_path = MODEL_DIR / 'head_best.pt'

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, criterion,
                             optimizer, scheduler, DEVICE,
                             desc=f"Epoch {epoch}/{args.epochs} Train")
        val_m   = run_epoch(model, val_loader, criterion,
                             device=DEVICE, desc=f"Epoch {epoch}/{args.epochs} Val  ")

        elapsed = time.time() - t0
        row = {'epoch': epoch, 'train': train_m, 'val': val_m, 'time': elapsed}
        history.append(row)

        print(f"\n  Epoch {epoch:2d}/{args.epochs} ({elapsed:.0f}s) | "
              f"Train Loss={train_m['loss']:.4f} F1={train_m['f1']:.4f} | "
              f"Val Loss={val_m['loss']:.4f} F1={val_m['f1']:.4f} "
              f"Prec={val_m['prec']:.4f} Rec={val_m['rec']:.4f}")

        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            torch.save({
                'model_state': model.state_dict(),
                'config': {'hidden': HIDDEN_SIZE, 'n_layers': 2, 'n_heads': 8},
                'epoch': epoch,
                'val_f1': best_f1
            }, best_path)
            print(f"  ✅ Best model disimpan (Val F1: {best_f1:.4f})")

    # Simpan history
    log_path = LOG_DIR / f'train_light_{int(time.time())}.json'
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2, default=float)
    print(f"\n  Log disimpan ke: {log_path}")

    # ── Evaluasi ROUGE ───────────────────────────────────────────
    print("\n[5/5] Evaluasi ROUGE pada test set (200 sampel)...")
    # Muat best model
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    rouge = evaluate_rouge_sample(model, splits['test'], n_sample=200)
    print(f"\n  📊 HASIL EVALUASI ROUGE:")
    print(f"  ROUGE-1 : {rouge['rouge1']:.4f}")
    print(f"  ROUGE-2 : {rouge['rouge2']:.4f}")
    print(f"  ROUGE-L : {rouge['rougeL']:.4f}")
    print(f"\n  Best Val F1  : {best_f1:.4f}")

    # Simpan config final
    config_path = MODEL_DIR / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'epochs': args.epochs,
            'best_val_f1': best_f1,
            'rouge': rouge,
            'model_path': str(best_path),
            'type': 'head_only',
            'bert_model': 'indobenchmark/indobert-base-p1'
        }, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  TRAINING SELESAI!")
    print(f"  Model disimpan: {best_path}")
    print(f"  Langkah selanjutnya:")
    print(f"  streamlit run app/streamlit_app.py")
    print("=" * 65)

    return model, history, rouge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='indosum',
                        choices=['indosum', 'combined'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_sents', type=int, default=30)
    parser.add_argument('--quick', action='store_true',
                        help='Demo cepat: 500 sampel, 3 epoch')
    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 3)

    main(args)
