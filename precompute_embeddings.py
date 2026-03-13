"""
precompute_embeddings.py
========================
Pre-komputasi embedding kalimat menggunakan IndoBERT.

Strategi ini:
  1. Jalankan IndoBERT sekali pada semua kalimat → simpan embedding ke disk
  2. Training head (Transformer + Classifier) menggunakan embedding yang sudah disimpan
  3. JAUH lebih cepat untuk CPU karena BERT tidak dijalankan ulang setiap epoch

Estimasi waktu:
  ~ 15-20 menit untuk mengencode 71,345 dokumen IndoSum
  ~ Training head setelah ini: 30-60 menit (5 epoch)

Penggunaan:
    python precompute_embeddings.py --dataset indosum
    python precompute_embeddings.py --dataset combined
"""

import sys
import os
import time
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent
DATA_DIR  = ROOT / 'data' / 'processed'
EMB_DIR   = ROOT / 'data' / 'embeddings'
EMB_DIR.mkdir(parents=True, exist_ok=True)

BERT_MODEL = 'indobenchmark/indobert-base-p1'
MAX_SENT_LEN  = 128
ENCODE_BATCH  = 32   # Kalimat per batch encoding


def mean_pooling(token_embeddings: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling dengan mask padding."""
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode_all_sentences(examples, tokenizer, bert_model, device,
                          max_sent_len: int = 128,
                          batch_size: int = 32) -> list:
    """
    Enkode semua kalimat dalam daftar dokumen.

    Returns:
        List of np.ndarray: embedding tiap dokumen (N_sents, 768)
    """
    bert_model.eval()
    all_doc_embeddings = []

    for example in tqdm(examples, desc="Encoding dokumen", leave=True):
        sentences = example['sentences']
        doc_embeddings = []

        # Proses dalam batch untuk efisiensi
        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i:i + batch_size]
            enc = tokenizer(
                batch_sents,
                padding='max_length',
                truncation=True,
                max_length=max_sent_len,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            
            output = bert_model(**enc)
            embs = mean_pooling(
                output.last_hidden_state,
                enc['attention_mask']
            )  # (batch, 768)
            doc_embeddings.append(embs.cpu().numpy())

        # Gabungkan semua batch kalimat dalam dokumen
        doc_emb = np.concatenate(doc_embeddings, axis=0)  # (N_sents, 768)
        all_doc_embeddings.append(doc_emb)

    return all_doc_embeddings


def precompute_split(split_name: str, dataset_prefix: str, args):
    """Pre-komputasi embedding untuk satu split."""
    out_path = EMB_DIR / f"{dataset_prefix}_{split_name}_emb.pkl"

    if out_path.exists() and not args.overwrite:
        print(f"  [Skip] {out_path.name} sudah ada (--overwrite untuk menimpa)")
        return True

    data_path = DATA_DIR / f"{dataset_prefix}_{split_name}.pkl"
    if not data_path.exists():
        print(f"  [Error] Data tidak ditemukan: {data_path}")
        return False

    with open(data_path, 'rb') as f:
        examples = pickle.load(f)

    if args.max_samples and split_name == 'train':
        examples = examples[:args.max_samples]
        print(f"  Membatasi training ke {len(examples):,} sampel")

    print(f"  Encoding {len(examples):,} dokumen untuk split={split_name}...")
    t0 = time.time()

    embeddings = encode_all_sentences(
        examples, tokenizer, bert_model,
        device=args.device,
        max_sent_len=MAX_SENT_LEN,
        batch_size=ENCODE_BATCH
    )

    # Simpan embeddings bersama labels
    data_to_save = []
    for ex, emb in zip(examples, embeddings):
        data_to_save.append({
            'id': ex.get('id', ''),
            'embeddings': emb,     # (N_sents, 768) float32
            'labels': ex['labels'][:emb.shape[0]],
            'sentences': ex['sentences'][:emb.shape[0]],
            'n_sents': emb.shape[0]
        })

    with open(out_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    elapsed = time.time() - t0
    print(f"  ✅ Disimpan: {out_path.name} ({len(data_to_save):,} dok) "
          f"dalam {elapsed/60:.1f} menit")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='indosum',
                        choices=['indosum', 'combined'])
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'val', 'test'])
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Batasi jumlah sampel training (None = semua)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print("=" * 65)
    print("  PRE-KOMPUTASI EMBEDDING IndoBERT")
    print("=" * 65)
    print(f"  Dataset : {args.dataset}")
    print(f"  Splits  : {args.splits}")
    print(f"  Device  : {args.device}")
    print(f"  Output  : {EMB_DIR}")
    print("=" * 65)

    # Muat tokenizer dan model BERT (frozen / read-only)
    print(f"\nMemuat {BERT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    bert_model = AutoModel.from_pretrained(BERT_MODEL)
    bert_model.eval()
    bert_model.to(args.device)

    # Hitung total parameter
    n_params = sum(p.numel() for p in bert_model.parameters())
    print(f"BERT parameter: {n_params/1e6:.1f}M (frozen - hanya forward, tanpa gradient)")

    t_total = time.time()
    for split in args.splits:
        print(f"\n[{split.upper()}]")
        precompute_split(split, args.dataset, args)

    print(f"\n{'='*65}")
    print(f"Semua selesai dalam {(time.time()-t_total)/60:.1f} menit")
    print(f"\nLangkah selanjutnya:")
    print(f"  python train_light.py --dataset {args.dataset}")
    print("=" * 65)
