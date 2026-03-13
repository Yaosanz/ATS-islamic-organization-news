"""
prepare_data.py
===============
Script untuk menyiapkan dataset training.
Jalankan SEKALI sebelum training untuk menyimpan data terproses.

Penggunaan:
    # Mode cepat - hanya IndoSum (~10 menit)
    python prepare_data.py --mode fast

    # Mode sedang - IndoSum + 5000 sample Ormas (~30 menit)
    python prepare_data.py --mode medium

    # Mode lengkap - IndoSum + semua Ormas (~2-3 jam)
    python prepare_data.py --mode full
"""

import sys
import os
import pickle
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocess import (
    load_indosum,
    load_ormas_csv,
    split_and_save,
    get_statistics
)

# ─────────────────────────────────────────────────────────────────
#  KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
INDOSUM_DIR  = ROOT / 'indosum'
ORMAS_CSV    = ROOT / 'ormas_liputan6.csv'
OUTPUT_DIR   = ROOT / 'data' / 'processed'


def prepare_indosum_only(output_dir: str) -> int:
    """
    Siapkan dataset IndoSum saja.
    Sudah memiliki extractive labels → paling cepat.
    """
    print("\n📂 Mode: IndoSum only")
    all_examples = []

    for split in ['train', 'dev', 'test']:
        examples = load_indosum(str(INDOSUM_DIR), split=split)
        all_examples.extend(examples)
        print(f"   {split:5s}: {len(examples):,} dokumen")

    print(f"\n   Total: {len(all_examples):,} dokumen")

    # Simpan - gunakan dev sebagai val, test sebagai test, train sebagai train
    train_ex = load_indosum(str(INDOSUM_DIR), split='train')
    val_ex   = load_indosum(str(INDOSUM_DIR), split='dev')
    test_ex  = load_indosum(str(INDOSUM_DIR), split='test')

    os.makedirs(output_dir, exist_ok=True)
    for name, data in [('train', train_ex), ('val', val_ex), ('test', test_ex)]:
        path = Path(output_dir) / f"indosum_{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"   ✅ Disimpan: {path.name} ({len(data):,} dokumen)")

    return len(train_ex) + len(val_ex) + len(test_ex)


def prepare_combined(output_dir: str, ormas_samples: int = 5000) -> int:
    """
    Siapkan dataset gabungan IndoSum + Ormas Liputan6.
    """
    print(f"\n📂 Mode: IndoSum + Ormas (maks {ormas_samples:,} sampel)")

    # Muat IndoSum
    print("\n[1/3] Memuat IndoSum...")
    indosum_train = load_indosum(str(INDOSUM_DIR), split='train')
    indosum_val   = load_indosum(str(INDOSUM_DIR), split='dev')
    indosum_test  = load_indosum(str(INDOSUM_DIR), split='test')

    # Muat Ormas CSV
    print(f"\n[2/3] Memuat Ormas CSV ({ormas_samples:,} sampel)...")
    ormas_all = load_ormas_csv(
        str(ORMAS_CSV),
        max_samples=ormas_samples,
        max_select=3,
        save_cache=True,
        cache_dir=str(output_dir)
    )

    # Gabung dan split Ormas
    from sklearn.model_selection import train_test_split
    import random
    random.seed(42)
    random.shuffle(ormas_all)

    n_val  = max(100, int(len(ormas_all) * 0.1))
    n_test = max(100, int(len(ormas_all) * 0.1))
    ormas_test     = ormas_all[:n_test]
    ormas_val      = ormas_all[n_test:n_test + n_val]
    ormas_train    = ormas_all[n_test + n_val:]

    # Gabungkan
    combined_train = indosum_train + ormas_train
    combined_val   = indosum_val   + ormas_val
    combined_test  = indosum_test  + ormas_test

    random.shuffle(combined_train)
    print(f"\n[3/3] Menyimpan dataset...")

    os.makedirs(output_dir, exist_ok=True)
    for name, data in [
        ('train', combined_train),
        ('val',   combined_val),
        ('test',  combined_test)
    ]:
        path = Path(output_dir) / f"combined_{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"   ✅ {path.name} → {len(data):,} dokumen")

    return len(combined_train) + len(combined_val) + len(combined_test)


def print_statistics(output_dir: str, prefix: str):
    """Tampilkan statistik dataset yang sudah disiapkan."""
    from src.preprocess import get_statistics
    print(f"\n📊 Statistik dataset '{prefix}':")
    for split in ['train', 'val', 'test']:
        path = Path(output_dir) / f"{prefix}_{split}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            stats = get_statistics(data)
            print(f"\n  [{split.upper()}] {stats['total_docs']:,} dokumen")
            print(f"    Rata-rata kalimat/dok    : {stats['avg_sentences']:.1f}")
            print(f"    Rata-rata label positif  : {stats['avg_positive_labels']:.1f}")
            print(f"    Rasio kalimat positif    : {stats['positive_ratio']*100:.1f}%")
            print(f"    Rata-rata panjang kalimat: {stats['avg_sentence_len_words']:.1f} kata")


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Persiapan data training')
    parser.add_argument(
        '--mode', default='fast',
        choices=['fast', 'medium', 'full'],
        help=(
            'fast   = IndoSum saja (~10 menit)\n'
            'medium = IndoSum + 5000 Ormas (~30 menit)\n'
            'full   = IndoSum + 15000 Ormas (~90 menit)'
        )
    )
    parser.add_argument('--output', default=str(OUTPUT_DIR),
                        help='Direktori output')
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 65)
    print("  PERSIAPAN DATA - IndoBERT Summarizer")
    print("=" * 65)
    print(f"  Mode   : {args.mode}")
    print(f"  Output : {args.output}")
    print("=" * 65)

    if args.mode == 'fast':
        total = prepare_indosum_only(args.output)
        dataset_prefix = 'indosum'
    elif args.mode == 'medium':
        total = prepare_combined(args.output, ormas_samples=5000)
        dataset_prefix = 'combined'
    else:  # full
        total = prepare_combined(args.output, ormas_samples=15000)
        dataset_prefix = 'combined'

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  ✅ Selesai dalam {elapsed/60:.1f} menit")
    print(f"  Total dokumen: {total:,}")
    print_statistics(args.output, dataset_prefix)
    print(f"\n  Langkah selanjutnya:")
    if dataset_prefix == 'indosum':
        print(f"  python src/train.py --dataset indosum --epochs1 5 --epochs2 3")
    else:
        print(f"  python src/train.py --dataset combined --epochs1 5 --epochs2 3")
    print("=" * 65)
