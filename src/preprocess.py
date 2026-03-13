"""
preprocess.py
=============
Modul preprocessing data untuk proyek skripsi:
  Peringkasan Otomatis Artikel Ormas Islam menggunakan IndoBERT

Fungsi:
    - Membersihkan teks artikel
    - Memecah artikel menjadi kalimat
    - Memuat dataset IndoSum (JSONL)
    - Memuat dataset Ormas Liputan6 (CSV)
    - Membuat label ekstraktif dari ringkasan abstraktif (greedy oracle)
    - Menyimpan data terproses
"""

import json
import re
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ── Dependensi opsional ───────────────────────────────────────────
try:
    from rouge_score import rouge_scorer as rs
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("[WARN] rouge-score tidak ditemukan. Install dengan: pip install rouge-score")

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _factory = StemmerFactory()
    _stemmer = _factory.create_stemmer()
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
#  TEXT CLEANING
# ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Bersihkan teks artikel dari HTML, URL, dan whitespace berlebih.

    Args:
        text: Teks mentah

    Returns:
        Teks bersih
    """
    if not isinstance(text, str):
        return ""
    # Hapus HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Hapus URL
    text = re.sub(r'https?://\S+', '', text)
    # Hapus karakter khusus berlebihan tapi pertahankan punctuation penting
    text = re.sub(r'[^\w\s.,!?;:()\-\'"//]', ' ', text)
    # Hapus baris iklan umum Liputan6
    text = re.sub(r'Advertisement', '', text)
    text = re.sub(r'BACA JUGA:.*?\n', '', text)
    # Normalisasi whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text: str, min_len: int = 20) -> List[str]:
    """
    Pecah teks menjadi daftar kalimat.

    Args:
        text: Teks artikel
        min_len: Panjang minimal kalimat (karakter) agar diikutsertakan

    Returns:
        Daftar kalimat
    """
    # Pisah pada akhir kalimat → titik, !, ?
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sentences:
        # Juga pecah berdasarkan newline
        for sub in s.split('\n'):
            sub = sub.strip()
            if len(sub) >= min_len:
                result.append(sub)
    return result


# ─────────────────────────────────────────────────────────────────
#  GREEDY ORACLE (Label Ekstraktif dari Ringkasan Abstraktif)
# ─────────────────────────────────────────────────────────────────

def greedy_extractive_labels(
    sentences: List[str],
    summary: str,
    max_select: int = 3,
    scorer=None
) -> List[int]:
    """
    Algoritma greedy oracle untuk menghasilkan label ekstraktif.

    Strategi: Secara iteratif, pilih kalimat yang paling meningkatkan
    skor ROUGE gabungan (R1 + R2 + RL) terhadap ringkasan referensi.

    Referensi: Liu & Lapata (2019) "Text Summarization with Pretrained Encoders"

    Args:
        sentences: Daftar kalimat dari artikel
        summary: Ringkasan referensi (abstraktif)
        max_select: Jumlah maksimal kalimat yang dipilih
        scorer: rouge_scorer instance (opsional, dibuat jika None)

    Returns:
        Daftar label biner (0/1) sepanjang daftar kalimat
    """
    if not ROUGE_AVAILABLE:
        raise ImportError("rouge-score diperlukan. Install: pip install rouge-score")
    if scorer is None:
        scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    labels = [0] * len(sentences)
    selected_idxs = []

    for _ in range(min(max_select, len(sentences))):
        best_combined = -1.0
        best_idx = -1

        for i, sent in enumerate(sentences):
            if labels[i] == 1:
                continue
            candidate_idxs = sorted(selected_idxs + [i])
            candidate = ' '.join(sentences[j] for j in candidate_idxs)
            scores = scorer.score(summary, candidate)
            combined = (
                scores['rouge1'].fmeasure +
                scores['rouge2'].fmeasure +
                scores['rougeL'].fmeasure
            ) / 3.0
            if combined > best_combined:
                best_combined = combined
                best_idx = i

        if best_idx >= 0:
            labels[best_idx] = 1
            selected_idxs.append(best_idx)

    return labels


# ─────────────────────────────────────────────────────────────────
#  LOAD INDOSUM
# ─────────────────────────────────────────────────────────────────

def load_indosum(indosum_dir: str, split: str = 'train') -> List[Dict]:
    """
    Muat dataset IndoSum dari file JSONL.

    Format IndoSum:
        paragraphs: [ [ [token,...], [token,...] ], ... ]
        gold_labels: [ [bool, bool], ... ]

    Args:
        indosum_dir: Path ke folder indosum/
        split: 'train', 'dev', atau 'test'

    Returns:
        List of dict: {id, sentences, labels, category}
    """
    indosum_dir = Path(indosum_dir)
    examples = []

    for part in range(1, 6):
        filepath = indosum_dir / f"{split}.0{part}.jsonl"
        if not filepath.exists():
            print(f"[WARN] File tidak ditemukan: {filepath}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Memuat IndoSum {split}.0{part}", leave=False):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sentences = []
                labels = []

                for para_idx, paragraph in enumerate(data.get('paragraphs', [])):
                    para_labels = data.get('gold_labels', [[]])[para_idx] \
                        if para_idx < len(data.get('gold_labels', [])) else []

                    for sent_idx, token_list in enumerate(paragraph):
                        sent_text = ' '.join(token_list).strip()
                        if len(sent_text) < 10:
                            continue
                        sentences.append(sent_text)
                        if sent_idx < len(para_labels):
                            labels.append(1 if para_labels[sent_idx] else 0)
                        else:
                            labels.append(0)

                if len(sentences) >= 2 and sum(labels) > 0:
                    examples.append({
                        'id': data.get('id', ''),
                        'category': data.get('category', ''),
                        'sentences': sentences,
                        'labels': labels,
                        'source': 'indosum'
                    })

    print(f"[INFO] IndoSum '{split}': {len(examples)} dokumen dimuat")
    return examples


# ─────────────────────────────────────────────────────────────────
#  LOAD ORMAS LIPUTAN6 CSV
# ─────────────────────────────────────────────────────────────────

def load_ormas_csv(
    csv_path: str,
    max_samples: Optional[int] = None,
    max_select: int = 3,
    save_cache: bool = True,
    cache_dir: str = 'data/processed'
) -> List[Dict]:
    """
    Muat dataset Ormas Liputan6 dari CSV dan hasilkan label ekstraktif.

    Args:
        csv_path: Path ke ormas_liputan6.csv
        max_samples: Batasi jumlah sampel (None = semua)
        max_select: Jumlah kalimat ringkasan yang dipilih
        save_cache: Simpan hasil ke cache pickle
        cache_dir: Direktori untuk menyimpan cache

    Returns:
        List of dict: {id, sentences, labels, title, summary, source}
    """
    cache_path = Path(cache_dir) / f"ormas_processed_{max_samples or 'all'}.pkl"

    # Cek cache
    if cache_path.exists():
        print(f"[INFO] Memuat dari cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"[INFO] Memuat {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['content', 'summary'])
    df = df[df['content'].str.len() > 100]
    df = df[df['summary'].str.len() > 20]

    if max_samples is not None:
        df = df.sample(min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    print(f"[INFO] Total artikel: {len(df)}")

    scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False) \
        if ROUGE_AVAILABLE else None

    examples = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Ormas CSV"):
        content = clean_text(str(row['content']))
        summary = clean_text(str(row['summary']))

        sentences = split_sentences(content)
        if len(sentences) < 2:
            skipped += 1
            continue

        # Buat label ekstraktif
        if ROUGE_AVAILABLE and scorer:
            labels = greedy_extractive_labels(sentences, summary, max_select, scorer)
        else:
            # Fallback: ambil kalimat pertama dan terakhir
            labels = [0] * len(sentences)
            labels[0] = 1
            if len(sentences) > 1:
                labels[-1] = 1

        if sum(labels) == 0:
            skipped += 1
            continue

        examples.append({
            'id': str(row.get('url', idx)),
            'title': str(row.get('title', '')),
            'summary': summary,
            'sentences': sentences[:50],  # Batasi 50 kalimat per dokumen
            'labels': labels[:50],
            'date': str(row.get('date', '')),
            'source': 'ormas'
        })

    print(f"[INFO] Ormas: {len(examples)} dokumen | {skipped} dilewati")

    # Simpan cache
    if save_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(examples, f)
        print(f"[INFO] Cache disimpan ke: {cache_path}")

    return examples


# ─────────────────────────────────────────────────────────────────
#  LOAD LIPUTAN6 CANONICAL JSON
# ─────────────────────────────────────────────────────────────────

def load_liputan6_canonical(data_dir: str, split: str = 'train',
                             max_samples: Optional[int] = None) -> List[Dict]:
    """
    Muat dataset Liputan6 canonical dari folder JSON.

    Args:
        data_dir: Path ke liputan6_data/canonical/
        split: 'train', 'dev', atau 'test'
        max_samples: Batasi jumlah sampel

    Returns:
        List of dict: {id, sentences, labels, clean_summary}
    """
    data_dir = Path(data_dir) / split
    if not data_dir.exists():
        print(f"[WARN] Direktori tidak ditemukan: {data_dir}")
        return []

    json_files = sorted(data_dir.glob("*.json"))
    if max_samples:
        json_files = json_files[:max_samples]

    examples = []
    scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False) \
        if ROUGE_AVAILABLE else None

    for jf in tqdm(json_files, desc=f"Memuat Liputan6 canonical {split}"):
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # clean_article berupa list of sentences (tokens dipisahkan spasi)
        raw_sents = data.get('clean_article', [])
        sentences = [' '.join(s) if isinstance(s, list) else str(s)
                     for s in raw_sents]
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 15]

        raw_summ = data.get('clean_summary', [])
        summary = ' '.join([' '.join(s) if isinstance(s, list) else str(s)
                            for s in raw_summ])

        extractive_idxs = data.get('extractive_summary', [])

        if extractive_idxs:
            # Gunakan label dari anotasi
            labels = [0] * len(sentences)
            for idx in extractive_idxs:
                if idx < len(labels):
                    labels[idx] = 1
        elif ROUGE_AVAILABLE and scorer and summary:
            labels = greedy_extractive_labels(sentences, summary, scorer=scorer)
        else:
            if not sentences:
                continue
            labels = [0] * len(sentences)
            labels[0] = 1

        if len(sentences) >= 2 and sum(labels) > 0:
            examples.append({
                'id': str(data.get('id', jf.stem)),
                'sentences': sentences[:50],
                'labels': labels[:50],
                'clean_summary': summary,
                'source': 'liputan6_canonical'
            })

    print(f"[INFO] Liputan6 canonical '{split}': {len(examples)} dokumen")
    return examples


# ─────────────────────────────────────────────────────────────────
#  SPLIT & SIMPAN DATASET
# ─────────────────────────────────────────────────────────────────

def split_and_save(
    examples: List[Dict],
    output_dir: str,
    name: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Bagi dataset menjadi train/val/test dan simpan ke file pickle.

    Args:
        examples: Daftar contoh terproses
        output_dir: Direktori output
        name: Nama prefix file
        val_ratio: Proporsi data validasi
        test_ratio: Proporsi data test
        seed: Random seed
    """
    os.makedirs(output_dir, exist_ok=True)

    train_val, test = train_test_split(examples, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_ratio / (1 - test_ratio), random_state=seed)

    splits = {'train': train, 'val': val, 'test': test}
    for split_name, data in splits.items():
        path = Path(output_dir) / f"{name}_{split_name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[INFO] {split_name}: {len(data)} sampel → {path}")

    return splits


def get_statistics(examples: List[Dict]) -> Dict:
    """Hitung statistik dataset."""
    if not examples:
        return {}

    num_sents = [len(e['sentences']) for e in examples]
    num_positives = [sum(e['labels']) for e in examples]
    sent_lens = [len(s.split()) for e in examples for s in e['sentences']]

    return {
        'total_docs': len(examples),
        'avg_sentences': np.mean(num_sents),
        'max_sentences': np.max(num_sents),
        'min_sentences': np.min(num_sents),
        'avg_positive_labels': np.mean(num_positives),
        'avg_sentence_len_words': np.mean(sent_lens),
        'positive_ratio': np.mean([l for e in examples for l in e['labels']])
    }


# ─────────────────────────────────────────────────────────────────
#  MAIN – jalankan langsung untuk preprocess semua data
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    BASE = Path(__file__).parent.parent
    OUT = BASE / 'data' / 'processed'

    print("=" * 60)
    print("  PREPROCESSING DATA - IndoBERT Skripsi")
    print("=" * 60)

    # 1. IndoSum
    print("\n[1/3] Memuat IndoSum ...")
    indosum_train = load_indosum(BASE / 'indosum', split='train')
    indosum_dev = load_indosum(BASE / 'indosum', split='dev')
    indosum_test = load_indosum(BASE / 'indosum', split='test')

    # Gabungkan dan simpan
    all_indosum = indosum_train + indosum_dev
    split_and_save(all_indosum, OUT, 'indosum',
                   val_ratio=0.1, test_ratio=0.1)

    stats = get_statistics(all_indosum)
    print(f"  Statistik IndoSum: {stats}")

    # 2. Ormas (ambil 5000 untuk domain adaptation)
    print("\n[2/3] Memuat Ormas Liputan6 CSV ...")
    ormas_data = load_ormas_csv(
        BASE / 'ormas_liputan6.csv',
        max_samples=5000,
        cache_dir=str(OUT)
    )
    split_and_save(ormas_data, OUT, 'ormas',
                   val_ratio=0.1, test_ratio=0.1)

    stats = get_statistics(ormas_data)
    print(f"  Statistik Ormas: {stats}")

    # 3. Combined
    print("\n[3/3] Menggabungkan semua dataset ...")
    combined = indosum_train + ormas_data
    split_and_save(combined, OUT, 'combined',
                   val_ratio=0.1, test_ratio=0.1)
    print(f"\n✅ Preprocessing selesai! Data tersimpan di: {OUT}")
