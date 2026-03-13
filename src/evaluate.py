"""
evaluate.py
===========
Evaluasi model IndoBERT Summarizer menggunakan metrik ROUGE.

Metrik:
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest Common Subsequence
  - Precision, Recall, F1 per kalimat (klasifikasi biner)
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)

try:
    from rouge_score import rouge_scorer as rs
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
#  METRIK KLASIFIKASI (SENTENCE LEVEL)
# ─────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Hitung metrik klasifikasi biner per kalimat.

    Args:
        y_true: Label ground truth (0/1)
        y_pred: Prediksi model (0/1)

    Returns:
        Dict berisi accuracy, precision, recall, f1
    """
    if len(y_true) == 0:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────────
#  METRIK ROUGE (DOCUMENT LEVEL)
# ─────────────────────────────────────────────────────────────────

def compute_rouge(
    hypothesis: str,
    reference: str,
    scorer=None
) -> Dict[str, float]:
    """
    Hitung ROUGE untuk satu pasang ringkasan & referensi.

    Args:
        hypothesis: Teks ringkasan yang dihasilkan
        reference: Teks ringkasan referensi
        scorer: RougeScorer instance (dibuat baru jika None)

    Returns:
        Dict: {'rouge1': {...}, 'rouge2': {...}, 'rougeL': {...}}
    """
    if not ROUGE_AVAILABLE:
        raise ImportError("rouge-score tidak tersedia. Install: pip install rouge-score")
    if scorer is None:
        scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1_p': scores['rouge1'].precision,
        'rouge1_r': scores['rouge1'].recall,
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_p': scores['rouge2'].precision,
        'rouge2_r': scores['rouge2'].recall,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_p': scores['rougeL'].precision,
        'rougeL_r': scores['rougeL'].recall,
        'rougeL_f': scores['rougeL'].fmeasure,
    }


def compute_rouge_corpus(
    hypotheses: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Hitung rata-rata ROUGE untuk korpus dokumen.

    Args:
        hypotheses: Daftar teks ringkasan yang dihasilkan
        references: Daftar teks ringkasan referensi

    Returns:
        Dict berisi rata-rata semua metrik ROUGE
    """
    if not ROUGE_AVAILABLE:
        raise ImportError("rouge-score tidak tersedia.")

    scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    all_scores = []

    for hyp, ref in zip(hypotheses, references):
        if not hyp.strip() or not ref.strip():
            continue
        scores = compute_rouge(hyp, ref, scorer)
        all_scores.append(scores)

    if not all_scores:
        return {}

    avg = {}
    for key in all_scores[0]:
        avg[key] = float(np.mean([s[key] for s in all_scores]))

    return avg


# ─────────────────────────────────────────────────────────────────
#  EVALUASI ROUGE DARI SKOR MODEL
# ─────────────────────────────────────────────────────────────────

def scores_to_summary(
    sentences: List[str],
    scores: List[float],
    ratio: float = 0.3,
    min_sentences: int = 1,
    max_sentences: int = 5
) -> str:
    """
    Konversi skor kalimat menjadi teks ringkasan.

    Args:
        sentences: Daftar kalimat dokumen
        scores: Skor relevansi per kalimat (0-1)
        ratio: Proporsi kalimat yang dipilih
        min_sentences: Min kalimat dalam ringkasan
        max_sentences: Max kalimat dalam ringkasan

    Returns:
        Ringkasan sebagai string
    """
    n = len(sentences)
    k = max(min_sentences, min(max_sentences, int(n * ratio)))

    # Pilih k kalimat dengan skor tertinggi
    top_idxs = sorted(
        range(n),
        key=lambda i: scores[i] if i < len(scores) else 0,
        reverse=True
    )[:k]

    # Urutkan kembali berdasarkan posisi asli
    top_idxs_sorted = sorted(top_idxs)

    return ' '.join(sentences[i] for i in top_idxs_sorted)


def evaluate_rouge_batch(
    model,
    examples: List[Dict],
    tokenizer,
    device: torch.device,
    max_sent_len: int = 128,
    max_sentences: int = 40,
    ratio: float = 0.3,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Evaluasi ROUGE model pada dataset.

    Args:
        model: IndoBERTSumExtractor
        examples: Daftar contoh dengan 'sentences', 'labels', dan opsional 'summary'
        tokenizer: Tokenizer IndoBERT
        device: Device
        max_sent_len: Max token per kalimat
        max_sentences: Max kalimat per dokumen
        ratio: Proporsi kalimat dipilih
        batch_size: Batch size evaluasi

    Returns:
        Dict berisi rata-rata ROUGE
    """
    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():
        for example in examples:
            sents = example['sentences'][:max_sentences]
            labels = example['labels'][:max_sentences]

            if not sents:
                continue

            # Tokenisasi
            encodings = tokenizer(
                sents,
                padding='max_length',
                truncation=True,
                max_length=max_sent_len,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].unsqueeze(0).to(device)
            attention_mask = encodings['attention_mask'].unsqueeze(0).to(device)

            # Prediksi
            scores = model(input_ids, attention_mask)  # (1, N)
            scores = scores[0].cpu().numpy().tolist()

            # Hasilkan ringkasan dari prediksi
            hypothesis = scores_to_summary(sents, scores, ratio)

            # Hasilkan referensi dari gold labels
            gold_idxs = [i for i, l in enumerate(labels) if l == 1]
            if gold_idxs:
                reference = ' '.join(sents[i] for i in sorted(gold_idxs))
            elif 'summary' in example:
                reference = example['summary']
            else:
                continue

            hypotheses.append(hypothesis)
            references.append(reference)

    if not hypotheses:
        return {}

    rouge_scores = compute_rouge_corpus(hypotheses, references)
    print(f"\n  ROUGE-1 F: {rouge_scores.get('rouge1_f', 0):.4f} | "
          f"ROUGE-2 F: {rouge_scores.get('rouge2_f', 0):.4f} | "
          f"ROUGE-L F: {rouge_scores.get('rougeL_f', 0):.4f}")
    return rouge_scores


# ─────────────────────────────────────────────────────────────────
#  LAPORAN EVALUASI LENGKAP
# ─────────────────────────────────────────────────────────────────

def full_evaluation_report(
    model,
    test_data: List[Dict],
    tokenizer,
    device: torch.device,
    save_path: Optional[str] = None
) -> Dict:
    """
    Buat laporan evaluasi lengkap (ROUGE + klasifikasi).

    Args:
        model: Model terlatih
        test_data: Data test
        tokenizer: Tokenizer
        device: Device
        save_path: Path untuk menyimpan laporan (opsional)

    Returns:
        Dict berisi semua metrik
    """
    print("\n" + "=" * 55)
    print("  LAPORAN EVALUASI LENGKAP")
    print("=" * 55)

    # 1. Metrik ROUGE
    print("\n[1/2] Menghitung ROUGE scores ...")
    rouge_scores = evaluate_rouge_batch(
        model, test_data, tokenizer, device
    )

    # 2. Metrik klasifikasi
    print("\n[2/2] Menghitung metrik klasifikasi kalimat ...")
    all_preds = []
    all_labels = []
    model.eval()

    from src.dataset import create_dataloader
    test_loader = create_dataloader(
        test_data, tokenizer, batch_size=4, shuffle=False
    )

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            sent_mask = batch['sent_mask'].to(device)

            scores = model(input_ids, attention_mask, sent_mask)
            valid_mask = ~sent_mask

            preds = (scores[valid_mask] > 0.5).float().cpu().numpy()
            lbls = labels[valid_mask].cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.tolist())

    clf_metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds)
    )

    # Gabungkan semua metrik
    report = {**rouge_scores, **clf_metrics}

    # Tampilkan
    print("\n  ── Metrik ROUGE ──")
    for k in ['rouge1_f', 'rouge2_f', 'rougeL_f']:
        print(f"    {k:20s}: {report.get(k, 0):.4f}")
    print("\n  ── Metrik Klasifikasi ──")
    for k in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"    {k:20s}: {report.get(k, 0):.4f}")

    # Simpan laporan
    if save_path:
        import json, os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Laporan disimpan ke: {save_path}")

    return report


if __name__ == '__main__':
    # Test fungsi ROUGE
    hyp = "IndoBERT adalah model bahasa Indonesia berbasis BERT yang kuat."
    ref = "IndoBERT merupakan model BERT untuk bahasa Indonesia."
    scores = compute_rouge(hyp, ref)
    print("ROUGE scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
