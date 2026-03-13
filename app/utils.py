"""
utils.py
========
Utilitas pendukung untuk aplikasi Streamlit.
"""
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter


def detect_device() -> str:
    """Deteksi device terbaik."""
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU"


def get_model_dir() -> Optional[str]:
    """Cari direktori model yang tersedia (pakai absolute path)."""
    _root = Path(__file__).resolve().parent.parent  # d:\Joki
    candidates = [
        _root / 'models' / 'indobert_summarizer',
        _root / 'models' / 'checkpoints',
    ]
    for c in candidates:
        if c.exists() and any(c.glob('*.pt')):
            return str(c)
    return None


def format_highlight_html(
    sentences: List[str],
    scores: List[float],
    selected_idxs: List[int]
) -> str:
    """
    Buat HTML dengan kalimat yang di-highlight berdasarkan skor.

    Args:
        sentences: Daftar kalimat
        scores: Skor relevansi per kalimat
        selected_idxs: Indeks kalimat terpilih

    Returns:
        HTML string siap render
    """
    selected_set = set(selected_idxs)
    max_score = max(scores) if scores else 1.0

    html_parts = []
    for i, sent in enumerate(sentences):
        score = scores[i] if i < len(scores) else 0.0
        normalized = score / max(max_score, 1e-9)

        if i in selected_set:
            # Kalimat terpilih: highlight kuning-hijau
            intensity = int(100 + normalized * 155)
            bg_color = f"rgb(255, {intensity}, 100)"
            border = "2px solid #2e7d32"
            font_weight = "bold"
            badge = f'<span style="background:#2e7d32;color:white;border-radius:4px;padding:1px 5px;font-size:0.7em;margin-right:4px;">✓ {score:.2f}</span>'
        else:
            # Kalimat tidak terpilih: opacity berdasarkan skor
            opacity = max(0.4, normalized * 0.6 + 0.2)
            bg_color = "transparent"
            border = "none"
            font_weight = "normal"
            badge = f'<span style="color:#aaa;font-size:0.7em;margin-right:4px;">{score:.2f}</span>'

        html_parts.append(
            f'<span style="background:{bg_color};border:{border};'
            f'font-weight:{font_weight};padding:3px 5px;border-radius:4px;'
            f'display:inline;line-height:2.0;">'
            f'{badge}{sent}</span> '
        )

    return '<div style="line-height:2.2;text-align:justify;">' + ''.join(html_parts) + '</div>'


def compute_simple_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """
    Hitung ROUGE sederhana tanpa library eksternal.

    Args:
        hypothesis: Teks yang dihasilkan
        reference: Teks referensi

    Returns:
        Dict berisi ROUGE-1 dan ROUGE-2 F1
    """
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return {'rouge1': 0.0, 'rouge2': 0.0}

    # ROUGE-1 dengan Counter (multiset) agar lebih akurat
    hyp_ctr = Counter(hyp_tokens)
    ref_ctr = Counter(ref_tokens)
    overlap_1 = sum((hyp_ctr & ref_ctr).values())
    p1 = overlap_1 / len(hyp_tokens) if hyp_tokens else 0
    r1 = overlap_1 / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0

    # ROUGE-2 dengan Counter (multiset)
    hyp_bg = list(zip(hyp_tokens, hyp_tokens[1:]))
    ref_bg = list(zip(ref_tokens, ref_tokens[1:]))
    hyp_bg_ctr = Counter(hyp_bg)
    ref_bg_ctr = Counter(ref_bg)
    overlap_2 = sum((hyp_bg_ctr & ref_bg_ctr).values())
    p2 = overlap_2 / len(hyp_bg) if hyp_bg else 0
    r2 = overlap_2 / len(ref_bg) if ref_bg else 0
    f2 = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else 0

    return {'rouge1': f1, 'rouge2': f2}


def truncate_text(text: str, max_words: int = 100) -> str:
    """Potong teks ke N kata pertama."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + ' ...'
