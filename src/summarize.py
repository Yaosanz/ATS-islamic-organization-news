"""
summarize.py
============
Modul inferensi: menghasilkan ringkasan dari artikel menggunakan
model IndoBERT yang sudah dilatih.

Fungsi utama:
    summarize(text, model, tokenizer, ratio=0.3) -> str
    summarize_with_highlights(text, ...) -> { summary, sentences, scores }
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from src.preprocess import clean_text, split_sentences
from src.model import IndoBERTSumExtractor, load_model


# ─────────────────────────────────────────────────────────────────
#  PIPELINE RINGKASAN
# ─────────────────────────────────────────────────────────────────

class SummarizationPipeline:
    """
    Pipeline lengkap untuk peringkasan artikel.

    Contoh Penggunaan:
        pipe = SummarizationPipeline.from_pretrained('models/indobert_summarizer')
        result = pipe.summarize(artikel, ratio=0.3)
        print(result['summary'])
    """

    def __init__(
        self,
        model: IndoBERTSumExtractor,
        tokenizer,
        device: Optional[torch.device] = None,
        max_sent_len: int = 128,
        max_sentences: int = 40
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cpu')
        self.max_sent_len = max_sent_len
        self.max_sentences = max_sentences

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: Optional[str] = None
    ) -> 'SummarizationPipeline':
        """
        Muat pipeline dari direktori model.

        Args:
            model_dir: Direktori berisi model_best.pt dan vocab/tokenizer
            device: 'cuda', 'cpu', atau None (auto-detect)
        """
        import json
        from transformers import AutoTokenizer

        model_path = Path(model_dir)

        # Deteksi device
        if device is None:
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            _device = torch.device(device)

        # Muat konfigurasi
        config_path = model_path / 'config.json'
        bert_name = 'indobenchmark/indobert-base-p1'
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            bert_name = cfg.get('bert_model_name', bert_name)

        # Cari file model
        model_files = list(model_path.glob('model_best*.pt'))
        if not model_files:
            model_files = list(model_path.glob('model*.pt'))
        if not model_files:
            raise FileNotFoundError(f"File model tidak ditemukan di {model_path}")

        # Pilih model terbaru
        model_path = sorted(model_files)[-1]
        print(f"[INFO] Memuat model: {model_path}")

        model = load_model(str(model_path), bert_name, str(_device))

        # Muat tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except Exception:
            print(f"[WARN] Tokenizer tidak ditemukan di {model_path}, "
                  f"memuat dari HuggingFace: {bert_name}")
            tokenizer = AutoTokenizer.from_pretrained(bert_name)

        return cls(model, tokenizer, _device)

    def get_sentence_scores(
        self,
        sentences: List[str]
    ) -> np.ndarray:
        """
        Dapatkan skor relevansi untuk setiap kalimat.

        Args:
            sentences: Daftar kalimat artikel

        Returns:
            Array skor (0-1) per kalimat
        """
        sents = sentences[:self.max_sentences]

        encodings = self.tokenizer(
            sents,
            padding='max_length',
            truncation=True,
            max_length=self.max_sent_len,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = encodings['attention_mask'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores = self.model(input_ids, attention_mask)  # (1, N)
        scores = scores[0].cpu().numpy()

        # Pad kembali ke panjang asli
        full_scores = np.zeros(len(sentences))
        full_scores[:len(sents)] = scores[:len(sents)]
        return full_scores

    def summarize(
        self,
        text: str,
        ratio: float = 0.3,
        min_sentences: int = 1,
        max_sentences: int = 5,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Hasilkan ringkasan dari teks artikel.

        Args:
            text: Teks artikel lengkap
            ratio: Proporsi kalimat yang dipilih (0.0 - 1.0)
            min_sentences: Minimal kalimat dalam ringkasan
            max_sentences: Maksimal kalimat dalam ringkasan
            threshold: Threshold skor (jika None, gunakan ratio)

        Returns:
            Dict berisi:
                summary: Teks ringkasan
                sentences: Daftar semua kalimat
                scores: Skor per kalimat
                selected_idxs: Indeks kalimat yang dipilih
                stats: Statistik ringkasan
        """
        # Bersihkan dan pecah teks
        clean = clean_text(text)
        sentences = split_sentences(clean)

        if not sentences:
            return {
                'summary': text[:500] + '...' if len(text) > 500 else text,
                'sentences': [text],
                'scores': [1.0],
                'selected_idxs': [0],
                'stats': {}
            }

        # Dapatkan skor
        scores = self.get_sentence_scores(sentences)

        # Pilih kalimat
        n = len(sentences)

        if threshold is not None:
            # Pilih berdasarkan threshold
            selected_idxs = [i for i, s in enumerate(scores) if s >= threshold]
            if not selected_idxs:
                selected_idxs = [int(np.argmax(scores))]
        else:
            # Pilih berdasarkan ratio
            k = max(min_sentences, min(max_sentences, int(n * ratio)))
            top_k = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
            selected_idxs = sorted(top_k)

        summary = ' '.join(sentences[i] for i in selected_idxs)

        # Hitung statistik
        original_words = len(clean.split())
        summary_words = len(summary.split())
        compression_ratio = 1 - (summary_words / max(original_words, 1))

        return {
            'summary': summary,
            'sentences': sentences,
            'scores': scores.tolist(),
            'selected_idxs': selected_idxs,
            'stats': {
                'original_sentences': n,
                'selected_sentences': len(selected_idxs),
                'original_words': original_words,
                'summary_words': summary_words,
                'compression_ratio': compression_ratio,
                'avg_score': float(np.mean(scores)),
                'max_score': float(np.max(scores)),
            }
        }


# ─────────────────────────────────────────────────────────────────
#  FUNGSI STANDALONE
# ─────────────────────────────────────────────────────────────────

def summarize_text(
    text: str,
    model_dir: str = 'models/indobert_summarizer',
    ratio: float = 0.3,
    device: Optional[str] = None
) -> str:
    """
    Fungsi singkat untuk meringkas teks.

    Args:
        text: Teks artikel
        model_dir: Direktori model
        ratio: Proporsi kalimat
        device: Device ('cpu' atau 'cuda')

    Returns:
        Teks ringkasan
    """
    pipe = SummarizationPipeline.from_pretrained(model_dir, device)
    result = pipe.summarize(text, ratio=ratio)
    return result['summary']


# ─────────────────────────────────────────────────────────────────
#  PIPELINE RINGAN (head-only model dari train_light.py)
# ─────────────────────────────────────────────────────────────────

class HeadOnlySummarizationPipeline:
    """
    Pipeline inferensi untuk model head-only (dari train_light.py).
    Memerlukan IndoBERT untuk encoding + SummarizationHead yang sudah dilatih.
    """

    def __init__(self, bert_model, tokenizer, head_model, device=None,
                 max_sent_len=128, max_sentences=40):
        self.bert      = bert_model
        self.tokenizer = tokenizer
        self.head      = head_model
        self.device    = device or torch.device('cpu')
        self.max_sent_len = max_sent_len
        self.max_sentences = max_sentences
        self.bert.to(self.device).eval()
        self.head.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str,
                        device=None) -> 'HeadOnlySummarizationPipeline':
        """Muat dari file checkpoint (head_best.pt)."""
        import importlib.util
        from pathlib import Path as _Path
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            _dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            _dev = torch.device(device)

        cp = _Path(checkpoint_path)
        if cp.is_dir():
            pts = list(cp.glob('head*.pt'))
            if not pts:
                raise FileNotFoundError(f"Tidak ada head*.pt di {cp}")
            cp = sorted(pts)[-1]

        print(f"[INFO] Memuat head model: {cp}")
        ckpt = torch.load(cp, map_location=_dev, weights_only=False)
        config = ckpt.get('config', {'hidden': 768, 'n_layers': 2, 'n_heads': 8})

        # Import SummarizationHead dari train_light.py
        tl_path = _Path(__file__).resolve().parent.parent / 'train_light.py'
        spec = importlib.util.spec_from_file_location("train_light", str(tl_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Gagal memuat modul train_light.py dari {tl_path}")
        tl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tl)

        head = tl.SummarizationHead(
            hidden=config.get('hidden', 768),
            n_layers=config.get('n_layers', 2),
            n_heads=config.get('n_heads', 8)
        )
        head.load_state_dict(ckpt['model_state'])
        head.eval()

        BERT_NAME = 'indobenchmark/indobert-base-p1'
        print(f"[INFO] Memuat BERT tokenizer + model: {BERT_NAME}")
        tokenizer  = AutoTokenizer.from_pretrained(BERT_NAME)
        bert_model = AutoModel.from_pretrained(BERT_NAME)

        return cls(bert_model, tokenizer, head, _dev)

    @torch.no_grad()
    def encode_sentences(self, sentences: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            sentences, padding='max_length', truncation=True,
            max_length=self.max_sent_len, return_tensors='pt'
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.bert(**enc)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(1e-9)

    def get_sentence_scores(self, sentences: List[str]) -> np.ndarray:
        sents = sentences[:self.max_sentences]
        with torch.no_grad():
            embs   = self.encode_sentences(sents).unsqueeze(0)   # (1, N, 768)
            scores = self.head(embs).squeeze(0).detach().cpu().numpy()  # (N,)
        full = np.zeros(len(sentences))
        full[:len(sents)] = scores[:len(sents)]
        return full

    def summarize(self, text: str, ratio: float = 0.3,
                  min_sentences: int = 1, max_sentences: int = 5,
                  threshold: Optional[float] = None) -> Dict:
        clean = clean_text(text)
        sentences = split_sentences(clean)
        if not sentences:
            return {'summary': text, 'sentences': [text], 'scores': [1.0],
                    'selected_idxs': [0], 'stats': {}}
        scores = self.get_sentence_scores(sentences)
        n = len(sentences)
        if threshold is not None:
            selected_idxs = ([i for i, s in enumerate(scores) if s >= threshold]
                             or [int(np.argmax(scores))])
        else:
            k = max(min_sentences, min(max_sentences, int(n * ratio)))
            selected_idxs = sorted(
                sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
            )
        summary = ' '.join(sentences[i] for i in selected_idxs)
        orig_words = len(clean.split())
        summ_words = len(summary.split())
        return {
            'summary': summary, 'sentences': sentences,
            'scores': scores.tolist(), 'selected_idxs': selected_idxs,
            'stats': {
                'original_sentences': n, 'selected_sentences': len(selected_idxs),
                'original_words': orig_words, 'summary_words': summ_words,
                'compression_ratio': 1 - summ_words / max(orig_words, 1),
                'avg_score': float(np.mean(scores)), 'max_score': float(np.max(scores)),
            }
        }


def auto_load_pipeline(model_dir: str, device: Optional[str] = None):
    """
    Muat pipeline yang sesuai berdasarkan file yang tersedia.
    Prioritas: head_best.pt (ringan) → model_best.pt (full model).
    """
    from pathlib import Path as _Path
    d = _Path(model_dir)
    if not d.exists():
        raise FileNotFoundError(f"Direktori tidak ditemukan: {model_dir}")
    if list(d.glob('head*.pt')):
        return HeadOnlySummarizationPipeline.from_checkpoint(str(d), device)
    if list(d.glob('model*.pt')):
        return SummarizationPipeline.from_pretrained(str(d), device)
    raise FileNotFoundError(
        f"Tidak ada file model di {model_dir}.\n"
        "Jalankan: python precompute_embeddings.py && python train_light.py"
    )



def highlight_sentences(
    sentences: List[str],
    scores: List[float],
    selected_idxs: List[int]
) -> List[Dict]:
    """
    Siapkan data untuk menampilkan teks dengan highlight.

    Args:
        sentences: Daftar kalimat
        scores: Skor per kalimat
        selected_idxs: Indeks kalimat terpilih

    Returns:
        Daftar dict: {text, score, selected, rank}
    """
    selected_set = set(selected_idxs)
    ranked_idxs = sorted(range(len(sentences)),
                         key=lambda i: scores[i] if i < len(scores) else 0,
                         reverse=True)
    rank_map = {idx: rank + 1 for rank, idx in enumerate(ranked_idxs)}

    return [
        {
            'text': sentences[i],
            'score': float(scores[i]) if i < len(scores) else 0.0,
            'selected': i in selected_set,
            'rank': rank_map.get(i, len(sentences))
        }
        for i in range(len(sentences))
    ]


if __name__ == '__main__':
    # Test dengan teks dummy
    sample = """
    Liputan6.com, Jakarta - Nahdlatul Ulama (NU) merupakan organisasi Islam
    terbesar di Indonesia yang didirikan pada tahun 1926. Organisasi ini memiliki
    lebih dari 90 juta anggota yang tersebar di seluruh nusantara.

    Ketua Umum Pengurus Besar Nahdlatul Ulama (PBNU) menegaskan komitmen
    organisasi untuk terus berkontribusi dalam pembangunan bangsa dan negara.
    NU selalu mendukung program pemerintah yang bertujuan mensejahterakan rakyat.

    Selain NU, Muhammadiyah juga merupakan organisasi Islam besar di Indonesia
    yang didirikan oleh KH Ahmad Dahlan pada tahun 1912 di Yogyakarta.
    Muhammadiyah fokus pada bidang pendidikan, kesehatan, dan sosial.

    Kedua organisasi ini menjadi pilar penting demokrasi dan toleransi
    di Indonesia serta berperan dalam menjaga keutuhan NKRI.
    """
    print("Artikel:")
    print(sample.strip())
    print("\n" + "─" * 50)
    print("Mencoba summarize dengan mode fallback (ROUGE-based) ...")

    # Test tanpa model (gunakan ROUGE-based selection sebagai fallback)
    from src.preprocess import split_sentences, clean_text
    sents = split_sentences(clean_text(sample))
    print(f"\nKalimat artikel ({len(sents)}):")
    for i, s in enumerate(sents):
        print(f"  [{i}] {s}")
    print("\n(Untuk inferensi model penuh, muat model terlatih terlebih dahulu)")
