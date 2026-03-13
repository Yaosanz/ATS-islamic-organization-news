"""
streamlit_app.py
================
Dashboard UI untuk Peringkasan Otomatis Artikel Ormas Islam
menggunakan IndoBERT

Jalankan dengan:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import torch
import numpy as np
import streamlit as st
from pathlib import Path

# Tambah root path supaya bisa import modul src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────
#  KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Peringkas Artikel Ormas Islam - IndoBERT",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Sistem Peringkasan Otomatis Artikel Ormas Islam\n"
                 "menggunakan IndoBERT\n\n"
                 "Skripsi - NLP & Automatic Text Summarization"
    }
)

# ─────────────────────────────────────────────────────────────────
#  CSS KUSTOM
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(26,35,126,0.3);
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.5rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Kartu statistik */
    .stat-card {
        background: white;
        border-left: 4px solid #3f51b5;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 0.3rem 0;
    }
    .stat-card .label { font-size: 0.75rem; color: #666; text-transform: uppercase; }
    .stat-card .value { font-size: 1.4rem; font-weight: 700; color: #1a237e; }

    /* Badge kalimat terpilih */
    .selected-badge {
        background: #e8f5e9;
        border-left: 3px solid #2e7d32;
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        margin: 0.4rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Area ringkasan */
    .summary-box {
        background: #f3f4ff;
        border: 1px solid #c5cae9;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        font-size: 1rem;
        line-height: 1.8;
        color: #1a1a2e;
    }

    /* Progress bar ROUGE */
    .rouge-bar { margin: 0.3rem 0; }
    .rouge-label { font-size: 0.8rem; color: #555; margin-bottom: 2px; }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        padding: 1rem 0 0.5rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }

    /* Sembunyikan menu Streamlit default */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Tombol */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white;
        border: none;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #283593, #5c6bc0);
        box-shadow: 0 4px 12px rgba(63,81,181,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  MUAT MODEL (cached)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline(model_dir: str):
    """Muat pipeline dan cache. Raises exception agar tidak disimpan saat gagal."""
    from src.summarize import auto_load_pipeline
    # Resolve ke absolute path jika relatif
    p = Path(model_dir)
    if not p.is_absolute():
        p = ROOT / model_dir
    return auto_load_pipeline(str(p))


def load_pipeline_safe(model_dir: str):
    """Wrapper dengan error handling. Saat gagal tidak di-cache → retry otomatis."""
    try:
        return load_pipeline(model_dir), None
    except Exception as e:
        return None, str(e)


def get_available_model_dir() -> str | None:
    """Cari direktori model yang tersedia."""
    candidates = [
        ROOT / 'models' / 'indobert_summarizer',
        ROOT / 'models' / 'checkpoints',
    ]
    for c in candidates:
        pts = list(c.glob('*.pt')) if c.exists() else []
        if pts:
            return str(c)
    return None


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render sidebar dengan pengaturan."""
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")
        st.markdown("---")

        # Pengaturan model
        st.markdown("### 🤖 Model")
        model_dir_input = st.text_input(
            "Direktori Model",
            value=get_available_model_dir() or "models/indobert_summarizer",
            help="Path ke folder berisi file model .pt yang sudah dilatih"
        )

        device_info = "GPU ✅" if torch.cuda.is_available() else "CPU ⚠️"
        st.info(f"Device: {device_info}")

        # Status model
        _mp = Path(model_dir_input)
        if not _mp.is_absolute():
            _mp = ROOT / model_dir_input
        _pts = list(_mp.glob('*.pt')) if _mp.exists() else []
        if _pts:
            st.success(f"✅ Model tersedia: `{_pts[0].name}`")
        else:
            st.error("❌ Model belum ada — jalankan training dulu")

        if st.button("🔄 Reload Model Setelah Training",
                     help="Tekan setelah local_demo_train.py selesai"):
            load_pipeline.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Parameter Ringkasan")

        ratio = st.slider(
            "Rasio Ringkasan",
            min_value=0.1, max_value=0.7, value=0.3, step=0.05,
            help="Proporsi kalimat yang dipilih (0.3 = 30% kalimat terpenting)"
        )
        max_sents = st.number_input(
            "Maks. Kalimat Ringkasan",
            min_value=1, max_value=10, value=5,
            help="Batas atas jumlah kalimat dalam ringkasan"
        )
        min_sents = st.number_input(
            "Min. Kalimat Ringkasan",
            min_value=1, max_value=5, value=2,
            help="Batas bawah jumlah kalimat dalam ringkasan"
        )

        st.markdown("---")
        st.markdown("### ℹ️ Tentang")
        st.markdown("""
        **Metode:** IndoBERT Extractive Summarization

        **Model:** `indobenchmark/indobert-base-p1`

        **Dataset:** IndoSum + Ormas Liputan6

        **Metrik Evaluasi:** ROUGE-1, ROUGE-2, ROUGE-L
        """)

    return model_dir_input, ratio, int(max_sents), int(min_sents)


# ─────────────────────────────────────────────────────────────────
#  KOMPONEN UI
# ─────────────────────────────────────────────────────────────────

def render_stats(stats: dict):
    """Tampilkan statistik ringkasan dalam 4 kolom."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kalimat Asli", stats.get('original_sentences', 0))
    c2.metric("Kalimat Dipilih", stats.get('selected_sentences', 0))
    cr = stats.get('compression_ratio', 0)
    c3.metric("Kompresi", f"{cr*100:.0f}%")
    orig_w = stats.get('original_words', 0)
    summ_w = stats.get('summary_words', 0)
    c4.metric("Kata (asli→ringkas)", f"{orig_w}→{summ_w}")


def render_highlighted_text(sentences, scores, selected_idxs):
    """Tampilkan teks dengan highlight kalimat terpilih."""
    from app.utils import format_highlight_html
    html = format_highlight_html(sentences, scores, selected_idxs)
    st.markdown(html, unsafe_allow_html=True)


def render_score_chart(sentences, scores, selected_idxs):
    """Tampilkan bar chart skor kalimat."""
    import plotly.graph_objects as go

    selected_set = set(selected_idxs)
    n = len(sentences)
    display_n = min(n, 25)  # Batasi tampilan

    labels = [f"K{i+1}" for i in range(display_n)]
    values = scores[:display_n]
    colors = ['#2e7d32' if i in selected_set else '#90a4ae' for i in range(display_n)]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition='outside',
        textfont_size=10
    ))
    fig.update_layout(
        title="Skor Relevansi Per Kalimat",
        xaxis_title="Kalimat",
        yaxis_title="Skor (0-1)",
        yaxis_range=[0, 1.15],
        height=280,
        margin=dict(l=20, r=20, t=40, b=30),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="Threshold 0.5")
    return fig


# ─────────────────────────────────────────────────────────────────
#  HALAMAN UTAMA
# ─────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📜 Peringkasan Otomatis Artikel Ormas Islam</h1>
        <p>Menggunakan Algoritma IndoBERT · Natural Language Processing · Automatic Text Summarization</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    model_dir, ratio, max_sents, min_sents = render_sidebar()

    # ── Tab utama ──────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📝 Ringkasan Artikel", "📁 Upload File", "📊 Tentang Penelitian"])

    # ════════════════════════════════════════════════════════════
    #  TAB 1: INPUT TEKS
    # ════════════════════════════════════════════════════════════
    with tab1:
        col_input, col_output = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("### 📄 Masukkan Artikel")

            # Contoh artikel Ormas Islam
            SAMPLE_TEXT = """Liputan6.com, Jakarta - Nahdlatul Ulama (NU) kembali menegaskan komitmennya sebagai organisasi Islam terbesar di Indonesia untuk selalu berkontribusi dalam pembangunan bangsa. Ketua Umum PBNU mengungkapkan hal tersebut dalam acara Harlah ke-99 NU yang digelar di Jakarta.

Dalam sambutannya, Ketua PBNU menekankan pentingnya peran ulama dan santri dalam memperkuat persatuan nasional di tengah berbagai tantangan global. NU selama hampir satu abad telah membuktikan diri sebagai garda terdepan dalam menjaga keutuhan NKRI dan Pancasila.

Sementara itu, Muhammadiyah juga menyatakan dukungannya terhadap program moderasi beragama yang dijalankan pemerintah. Ketua PP Muhammadiyah menegaskan bahwa Islam rahmatan lil alamin harus diwujudkan dalam kehidupan berbangsa dan bernegara.

Kedua organisasi Islam terbesar di Indonesia ini sepakat bahwa toleransi dan kerukunan antarumat beragama merupakan fondasi utama bagi kemajuan Indonesia. Mereka berkomitmen untuk terus bekerja sama dalam mencegah radikalisme dan ekstremisme yang mengancam persatuan bangsa.

Menteri Agama turut mengapresiasi kontribusi NU dan Muhammadiyah dalam memperkuat moderasi beragama di Indonesia. Pemerintah berharap kedua ormas ini terus menjadi pilar utama dalam mewujudkan Indonesia yang damai, adil, dan makmur."""

            use_sample = st.checkbox("Gunakan contoh artikel Ormas Islam", value=False)
            article_text = st.text_area(
                "Teks Artikel",
                value=SAMPLE_TEXT if use_sample else "",
                height=350,
                placeholder="Salin dan tempel artikel tentang Ormas Islam di sini...",
                label_visibility="collapsed"
            )

            ref_summary = st.text_area(
                "Ringkasan Referensi (opsional, untuk menghitung ROUGE)",
                height=80,
                placeholder="Masukkan ringkasan referensi untuk evaluasi ROUGE (opsional)..."
            )

            run_btn = st.button("🚀 Hasilkan Ringkasan", type="primary")

        # ── Proses ringkasan ──────────────────────────────────
        with col_output:
            st.markdown("### 📋 Hasil Ringkasan")

            if run_btn:
                if not article_text.strip():
                    st.warning("⚠️ Masukkan artikel terlebih dahulu.")
                else:
                    with st.spinner("⏳ Memproses artikel dengan IndoBERT..."):
                        # Coba muat model
                        pipeline, err = load_pipeline_safe(model_dir)

                        if pipeline is not None:
                            # Gunakan model IndoBERT
                            result = pipeline.summarize(
                                article_text,
                                ratio=ratio,
                                min_sentences=min_sents,
                                max_sentences=max_sents
                            )
                            model_used = "IndoBERT (Fine-tuned)"
                        else:
                            # Fallback: TF-IDF scoring
                            result = tfidf_summarize(
                                article_text, ratio, min_sents, max_sents
                            )
                            model_used = "TF-IDF (Fallback)"

                    # ── Tampilkan ringkasan ──
                        if model_used.startswith("TF-IDF"):
                            st.info(
                                "ℹ️ Menggunakan **TF-IDF Fallback** karena model IndoBERT "
                                "belum tersedia. Untuk hasil akurat, selesaikan training "
                                "lalu klik **🔄 Reload Model Setelah Training** di sidebar."
                            )
                        else:
                            st.success(f"✅ Ringkasan berhasil dibuat ({model_used})")
                    st.markdown(
                        f'<div class="summary-box">{result["summary"]}</div>',
                        unsafe_allow_html=True
                    )

                    # ── Statistik ──
                    st.markdown("---")
                    st.markdown("#### 📊 Statistik")
                    render_stats(result.get('stats', {}))

                    # ── Skor ROUGE jika ada referensi ──
                    if ref_summary.strip():
                        st.markdown("---")
                        st.markdown("#### 🎯 Evaluasi ROUGE")
                        try:
                            from app.utils import compute_simple_rouge
                            rouge = compute_simple_rouge(result['summary'], ref_summary)
                            rc1, rc2 = st.columns(2)
                            with rc1:
                                st.metric("ROUGE-1 F1", f"{rouge['rouge1']:.4f}")
                            with rc2:
                                st.metric("ROUGE-2 F1", f"{rouge['rouge2']:.4f}")
                        except Exception as e:
                            st.error(f"Gagal hitung ROUGE: {e}")

                    # Simpan hasil di session_state untuk tab visualisasi
                    st.session_state['last_result'] = result

            # ── Visualisasi (jika sudah ada hasil) ──
            if 'last_result' in st.session_state:
                result = st.session_state['last_result']
                sentences = result.get('sentences', [])
                scores = result.get('scores', [])
                selected = result.get('selected_idxs', [])

                if sentences and scores:
                    with st.expander("🔍 Visualisasi Skor Per Kalimat", expanded=False):
                        import plotly.graph_objects as go
                        fig = render_score_chart(sentences, scores, selected)
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("📖 Teks dengan Highlight", expanded=False):
                        st.caption(
                            "🟢 Kalimat Terpilih | Angka di depan kalimat = skor relevansi"
                        )
                        render_highlighted_text(sentences, scores, selected)

                    with st.expander("📜 Daftar Semua Kalimat & Skor", expanded=False):
                        import pandas as pd
                        df_sents = pd.DataFrame([
                            {
                                'No': i + 1,
                                'Kalimat': s[:120] + ('...' if len(s) > 120 else ''),
                                'Skor': f"{scores[i]:.4f}" if i < len(scores) else '-',
                                'Dipilih': '✅' if i in set(selected) else ''
                            }
                            for i, s in enumerate(sentences)
                        ])
                        st.dataframe(df_sents, use_container_width=True, height=300)

    # ════════════════════════════════════════════════════════════
    #  TAB 2: UPLOAD FILE
    # ════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📁 Upload File Artikel")
        uploaded_file = st.file_uploader(
            "Upload file .txt atau .csv (kolom 'content')",
            type=['txt', 'csv'],
            help="File .txt: teks artikel langsung. File .csv: harus ada kolom 'content'"
        )

        if uploaded_file is not None:
            import pandas as pd
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'content' not in df.columns:
                    st.error("CSV harus memiliki kolom 'content'")
                else:
                    st.success(f"✅ {len(df)} artikel dimuat")
                    selected_idx = st.selectbox(
                        "Pilih artikel:",
                        range(len(df)),
                        format_func=lambda i: (
                            df.iloc[i].get('title', f'Artikel #{i+1}')[:80]
                            if 'title' in df.columns else f'Artikel #{i+1}'
                        )
                    )
                    article_text = str(df.iloc[selected_idx]['content'])
                    ref = str(df.iloc[selected_idx].get('summary', '')) \
                        if 'summary' in df.columns else ''
                    st.text_area("Preview Artikel", article_text[:1000] + "...", height=200)
            else:
                article_text = uploaded_file.read().decode('utf-8')
                ref = ''
                st.text_area("Isi File", article_text[:1000] + "...", height=200)

            if st.button("🚀 Ringkas Artikel Ini", type="primary"):
                with st.spinner("Memproses..."):
                    pipeline, err = load_pipeline_safe(model_dir)
                    if pipeline:
                        result = pipeline.summarize(article_text, ratio=ratio,
                                                     min_sentences=min_sents,
                                                     max_sentences=max_sents)
                    else:
                        result = tfidf_summarize(article_text, ratio, min_sents, max_sents)

                st.markdown("#### Ringkasan:")
                st.markdown(
                    f'<div class="summary-box">{result["summary"]}</div>',
                    unsafe_allow_html=True
                )

                # Download button
                st.download_button(
                    "⬇️ Unduh Ringkasan (.txt)",
                    data=result['summary'],
                    file_name="ringkasan.txt",
                    mime="text/plain"
                )

    # ════════════════════════════════════════════════════════════
    #  TAB 3: TENTANG PENELITIAN
    # ════════════════════════════════════════════════════════════
    with tab3:
        col_a, col_b = st.columns([1, 1], gap="large")

        with col_a:
            st.markdown("""
            ## 📚 Tentang Penelitian

            ### Judul
            **Peringkasan Otomatis Artikel Tentang Ormas Islam di Media Online
            Menggunakan Algoritma IndoBERT**

            ### Tujuan Penelitian
            1. Mengimplementasikan algoritma **IndoBERT** untuk peringkasan otomatis
               artikel Ormas Islam dari media online Liputan6.com
            2. Merancang dan membangun **antarmuka pengguna** sederhana yang
               menampilkan hasil peringkasan secara interaktif
            3. Menganalisis hasil peringkasan menggunakan metrik **ROUGE** untuk
               mengukur kualitas ringkasan yang dihasilkan

            ### Metode
            | Komponen | Detail |
            |----------|--------|
            | Model Dasar | `indobenchmark/indobert-base-p1` |
            | Pendekatan | Extractive Summarization |
            | Arsitektur | BertSum + Inter-Sentence Transformer |
            | Dataset Training | IndoSum + Ormas Liputan6 |
            | Evaluasi | ROUGE-1, ROUGE-2, ROUGE-L |

            ### Arsitektur Model
            ```
            Artikel
              ↓
            Split Kalimat
              ↓
            IndoBERT Encoder (per kalimat)
              ↓
            Mean Pooling → Sentence Embeddings
              ↓
            Positional Encoding
              ↓
            Inter-Sentence Transformer (2 layer)
              ↓
            Linear Classifier → Skor Relevansi
              ↓
            Top-K Selection → Ringkasan
            ```
            """)

        with col_b:
            st.markdown("""
            ### 📊 Dataset

            | Dataset | Jumlah | Keterangan |
            |---------|--------|------------|
            | Ormas Liputan6 | ~35.000 | Artikel Ormas Islam |
            | IndoSum | ~20.000 | Extractive summarization |
            | Liputan6 Canonical | ~200.000 | Artikel umum |

            ### 🔧 Teknologi
            - **Python** 3.9+
            - **PyTorch** 2.0
            - **HuggingFace Transformers** 4.35
            - **Streamlit** 1.28 (UI)
            - **ROUGE Score** (evaluasi)

            ### 📈 Alur Penelitian
            1. **Pengumpulan Data** — Scraping & dataset Liputan6
            2. **Preprocessing** — Membersihkan teks, split kalimat,
               generate label ekstraktif (greedy oracle)
            3. **Training** — Fine-tune IndoBERT 2 tahap
               (frozen BERT → full fine-tune)
            4. **Evaluasi** — ROUGE-1, ROUGE-2, ROUGE-L,
               Precision, Recall, F1
            5. **Deployment** — Streamlit dashboard

            ### 📖 Referensi
            - Liu & Lapata (2019). *Text Summarization with Pretrained Encoders*
            - Wilie et al. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian NLP*
            - Kurniawan & Louvan (2018). *IndoSum: A New Benchmark for Indonesian Automatic Text Summarization*
            """)

    # Footer
    st.markdown("""
    <div class="footer">
        Sistem Peringkasan Otomatis Artikel Ormas Islam · IndoBERT · Skripsi NLP
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  FALLBACK: TF-IDF SUMMARIZER
# ─────────────────────────────────────────────────────────────────

def tfidf_summarize(text: str, ratio: float, min_sents: int, max_sents: int) -> dict:
    """
    Fallback summarizer berbasis TF-IDF ketika model belum dilatih.
    """
    import re
    from collections import Counter

    def clean(t):
        t = re.sub(r'<[^>]+>', ' ', t)
        t = re.sub(r'https?://\S+', '', t)
        t = re.sub(r'\s+', ' ', t)
        return t.strip()

    def split_sentences(t):
        sents = re.split(r'(?<=[.!?])\s+', t)
        return [s.strip() for s in sents if len(s.strip()) >= 20]

    clean_text = clean(text)
    sentences = split_sentences(clean_text)
    if not sentences:
        return {'summary': text[:300], 'sentences': [text], 'scores': [1.0],
                'selected_idxs': [0], 'stats': {}}

    # Hitung TF-IDF sederhana
    stopwords = set(['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan',
                     'untuk', 'adalah', 'dalam', 'tidak', 'akan', 'pada', 'juga',
                     'sebagai', 'atau', 'oleh', 'dapat', 'saat', 'telah', 'sudah',
                     'karena', 'lebih', 'ada', 'bahwa', 'kita', 'mereka', 'ia'])

    def tokenize(s):
        return [w.lower() for w in re.findall(r'\b\w+\b', s) if w.lower() not in stopwords]

    all_words = [w for s in sentences for w in tokenize(s)]
    word_freq = Counter(all_words)
    max_freq = max(word_freq.values()) if word_freq else 1

    scores = []
    for sent in sentences:
        words = tokenize(sent)
        score = sum(word_freq.get(w, 0) / max_freq for w in words) / max(len(words), 1)
        scores.append(score)

    # Pilih kalimat
    n = len(sentences)
    k = max(min_sents, min(max_sents, int(n * ratio)))
    top_k = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    selected_idxs = sorted(top_k)
    summary = ' '.join(sentences[i] for i in selected_idxs)

    orig_words = len(clean_text.split())
    summ_words = len(summary.split())

    return {
        'summary': summary,
        'sentences': sentences,
        'scores': scores,
        'selected_idxs': selected_idxs,
        'stats': {
            'original_sentences': n,
            'selected_sentences': len(selected_idxs),
            'original_words': orig_words,
            'summary_words': summ_words,
            'compression_ratio': 1 - (summ_words / max(orig_words, 1)),
            'avg_score': float(sum(scores) / max(len(scores), 1)),
            'max_score': float(max(scores)) if scores else 0.0,
        }
    }


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
