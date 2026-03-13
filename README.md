# 📜 Peringkasan Otomatis Artikel Ormas Islam

## Menggunakan Algoritma IndoBERT

> **Skripsi** — Natural Language Processing · Automatic Text Summarization · IndoBERT

---

## 📋 Deskripsi

Sistem peringkasan otomatis artikel tentang Organisasi Kemasyarakatan (Ormas) Islam dari media online Liputan6.com menggunakan model **IndoBERT** (_indobenchmark/indobert-base-p1_) dengan pendekatan **Extractive Text Summarization**.

### Tujuan Penelitian

1. Mengimplementasikan IndoBERT untuk peringkasan otomatis artikel Ormas Islam
2. Merancang antarmuka pengguna berbasis Streamlit
3. Menganalisis hasil peringkasan menggunakan metrik ROUGE

---

## 🏗️ Struktur Proyek

```
d:\Joki\
├── 🐍 prepare_data.py              ← Siapkan data (pickle)
├── 🧮 precompute_embeddings.py     ← Pre-komputasi embedding IndoBERT
├── ⚡ train_light.py               ← Training head-only (CPU-friendly)
├── 🚀 local_demo_train.py          ← Demo training lokal cepat
├── 📓 notebooks/
│   └── skripsi_indobert.ipynb    ← Notebook training (seperti Google Colab)
├── 🐍 src/
│   ├── __init__.py
│   ├── preprocess.py             ← Preprocessing & label generation
│   ├── dataset.py                ← PyTorch Dataset & DataLoader
│   ├── model.py                  ← Arsitektur IndoBERT Summarizer
│   ├── train.py                  ← Script training
│   ├── evaluate.py               ← Evaluasi ROUGE
│   └── summarize.py              ← Inferensi & pipeline
├── 🌐 app/
│   ├── streamlit_app.py          ← Dashboard UI Streamlit
│   └── utils.py                  ← Utilitas UI
├── 📊 data/
│   ├── processed/                ← Data terproses (otomatis dibuat)
│   ├── embeddings/               ← Cache embedding (otomatis dibuat)
│   └── raw/                      ← Data mentah
├── 🤖 models/
│   └── indobert_summarizer/      ← Model terlatih (otomatis dibuat)
├── 📈 results/                   ← Grafik & laporan evaluasi
├── 📋 logs/                      ← Log training
├── ormas_liputan6.csv            ← Dataset utama Ormas Islam
├── indosum/                      ← Dataset IndoSum (JSONL)
├── liputan6_data/                ← Dataset Liputan6 (JSON)
├── requirements.txt
└── setup_env.bat                 ← Script setup Windows
```

---

## ⚡ Quick Start

### 1. Setup Lingkungan

```bash
# Clone atau buka folder proyek di VS Code
# Kemudian jalankan setup otomatis:
setup_env.bat
```

Atau manual:

```bash
# Buat virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt

# Register kernel Jupyter
python -m ipykernel install --user --name=indobert_skripsi --display-name "Python (IndoBERT Skripsi)"
```

### 2. Training (Cara 1: Notebook — Disarankan)

1. Buka VS Code
2. Buka file `notebooks/skripsi_indobert.ipynb`
3. Pilih kernel: **Python (IndoBERT Skripsi)**
4. Jalankan sel satu per satu dengan **Shift+Enter**

### 3. Training (Cara 2: Command Line)

```bash
# Aktifkan environment
venv\Scripts\activate

# Step 1: Siapkan data IndoSum (cepat)
python prepare_data.py --mode fast

# Step 2A (disarankan di CPU): training demo cepat (~35 menit)
python local_demo_train.py

# atau Step 2B: training head-only dengan embedding cache
python precompute_embeddings.py --dataset indosum
python train_light.py --dataset indosum --epochs 5

# Step 2C (untuk GPU): training end-to-end
python src/train.py --dataset combined --epochs1 5 --epochs2 3 --batch_size 4

# Step 3: Evaluasi
python src/evaluate.py
```

### 4. Jalankan Streamlit App

```bash
# Aktifkan environment dulu
venv\Scripts\activate

# Jalankan dashboard
streamlit run app/streamlit_app.py
```

Buka browser di: **http://localhost:8501**

---

## 🤖 Arsitektur Model

```
Artikel Teks
    ↓
Split Kalimat → [K1] [K2] [K3] ... [Kn]
    ↓               ↓      ↓           ↓
             IndoBERT (shared weights)
                   (indobert-base-p1)
    ↓               ↓      ↓           ↓
             Mean Pooling [CLS token]
    ↓
Positional Encoding (sinusoidal)
    ↓
Inter-Sentence Transformer (2 layer, 8 head)
    ↓
Linear Classifier → Sigmoid
    ↓
Skor Relevansi: [0.8, 0.2, 0.9, 0.1, ...]
    ↓
Top-K Selection (misal: 30% kalimat)
    ↓
Ringkasan Ekstraktif
```

### Parameter Model

| Komponen                    | Detail                           |
| --------------------------- | -------------------------------- |
| Base Model                  | `indobenchmark/indobert-base-p1` |
| Hidden Size                 | 768                              |
| Max Token/Kalimat           | 128                              |
| Max Kalimat/Dokumen         | 40                               |
| Inter-Sent Layers           | 2                                |
| Attention Heads             | 8                                |
| Total Parameter             | ~111M                            |
| Parameter Dilatih (Tahap 1) | ~5M                              |

---

## 📊 Dataset

| Dataset                | Jumlah   | Format | Keterangan                            |
| ---------------------- | -------- | ------ | ------------------------------------- |
| **Ormas Liputan6**     | 35.065   | CSV    | Artikel Ormas Islam dari Liputan6.com |
| **IndoSum**            | 93.860   | JSONL  | Train 71.345, Val 3.743, Test 18.772  |
| **Liputan6 Canonical** | ~50.000+ | JSON   | Artikel berita umum                   |

### Format Data Ormas CSV

```csv
content,date,summary,title,url
"Teks artikel...", "2025-01-01", "Teks ringkasan...", "Judul", "URL"
```

### Format IndoSum JSONL

```json
{
  "id": "...",
  "paragraphs": [[[token, ...], [token, ...]], ...],
  "gold_labels": [[false, true], [true, false], ...]
}
```

---

## 🏋️ Training

### Strategi 2 Tahap

**Tahap 1** (5 epoch): BERT dibekukan → Latih Inter-Sentence Transformer + Classifier

- Learning Rate: 5e-4
- Grad Accumulation: 4 (efektif batch = 16)
- Loss: Weighted BCE (pos_weight=3.0)

**Tahap 2** (3 epoch): BERT dicairkan → Fine-tune end-to-end

- BERT LR: 1e-5 (lebih kecil untuk stabilitas)
- Head LR: 2e-5
- Scheduler: Cosine decay dengan warmup

### Konfigurasi Default

```python
config = {
    'bert_model_name': 'indobenchmark/indobert-base-p1',
    'batch_size': 4,
    'grad_accumulation': 4,
    'num_epochs_stage1': 5,
    'num_epochs_stage2': 3,
    'max_sent_len': 128,
    'max_sentences': 40,
    'pos_weight': 3.0,  # Class imbalance weight
}
```

---

## 📈 Evaluasi

### Metrik

| Metrik      | Deskripsi                                    |
| ----------- | -------------------------------------------- |
| **ROUGE-1** | Unigram overlap antara ringkasan & referensi |
| **ROUGE-2** | Bigram overlap                               |
| **ROUGE-L** | Longest Common Subsequence                   |
| **F1**      | F1 score klasifikasi kalimat                 |

### Output Evaluasi

```
ROUGE-1 F1: 0.XXXX
ROUGE-2 F1: 0.XXXX
ROUGE-L F1: 0.XXXX
Accuracy  : 0.XXXX
Precision : 0.XXXX
Recall    : 0.XXXX
F1        : 0.XXXX
```

---

## 🌐 Streamlit Dashboard

Fitur dashboard:

- ✅ Input teks artikel langsung atau upload file
- ✅ Pengaturan rasio ringkasan (10%-70%)
- ✅ Highlight kalimat terpilih dengan skor
- ✅ Bar chart skor relevansi per kalimat
- ✅ Statistik: kompresi ratio, jumlah kata, dll.
- ✅ Evaluasi ROUGE jika ada referensi
- ✅ Download ringkasan sebagai .txt
- ✅ Mode fallback TF-IDF (sebelum model dilatih)
- ✅ Halaman "Tentang Penelitian"

---

## 🔧 Troubleshooting

### Error: CUDA out of memory

```bash
# Kurangi batch size
python src/train.py --batch_size 2
# Atau kurangi max_sentences di konfigurasi
```

### Error: Model tidak ditemukan di Streamlit

```
Pastikan training sudah selesai. Model disimpan di salah satu file:
models/indobert_summarizer/head_best.pt
models/indobert_summarizer/model_best.pt

Jika belum training, app akan menggunakan TF-IDF sebagai fallback.
```

### Training sangat lambat di CPU

```
Full fine-tuning IndoBERT membutuhkan GPU untuk kecepatan optimal.
Tanpa GPU (CPU saja), gunakan pipeline head-only agar lebih realistis.

Alternatif:
1. Gunakan Google Colab (GPU gratis)
2. Gunakan demo lokal: python local_demo_train.py
3. Gunakan train_light.py + precompute_embeddings.py
```

### Download model gagal

```bash
# Pastikan koneksi internet
# Atau download manual dan simpan ke models/indobert_summarizer/
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('indobenchmark/indobert-base-p1')"
```

---

## 📚 Referensi

1. **Liu, Y. & Lapata, M.** (2019). _Text Summarization with Pretrained Encoders_. EMNLP 2019. [arXiv:1908.08345](https://arxiv.org/abs/1908.08345)

2. **Wilie, B. et al.** (2020). _IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding_. AACL-IJCNLP 2020.

3. **Kurniawan, K. & Louvan, S.** (2018). _IndoSum: A New Benchmark Dataset for the Indonesian Automatic Text Summarization Task_. IALP 2018.

4. **Devlin, J. et al.** (2018). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_. NAACL 2019.

5. **Lin, C.Y.** (2004). _ROUGE: A Package for Automatic Evaluation of Summaries_. ACL 2004.

---

## 🎓 Informasi Skripsi

**Judul**: Peringkasan Otomatis Artikel Tentang Ormas Islam di Media Online Menggunakan Algoritma IndoBERT

**Metode**:

- Natural Language Processing (NLP)
- Automatic Text Summarization (ATS)
- BERT & IndoBERT (Bidirectional Encoder Representations from Transformers)

**Tools**:

- Python 3.9+ · PyTorch · HuggingFace Transformers · Streamlit · VS Code
# ATS-islamic-organization-news
