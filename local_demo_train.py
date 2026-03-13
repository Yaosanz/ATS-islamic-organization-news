"""
local_demo_train.py
===================
Training demo lokal untuk CPU.
Menggunakan 1000 sampel training untuk demo yang selesai dalam ~1 jam.

Langkah:
  1. Encode 1000 dokumen training + 500 val + test dengan IndoBERT  
  2. Train head 5 epoch
  3. Evaluasi ROUGE
  4. Simpan model untuk Streamlit

Perkiraan waktu:
  Encoding: 1500 dok × 1.2s = ~30 menit
  Training: ~5 menit
  Total  : ~35 menit

CATATAN: Untuk hasil akademis terbaik, gunakan train_light.py 
         dengan --dataset indosum (full 71K sampel)  
         atau gunakan Google Colab dengan GPU.
"""

import sys, os, time, pickle, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(str(Path(__file__).resolve().parent))

DATA_DIR = Path('data/processed')
EMB_DIR  = Path('data/embeddings')
MODEL_DIR = Path('models/indobert_summarizer')
EMB_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BERT_NAME   = 'indobenchmark/indobert-base-p1'
DEVICE      = torch.device('cpu')
MAX_SENTS   = 20
MAX_SENT_LEN = 64
BATCH_ENC   = 64   # sentences per BERT call (across docs, faster!)

# ─── How many samples to use ───────────────────────────────────
N_TRAIN = 1000   # increase for better model (but slower)
N_VAL   = 300
N_TEST  = 500

print("=" * 65)
print("  DEMO TRAINING LOKAL - IndoBERT Summarizer (CPU)")
print("=" * 65)
print(f"  Train  : {N_TRAIN:,} dokumen")
print(f"  Val    : {N_VAL:,} dokumen")
print(f"  Test   : {N_TEST:,} dokumen")
print(f"  Device : {DEVICE}")
print("=" * 65)

# ─── STEP 1: Load data ──────────────────────────────────────────
print("\n[1/5] Memuat data...")
with open(DATA_DIR / 'indosum_train.pkl', 'rb') as f:
    train_data = pickle.load(f)[:N_TRAIN]
with open(DATA_DIR / 'indosum_val.pkl', 'rb') as f:
    val_data = pickle.load(f)[:N_VAL]
with open(DATA_DIR / 'indosum_test.pkl', 'rb') as f:
    test_data = pickle.load(f)[:N_TEST]

print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

# ─── STEP 2: Encode with BERT (cross-document batching) ─────────
def encode_batch_efficient(examples, tokenizer, bert, max_sent_len=64, batch_size=64):
    """Encode semua kalimat dari semua dokumen dalam batch besar."""
    bert.eval()
    
    # Kumpulkan semua kalimat dengan doc_id
    all_sents = []
    doc_ranges = []
    
    for doc_idx, ex in enumerate(examples):
        sents = ex['sentences'][:MAX_SENTS]
        start = len(all_sents)
        all_sents.extend(sents)
        doc_ranges.append((start, start + len(sents)))
    
    # Encode dalam batch besar
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_sents), batch_size),
                      desc="  Encoding kalimat", leave=False):
            batch_sents = all_sents[i:i + batch_size]
            enc = tokenizer(batch_sents, padding='max_length', truncation=True,
                            max_length=max_sent_len, return_tensors='pt')
            out = bert(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(1e-9)
            all_embs.append(embs.numpy())
    
    all_embs_flat = np.concatenate(all_embs, axis=0)  # (total_sents, 768)
    
    # Pisahkan kembali per dokumen
    results = []
    for doc_idx, ex in enumerate(examples):
        start, end = doc_ranges[doc_idx]
        doc_emb = all_embs_flat[start:end]  # (N, 768)
        results.append({
            'id': ex.get('id', ''),
            'embeddings': doc_emb,
            'labels': ex['labels'][:doc_emb.shape[0]],
            'sentences': ex['sentences'][:doc_emb.shape[0]],
            'n_sents': doc_emb.shape[0]
        })
    return results

print("\n[2/5] Pre-komputasi embedding (cross-document batching)...")
print(f"  Memuat IndoBERT encoder...")
tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
bert = AutoModel.from_pretrained(BERT_NAME)

t0 = time.time()
train_emb = encode_batch_efficient(train_data, tokenizer, bert, MAX_SENT_LEN, BATCH_ENC)
val_emb   = encode_batch_efficient(val_data,   tokenizer, bert, MAX_SENT_LEN, BATCH_ENC)
test_emb  = encode_batch_efficient(test_data,  tokenizer, bert, MAX_SENT_LEN, BATCH_ENC)
enc_time  = time.time() - t0

# Save embeddings to disk
for name, data in [('demo_train', train_emb), ('demo_val', val_emb), ('demo_test', test_emb)]:
    with open(EMB_DIR / f'{name}_emb.pkl', 'wb') as f:
        pickle.dump(data, f)

print(f"  Selesai dalam {enc_time/60:.1f} menit")
print(f"  ({enc_time/len(train_emb+val_emb+test_emb)*1000:.0f}ms per dokumen)")

# ─── STEP 3: Train head ─────────────────────────────────────────
print("\n[3/5] Training Summarization Head...")

# Import train_light utilities
import importlib.util
tl_spec = importlib.util.spec_from_file_location("train_light", "train_light.py")
tl = importlib.util.module_from_spec(tl_spec)
tl_spec.loader.exec_module(tl)

EmbeddingDataset = tl.EmbeddingDataset
collate_emb      = tl.collate_emb
SummarizationHead = tl.SummarizationHead
WeightedBCELoss  = tl.WeightedBCELoss

train_ds = EmbeddingDataset(train_emb, MAX_SENTS)
val_ds   = EmbeddingDataset(val_emb,   MAX_SENTS)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  collate_fn=collate_emb)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, collate_fn=collate_emb)

head = SummarizationHead(n_layers=2, n_heads=8, dropout=0.1).to(DEVICE)
print(f"  Model Head Parameter: {head.count_params():,}")

criterion = WeightedBCELoss(pos_weight=3.0)
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, int(len(train_loader)*5*0.1), len(train_loader)*5
)

best_f1 = 0.0
best_path = MODEL_DIR / 'head_best.pt'
history = []

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    # Train
    head.train()
    total_loss, all_p, all_l = 0, [], []
    for batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS} Train", leave=False):
        embs  = batch['embeddings'].to(DEVICE)
        lbls  = batch['labels'].to(DEVICE)
        mask  = batch['sent_mask'].to(DEVICE)
        scores = head(embs, mask)
        valid  = ~mask
        loss   = criterion(scores[valid], lbls[valid])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        all_p.extend((scores[valid].detach() > 0.5).float().cpu().numpy().tolist())
        all_l.extend(lbls[valid].cpu().numpy().tolist())
    
    train_f1 = f1_score(np.array(all_l), np.array(all_p), zero_division=0)
    
    # Val
    head.eval()
    v_loss, vp, vl = 0, [], []
    with torch.no_grad():
        for batch in val_loader:
            embs = batch['embeddings'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            mask = batch['sent_mask'].to(DEVICE)
            s = head(embs, mask)
            v = ~mask
            v_loss += criterion(s[v], lbls[v]).item()
            vp.extend((s[v] > 0.5).float().cpu().numpy().tolist())
            vl.extend(lbls[v].cpu().numpy().tolist())
    
    val_f1 = f1_score(np.array(vl), np.array(vp), zero_division=0)
    ep_time = time.time() - t_ep
    
    print(f"  Epoch {epoch:2d}/{EPOCHS} ({ep_time:.0f}s) | "
          f"Train Loss={total_loss/len(train_loader):.4f} F1={train_f1:.4f} | "
          f"Val F1={val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save({'model_state': head.state_dict(),
                    'config': {'hidden': 768, 'n_layers': 2, 'n_heads': 8},
                    'epoch': epoch, 'val_f1': best_f1}, best_path)
        print(f"  ✅ Best model disimpan (Val F1: {best_f1:.4f})")

# ─── STEP 4: ROUGE Evaluation ──────────────────────────────────
print("\n[4/5] Evaluasi ROUGE (200 sampel dari test set)...")
from rouge_score import rouge_scorer as rs
scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
head.load_state_dict(ckpt['model_state'])
head.eval()

r1, r2, rl = [], [], []
for item in tqdm(test_emb[:200], desc="  ROUGE eval", leave=False):
    sents = item['sentences']
    if not sents:
        continue
    emb = torch.tensor(item['embeddings'], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        scores = head(emb).squeeze(0).numpy()
    n_sel = max(1, int(len(sents) * 0.3))
    top_idx = sorted(np.argsort(scores)[-n_sel:])
    pred = ' '.join(sents[i] for i in top_idx if i < len(sents))
    ref_idx = [i for i, l in enumerate(item['labels'][:len(sents)]) if l == 1]
    if not ref_idx:
        continue
    ref = ' '.join(sents[i] for i in ref_idx if i < len(sents))
    sc = scorer.score(ref, pred)
    r1.append(sc['rouge1'].fmeasure)
    r2.append(sc['rouge2'].fmeasure)
    rl.append(sc['rougeL'].fmeasure)

print(f"\n  📊 HASIL EVALUASI:")
print(f"  ROUGE-1 : {np.mean(r1):.4f}")
print(f"  ROUGE-2 : {np.mean(r2):.4f}")
print(f"  ROUGE-L : {np.mean(rl):.4f}")
print(f"  Best Val F1 : {best_f1:.4f}")

# Save training config
with open(MODEL_DIR / 'training_config.json', 'w') as f:
    json.dump({'dataset': 'indosum_demo', 'n_train': N_TRAIN,
               'best_val_f1': best_f1,
               'rouge': {'rouge1': np.mean(r1), 'rouge2': np.mean(r2), 'rougeL': np.mean(rl)},
               'enc_time_min': enc_time/60, 'type': 'head_only',
               'bert_model': BERT_NAME}, f, indent=2)

# ─── STEP 5: Inference Test ─────────────────────────────────────
print("\n[5/5] Demo Inferensi...")

sample_article = """Nahdlatul Ulama (NU) mengadakan kongres nasional yang dihadiri 
oleh ribuan pengurus dari seluruh Indonesia. Kongres yang berlangsung di 
Surabaya ini membahas isu-isu penting mengenai peran Islam dalam kehidupan 
berbangsa dan bernegara. Ketua Umum PBNU, dalam pidatonya, menegaskan 
komitmen NU untuk terus berkontribusi dalam pembangunan bangsa. 
Para ulama sepakat bahwa Islam rahmatan lil alamin harus menjadi landasan 
dalam setiap kebijakan organisasi. Kongres juga memilih pengurus baru 
yang akan memimpin NU untuk periode lima tahun ke depan. 
Muhammadiyah turut mengirimkan delegasi sebagai bentuk solidaritas 
antar organisasi Islam di Indonesia."""

from src.summarize import HeadOnlySummarizationPipeline
import importlib.util
tl_spec2 = importlib.util.spec_from_file_location("train_light2", "train_light.py")
tl2 = importlib.util.module_from_spec(tl_spec2)
tl_spec2.loader.exec_module(tl2)

head2 = tl2.SummarizationHead()
head2.load_state_dict(ckpt['model_state'])

from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(BERT_NAME)
bert2 = AutoModel.from_pretrained(BERT_NAME)

pipeline = HeadOnlySummarizationPipeline(bert2, tok, head2, DEVICE,
                                          max_sent_len=MAX_SENT_LEN)

result = pipeline.summarize(sample_article, ratio=0.4)
print(f"\n  Artikel asli: {len(sample_article)} karakter")
print(f"  Ringkasan  :")
print(f"  {result['summary']}")
print(f"  Kalimat dipilih: {result['stats'].get('selected_sentences', '?')}/{result['stats'].get('original_sentences', '?')}")
print(f"  Kompresi: {result['stats'].get('compression_ratio', 0)*100:.1f}%")

print(f"\n{'='*65}")
print(f"  ✅ DEMO TRAINING SELESAI!")
print(f"  Model tersimpan di: {best_path}")
print(f"\n  Langkah selanjutnya - uji Streamlit:")
print(f"  cd d:\\Joki && .\\venv\\Scripts\\activate && streamlit run app/streamlit_app.py")
print("=" * 65)
