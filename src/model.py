"""
model.py
========
Arsitektur Model: IndoBERT Extractive Summarizer

Arsitektur (BertSum-style, Liu & Lapata 2019):
 1. IndoBERT mengenkode setiap kalimat secara independen
    → embedding kalimat (mean pooling) ukuran 768
 2. Positional Encoding ditambahkan ke setiap embedding kalimat
 3. Inter-Sentence Transformer (2 layer) untuk menangkap konteks antar kalimat
 4. Linear Classifier → skor relevansi per kalimat (sigmoid)
"""

import torch
import torch.nn as nn
import math
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────
#  POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding untuk urutan kalimat.
    Memberikan informasi posisi kalimat dalam dokumen.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Hitung encoding sinusoidal
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────
#  MODEL UTAMA
# ─────────────────────────────────────────────────────────────────

class IndoBERTSumExtractor(nn.Module):
    """
    IndoBERT-based Extractive Summarizer.

    Pipeline:
        Kalimat → IndoBERT → Mean Pooling → Positional Encoding
            → Inter-Sentence Transformer → Linear → Sigmoid Score

    Penggunaan:
        model = IndoBERTSumExtractor()
        scores = model(input_ids, attention_mask, sent_mask)
        # scores: (batch, max_sents) nilai 0-1, threshold untuk seleksi
    """

    def __init__(
        self,
        bert_model_name: str = 'indobenchmark/indobert-base-p1',
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_bert_layers: int = 8  # Bekukan N layer pertama BERT
    ):
        """
        Args:
            bert_model_name: Nama model dari HuggingFace Hub
            num_transformer_layers: Jumlah layer inter-sentence transformer
            num_heads: Jumlah attention head
            dropout: Dropout rate
            freeze_bert_layers: Bekukan N layer awal BERT (0 = tidak dibekukan sama sekali)
        """
        super().__init__()

        # ── Muat IndoBERT ────────────────────────────────────────
        print(f"[INFO] Memuat model BERT: {bert_model_name}")
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Bekukan layer awal BERT untuk efisiensi
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)

        # ── Positional Encoding ──────────────────────────────────
        self.pos_encoding = PositionalEncoding(self.hidden_size, dropout)

        # ── Inter-Sentence Transformer ───────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN (lebih stabil untuk training)
        )
        self.inter_sent_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False
        )

        # ── Classifier ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def _freeze_bert_layers(self, n_layers: int):
        """Bekukan embedding dan N layer encoder pertama."""
        # Bekukan embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        # Bekukan N layer encoder pertama
        for layer in self.bert.encoder.layer[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"[INFO] {n_layers} layer BERT pertama dibekukan")

    def unfreeze_bert(self):
        """Cairkan semua parameter BERT (untuk fine-tuning tahap 2)."""
        for param in self.bert.parameters():
            param.requires_grad = True
        print("[INFO] Semua layer BERT dicairkan")

    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling dengan memperhitungkan padding mask.

        Args:
            token_embeddings: (N, T, H)
            attention_mask: (N, T)

        Returns:
            sentence_embeddings: (N, H)
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_emb = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_emb / sum_mask

    def encode_sentences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Enkode batch kalimat dengan IndoBERT.

        Args:
            input_ids: (total_sents, max_sent_len)
            attention_mask: (total_sents, max_sent_len)

        Returns:
            sentence_embeddings: (total_sents, hidden_size)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.mean_pooling(outputs.last_hidden_state, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sent_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch, max_sents, max_sent_len)
            attention_mask: (batch, max_sents, max_sent_len)
            sent_mask: (batch, max_sents) - True pada padding kalimat

        Returns:
            scores: (batch, max_sents) - skor relevansi [0, 1]
        """
        batch_size, max_sents, max_sent_len = input_ids.shape

        # Reshape untuk enkode semua kalimat sekaligus
        # (batch * max_sents, max_sent_len)
        flat_ids = input_ids.view(-1, max_sent_len)
        flat_mask = attention_mask.view(-1, max_sent_len)

        # Enkode kalimat dengan IndoBERT
        sent_embs = self.encode_sentences(flat_ids, flat_mask)
        # Reshape kembali: (batch, max_sents, hidden_size)
        sent_embs = sent_embs.view(batch_size, max_sents, self.hidden_size)
        sent_embs = self.dropout(sent_embs)

        # Tambah positional encoding
        sent_embs = self.pos_encoding(sent_embs)

        # Inter-sentence transformer
        # sent_mask: True = abaikan (padding kalimat)
        sent_embs = self.inter_sent_transformer(
            sent_embs,
            src_key_padding_mask=sent_mask
        )

        # Klasifikasi per kalimat
        logits = self.classifier(sent_embs).squeeze(-1)  # (batch, max_sents)
        scores = torch.sigmoid(logits)

        return scores

    def get_trainable_params(self) -> int:
        """Hitung jumlah parameter yang bisa dilatih."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Hitung total parameter."""
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────
#  FUNGSI UTILITAS MODEL
# ─────────────────────────────────────────────────────────────────

def save_model(model: IndoBERTSumExtractor, save_dir: str, epoch: int = None):
    """
    Simpan model ke direktori.

    Args:
        model: Model yang sudah dilatih
        save_dir: Direktori tujuan
        epoch: Epoch saat ini (untuk checkpoint)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Simpan state dict model
    suffix = f"_epoch{epoch}" if epoch is not None else "_best"
    model_path = f"{save_dir}/model{suffix}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model disimpan ke: {model_path}")

    # Simpan konfigurasi
    config = {
        'bert_model_name': model.bert.config.name_or_path,
        'hidden_size': model.hidden_size,
    }
    import json
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return model_path


def load_model(
    model_path: str,
    bert_model_name: str = 'indobenchmark/indobert-base-p1',
    device: str = 'cpu',
    freeze_bert_layers: int = 0
) -> IndoBERTSumExtractor:
    """
    Muat model dari file checkpoint.

    Args:
        model_path: Path ke file .pt
        bert_model_name: Nama model BERT
        device: Device untuk loading
        freeze_bert_layers: Lapisan BERT yang dibekukan

    Returns:
        Model yang sudah dimuat
    """
    model = IndoBERTSumExtractor(
        bert_model_name=bert_model_name,
        freeze_bert_layers=freeze_bert_layers
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Model dimuat dari: {model_path}")
    return model


if __name__ == '__main__':
    # Test model
    print("Testing IndoBERTSumExtractor ...")
    model = IndoBERTSumExtractor(freeze_bert_layers=8)
    total = model.get_total_params()
    trainable = model.get_trainable_params()
    print(f"Total parameter: {total:,}")
    print(f"Parameter yang bisa dilatih: {trainable:,}")
    print(f"Proporsi yang dilatih: {trainable/total:.1%}")

    # Forward test dengan data dummy
    batch_size, max_sents, max_len = 2, 5, 32
    dummy_ids = torch.randint(0, 1000, (batch_size, max_sents, max_len))
    dummy_mask = torch.ones(batch_size, max_sents, max_len, dtype=torch.long)
    dummy_sent_mask = torch.zeros(batch_size, max_sents, dtype=torch.bool)
    dummy_sent_mask[:, 3:] = True  # Padding pada kalimat 3-4

    with torch.no_grad():
        scores = model(dummy_ids, dummy_mask, dummy_sent_mask)
    print(f"Output shape: {scores.shape}")  # Expected: (2, 5)
    print(f"Sample scores: {scores[0].tolist()}")
    print("✅ Model test berhasil!")
