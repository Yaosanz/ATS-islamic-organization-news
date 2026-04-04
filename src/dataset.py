"""
dataset.py
==========
PyTorch Dataset dan DataLoader untuk IndoBERT Extractive Summarizer.

Setiap contoh adalah satu dokumen dengan daftar kalimat dan label biner.
Kalimat dienkode secara individual dengan tokenizer IndoBERT.
"""

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path


class SummarizationDataset(Dataset):
    """
    Dataset untuk fine-tuning IndoBERT pada tugas ekstraktif summarisasi.

    Setiap item mewakili SATU DOKUMEN yang berisi:
        - sentences: daftar teks kalimat
        - labels: daftar label biner (0/1) per kalimat
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer_name: str = 'indobenchmark/indobert-base-p1',
        max_sent_len: int = 128,
        max_sentences: int = 40,
        tokenizer=None
    ):
        """
        Args:
            examples: Daftar dict {sentences, labels, ...}
            tokenizer_name: Nama model HuggingFace
            max_sent_len: Panjang max token per kalimat (default 128)
            max_sentences: Panjang max kalimat per dokumen (default 40)
            tokenizer: Tokenizer yang sudah dimuat (opsional)
        """
        self.examples = examples
        self.max_sent_len = max_sent_len
        self.max_sentences = max_sentences

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            print(f"[INFO] Memuat tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        sentences = example['sentences'][:self.max_sentences]
        labels = example['labels'][:self.max_sentences]

        # Tokenisasi setiap kalimat
        encodings = self.tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=self.max_sent_len,
            return_tensors='pt'
        )

        n_sents = len(sentences)

        return {
            'input_ids': encodings['input_ids'],            # (N, max_sent_len)
            'attention_mask': encodings['attention_mask'],  # (N, max_sent_len)
            'labels': torch.tensor(labels, dtype=torch.float32),  # (N,)
            'n_sents': n_sents,
            'doc_id': example.get('id', str(idx))
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Fungsi collate untuk menggabungkan dokumen dengan jumlah kalimat berbeda.
    Melakukan padding pada dimensi kalimat (N) ke ukuran maksimum dalam batch.
    """
    max_sents = max(item['n_sents'] for item in batch)
    max_sent_len = batch[0]['input_ids'].shape[1]

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    sent_mask_list = []  # True = padding sentence (untuk diabaikan)

    for item in batch:
        n = item['n_sents']
        pad_size = max_sents - n

        # Pad input_ids dan attention_mask pada dimensi kalimat
        input_ids = item['input_ids']           # (N, T)
        att_mask = item['attention_mask']        # (N, T)
        lbls = item['labels']                    # (N,)

        if pad_size > 0:
            # Pad kalimat dengan zeros
            pad_ids = torch.zeros(pad_size, max_sent_len, dtype=torch.long)
            pad_att = torch.zeros(pad_size, max_sent_len, dtype=torch.long)
            pad_lbl = torch.zeros(pad_size, dtype=torch.float32)

            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            att_mask = torch.cat([att_mask, pad_att], dim=0)
            lbls = torch.cat([lbls, pad_lbl], dim=0)

        # sent_mask: True pada kalimat padding
        sent_mask = torch.zeros(max_sents, dtype=torch.bool)
        sent_mask[n:] = True

        input_ids_list.append(input_ids)
        attention_mask_list.append(att_mask)
        labels_list.append(lbls)
        sent_mask_list.append(sent_mask)

    return {
        'input_ids': torch.stack(input_ids_list),        # (B, max_sents, T)
        'attention_mask': torch.stack(attention_mask_list),  # (B, max_sents, T)
        'labels': torch.stack(labels_list),              # (B, max_sents)
        'sent_mask': torch.stack(sent_mask_list),        # (B, max_sents)
        'n_sents': [item['n_sents'] for item in batch]
    }


def create_dataloader(
    examples: List[Dict],
    tokenizer,
    batch_size: int = 4,
    max_sent_len: int = 128,
    max_sentences: int = 40,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Buat DataLoader dari daftar contoh.

    Args:
        examples: Data terproses
        tokenizer: Tokenizer IndoBERT
        batch_size: Ukuran batch (dalam dokumen)
        max_sent_len: Max token per kalimat
        max_sentences: Max kalimat per dokumen
        shuffle: Acak data
        num_workers: Worker untuk loading paralel

    Returns:
        DataLoader siap pakai
    """
    dataset = SummarizationDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_sent_len=max_sent_len,
        max_sentences=max_sentences
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def load_processed_data(data_dir: Union[str, Path], name: str) -> Tuple[List, List, List]:
    """
    Muat data yang sudah diproses dari file pickle.

    Args:
        data_dir: Direktori data
        name: Prefix nama file (mis. 'combined')

    Returns:
        (train_data, val_data, test_data)
    """
    data_dir = Path(data_dir)
    splits = {}
    for split in ['train', 'val', 'test']:
        path = data_dir / f"{name}_{split}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                splits[split] = pickle.load(f)
            print(f"[INFO] Dimuat {split}: {len(splits[split])} contoh dari {path}")
        else:
            print(f"[WARN] File tidak ditemukan: {path}")
            splits[split] = []

    return splits.get('train', []), splits.get('val', []), splits.get('test', [])
