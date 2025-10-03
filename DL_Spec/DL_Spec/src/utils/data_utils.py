from __future__ import annotations
import random
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset


class SyntheticMNISTLike(Dataset):
    """A tiny synthetic dataset: 28x28 grayscale blobs with two classes."""
    def __init__(self, n: int = 256, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 1, 28, 28, generator=g)
        # Simple rule: mean pixel > 0 => class 1 else 0
        self.y = (self.X.mean(dim=(1, 2, 3)) > 0).long()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Simple whitespace tokenizer + vocab
class Vocab:
    def __init__(self, tokens: List[str], min_freq: int = 1, specials: List[str] = None):
        specials = specials or ["<pad>", "<unk>"]
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        # order: specials then words by frequency
        self.itos = list(specials)
        for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and w not in specials:
                self.itos.append(w)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_index = self.stoi["<pad>"]
        self.unk_index = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        ids = []
        for tok in text.strip().split():
            ids.append(self.stoi.get(tok, self.unk_index))
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.itos[i] for i in ids)


def pad_sequences(batch: List[List[int]], pad_index: int, max_len: int = None) -> torch.Tensor:
    if max_len is None:
        max_len = max(len(x) for x in batch)
    out = torch.full((len(batch), max_len), fill_value=pad_index, dtype=torch.long)
    for i, seq in enumerate(batch):
        L = min(len(seq), max_len)
        out[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
    return out


class ToySentimentDataset(Dataset):
    """Very small, deterministic sentiment dataset for quick demos."""
    def __init__(self):
        pos = [
            "i love this movie",
            "this product is great",
            "what a fantastic day",
            "amazing experience overall",
        ]
        neg = [
            "i hate this movie",
            "this product is terrible",
            "what a horrible day",
            "awful experience overall",
        ]
        self.texts = pos + neg
        self.labels = [1] * len(pos) + [0] * len(neg)
        tokens = [tok for s in self.texts for tok in s.split()]
        self.vocab = Vocab(tokens)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def collate(self, batch):
        texts, labels = zip(*batch)
        ids = [self.vocab.encode(t) for t in texts]
        x = pad_sequences(ids, pad_index=self.vocab.pad_index)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y


class ToyChatDataset(Dataset):
    """Pairs of (input, response) for a toy chatbot."""
    def __init__(self):
        pairs = [
            ("hi", "hello"),
            ("hello", "hi"),
            ("how are you", "i am fine"),
            ("what is your name", "i am a bot"),
            ("bye", "goodbye"),
        ]
        self.src_texts, self.tgt_texts = zip(*pairs)
        tokens = [tok for s in (list(self.src_texts) + list(self.tgt_texts)) for tok in s.split()]
        self.vocab = Vocab(tokens, specials=["<pad>", "<unk>", "<bos>", "<eos>"])
        self.bos = self.vocab.stoi["<bos>"]
        self.eos = self.vocab.stoi["<eos>"]

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return self.src_texts[idx], self.tgt_texts[idx]

    def collate(self, batch):
        src, tgt = zip(*batch)
        src_ids = [self.vocab.encode(s) for s in src]
        tgt_ids = [[self.bos] + self.vocab.encode(t) + [self.eos] for t in tgt]
        src_pad = pad_sequences(src_ids, pad_index=self.vocab.pad_index)
        tgt_pad = pad_sequences(tgt_ids, pad_index=self.vocab.pad_index)
        return src_pad, tgt_pad
