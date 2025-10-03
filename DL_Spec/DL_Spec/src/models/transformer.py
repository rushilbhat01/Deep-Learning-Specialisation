import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T]


class SimpleTransformerClassifier(nn.Module):
    """
    Tiny Transformer encoder classifier with dropout.
    """
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.posenc = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T)
        emb = self.embedding(x)
        emb = self.posenc(emb)
        enc = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        pooled = enc.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.cls(pooled)
