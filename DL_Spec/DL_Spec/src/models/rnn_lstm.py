import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)
        out, h = self.rnn(emb)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 128, num_layers: int = 1, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)
