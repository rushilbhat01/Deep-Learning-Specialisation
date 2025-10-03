import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from src.utils.data_utils import ToyChatDataset


class TinySeq2Seq(nn.Module):
    """A minimal encoder-decoder with shared embedding for toy chatbot."""
    def __init__(self, vocab_size: int, emb: int = 64, hid: int = 128, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.encoder = nn.LSTM(emb, hid, batch_first=True)
        self.decoder = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt_in):
        # src: (B, S) ; tgt_in: (B, T)
        src_emb = self.dropout(self.embedding(src))
        _, (h, c) = self.encoder(src_emb)
        tgt_emb = self.dropout(self.embedding(tgt_in))
        out, _ = self.decoder(tgt_emb, (h, c))
        logits = self.fc(out)  # (B, T, V)
        return logits


def main():
    dataset = ToyChatDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate)

    V = len(dataset.vocab)
    model = TinySeq2Seq(V)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.pad_index)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train for a couple of epochs on tiny data
    model.train()
    for epoch in range(2):
        for src, tgt in loader:
            # Teacher forcing: predict next token given previous target tokens
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1} loss={loss.item():.4f}")

    # Greedy decode a couple of samples
    model.eval()
    with torch.no_grad():
        for prompt in ["hi", "how are you", "bye"]:
            ids = dataset.vocab.encode(prompt)
            src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            src_pad = torch.full((1, max(1, src.size(1))), dataset.vocab.pad_index, dtype=torch.long)
            src_pad[0, : src.size(1)] = src

            # Encode
            emb = model.embedding(src_pad)
            _, (h, c) = model.encoder(emb)

            # Decode
            out_ids = []
            cur = torch.tensor([[dataset.vocab.stoi["<bos>"]]], dtype=torch.long)
            for _ in range(10):
                emb = model.embedding(cur)
                o, (h, c) = model.decoder(emb, (h, c))
                logit = model.fc(o[:, -1])
                next_id = logit.argmax(dim=-1)
                ni = next_id.item()
                if ni == dataset.vocab.stoi.get("<eos>", -1):
                    break
                out_ids.append(ni)
                cur = torch.tensor([[ni]], dtype=torch.long)
            print(f"User: {prompt}\nBot:  {dataset.vocab.decode(out_ids)}\n")


if __name__ == "__main__":
    main()
