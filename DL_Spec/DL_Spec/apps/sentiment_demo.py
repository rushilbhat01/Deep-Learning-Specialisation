import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.models.rnn_lstm import SimpleLSTM
from src.models.transformer import SimpleTransformerClassifier
from src.utils.data_utils import ToySentimentDataset


def run_with_model(model, dataset):
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Simple single-epoch loop inline to keep it brief
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
    print(f"Finished one pass with {model.__class__.__name__}")


def main():
    dataset = ToySentimentDataset()
    vocab_size = len(dataset.vocab)

    lstm = SimpleLSTM(vocab_size=vocab_size, num_classes=2, dropout=0.3)
    run_with_model(lstm, dataset)

    transformer = SimpleTransformerClassifier(vocab_size=vocab_size, num_classes=2, dropout=0.2)
    run_with_model(transformer, dataset)


if __name__ == "__main__":
    main()
