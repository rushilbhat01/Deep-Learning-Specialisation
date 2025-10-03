import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.models.cnn import SimpleCNN
from src.models.rnn_lstm import SimpleRNN, SimpleLSTM
from src.models.transformer import SimpleTransformerClassifier
from src.utils.data_utils import SyntheticMNISTLike, ToySentimentDataset


def test_cnn_one_step():
    ds = SyntheticMNISTLike(n=32)
    dl = DataLoader(ds, batch_size=8)
    model = SimpleCNN(num_classes=2)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    xb, yb = next(iter(dl))
    logits = model(xb)
    loss = loss_fn(logits, yb)
    loss.backward()
    opt.step()
    assert logits.shape == (8, 2)


def test_nlp_models_one_step():
    ds = ToySentimentDataset()
    dl = DataLoader(ds, batch_size=4, collate_fn=ds.collate)
    vocab_size = len(ds.vocab)

    for model in [
        SimpleRNN(vocab_size=vocab_size, num_classes=2),
        SimpleLSTM(vocab_size=vocab_size, num_classes=2),
        SimpleTransformerClassifier(vocab_size=vocab_size, num_classes=2),
    ]:
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        xb, yb = next(iter(dl))
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        assert logits.shape[0] == xb.shape[0]
