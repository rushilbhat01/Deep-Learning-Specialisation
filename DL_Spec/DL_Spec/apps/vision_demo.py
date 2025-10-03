import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.models.cnn import SimpleCNN
from src.trainers.simple_trainer import SimpleTrainer, TrainConfig
from src.utils.data_utils import SyntheticMNISTLike


def main():
    dataset = SyntheticMNISTLike(n=256)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleCNN(num_classes=2, dropout=0.3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = SimpleTrainer(model, loss_fn, optimizer)
    history = trainer.fit(train_loader, None, TrainConfig(epochs=2))
    print("Vision demo history:", history)


if __name__ == "__main__":
    main()
