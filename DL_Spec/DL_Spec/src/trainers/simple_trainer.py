from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 1e-3
    device: str = "cpu"
    log_every: int = 50


def default_collate(batch):
    # Identity collate for pre-tensorized batches
    return tuple(zip(*batch))


class SimpleTrainer:
    def __init__(self, model: nn.Module, loss_fn: Callable, optimizer: torch.optim.Optimizer, device: Optional[str] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in dataloader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            if logits.ndim == 2 and logits.size(1) > 1:
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return {"loss": avg_loss, "acc": acc}

    @torch.no_grad()
    def eval_epoch(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in dataloader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            if logits.ndim == 2 and logits.size(1) > 1:
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return {"loss": avg_loss, "acc": acc}

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, config: Optional[TrainConfig] = None):
        cfg = config or TrainConfig()
        hist = []
        for epoch in range(cfg.epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader) if val_loader is not None else None
            dt = time.time() - t0
            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
                "time_sec": round(dt, 2)
            }
            hist.append(record)
        return hist
