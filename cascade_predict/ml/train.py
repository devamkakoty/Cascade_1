"""
Training pipeline for the cascade PINN.

Combines data loss (MSE on temperature + BCE on runaway classification)
with physics loss (Arrhenius consistency, spatial coupling, bounds).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .pinn import CascadePINN, PhysicsLoss
from .dataset import CascadeDataset


class Trainer:
    """Trains the CascadePINN with combined data + physics losses."""

    def __init__(
        self,
        model: CascadePINN,
        lr: float = 1e-3,
        physics_weight: float = 0.3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        self.physics_loss_fn = PhysicsLoss()
        self.physics_weight = physics_weight

        # Data losses
        self.temp_loss_fn = nn.MSELoss()
        self.runaway_loss_fn = nn.BCELoss()

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train one epoch. Returns loss dict."""
        self.model.train()
        epoch_losses = {
            "total": 0, "temp": 0, "runaway": 0,
            "physics": 0, "arrhenius": 0, "spatial": 0, "bounds": 0,
        }
        n_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            pred = self.model(inputs)

            # Data losses
            temp_loss = self.temp_loss_fn(pred[..., 0], targets[..., 0])
            runaway_loss = self.runaway_loss_fn(pred[..., 1], targets[..., 1])
            data_loss = temp_loss + runaway_loss

            # Physics loss
            physics_loss, physics_components = self.physics_loss_fn(inputs, pred)

            # Combined
            total_loss = data_loss + self.physics_weight * physics_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses["total"] += total_loss.item()
            epoch_losses["temp"] += temp_loss.item()
            epoch_losses["runaway"] += runaway_loss.item()
            epoch_losses["physics"] += physics_loss.item()
            for k, v in physics_components.items():
                epoch_losses[k] += v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        total_temp_err = 0
        total_runaway_acc = 0
        n_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            pred = self.model(inputs)

            # Temperature MAE
            temp_err = torch.mean(torch.abs(pred[..., 0] - targets[..., 0]))
            total_temp_err += temp_err.item()

            # Runaway classification accuracy
            pred_runaway = (pred[..., 1] > 0.5).float()
            acc = (pred_runaway == targets[..., 1]).float().mean()
            total_runaway_acc += acc.item()

            n_batches += 1

        return {
            "temp_mae": total_temp_err / max(n_batches, 1),
            "runaway_accuracy": total_runaway_acc / max(n_batches, 1),
        }

    def fit(
        self,
        train_dataset: CascadeDataset,
        val_dataset: CascadeDataset | None = None,
        n_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[dict]:
        """Full training loop."""
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        history = []
        for epoch in range(n_epochs):
            train_losses = self.train_epoch(train_loader)
            self.scheduler.step(train_losses["total"])

            record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_losses.items()}}

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                record.update({f"val_{k}": v for k, v in val_metrics.items()})

            history.append(record)

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch:3d} | loss={train_losses['total']:.4f}"
                msg += f" temp={train_losses['temp']:.4f}"
                msg += f" phys={train_losses['physics']:.4f}"
                if val_loader:
                    msg += f" | val_mae={record['val_temp_mae']:.1f}K"
                    msg += f" val_acc={record['val_runaway_accuracy']:.3f}"
                print(msg)

        return history
