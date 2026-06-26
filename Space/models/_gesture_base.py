"""Shared base class and utilities for all gesture-level models."""

import numpy as np
import torch
import torch.nn as nn
from typing import List


def sample_frames(features_list: List[List[float]], target_len: int) -> np.ndarray:
    """Uniformly sample *target_len* frames from a variable-length sequence."""
    T = len(features_list)
    arr = np.array(features_list, dtype=np.float32)
    if T == 0:
        return np.zeros((target_len, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32)
    if T < target_len:
        pad = np.tile(arr[-1:], (target_len - T, 1))
        arr = np.concatenate([arr, pad], axis=0)
        return arr
    indices = np.linspace(0, T - 1, target_len, dtype=int)
    return arr[indices]


def _make_gesture_tensors(data, target_frames: int, device: str):
    """Convert gesture data into (N, C, T) tensor + labels."""
    X = np.stack(
        [sample_frames(feats, target_frames) for feats, _ in data], axis=0
    )
    y = np.array([label for _, label in data], dtype=np.int64)
    X = torch.tensor(X, dtype=torch.float32, device=device).permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return X, y


class GestureModelBase:
    """Shared training / eval / predict / save / load loop for all gesture models."""

    def __init__(self, model: nn.Module, target_frames: int, lr: float,
                 weight_decay: float, device: str, name: str):
        self.name = name
        self.target_frames = target_frames
        self.lr = lr
        self.device = device
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

    # ------------------------------------------------------------------
    def train(self, train_data, val_data=None, epoch=0, total_epochs=0):
        self.model.train()
        X, y = _make_gesture_tensors(train_data, self.target_frames, self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), batch_size=16, shuffle=True
        )

        total_loss, correct, total = 0.0, 0, 0
        n_batches = len(loader)
        report_every = max(1, n_batches // 10)
        label = f"Epoch {epoch}/{total_epochs}" if total_epochs else "Training"

        for i, (bx, by) in enumerate(loader):
            self.optimizer.zero_grad()
            logits = self.model(bx)
            loss = self.criterion(logits, by)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

            if (i + 1) % report_every == 0 or i == n_batches - 1:
                pct = (i + 1) / n_batches * 100
                filled = int(20 * (i + 1) / n_batches)
                bar = "█" * filled + "░" * (20 - filled)
                rl, ra = total_loss / total, correct / total
                print(f"\r  {label}  {bar}  {pct:3.0f}%  loss={rl:.4f}  acc={ra:.3f}",
                      end="", flush=True)
        print()

        self.scheduler.step()
        result = {"train_loss": total_loss / total, "train_acc": correct / total}
        if val_data:
            result["val_loss"], result["val_acc"] = self._evaluate(val_data)
        return result

    def _evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            X, y = _make_gesture_tensors(data, self.target_frames, self.device)
            logits = self.model(X)
            loss = self.criterion(logits, y).item()
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        return loss, acc

    def predict(self, data):
        if not data:
            return []
        self.model.eval()
        with torch.no_grad():
            X, _ = _make_gesture_tensors(data, self.target_frames, self.device)
            return self.model(X).argmax(dim=1).tolist()

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
