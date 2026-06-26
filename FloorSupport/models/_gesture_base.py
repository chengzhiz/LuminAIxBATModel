"""Shared base class and utilities for all gesture-level models."""

import numpy as np
import torch
import torch.nn as nn
from typing import List


# ═══════════════════════════════════════════════════════════════════
# Height normalization — baked into the model so raw features work
# ═══════════════════════════════════════════════════════════════════

class HeightNormalize(nn.Module):
    """Normalize body positions to unit height — makes the model
    height-invariant (adults, kids, all skeleton sizes).

    Position layout (732-dim vector):
        [0:208]   bodyFrameHuman      (52 joints × 4 quat)
        [208:364] bodyFrameHumanHeading (52 joints × 3 heading)
        [364:520] bodyFrameHumanPos   (52 joints × 3 position)  ← normalized
        [520:524] initialRootRotation (4 quat)
        [524:732] initialRotations    (52 joints × 4 quat)

    The Y-coordinates are every 3rd value within [364:520]:
        indices 365, 368, 371, ..., 518
    """

    def __init__(self):
        super().__init__()
        # Pre-compute indices for efficient access
        pos_start, pos_end = 364, 520
        self.register_buffer("pos_start", torch.tensor(pos_start, dtype=torch.long))
        self.register_buffer("pos_end", torch.tensor(pos_end, dtype=torch.long))
        # Y indices within the position block: every 3rd starting from 1
        y_indices = torch.tensor([pos_start + 1 + i * 3 for i in range(52)], dtype=torch.long)
        self.register_buffer("y_indices", y_indices)

    def forward(self, x):
        """x: (B, C, T) — normalize position channels by per-sample height."""
        # Gather Y values from position block
        y_vals = x[:, self.y_indices, :]                         # (B, 52, T)
        y_min = y_vals.amin(dim=(1, 2), keepdim=True)           # (B, 1, 1)
        y_max = y_vals.amax(dim=(1, 2), keepdim=True)
        height = (y_max - y_min).clamp(min=0.001)                # avoid /0

        # Divide position channels by height
        x_norm = x.clone()
        x_norm[:, self.pos_start:self.pos_end, :] /= height
        return x_norm


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


# ═══════════════════════════════════════════════════════════════════════
# Multi-label variant  —  independent sigmoid per code, BCEWithLogitsLoss
# ═══════════════════════════════════════════════════════════════════════

def _make_multilabel_tensors(data, target_frames: int, num_codes: int, device: str):
    """Convert gesture data into (N, C, T) tensor + multi-hot label matrix."""
    X = np.stack(
        [sample_frames(feats, target_frames) for feats, _ in data], axis=0
    )
    y = np.array([label for _, label in data], dtype=np.float32)
    # Ensure y is (N, num_codes) — pad/truncate if needed
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] < num_codes:
        y = np.pad(y, ((0, 0), (0, num_codes - y.shape[1])), constant_values=0.0)
    elif y.shape[1] > num_codes:
        y = y[:, :num_codes]
    X = torch.tensor(X, dtype=torch.float32, device=device).permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    return X, y


class MultiLabelGestureModelBase:
    """Shared training / eval / predict / save / load loop for multi-label gesture models.

    Uses BCEWithLogitsLoss — each output is an independent sigmoid.
    Labels are multi-hot float vectors of shape (N, num_codes).
    """

    def __init__(self, model: nn.Module, num_codes: int, target_frames: int,
                 lr: float, weight_decay: float, device: str, name: str):
        self.name = name
        self.num_codes = num_codes
        self.target_frames = target_frames
        self.lr = lr
        self.device = device
        self.model = model.to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

    # ------------------------------------------------------------------
    def train(self, train_data, val_data=None, epoch=0, total_epochs=0):
        self.model.train()
        X, y = _make_multilabel_tensors(
            train_data, self.target_frames, self.num_codes, self.device
        )
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), batch_size=16, shuffle=True
        )

        total_loss, correct, total = 0.0, 0, 0
        total_possible = 0
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
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == by).float().sum().item()
            total_possible += by.numel()
            total += bx.size(0)

            if (i + 1) % report_every == 0 or i == n_batches - 1:
                pct = (i + 1) / n_batches * 100
                filled = int(20 * (i + 1) / n_batches)
                bar = "█" * filled + "░" * (20 - filled)
                rl = total_loss / total
                ra = correct / total_possible if total_possible else 0.0
                print(f"\r  {label}  {bar}  {pct:3.0f}%  loss={rl:.4f}  acc={ra:.3f}",
                      end="", flush=True)
        print()

        self.scheduler.step()
        result = {"train_loss": total_loss / total,
                  "train_acc": correct / total_possible if total_possible else 0.0}
        if val_data:
            result["val_loss"], result["val_acc"] = self._evaluate(val_data)
        return result

    def _evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            X, y = _make_multilabel_tensors(
                data, self.target_frames, self.num_codes, self.device
            )
            logits = self.model(X)
            loss = self.criterion(logits, y).item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean().item()
        return loss, acc

    def predict(self, data):
        """Return binary label vectors  —  List[List[int]]."""
        if not data:
            return []
        self.model.eval()
        with torch.no_grad():
            X, _ = _make_multilabel_tensors(
                data, self.target_frames, self.num_codes, self.device
            )
            return (torch.sigmoid(self.model(X)) > 0.5).int().tolist()

    def predict_proba(self, data):
        """Return raw sigmoid probabilities  —  List[List[float]]."""
        if not data:
            return []
        self.model.eval()
        with torch.no_grad():
            X, _ = _make_multilabel_tensors(
                data, self.target_frames, self.num_codes, self.device
            )
            return torch.sigmoid(self.model(X)).tolist()

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
