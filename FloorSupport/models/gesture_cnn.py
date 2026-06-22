"""Gesture-level 1D CNN: classifies an entire motion recording (one JSON → one label).

Unlike the frame-level CNN1DModel which treats every bodyframe independently,
this model processes the full temporal sequence of a gesture.

Variable-length recordings are handled by uniform sampling to a fixed number of
frames — no padding waste, no synthetic interpolation.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Union


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames(features_list: List[List[float]], target_len: int) -> np.ndarray:
    """Uniformly sample *target_len* frames from a variable-length sequence.

    - If T < target_len  → pad by repeating the last frame
    - If T >= target_len → take evenly-spaced indices (no interpolation)

    Returns (target_len, feature_dim) float32 array.
    """
    T = len(features_list)
    arr = np.array(features_list, dtype=np.float32)
    if T == 0:
        return np.zeros((target_len, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32)
    if T < target_len:
        # Pad with copies of the last frame
        pad = np.tile(arr[-1:], (target_len - T, 1))
        arr = np.concatenate([arr, pad], axis=0)
        return arr
    # Uniform sampling
    indices = np.linspace(0, T - 1, target_len, dtype=int)
    return arr[indices]


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------

class GestureCNN1D(nn.Module):
    """1D CNN over the time axis.

    Input:  (B, channels, time)  e.g. (B, 732, 128)
    Output: (B, num_classes)
    """

    def __init__(self, in_channels: int = 732, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.in_channels = in_channels

        self.conv = nn.Sequential(
            # Block 1: 732 → 128 channels, kernel=7 over time
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # time / 2

            # Block 2: 128 → 256
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # time / 4

            # Block 3: 256 → 512
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),              # → (B, 512, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, channels, time)
        return self.classifier(self.conv(x))


# ---------------------------------------------------------------------------
# Trainer-compatible wrapper
# ---------------------------------------------------------------------------

class GestureCNNModel:
    """Wraps GestureCNN1D to match the BaseModel / Trainer interface."""

    def __init__(
        self,
        num_classes: int = 3,
        target_frames: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.5,
        device: str = "cpu",
        name: str = "gesture_cnn",
    ):
        self.name = name
        self.num_classes = num_classes
        self.target_frames = target_frames
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

        self.model = GestureCNN1D(
            in_channels=732,
            num_classes=num_classes,
            dropout=dropout,
        ).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )
        self._gestures = None   # cached sampled gestures

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _prepare_gestures(self, data):
        """Convert per-JSON feature lists into (N, channels, target_frames) tensor.

        *data* is a list of (features_list, label) where features_list is the
        raw variable-length list-of-lists returned by load_feature_vectors_from_json.
        """
        X = np.stack(
            [sample_frames(feats, self.target_frames) for feats, _ in data],
            axis=0,
        )  # (N, target_frames, 732)
        y = np.array([label for _, label in data], dtype=np.int64)

        # Convert to (N, channels, time) for Conv1d
        X = torch.tensor(X, dtype=torch.float32, device=self.device).permute(0, 2, 1)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_data, val_data=None, epoch=0, total_epochs=0):
        """Train one epoch on gesture-level data."""
        self.model.train()

        X, y = self._prepare_gestures(train_data)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = len(loader)
        report_every = max(1, n_batches // 10)
        epoch_label = f"Epoch {epoch}/{total_epochs}" if total_epochs else "Training"

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
                bar_width = 20
                filled = int(bar_width * (i + 1) / n_batches)
                bar = "█" * filled + "░" * (bar_width - filled)
                running_loss = total_loss / total
                running_acc = correct / total
                print(
                    f"\r  {epoch_label}  {bar}  {pct:3.0f}%  "
                    f"loss={running_loss:.4f}  acc={running_acc:.3f}",
                    end="", flush=True,
                )
        print()

        self.scheduler.step()

        result = {
            "train_loss": total_loss / total if total else 0.0,
            "train_acc": correct / total if total else 0.0,
        }

        if val_data:
            val_loss, val_acc = self._evaluate(val_data)
            result["val_loss"] = val_loss
            result["val_acc"] = val_acc

        return result

    def _evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            X, y = self._prepare_gestures(data)
            logits = self.model(X)
            loss = self.criterion(logits, y).item()
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        return loss, acc

    def predict(self, data):
        """Predict class for a list of (feature_list, label) gesture samples."""
        if not data:
            return []
        self.model.eval()
        with torch.no_grad():
            X, _ = self._prepare_gestures(data)
            logits = self.model(X)
            return logits.argmax(dim=1).tolist()

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
