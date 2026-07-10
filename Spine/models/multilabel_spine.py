"""Single-label model for Spine — 6-class softmax: E, F, HG, LF, SR, U.

Matches Unity `spine_codes` order in BATModelRunner.cs.
Uses softmax — codes are mutually exclusive (one spine position at a time).
SR and U have no training data but are included as classes 4 and 5.

Architecture: BiLSTM + attention pooling + deep classifier with LayerNorm.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List


class _SpineClassifier(nn.Module):
    """Gesture-level single-label model.  Input: (B, 73, T) -> Output: (B, 6).

    Uses attention pooling (instead of mean pooling) so the model can focus on
    the most discriminative frames — e.g. the peak of a flexion vs the recovery.
    Classifier kept shallow with LayerNorm to avoid overfitting on 283 samples.
    """
    def __init__(self, in_features=73, num_classes=6, hidden=256, num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features, hidden, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout,
        )
        lstm_out = hidden * 2  # 512

        # ── Attention pooling — learn which frames matter ──────────────
        self.attn = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # ── Classifier with LayerNorm (kept shallow to avoid overfitting) ─
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x.permute(0, 2, 1))   # (B, T, 512)

        # Attention-weighted pooling
        w = self.attn(out)                         # (B, T, 1)
        w = torch.softmax(w, dim=1)                # normalise over time
        pooled = (out * w).sum(dim=1)              # (B, 512)

        return self.classifier(pooled)              # (B, 6)


# ── Frame sampling ─────────────────────────────────────────────────
def sample_frames(features_list, target_len, jitter=False):
    """Uniformly sample target_len frames.  With jitter=True, sampling
    positions are randomly perturbed (training-time augmentation)."""
    T = len(features_list)
    arr = np.array(features_list, dtype=np.float32)
    if T == 0:
        return np.zeros((target_len, arr.shape[1] if arr.ndim == 2 else 0),
                        dtype=np.float32)
    if T < target_len:
        arr = np.concatenate([arr, np.tile(arr[-1:], (target_len - T, 1))], axis=0)
        return arr
    if jitter:
        base = np.linspace(0, T - 1, target_len)
        base = base + np.random.uniform(-0.5, 0.5, target_len) * (T - 1) / target_len
        return arr[np.clip(np.round(base).astype(int), 0, T - 1)]
    return arr[np.linspace(0, T - 1, target_len, dtype=int)]


def _make_tensors(data, target_frames, device, augment=False):
    """Convert gestures → tensors.  augment=True adds training-time
    augmentation: temporal jitter, Gaussian noise, and frame dropout —
    effective regularisers for small datasets (<300 samples)."""
    X = np.stack(
        [sample_frames(feats, target_frames, jitter=augment) for feats, _ in data],
        axis=0,
    )
    y = np.array([label for _, label in data], dtype=np.int64)
    X = torch.tensor(X, dtype=torch.float32, device=device).permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.long, device=device)

    if augment:
        # Gaussian noise — 2 % of global feature std
        X = X + torch.randn_like(X) * (0.02 * X.std())
        # Frame dropout — replace ~5 % of frames with their predecessor
        B, C, T = X.shape
        drop = (torch.rand(B, 1, T, device=X.device) < 0.05)
        X_prev = torch.cat([X[:, :, :1], X[:, :, :-1]], dim=2)
        X = torch.where(drop, X_prev, X)

    return X, y


class MultiLabelSpineModel:
    """Single-label 6-class softmax model for Spine.

    Output classes (in order): E=0, F=1, HG=2, LF=3, SR=4, U=5

    Uses class weights to compensate for imbalanced training data:
    LF (40 samples) gets extra weight vs E (72 samples).
    """
    def __init__(self, num_classes=6, target_frames=256, lr=1e-3,
                 weight_decay=1e-4, dropout=0.5, device="cpu",
                 name="spine_classifier_lstm"):
        self.name = name
        self.num_classes = num_classes
        self.target_frames = target_frames
        self.lr = lr
        self.device = device
        self.model = _SpineClassifier(in_features=73, num_classes=num_classes,
                                       dropout=dropout).to(device)

        # Class weights to help minority classes (LF=40 samples, others 72-87)
        # Higher weight -> model penalised more for getting that class wrong
        class_weight = torch.tensor([1.0, 1.2, 1.0, 3.0, 1.0, 1.0],
                                    dtype=torch.float32, device=device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

    def train(self, train_data, val_data=None, epoch=0, total_epochs=0):
        self.model.train()
        # augment=False — tested jitter+noise+frame-dropout: 71.2% vs 74.6%
        # baseline.  Augmentation amplifies label noise; fix labels first.
        X, y = _make_tensors(train_data, self.target_frames, self.device)
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
            X, y = _make_tensors(data, self.target_frames, self.device)
            logits = self.model(X)
            loss = self.criterion(logits, y).item()
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        return loss, acc

    def predict(self, data):
        """Return class indices 0-5."""
        if not data:
            return []
        self.model.eval()
        with torch.no_grad():
            X, _ = _make_tensors(data, self.target_frames, self.device)
            return self.model(X).argmax(dim=1).tolist()

    def predict_proba(self, data):
        """Return softmax probabilities (B, 6) for ONNX-compatible output."""
        if not data:
            return []
        self.model.eval()
        with torch.no_grad():
            X, _ = _make_tensors(data, self.target_frames, self.device)
            return torch.softmax(self.model(X), dim=1).tolist()

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
