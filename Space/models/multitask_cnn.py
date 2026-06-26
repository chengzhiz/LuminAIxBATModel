"""Multi-task gesture model for Space: two independent softmax heads.

Head 1 — Energy:    High(0), Medium(1), Low(2)   [3 classes, only High in data]
Head 2 — Movement:  Revolution(0), Spring(1), Stationary(2)  [3 classes]
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "FloorSupport"))
from trainers import load_feature_vectors_from_json


# ── Label parsing ─────────────────────────────────────────────────
def parse_dual_label(dir_name: str) -> Tuple[int, int]:
    """Parse 'h_rv_high_revolution' → (energy=0, movement=0)."""
    parts = dir_name.split("_")
    energy = 0 if parts[0] == "h" else (1 if parts[0] == "m" else 2)
    movement = {"rv": 0, "sp": 1, "st": 2}[parts[1]]
    return energy, movement


# ── Frame sampling ────────────────────────────────────────────────
def sample_frames(features_list, target_len):
    T = len(features_list)
    arr = np.array(features_list, dtype=np.float32)
    if T == 0:
        return np.zeros((target_len, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32)
    if T < target_len:
        arr = np.concatenate([arr, np.tile(arr[-1:], (target_len - T, 1))], axis=0)
        return arr
    return arr[np.linspace(0, T - 1, target_len, dtype=int)]


# ── Model ─────────────────────────────────────────────────────────
class _HeightNorm(nn.Module):
    """Normalize position channels (indices 364-519) by per-sample height."""
    def __init__(self):
        super().__init__()
        y_idx = torch.tensor([364 + 1 + i * 3 for i in range(52)], dtype=torch.long)
        self.register_buffer("y_idx", y_idx)
        self.register_buffer("p_start", torch.tensor(364, dtype=torch.long))
        self.register_buffer("p_end", torch.tensor(520, dtype=torch.long))

    def forward(self, x):
        y = x[:, self.y_idx, :]
        h = (y.amax(dim=(1,2), keepdim=True) - y.amin(dim=(1,2), keepdim=True)).clamp(min=0.001)
        x = x.clone()
        x[:, self.p_start:self.p_end, :] /= h
        return x


class _MultiTaskCNN(nn.Module):
    def __init__(self, in_channels=732, dropout=0.5):
        super().__init__()
        self.normalize = _HeightNorm()
        self.trunk = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, 3, padding=1), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, 3),  # H/M/L
        )
        self.movement_head = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, 3),  # RV/SP/ST
        )

    def forward(self, x):
        x = self.normalize(x)
        feats = self.trunk(x)
        return self.energy_head(feats), self.movement_head(feats)


# ── Trainer wrapper ───────────────────────────────────────────────
class MultiTaskCNN:
    def __init__(self, target_frames=128, lr=1e-3, weight_decay=1e-4,
                 dropout=0.5, device="cpu", name="multitask_cnn"):
        self.name = name
        self.target_frames = target_frames
        self.lr = lr
        self.device = device
        self.model = _MultiTaskCNN(in_channels=732, dropout=dropout).to(device)
        self.ce = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

    def _prepare(self, data):
        X = np.stack([sample_frames(feats, self.target_frames) for feats, _ in data], axis=0)
        e = np.array([lbl[0] for _, lbl in data], dtype=np.int64)
        m = np.array([lbl[1] for _, lbl in data], dtype=np.int64)
        X = torch.tensor(X, dtype=torch.float32, device=self.device).permute(0, 2, 1)
        return X, torch.tensor(e, dtype=torch.long, device=self.device), \
                   torch.tensor(m, dtype=torch.long, device=self.device)

    def train(self, train_data, val_data=None, epoch=0, total_epochs=0):
        self.model.train()
        X, e, m = self._prepare(train_data)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, e, m), batch_size=16, shuffle=True
        )

        total_loss, total = 0.0, 0
        n_batches = len(loader)
        report_every = max(1, n_batches // 10)
        label = f"Epoch {epoch}/{total_epochs}" if total_epochs else "Training"

        for i, (bx, be, bm) in enumerate(loader):
            self.optimizer.zero_grad()
            e_out, m_out = self.model(bx)
            loss = self.ce(e_out, be) + self.ce(m_out, bm)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * bx.size(0)
            total += bx.size(0)

            if (i + 1) % report_every == 0 or i == n_batches - 1:
                pct = (i + 1) / n_batches * 100
                filled = int(20 * (i + 1) / n_batches)
                bar = "█" * filled + "░" * (20 - filled)
                print(f"\r  {label}  {bar}  {pct:3.0f}%  loss={total_loss/total:.4f}",
                      end="", flush=True)
        print()

        self.scheduler.step()
        result = {"train_loss": total_loss / total}
        e_acc, m_acc = self._accuracy(train_data)
        result["train_acc"] = (e_acc + m_acc) / 2
        if val_data:
            vloss, v_e_acc, v_m_acc = self._evaluate(val_data)
            result["val_loss"] = vloss
            result["val_acc"] = (v_e_acc + v_m_acc) / 2
        return result

    def _evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            X, e, m = self._prepare(data)
            e_out, m_out = self.model(X)
            loss = (self.ce(e_out, e) + self.ce(m_out, m)).item()
            e_acc = (e_out.argmax(1) == e).float().mean().item()
            m_acc = (m_out.argmax(1) == m).float().mean().item()
        return loss, e_acc, m_acc

    def _accuracy(self, data):
        self.model.eval()
        with torch.no_grad():
            X, e, m = self._prepare(data)
            e_out, m_out = self.model(X)
            e_acc = (e_out.argmax(1) == e).float().mean().item()
            m_acc = (m_out.argmax(1) == m).float().mean().item()
        return e_acc, m_acc

    def predict(self, data):
        if not data: return []
        self.model.eval()
        with torch.no_grad():
            X, _, _ = self._prepare(data)
            e_out, m_out = self.model(X)
            e_preds = e_out.argmax(1).tolist()
            m_preds = m_out.argmax(1).tolist()
        return list(zip(e_preds, m_preds))

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
