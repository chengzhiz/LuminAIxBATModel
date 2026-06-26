"""Multi-label gesture model: 4 binary sigmoid heads (contrasting pairs).

Each head encodes a mutually-exclusive pair:
    head 0: body      0=LowerBody(LB)  1=UpperBody(UB)
    head 1: limb      0=SingleLimb(SL) 1=DualLimb(DL)
    head 2: symmetry  0=Symmetric(SY)  1=Asymmetric(AS)
    head 3: contact   0=Ground(G)      1=Air(A)

Every gesture has exactly one from each pair.

For Unity (8-code output), the 4 heads are expanded:
    LB=1-head0, UB=head0, SL=1-head1, DL=head1,
    SY=1-head2, AS=head2, G=1-head3, A=head3
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple


# ── Attribute parsing ─────────────────────────────────────────────
def parse_attributes(dir_name: str) -> List[int]:
    """Parse directory name into 4-element binary vector (one per contrasting pair).

    Returns [body, limb, symmetry, contact] where:
        body:     0=LowerBody(LB)  1=UpperBody(UB)
        limb:     0=SingleLimb(SL) 1=DualLimb(DL)
        symmetry: 0=Symmetric(SY)  1=Asymmetric(AS)
        contact:  0=Ground(G)      1=Air(A)
    """
    parts = dir_name.split("_")
    return [
        0 if parts[0] == "lb" else 1,   # body: 0=LB, 1=UB
        0 if parts[1] == "sl" else 1,   # limb: 0=SL, 1=DL
        0 if parts[2] == "sy" else 1,   # symmetry: 0=SY, 1=AS
        0 if parts[3] == "g" else 1,    # contact: 0=G, 1=A
    ]


# ── Frame sampling ────────────────────────────────────────────────
def sample_frames(features_list: List[List[float]], target_len: int) -> np.ndarray:
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


# ── CNN backbone ──────────────────────────────────────────────────
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


class _MultiLabelCNN(nn.Module):
    """Shared CNN trunk + 4 independent sigmoid heads (one per contrasting pair).

    The 4 heads correspond to: body, limb, symmetry, contact.
    For Unity's 8-code output, use expand_4to8().
    """
    def __init__(self, in_channels=73, num_heads=4, dropout=0.5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, 3, padding=1), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        # 4 independent binary heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 64), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(64, 1),
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        feats = self.trunk(x)              # (B, 512)
        return torch.cat([head(feats) for head in self.heads], dim=1)  # (B, 4)


def expand_4to8(logits_4: torch.Tensor) -> torch.Tensor:
    """Expand 4 contrasting-pair logits to 8 independent code logits.

    4 inputs:  [body, limb, symmetry, contact]  (each is logit for the 1-case)
    8 outputs: [LB, SL, AS, A, G, UB, DL, SY]
    """
    probs = torch.sigmoid(logits_4)  # (B, 4) — probability of the 1-case
    # LB = 1 - p_UB,  UB = p_UB,  SL = 1 - p_DL,  DL = p_DL, etc.
    return torch.cat([
        -logits_4[:, 0:1],         # LB = sigmoid(-body_logit) = 1 - sigmoid(body_logit)
        -logits_4[:, 1:2],         # SL
         logits_4[:, 2:3],         # AS
         logits_4[:, 3:4],         # A
        -logits_4[:, 3:4],         # G  = sigmoid(-contact_logit)
         logits_4[:, 0:1],         # UB
         logits_4[:, 1:2],         # DL
        -logits_4[:, 2:3],         # SY
    ], dim=1)  # (B, 8)


def expand_probs_4to8(probs_4: torch.Tensor) -> torch.Tensor:
    """Expand 4 sigmoid probabilities to 8 probabilities for Unity consumption."""
    # probs_4: (B, 4) — [p_UB, p_DL, p_AS, p_A]
    return torch.cat([
        1.0 - probs_4[:, 0:1],     # LB
        1.0 - probs_4[:, 1:2],     # SL
        probs_4[:, 2:3],           # AS
        probs_4[:, 3:4],           # A
        1.0 - probs_4[:, 3:4],     # G
        probs_4[:, 0:1],           # UB
        probs_4[:, 1:2],           # DL
        1.0 - probs_4[:, 2:3],     # SY
    ], dim=1)  # (B, 8)


# ── Trainer-compatible wrapper ────────────────────────────────────
class MultiLabelCNN:
    """4-head multi-label model for LimbExpression (expands to 8 for Unity).

    Internal: 4 sigmoid heads — [body, limb, symmetry, contact]
    Unity output: 8 codes — [LB, SL, AS, A, G, UB, DL, SY]
    """
    def __init__(self, num_heads=4, target_frames=128, lr=1e-3, weight_decay=1e-4,
                 dropout=0.5, device="cpu", name="multilabel_cnn"):
        self.name = name
        self.num_codes = num_heads  # 4 for training (pairs)
        self.num_unity_codes = 8    # 8 for Unity output
        self.target_frames = target_frames
        self.lr = lr
        self.device = device
        self.model = _MultiLabelCNN(in_channels=73, num_heads=num_heads,
                                     dropout=dropout).to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

    # ── Data helpers ───────────────────────────────────────────────
    def _prepare(self, data):
        X = np.stack([sample_frames(feats, self.target_frames) for feats, _ in data], axis=0)
        y = np.array([label for _, label in data], dtype=np.float32)
        X = torch.tensor(X, dtype=torch.float32, device=self.device).permute(0, 2, 1)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        return X, y

    # ── Training ──────────────────────────────────────────────────
    def train(self, train_data, val_data=None, epoch=0, total_epochs=0):
        self.model.train()
        X, y = self._prepare(train_data)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), batch_size=16, shuffle=True
        )

        total_loss, total = 0.0, 0
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
            total += bx.size(0)

            if (i + 1) % report_every == 0 or i == n_batches - 1:
                pct = (i + 1) / n_batches * 100
                filled = int(20 * (i + 1) / n_batches)
                bar = "█" * filled + "░" * (20 - filled)
                print(f"\r  {label}  {bar}  {pct:3.0f}%  loss={total_loss/total:.4f}",
                      end="", flush=True)
        print()

        self.scheduler.step()
        result = {"train_loss": total_loss / total, "train_acc": self._accuracy(train_data)}
        if val_data:
            result["val_loss"], result["val_acc"] = self._evaluate(val_data)
        return result

    def _evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            X, y = self._prepare(data)
            logits = self.model(X)
            loss = self.criterion(logits, y).item()
            acc = self._compute_accuracy(logits, y)
        return loss, acc

    def _accuracy(self, data):
        self.model.eval()
        with torch.no_grad():
            X, y = self._prepare(data)
            return self._compute_accuracy(self.model(X), y)

    @staticmethod
    def _compute_accuracy(logits, y):
        preds = (torch.sigmoid(logits) > 0.5).float()
        return (preds == y).float().mean().item()  # exact match across all 4 bits

    def predict(self, data):
        """Return 4-bit predictions for each gesture (for trainer metrics)."""
        if not data:
            return []
        self.model.eval()
        with torch.no_grad():
            X, _ = self._prepare(data)
            return (torch.sigmoid(self.model(X)) > 0.5).int().tolist()

    def predict_proba(self, data):
        """Return 4-element sigmoid probabilities (for trainer loss metrics)."""
        if not data:
            return []
        self.model.eval()
        with torch.no_grad():
            X, _ = self._prepare(data)
            return torch.sigmoid(self.model(X)).tolist()

    def predict_unity(self, data):
        """Return 8-element probabilities for Unity CSV output.
        Order: LB, SL, AS, A, G, UB, DL, SY
        """
        if not data:
            return []
        self.model.eval()
        with torch.no_grad():
            X, _ = self._prepare(data)
            probs_4 = torch.sigmoid(self.model(X))  # (B, 4)
            probs_8 = expand_probs_4to8(probs_4)     # (B, 8)
            return probs_8.tolist()

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
