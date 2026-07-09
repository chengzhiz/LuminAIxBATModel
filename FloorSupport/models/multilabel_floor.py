"""Multi-label model for Floor Support — 4 sigmoid outputs: FT, HN, S, D.

Matches Unity `floor_codes` order in BATModelRunner.cs.
"""

import torch
import torch.nn as nn
from ._gesture_base import MultiLabelGestureModelBase
from ._gesture_base import HeightNormalize


class _MultiLabelFS(nn.Module):
    """Gesture-level multi-label model.  Input: (B, 73, T) → Output: (B, 4).

    No HeightNormalize needed — plumb-line features are already COG-relative.

    Uses two independent attention+classifier heads, one per pair group:
      Head A — FT/HN (group 0)
      Head B — S/D   (group 1)

    Each head can attend to different temporal patterns.  This is important
    because the frames that distinguish FT vs HN (body part) are different
    from the frames that distinguish S vs D (single/dual contact).  For
    example, Head B needs to focus on moments when both feet are contacting
    the ground to correctly identify the dual-foot (D) code.
    """
    def __init__(self, in_features=73, num_codes=4, hidden=256, num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features, hidden, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout,
        )
        lstm_out = hidden * 2  # 512

        # ── Head A: attention + classifier for FT/HN (indices 0,1) ─────
        self.attn_a = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.head_a = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        # ── Head B: attention + classifier for S/D (indices 2,3) ───────
        self.attn_b = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.head_b = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # x: (B, 73, T) — features, time
        out, _ = self.lstm(x.permute(0, 2, 1))   # (B, T, 512)

        # ── Head A: FT/HN ─────────────────────────────────────────────
        w_a = self.attn_a(out)                     # (B, T, 1)
        w_a = torch.softmax(w_a, dim=1)            # normalise over time
        pooled_a = (out * w_a).sum(dim=1)          # (B, 512)
        logits_a = self.head_a(pooled_a)           # (B, 2)

        # ── Head B: S/D ───────────────────────────────────────────────
        w_b = self.attn_b(out)                     # (B, T, 1)
        w_b = torch.softmax(w_b, dim=1)            # normalise over time
        pooled_b = (out * w_b).sum(dim=1)          # (B, 512)
        logits_b = self.head_b(pooled_b)           # (B, 2)

        return torch.cat([logits_a, logits_b], dim=1)  # (B, 4)


class MultiLabelFloorModel(MultiLabelGestureModelBase):
    """BiLSTM multi-label model for Floor Support.

    Output codes (in order): FT, HN, S, D

    Uses pair_groups to structurally enforce mutual exclusivity:
      Group 0: FT ↔ HN  (body part — exactly one active)
      Group 1: S  ↔ D   (count — exactly one active)

    Predictions are always one of 4 valid combos: FT+D, FT+S, HN+D, HN+S.
    CrossEntropyLoss is applied per group instead of independent sigmoid BCE.
    """
    def __init__(self, num_codes=4, target_frames=256, lr=1e-3,
                 weight_decay=1e-4, dropout=0.5, device="cpu",
                 name="multilabel_floor_lstm"):
        model = _MultiLabelFS(in_features=73, num_codes=num_codes, dropout=dropout)
        # Two independent binary choices:
        #   Group 0: FT vs HN  (body part)
        #   Group 1: S  vs D   (single/dual count)
        pair_groups = [(0, 1), (2, 3)]
        # Weight D 2.0× — tuned from a sweep.  The two-head architecture
        # (separate S/D attention) enables the model to actually identify D
        # frames (FT+D was 25 % → 75 % at 3.0×), but 3.0× overcorrected and
        # suppressed FT+S.  2.0× balances both.
        # Group positions: S=0, D=1 within logits[:, [2,3]]
        pair_weights = [None, torch.tensor([1.0, 2.0])]
        super().__init__(model, num_codes, target_frames, lr, weight_decay,
                        device, name, pair_groups=pair_groups,
                        pair_weights=pair_weights)
