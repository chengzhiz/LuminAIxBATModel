"""Multi-label model for Space — 7 outputs in 2 pair groups.

Movement group (softmax): ST, T, RV, SP  (indices 0-3)
Energy group (softmax):   H,  M, L      (indices 4-6)

12 valid combos = 4 movement x 3 energy.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import importlib.util

_floor = Path(__file__).resolve().parents[2] / "FloorSupport"
_mlgb_path = _floor / "models" / "_gesture_base.py"
_spec = importlib.util.spec_from_file_location("fs_gesture_base", _mlgb_path)
_fs_gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fs_gb)
MultiLabelGestureModelBase = _fs_gb.MultiLabelGestureModelBase


class _MultiLabelSpace(nn.Module):
    """Gesture-level multi-label model.  Input: (B, 73, T) -> Output: (B, 7).

    Two independent attention+classifier heads, one per pair group:
      Head A — movement (ST, T, RV, SP — indices 0-3)
      Head B — energy   (H, M, L      — indices 4-6)

    Attention pooling (instead of mean pooling) is critical for the movement
    head: a spring/jump (SP) is a brief event — maybe 10-20 frames out of
    256 — and mean pooling washes it out, making SP look like ST
    (stationary).  Attention lets the head focus on the few frames where the
    jump actually happens.
    """
    def __init__(self, in_features=73, num_codes=7, hidden=256, num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features, hidden, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout,
        )
        lstm_out = hidden * 2  # 512

        # ── Head A: attention + classifier for movement (indices 0-3) ──
        self.attn_a = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head_a = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )

        # ── Head B: attention + classifier for energy (indices 4-6) ────
        self.attn_b = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head_b = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        out, _ = self.lstm(x.permute(0, 2, 1))   # (B, T, 512)

        # ── Head A: movement ──────────────────────────────────────────
        w_a = torch.softmax(self.attn_a(out), dim=1)   # (B, T, 1)
        pooled_a = (out * w_a).sum(dim=1)               # (B, 512)
        logits_a = self.head_a(pooled_a)                # (B, 4)

        # ── Head B: energy ────────────────────────────────────────────
        w_b = torch.softmax(self.attn_b(out), dim=1)   # (B, T, 1)
        pooled_b = (out * w_b).sum(dim=1)               # (B, 512)
        logits_b = self.head_b(pooled_b)                # (B, 3)

        return torch.cat([logits_a, logits_b], dim=1)   # (B, 7)


class MultiLabelSpaceModel(MultiLabelGestureModelBase):
    """BiLSTM multi-label model for Space.

    Output codes: ST, T, RV, SP, H, M, L

    Uses pair_groups to structurally enforce mutual exclusivity:
      Group 0: ST, T, RV, SP  (movement — 4-way softmax)
      Group 1: H,  M, L       (energy   — 3-way softmax)

    Predictions are always one of 12 valid combos.
    """
    def __init__(self, num_codes=7, target_frames=256, lr=1e-3,
                 weight_decay=1e-4, dropout=0.5, device="cpu",
                 name="multilabel_space_lstm"):
        model = _MultiLabelSpace(in_features=73, num_codes=num_codes, dropout=dropout)
        pair_groups = [(0, 1, 2, 3), (4, 5, 6)]
        # Movement class weights — ST is the minority (65 train samples vs
        # SP 95, RV 77) and was frequently misclassified as SP.  Up-weight
        # ST so the model builds a proper boundary for it.
        # Group positions: ST=0, T=1, RV=2, SP=3.  Energy group unweighted.
        pair_weights = [torch.tensor([1.8, 1.0, 1.2, 1.0]), None]
        super().__init__(model, num_codes, target_frames, lr, weight_decay,
                        device, name, pair_groups=pair_groups,
                        pair_weights=pair_weights)
