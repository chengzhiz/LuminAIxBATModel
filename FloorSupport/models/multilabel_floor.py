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

    Uses attention pooling instead of mean pooling so the model can focus on
    the most discriminative frames (e.g. the few frames where both feet are
    visibly on the ground for the D / dual-foot code).
    """
    def __init__(self, in_features=73, num_codes=4, hidden=256, num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features, hidden, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout,
        )
        lstm_out = hidden * 2  # 512

        # ── Attention pooling: learn which frames matter ──────────────
        self.attn = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # ── Deeper classifier with LayerNorm for stability ────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_codes),
        )

    def forward(self, x):
        # x: (B, 73, T) — features, time
        out, _ = self.lstm(x.permute(0, 2, 1))   # (B, T, lstm_out)

        # Attention-weighted pooling — model learns which frames to focus on
        w = self.attn(out)                         # (B, T, 1)
        w = torch.softmax(w, dim=1)                # normalise over time
        pooled = (out * w).sum(dim=1)              # (B, lstm_out)

        return self.classifier(pooled)              # (B, num_codes)


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
        # Weight D 2× in the S/D group to penalise false negatives.
        # D (dual-foot) is harder to detect than S (single-foot), so
        # we up-weight it to push the model to learn the distinction.
        # Group positions: S=0, D=1 within logits[:, [2,3]]
        pair_weights = [None, torch.tensor([1.0, 2.0])]
        super().__init__(model, num_codes, target_frames, lr, weight_decay,
                        device, name, pair_groups=pair_groups,
                        pair_weights=pair_weights)
