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
    """Gesture-level multi-label model.  Input: (B, 73, T) -> Output: (B, 7)."""
    def __init__(self, in_features=73, num_codes=7, hidden=256, num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features, hidden, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_codes),
        )

    def forward(self, x):
        out, _ = self.lstm(x.permute(0, 2, 1))
        return self.classifier(out.mean(dim=1))


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
        super().__init__(model, num_codes, target_frames, lr, weight_decay,
                        device, name, pair_groups=pair_groups)
