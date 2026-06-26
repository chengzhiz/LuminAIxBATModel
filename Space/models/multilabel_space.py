"""Multi-label model for Space — 5 sigmoid outputs: RV, ST, SP, H, M.

Matches Unity `space_codes` order in BATModelRunner.cs.
"""

import torch.nn as nn
import sys
from pathlib import Path

_floor = Path(__file__).resolve().parents[2] / "FloorSupport"
# Import MultiLabelGestureModelBase via direct module loading to avoid
# conflict with Space's own models package
import importlib.util
_mlgb_path = _floor / "models" / "_gesture_base.py"
_spec = importlib.util.spec_from_file_location("fs_gesture_base", _mlgb_path)
_fs_gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fs_gb)
MultiLabelGestureModelBase = _fs_gb.MultiLabelGestureModelBase


class _MultiLabelSpace(nn.Module):
    """Gesture-level multi-label model.  Input: (B, 73, T) → Output: (B, 5)."""
    def __init__(self, in_features=73, num_codes=5, hidden=256, num_layers=2,
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

    Output codes (in order): RV, ST, SP, H, M
    """
    def __init__(self, num_codes=5, target_frames=256, lr=1e-3,
                 weight_decay=1e-4, dropout=0.5, device="cpu",
                 name="multilabel_space_lstm"):
        model = _MultiLabelSpace(in_features=73, num_codes=num_codes, dropout=dropout)
        super().__init__(model, num_codes, target_frames, lr, weight_decay,
                        device, name)
