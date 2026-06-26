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
    """
    def __init__(self, in_features=73, num_codes=4, hidden=256, num_layers=2,
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
        # x: (B, 73, T) — features, time
        out, _ = self.lstm(x.permute(0, 2, 1))   # (B, T, hidden*2)
        return self.classifier(out.mean(dim=1))   # (B, num_codes)  — temporal pooling


class MultiLabelFloorModel(MultiLabelGestureModelBase):
    """BiLSTM multi-label model for Floor Support.

    Output codes (in order): FT, HN, S, D
    """
    def __init__(self, num_codes=4, target_frames=256, lr=1e-3,
                 weight_decay=1e-4, dropout=0.5, device="cpu",
                 name="multilabel_floor_lstm"):
        model = _MultiLabelFS(in_features=73, num_codes=num_codes, dropout=dropout)
        super().__init__(model, num_codes, target_frames, lr, weight_decay,
                        device, name)
