"""Bidirectional LSTM for gesture classification  (~84.3% val acc)."""

import torch.nn as nn
from ._gesture_base import GestureModelBase, HeightNormalize


class _GestureLSTM(nn.Module):
    """Bidirectional LSTM.  Input: (B, T, 732) → Output: (B, 3)."""
    def __init__(self, in_features=732, hidden=256, num_layers=2, num_classes=3,
                 dropout=0.5):
        super().__init__()
        self.normalize = HeightNormalize()
        self.lstm = nn.LSTM(
            in_features, hidden, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.normalize(x)
        out, _ = self.lstm(x.permute(0, 2, 1))
        return self.classifier(out.mean(dim=1))


class GestureLSTM(GestureModelBase):
    """BiLSTM with 256 target frames.  Best-performing model so far."""
    def __init__(self, num_classes=3, target_frames=256, lr=1e-3, weight_decay=1e-4,
                 dropout=0.5, device="cpu", name="gesture_lstm"):
        model = _GestureLSTM(in_features=732, num_classes=num_classes, dropout=dropout)
        super().__init__(model, target_frames, lr, weight_decay, device, name)
