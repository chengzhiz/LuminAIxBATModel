"""Baseline 1D CNN over time  (~78.4% val acc)."""

import torch.nn as nn
from ._gesture_base import GestureModelBase, HeightNormalize


class _GestureCNN(nn.Module):
    """1D CNN over time axis.  Input: (B, 732, T) → Output: (B, 3)."""
    def __init__(self, in_channels=732, num_classes=3, dropout=0.5):
        super().__init__()
        self.normalize = HeightNormalize()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, 3, padding=1), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.normalize(x)
        return self.net(x)


class GestureCNN(GestureModelBase):
    """Baseline 1D CNN.  128 target frames, 3 conv blocks."""
    def __init__(self, num_classes=3, target_frames=128, lr=1e-3, weight_decay=1e-4,
                 dropout=0.5, device="cpu", name="gesture_cnn"):
        model = _GestureCNN(in_channels=732, num_classes=num_classes, dropout=dropout)
        super().__init__(model, target_frames, lr, weight_decay, device, name)
