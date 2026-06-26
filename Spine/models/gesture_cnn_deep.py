"""Deep residual 1D CNN for gesture classification  (~72.5% val acc)."""

import torch.nn as nn
from ._gesture_base import GestureModelBase, HeightNormalize


class _ResBlock(nn.Module):
    """1D residual block: Conv → BN → ReLU → Conv → BN, with skip connection."""
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class _GestureCNNDeep(nn.Module):
    """Deeper CNN with residual blocks.  256 target frames."""
    def __init__(self, in_channels=732, num_classes=3, dropout=0.5):
        super().__init__()
        self.normalize = HeightNormalize()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(_ResBlock(128, 5), nn.MaxPool1d(2))
        self.stage2 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), _ResBlock(256, 3), nn.MaxPool1d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), _ResBlock(512, 3), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.normalize(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


class GestureCNNDeep(GestureModelBase):
    """Deep residual CNN.  256 target frames, skip connections, double dropout."""
    def __init__(self, num_classes=3, target_frames=256, lr=1e-3, weight_decay=1e-4,
                 dropout=0.5, device="cpu", name="gesture_cnn_deep"):
        model = _GestureCNNDeep(in_channels=732, num_classes=num_classes, dropout=dropout)
        super().__init__(model, target_frames, lr, weight_decay, device, name)
