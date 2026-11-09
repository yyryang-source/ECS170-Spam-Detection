"""Architectures for ECS 170 Stage 3: baseline CNN and residual CNNs.

Written for learning rather than pretrained transfer learning.
"""
import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two convolution layers with optional residual shortcut."""
    def __init__(self, in_ch, out_ch, kernel_size=3, residual=False):
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch))
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                         if residual and in_ch != out_ch else nn.Identity())
        self.residual = residual
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.body(x)
        if self.residual:
            y = y + self.shortcut(x)
        return self.act(y)


class Stage3CNN(nn.Module):
    """2D CNN: stages -> global avg pooling -> 128-unit head -> logits.

    A baseline and two controlled configurations support ablation studies:
    - baseline: 2 non-residual blocks (32,64), kernel 3
    - residual: 3 residual blocks (32,64,128), kernel 3
    - kernel5: same residual model, kernel 5 (controlled kernel-size ablation)
    """
    def __init__(self, in_channels, num_classes, config="residual", dropout=0.3,
                 pool_size=1):
        super().__init__()
        if config == "baseline":
            widths, kernel, residual = (32, 64), 3, False
        elif config == "residual":
            widths, kernel, residual = (32, 64, 128), 3, True
        elif config == "kernel5":
            widths, kernel, residual = (32, 64, 128), 5, True
        else:
            raise ValueError(f"Unknown config {config!r}")
        layers = []
        incoming = in_channels
        for width in widths:
            layers.extend([
                ConvBlock(incoming, width, kernel, residual),
                nn.MaxPool2d(2)
            ])
            incoming = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_size, pool_size)), nn.Flatten(),
            nn.Linear(incoming * pool_size**2, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_classes))
        self.config = config

    def forward(self, images):
        return self.head(self.features(images))


if __name__ == "__main__":
    for channels, nclass, spatial in [(1, 10, 28), (1, 40, 64), (3, 10, 32)]:
        for name in ("baseline", "residual", "kernel5"):
            out = Stage3CNN(channels, nclass, name)(torch.rand(2, channels, spatial, spatial))
            assert tuple(out.shape) == (2, nclass)
    print("All Stage 3 architecture shape checks passed")
