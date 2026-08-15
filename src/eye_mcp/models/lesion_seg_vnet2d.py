"""
VNet2d - 2D V-Net for fundus lesion segmentation.

4 independent binary segmentation models, one per lesion type:
- MA: 微血管瘤 (Microaneurysm)
- HE: 出血 (Haemorrhage)
- EX: 硬性渗出 (Exudate)
- SE: 软性渗出/棉絮斑 (Soft Exudate)

Each model: VNet2d(in_channels=3, num_classes=1) → sigmoid → binary mask
"""

import torch
import torch.nn as nn


LESION_CLASSES = ["MA", "HE", "EX", "SE"]

LESION_NAMES_CN = {
    "MA": "微血管瘤",
    "HE": "出血",
    "EX": "硬性渗出",
    "SE": "软性渗出(棉絮斑)",
}


class ResBlock(nn.Module):
    def __init__(self, channels, num_convs):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.PReLU(channels))
        self.ops = nn.Sequential(*layers)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        return self.prelu(self.ops(x) + x)


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=1):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(out_channels)
        self.prelu_in = nn.PReLU(out_channels)
        self.res_block = ResBlock(out_channels, num_convs)

    def forward(self, x):
        out = self.prelu_in(self.bn_in(self.conv_in(x)))
        out = self.res_block(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.res_block = ResBlock(out_channels, num_convs)

    def forward(self, x):
        out = self.down(x)
        out = self.res_block(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.res_block = ResBlock(out_channels, num_convs)

    def forward(self, x, skip):
        out = self.up(x)
        out = torch.cat([out, skip], dim=1)
        out = self.conv_reduce(out)
        out = self.res_block(out)
        return out


class VNet2d(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc1 = InputBlock(in_channels, 16, num_convs=1)
        self.enc2 = DownBlock(16, 32, num_convs=2)
        self.enc3 = DownBlock(32, 64, num_convs=3)
        self.enc4 = DownBlock(64, 128, num_convs=3)
        self.bottleneck = DownBlock(128, 256, num_convs=3)
        self.dec4 = UpBlock(256, 128, num_convs=3)
        self.dec3 = UpBlock(128, 64, num_convs=3)
        self.dec2 = UpBlock(64, 32, num_convs=2)
        self.dec1 = UpBlock(32, 16, num_convs=1)
        self.out_conv = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)
        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)
