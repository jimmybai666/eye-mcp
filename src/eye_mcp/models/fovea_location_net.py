"""
Fovea 中央凹关键点定位模型
通过热力图回归定位中央凹位置，输出归一化坐标
"""
import torch
import torch.nn as nn
import torchvision.models as models


class FoveaHeatmapNet(nn.Module):
    """
    热力图回归网络：ResNet34 encoder + 上采样 decoder -> 预测 fovea 热力图
    输出一张单通道热力图，峰值位置即 fovea 坐标
    """

    def __init__(self, pretrained=False):
        super().__init__()
        resnet = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)

        # Encoder: ResNet34 去掉最后的 avgpool + fc
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,  # 64ch, /4
            resnet.layer2,  # 128ch, /8
            resnet.layer3,  # 256ch, /16
            resnet.layer4,  # 512ch, /32
        )

        # Decoder: 逐步上采样恢复分辨率到 1/4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # /16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # /8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # /4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 输出头: 1通道热力图
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.encoder(x)
        decoded = self.decoder(features)
        heatmap = self.head(decoded)
        return heatmap


def get_fovea_coords_from_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """
    从热力图中提取 fovea 坐标 (归一化 0~1)

    Args:
        heatmap: (B, 1, H, W) 预测热力图

    Returns:
        coords: (B, 2) 归一化坐标 [x, y]
    """
    B, _, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, -1)
    max_idx = heatmap_flat.argmax(dim=1)

    y = (max_idx // W).float() / H
    x = (max_idx % W).float() / W

    return torch.stack([x, y], dim=1)
