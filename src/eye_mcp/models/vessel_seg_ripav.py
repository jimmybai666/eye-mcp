"""
视网膜动静脉 (A/V) 血管分割模型 - RIP-AV (PGNet + ConvNeXt-Tiny)

模型来自:
  RIP-AV: Joint Representative Instance Pre-training and Context Aware
  Network for Retinal Artery/Vein Segmentation
  作者: Wei Dai
  来源: https://huggingface.co/spaces/weidai00/RIP-AV-su-lab
  许可证: MIT
  骨干网络: ConvNeXt-Tiny (ImageNet-1K 预训练)
  训练数据: AV-DRIVE / LES-AV / HRF

分割类别 (3通道输出):
  channel 0: artery (动脉)
  channel 1: vessel (全血管)
  channel 2: vein (静脉)
"""
from eye_mcp.models.ripav.network import PGNet
from eye_mcp.models.ripav.utils import (
    paint_border_overlap,
    extract_ordered_overlap,
    recompone_overlap,
    Normalize,
    sigmoid,
)

NUM_CLASSES = 3
CLASS_NAMES = ["artery", "vessel", "vein"]

# HuggingFace Space (用于回退下载)
HF_REPO_ID = "weidai00/RIP-AV-su-lab"
HF_WEIGHTS_FILENAME = "G_best.pkl"


def build_pgnet(use_cuda: bool = False, pretrained: bool = False):
    """
    构建 PGNet 血管分割模型

    Args:
        use_cuda: 是否使用 CUDA
        pretrained: 是否加载 ConvNeXt-Tiny ImageNet 预训练权重

    Returns:
        PGNet 模型实例
    """
    return PGNet(
        resnet="convnext_tiny",
        use_global_semantic=True,
        input_ch=3,
        num_classes=NUM_CLASSES,
        use_cuda=use_cuda,
        pretrained=pretrained,
        centerness=True,
        centerness_map_size=[128, 128],
    )
