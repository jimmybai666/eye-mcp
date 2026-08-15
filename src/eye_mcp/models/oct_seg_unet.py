"""
UNet 模型定义 - 使用 ResNet34 预训练编码器 (segmentation_models_pytorch)
用于 OMIHS 数据集 OCT 图像分割: bg/retina/choroid/irc/mh
"""
import segmentation_models_pytorch as smp


NUM_CLASSES = 5
CLASS_NAMES = ["background", "retina", "choroid", "irc", "mh"]


def build_unet(n_classes=NUM_CLASSES, encoder_name="resnet34", pretrained=False):
    """
    构建带预训练编码器的 UNet

    Args:
        n_classes: 输出类别数 (5: bg/retina/choroid/irc/mh)
        encoder_name: 编码器骨干网络
        pretrained: 是否使用 ImageNet 预训练权重
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=n_classes,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_norm="batchnorm",
    )
