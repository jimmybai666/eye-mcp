"""
视杯视盘分割模型 - SegFormer (HuggingFace Transformers)

预训练模型来自:
  pamixsun/segformer_for_optic_disc_cup_segmentation
  https://huggingface.co/pamixsun/segformer_for_optic_disc_cup_segmentation
  作者: Xu Sun
  许可证: Apache-2.0
  训练数据: REFUGE challenge dataset

分割类别:
  0: Background (背景)
  1: Optic disc (视盘)
  2: Optic cup (视杯)
"""
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

NUM_CLASSES = 3
CLASS_NAMES = ["background", "optic_disc", "optic_cup"]

# HuggingFace 模型标识（用于在线下载回退）
HF_MODEL_NAME = "pamixsun/segformer_for_optic_disc_cup_segmentation"


def load_segformer(model_dir: str = None, from_hub: bool = False):
    """
    加载 SegFormer 视杯视盘分割模型

    Args:
        model_dir: 本地模型目录路径（包含 config.json, model.safetensors, preprocessor_config.json）
        from_hub: 是否从 HuggingFace Hub 下载

    Returns:
        (processor, model): 预处理器和模型
    """
    source = HF_MODEL_NAME if from_hub else model_dir

    processor = AutoImageProcessor.from_pretrained(source)
    model = SegformerForSemanticSegmentation.from_pretrained(source)

    return processor, model
