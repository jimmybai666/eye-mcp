"""
视杯视盘分割推理服务 SDK
基于 SegFormer 模型，分割眼底图像中的视盘(optic disc)和视杯(optic cup)
计算杯盘比(CDR)等临床指标

预训练模型来自:
  pamixsun/segformer_for_optic_disc_cup_segmentation
  许可证: Apache-2.0
"""
import asyncio
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from eye_mcp.models.optic_disc_cup_seg import (
    load_segformer,
    HF_MODEL_NAME,
    NUM_CLASSES,
    CLASS_NAMES,
)

logger = logging.getLogger(__name__)

# 默认本地权重目录
DEFAULT_WEIGHTS_DIR = (
    Path(__file__).parent.parent.parent.parent / "weights" / "optic_disc_cup_seg"
)

# 支持的图片格式
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# 各类别可视化颜色 (RGB)
CLASS_COLORS = [
    (0, 0, 0),        # 0: background - 黑
    (51, 102, 255),   # 1: optic disc - 蓝色
    (0, 153, 51),     # 2: optic cup - 深绿色
]


class OpticDiscCupSegService:
    """视杯视盘分割服务"""

    def __init__(self, weights_dir: str = None, device: str = None):
        """
        初始化分割服务

        Args:
            weights_dir: 本地模型目录路径，默认使用项目内置权重；
                         若本地不存在则自动从 HuggingFace Hub 下载
            device: 推理设备 ("cuda" / "cpu")，默认自动选择
        """
        self.device = self._resolve_device(device)
        self.class_names = CLASS_NAMES
        self.num_classes = NUM_CLASSES

        # 加载模型（本地优先，找不到则从 HF Hub 下载）
        self.processor, self.model = self._load_model(
            weights_dir or str(DEFAULT_WEIGHTS_DIR)
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"OpticDiscCupSegService 初始化完成, device={self.device}")

    @staticmethod
    def _resolve_device(device: str = None) -> torch.device:
        """解析推理设备，CUDA 不可用时自动回退 CPU"""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA 不可用，回退到 CPU 推理")
        return torch.device("cpu")

    def _load_model(self, weights_dir: str):
        """
        加载模型：优先本地目录，找不到则从 HuggingFace Hub 下载

        Args:
            weights_dir: 本地模型目录

        Returns:
            (processor, model)
        """
        config_path = os.path.join(weights_dir, "config.json")

        if os.path.isfile(config_path):
            logger.info(f"从本地加载模型: {weights_dir}")
            return load_segformer(model_dir=weights_dir, from_hub=False)
        else:
            logger.warning(
                f"本地权重不存在: {weights_dir}，从 HuggingFace Hub 下载: {HF_MODEL_NAME}"
            )
            return load_segformer(from_hub=True)

    def _fallback_to_cpu(self):
        """将模型从 CUDA 迁移到 CPU"""
        logger.info("正在将模型迁移到 CPU...")
        self.device = torch.device("cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """
        对单张眼底图像进行视杯视盘分割

        Args:
            image: PIL Image 对象 (RGB)

        Returns:
            dict: {
                "mask": np.ndarray (H, W) 每个像素的类别索引,
                "class_names": list 类别名称,
                "statistics": dict 各类别像素占比,
                "cdr": float 杯盘比 (cup-to-disc ratio),
                "image_size": {"width": int, "height": int},
            }
        """
        image_rgb = image.convert("RGB")
        original_size = image_rgb.size  # (W, H)
        h, w = original_size[1], original_size[0]

        # 预处理
        inputs = self.processor(image_rgb, return_tensors="pt").to(self.device)

        # 推理（CUDA 失败时自动回退 CPU）
        try:
            outputs = self.model(**inputs)
        except RuntimeError as e:
            if self.device.type == "cuda":
                logger.warning(f"CUDA 推理失败: {e}，回退到 CPU 重试")
                self._fallback_to_cpu()
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
            else:
                raise

        logits = outputs.logits.cpu()

        # 插值回原图大小
        upsampled = nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        mask = upsampled.argmax(dim=1)[0].numpy().astype(np.uint8)

        # 统计各类别占比
        total_pixels = mask.size
        statistics = {}
        for i, name in enumerate(CLASS_NAMES):
            count = int((mask == i).sum())
            statistics[name] = {
                "pixel_count": count,
                "ratio": round(count / total_pixels, 4),
            }

        # 计算杯盘比 CDR
        disc_pixels = int((mask == 1).sum()) + int((mask == 2).sum())  # 视盘包含视杯
        cup_pixels = int((mask == 2).sum())
        cdr = round(cup_pixels / disc_pixels, 4) if disc_pixels > 0 else 0.0

        return {
            "mask": mask,
            "class_names": CLASS_NAMES,
            "statistics": statistics,
            "cdr": cdr,
            "image_size": {"width": w, "height": h},
        }

    def overlay(self, image: Image.Image, mask: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """
        将分割 mask 以半透明颜色叠加到原图上

        Args:
            image: 原始 PIL Image (RGB)
            mask: 分割结果 np.ndarray (H, W)
            alpha: 叠加透明度 (0~1)

        Returns:
            叠加后的 PIL Image (RGB)
        """
        image_rgb = image.convert("RGB").resize((mask.shape[1], mask.shape[0]))
        base = np.array(image_rgb, dtype=np.float32)

        color_layer = np.zeros_like(base)
        for class_idx in range(1, self.num_classes):  # 跳过 background
            region = mask == class_idx
            if region.any():
                color_layer[region] = CLASS_COLORS[class_idx]

        foreground = mask > 0
        blended = base.copy()
        blended[foreground] = (
            base[foreground] * (1 - alpha) + color_layer[foreground] * alpha
        )

        return Image.fromarray(blended.astype(np.uint8))

    def predict_with_overlay(self, image: Image.Image, alpha: float = 0.4) -> dict:
        """
        预测并生成叠加可视化图

        Args:
            image: PIL Image 对象 (RGB)
            alpha: 叠加透明度

        Returns:
            dict: predict() 的返回值 + "overlay_image": PIL Image
        """
        result = self.predict(image)
        result["overlay_image"] = self.overlay(image, result["mask"], alpha)
        return result

    def predict_from_path(self, image_path: str) -> dict:
        """
        从文件路径加载图像并预测

        Args:
            image_path: 图像文件路径

        Returns:
            同 predict() 返回值

        Raises:
            ValueError: 图片格式不支持
            FileNotFoundError: 文件不存在
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的图片格式: {suffix}，"
                f"支持的格式: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        image = Image.open(image_path).convert("RGB")
        return self.predict(image)

    async def predict_async(self, image: Image.Image) -> dict:
        """异步版本的 predict"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, image)

    async def predict_from_path_async(self, image_path: str) -> dict:
        """异步版本的 predict_from_path"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_from_path, image_path)

    async def predict_with_overlay_async(self, image: Image.Image, alpha: float = 0.4) -> dict:
        """异步版本的 predict_with_overlay"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_with_overlay, image, alpha)
