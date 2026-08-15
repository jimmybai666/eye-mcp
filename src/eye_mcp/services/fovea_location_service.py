"""
Fovea 中央凹定位推理服务 SDK
提供加载模型、预处理、推理、可视化的完整流程
基于 IDRiD 数据集训练的热力图回归模型，输出中央凹坐标和十字可视化
"""
import asyncio
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from eye_mcp.models.fovea_location_net import (
    FoveaHeatmapNet,
    get_fovea_coords_from_heatmap,
)

logger = logging.getLogger(__name__)

# 默认权重路径
DEFAULT_WEIGHTS_PATH = (
    Path(__file__).parent.parent.parent.parent / "weights" / "fovea_location.pth"
)

# ImageNet 标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 支持的图片格式
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# 十字可视化颜色 (RGB) - 深绿色
CROSS_COLOR = (0, 153, 51)


class FoveaLocationService:
    """Fovea 中央凹定位服务"""

    def __init__(self, weights_path: str = None, device: str = None, img_size: int = 512):
        """
        初始化定位服务

        Args:
            weights_path: 模型权重文件路径，默认使用项目内置权重
            device: 推理设备 ("cuda" / "cpu")，默认自动选择
            img_size: 输入图像尺寸，默认 512
        """
        self.img_size = img_size
        self.device = self._resolve_device(device)

        # 加载模型
        self.model = FoveaHeatmapNet(pretrained=False)
        self._load_weights(weights_path or str(DEFAULT_WEIGHTS_PATH))
        self.model.to(self.device)
        self.model.eval()

        # 预处理 transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        logger.info(f"FoveaLocationService 初始化完成, device={self.device}, img_size={img_size}")

    @staticmethod
    def _resolve_device(device: str = None) -> torch.device:
        """解析推理设备，CUDA 不可用时自动回退 CPU"""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA 不可用，回退到 CPU 推理")
        return torch.device("cpu")

    def _load_weights(self, weights_path: str):
        """加载模型权重"""
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    def _fallback_to_cpu(self):
        """将模型从 CUDA 迁移到 CPU"""
        logger.info("正在将模型迁移到 CPU...")
        self.device = torch.device("cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """
        对单张眼底图像进行 fovea 定位

        Args:
            image: PIL Image 对象 (RGB)

        Returns:
            dict: {
                "x": float, 归一化 x 坐标 (0~1),
                "y": float, 归一化 y 坐标 (0~1),
                "pixel_x": int, 像素 x 坐标（基于原图尺寸）,
                "pixel_y": int, 像素 y 坐标（基于原图尺寸）,
                "image_size": {"width": int, "height": int},
            }
        """
        original_size = image.size  # (W, H)

        # 预处理
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # 推理（CUDA 失败时自动回退 CPU）
        try:
            heatmap = self.model(input_tensor)
        except RuntimeError as e:
            if self.device.type == "cuda":
                logger.warning(f"CUDA 推理失败: {e}，回退到 CPU 重试")
                self._fallback_to_cpu()
                input_tensor = input_tensor.to(self.device)
                heatmap = self.model(input_tensor)
            else:
                raise

        # 从热力图提取坐标
        coords = get_fovea_coords_from_heatmap(heatmap)
        x_norm, y_norm = coords[0].cpu().numpy()

        # 转为像素坐标
        pixel_x = int(x_norm * original_size[0])
        pixel_y = int(y_norm * original_size[1])

        return {
            "x": float(x_norm),
            "y": float(y_norm),
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "image_size": {"width": original_size[0], "height": original_size[1]},
        }

    def overlay(self, image: Image.Image, result: dict, line_width: int = 5) -> Image.Image:
        """
        在原图上绘制深绿色十字标记可视化

        Args:
            image: 原始 PIL Image (RGB)
            result: predict() 的返回值
            line_width: 十字线宽度，默认 5px

        Returns:
            叠加十字标记后的 PIL Image (RGB)
        """
        img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)

        w, h = img.size
        px = result["pixel_x"]
        py = result["pixel_y"]

        # 十字臂长：短边的 5%
        arm = int(min(w, h) * 0.05)

        # 画十字（粗线，深绿色）
        draw.line([(px - arm, py), (px + arm, py)], fill=CROSS_COLOR, width=line_width)
        draw.line([(px, py - arm), (px, py + arm)], fill=CROSS_COLOR, width=line_width)

        # 中心点
        dot_r = line_width + 1
        draw.ellipse(
            [(px - dot_r, py - dot_r), (px + dot_r, py + dot_r)],
            fill=CROSS_COLOR,
        )

        return img

    def predict_with_overlay(self, image: Image.Image, line_width: int = 5) -> dict:
        """
        预测并生成十字可视化图

        Args:
            image: PIL Image 对象 (RGB)
            line_width: 十字线宽度

        Returns:
            dict: predict() 的返回值 + "overlay_image": PIL Image
        """
        result = self.predict(image)
        result["overlay_image"] = self.overlay(image, result, line_width)
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

    async def predict_with_overlay_async(self, image: Image.Image, line_width: int = 5) -> dict:
        """异步版本的 predict_with_overlay"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_with_overlay, image, line_width)
