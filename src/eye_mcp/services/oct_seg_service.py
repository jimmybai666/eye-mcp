"""
OCT 分割推理服务 SDK
提供加载模型、预处理、推理、后处理的完整流程
基于 OMIHS 数据集训练的 UNet 模型，支持 5 类分割:
  bg / retina / choroid / irc / mh
"""
import asyncio
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eye_mcp.models.oct_seg_unet import build_unet, NUM_CLASSES, CLASS_NAMES

logger = logging.getLogger(__name__)

# 默认权重路径（项目根目录下的 weights/）
DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent.parent.parent / "weights" / "oct_seg.pth"

# ImageNet 标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 支持的图片格式
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# 各类别可视化颜色 (R, G, B)，与训练时可视化一致
CLASS_COLORS = [
    (0, 0, 0),        # 0: background - 黑（不叠加）
    (51, 153, 255),   # 1: retina - 蓝
    (255, 102, 0),    # 2: choroid - 橙
    (0, 255, 102),    # 3: irc - 绿
    (255, 51, 153),   # 4: mh - 粉红
]


class OCTSegService:
    """OCT 图像分割服务"""

    def __init__(self, weights_path: str = None, device: str = None, img_size: int = 512):
        """
        初始化分割服务

        Args:
            weights_path: 模型权重文件路径，默认使用项目内置权重
            device: 推理设备 ("cuda" / "cpu")，默认自动选择
            img_size: 输入图像尺寸，默认 512
        """
        self.img_size = img_size
        self.device = self._resolve_device(device)
        self.class_names = CLASS_NAMES
        self.num_classes = NUM_CLASSES

        # 加载模型
        self.model = build_unet(n_classes=NUM_CLASSES, pretrained=False)
        self._load_weights(weights_path or str(DEFAULT_WEIGHTS_PATH))
        self.model.to(self.device)
        self.model.eval()

        # 预处理 transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        logger.info(f"OCTSegService 初始化完成, device={self.device}, img_size={img_size}")

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

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """
        对单张 OCT 图像进行分割预测

        Args:
            image: PIL Image 对象 (RGB)

        Returns:
            dict: {
                "mask": np.ndarray (H, W) 每个像素的类别索引,
                "probabilities": np.ndarray (C, H, W) 每个类别的概率图,
                "class_names": list 类别名称,
                "statistics": dict 各类别像素占比
            }
        """
        original_size = image.size  # (W, H)

        # 预处理
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # 推理（CUDA 失败时自动回退 CPU）
        try:
            output = self.model(input_tensor)
        except RuntimeError as e:
            if self.device.type == "cuda":
                logger.warning(f"CUDA 推理失败: {e}，回退到 CPU 重试")
                self._fallback_to_cpu()
                input_tensor = input_tensor.to(self.device)
                output = self.model(input_tensor)
            else:
                raise

        probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)
        mask = probs.argmax(axis=0)  # (H, W)

        # 恢复原始尺寸
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                original_size, resample=Image.NEAREST
            )
        )

        # 统计各类别占比
        total_pixels = mask_resized.size
        statistics = {}
        for i, name in enumerate(CLASS_NAMES):
            count = int((mask_resized == i).sum())
            statistics[name] = {
                "pixel_count": count,
                "ratio": round(count / total_pixels, 4),
            }

        return {
            "mask": mask_resized,
            "probabilities": probs,
            "class_names": CLASS_NAMES,
            "statistics": statistics,
        }

    async def predict_async(self, image: Image.Image) -> dict:
        """
        异步版本的 predict，将推理放入线程池避免阻塞事件循环

        Args:
            image: PIL Image 对象 (RGB)

        Returns:
            同 predict() 返回值
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, image)

    def _fallback_to_cpu(self):
        """将模型从 CUDA 迁移到 CPU"""
        logger.info("正在将模型迁移到 CPU...")
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def overlay(self, image: Image.Image, mask: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """
        将分割 mask 以半透明颜色叠加到原图上

        Args:
            image: 原始 PIL Image (RGB)
            mask: 分割结果 np.ndarray (H, W)，值为类别索引
            alpha: 叠加透明度 (0~1)，越大颜色越明显，默认 0.4

        Returns:
            叠加后的 PIL Image (RGB)
        """
        image_rgb = image.convert("RGB").resize((mask.shape[1], mask.shape[0]))
        base = np.array(image_rgb, dtype=np.float32)

        # 构建颜色叠加层
        color_layer = np.zeros_like(base)
        for class_idx in range(1, self.num_classes):  # 跳过 background
            region = mask == class_idx
            if region.any():
                color_layer[region] = CLASS_COLORS[class_idx]

        # 只在非背景区域叠加
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
            dict: predict() 的返回值 + "overlay_image": PIL Image 叠加后的图
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

    async def predict_from_path_async(self, image_path: str) -> dict:
        """
        异步版本的 predict_from_path

        Args:
            image_path: 图像文件路径

        Returns:
            同 predict() 返回值
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_from_path, image_path)
