"""
视网膜动静脉血管分割推理服务 SDK
基于 RIP-AV (PGNet + ConvNeXt-Tiny) 模型
分割眼底图像中的动脉(artery)、静脉(vein)、全血管(vessel)

模型来自:
  RIP-AV: Wei Dai
  许可证: MIT
"""
import asyncio
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from eye_mcp.models.vessel_seg_ripav import (
    build_pgnet,
    paint_border_overlap,
    extract_ordered_overlap,
    recompone_overlap,
    Normalize,
    sigmoid,
    NUM_CLASSES,
    CLASS_NAMES,
    HF_REPO_ID,
    HF_WEIGHTS_FILENAME,
)

logger = logging.getLogger(__name__)

# 默认权重路径
DEFAULT_WEIGHTS_PATH = (
    Path(__file__).parent.parent.parent.parent / "weights" / "vessel_seg.pkl"
)

# 支持的图片格式
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# 模型训练使用的固定分辨率 (宽, 高)
TARGET_RESOLUTION = (1620, 1444)

# 各类别可视化颜色 (RGB)
CLASS_COLORS = {
    "artery": (255, 50, 50),   # 动脉 - 红色
    "vein": (51, 102, 255),    # 静脉 - 蓝色
    "vessel": (0, 200, 100),   # 全血管 - 浅绿
}


class VesselSegService:
    """视网膜动静脉血管分割服务"""

    def __init__(self, weights_path: str = None, device: str = None):
        """
        初始化血管分割服务

        Args:
            weights_path: 模型权重文件路径，默认使用项目内置权重；
                         若本地不存在则自动从 HuggingFace 下载
            device: 推理设备 ("cuda" / "cpu")，默认自动选择
        """
        self.device = self._resolve_device(device)
        self.use_cuda = self.device.type == "cuda"
        self.class_names = CLASS_NAMES
        self.num_classes = NUM_CLASSES

        # 推理参数
        self.patch_size = 256
        self.stride_height = 50
        self.stride_width = 50
        self.batch_size = 8

        # 加载模型
        self.model = build_pgnet(use_cuda=self.use_cuda, pretrained=False)
        self._load_weights(weights_path or str(DEFAULT_WEIGHTS_PATH))
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()

        logger.info(f"VesselSegService 初始化完成, device={self.device}")

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
        """加载模型权重：本地优先，找不到从 HuggingFace 下载"""
        if not os.path.isfile(weights_path):
            logger.warning(f"本地权重不存在: {weights_path}，尝试从 HuggingFace 下载")
            try:
                from huggingface_hub import hf_hub_download
                weights_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=HF_WEIGHTS_FILENAME,
                    repo_type="space",
                )
                logger.info(f"从 HuggingFace 下载成功: {weights_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"权重文件不存在且 HuggingFace 下载失败: {e}"
                )

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

    def _fallback_to_cpu(self):
        """将模型从 CUDA 迁移到 CPU"""
        logger.info("正在将模型迁移到 CPU...")
        self.device = torch.device("cpu")
        self.use_cuda = False
        self.model.cpu()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """
        对单张眼底图像进行动静脉血管分割

        Args:
            image: PIL Image 对象 (RGB)

        Returns:
            dict: {
                "artery_mask": np.ndarray (H, W) bool,
                "vein_mask": np.ndarray (H, W) bool,
                "vessel_mask": np.ndarray (H, W) bool,
                "class_names": list,
                "statistics": dict 各类别像素占比,
                "image_size": {"width": int, "height": int},
            }
        """
        original_size = image.size  # (W, H)
        img_rgb = np.array(image.convert("RGB"))

        # 调整为训练尺寸
        img_resized = cv2.resize(img_rgb, TARGET_RESOLUTION)
        img_float = np.float32(img_resized / 255.0)

        # Padding
        img_enlarged = paint_border_overlap(
            img_float, self.patch_size, self.patch_size,
            self.stride_height, self.stride_width
        )

        # 提取 Patch
        patches_imgs, _ = extract_ordered_overlap(
            img_enlarged, self.patch_size, self.patch_size,
            self.stride_height, self.stride_width
        )
        patches_imgs = np.transpose(patches_imgs, (0, 3, 1, 2))
        patches_imgs = Normalize(patches_imgs)

        # 全局语义图
        global_img = cv2.resize(img_enlarged, (self.patch_size, self.patch_size))
        global_img = np.transpose(global_img, (2, 0, 1))
        global_img = np.expand_dims(global_img, axis=0)
        global_img = Normalize(global_img)

        # 分 batch 推理
        patch_num = patches_imgs.shape[0]
        max_iter = int(np.ceil(patch_num / float(self.batch_size)))
        pred_patches = np.zeros(
            (patch_num, NUM_CLASSES, self.patch_size, self.patch_size), np.float32
        )

        try:
            for i in range(max_iter):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, patch_num)

                batch_patch = torch.FloatTensor(patches_imgs[start:end])
                repeat_global = torch.FloatTensor(
                    np.repeat(global_img, end - start, axis=0)
                )

                if self.use_cuda:
                    batch_patch = batch_patch.cuda()
                    repeat_global = repeat_global.cuda()

                outputs, _ = self.model(batch_patch, repeat_global)
                pred_patches[start:end] = sigmoid(outputs.cpu().numpy())
        except RuntimeError as e:
            if self.device.type == "cuda":
                logger.warning(f"CUDA 推理失败: {e}，回退到 CPU 重试")
                self._fallback_to_cpu()
                return self.predict(image)  # 递归重试一次
            raise

        # 重建预测图
        pred_img = recompone_overlap(
            pred_patches, img_enlarged.shape[0], img_enlarged.shape[1],
            self.stride_height, self.stride_width
        )
        pred_img = pred_img[:, :TARGET_RESOLUTION[1], :TARGET_RESOLUTION[0]]

        # 生成 mask (阈值 0.5)
        artery_mask = pred_img[0] > 0.5
        vessel_mask = pred_img[1] > 0.5
        vein_mask = pred_img[2] > 0.5

        # 缩放回原图尺寸
        artery_mask = cv2.resize(
            artery_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        vein_mask = cv2.resize(
            vein_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        vessel_mask = cv2.resize(
            vessel_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # 统计
        total_pixels = original_size[0] * original_size[1]
        statistics = {
            "artery": {
                "pixel_count": int(artery_mask.sum()),
                "ratio": round(int(artery_mask.sum()) / total_pixels, 4),
            },
            "vein": {
                "pixel_count": int(vein_mask.sum()),
                "ratio": round(int(vein_mask.sum()) / total_pixels, 4),
            },
            "vessel": {
                "pixel_count": int(vessel_mask.sum()),
                "ratio": round(int(vessel_mask.sum()) / total_pixels, 4),
            },
        }

        return {
            "artery_mask": artery_mask,
            "vein_mask": vein_mask,
            "vessel_mask": vessel_mask,
            "class_names": CLASS_NAMES,
            "statistics": statistics,
            "image_size": {"width": original_size[0], "height": original_size[1]},
        }

    def overlay(self, image: Image.Image, result: dict, alpha: float = 0.5) -> Image.Image:
        """
        将动静脉分割结果以半透明颜色叠加到原图上

        Args:
            image: 原始 PIL Image (RGB)
            result: predict() 的返回值
            alpha: 叠加透明度 (0~1)

        Returns:
            叠加后的 PIL Image (RGB)
        """
        img_rgb = np.array(image.convert("RGB"), dtype=np.float32)
        color_layer = np.zeros_like(img_rgb)

        # 按优先级叠加：vessel -> vein -> artery (动脉最上层)
        color_layer[result["vessel_mask"]] = CLASS_COLORS["vessel"]
        color_layer[result["vein_mask"]] = CLASS_COLORS["vein"]
        color_layer[result["artery_mask"]] = CLASS_COLORS["artery"]

        # 混合
        any_vessel = (
            result["artery_mask"] | result["vein_mask"] | result["vessel_mask"]
        )
        blended = img_rgb.copy()
        blended[any_vessel] = (
            img_rgb[any_vessel] * (1 - alpha) + color_layer[any_vessel] * alpha
        )

        return Image.fromarray(blended.astype(np.uint8))

    def predict_with_overlay(self, image: Image.Image, alpha: float = 0.5) -> dict:
        """
        预测并生成叠加可视化图

        Returns:
            dict: predict() 的返回值 + "overlay_image": PIL Image
        """
        result = self.predict(image)
        result["overlay_image"] = self.overlay(image, result, alpha)
        return result

    def predict_from_path(self, image_path: str) -> dict:
        """从文件路径加载图像并预测"""
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

    async def predict_with_overlay_async(self, image: Image.Image, alpha: float = 0.5) -> dict:
        """异步版本的 predict_with_overlay"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_with_overlay, image, alpha)
