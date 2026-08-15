"""
眼底病灶分割服务 - 4 类病灶独立二值分割

每类病灶使用独立 VNet2d(num_classes=1) 模型:
- MA: 微血管瘤
- HE: 出血
- EX: 硬性渗出
- SE: 软性渗出(棉絮斑)
"""

import asyncio
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from eye_mcp.common.config import load_config
from eye_mcp.models.lesion_seg_vnet2d import VNet2d, LESION_CLASSES, LESION_NAMES_CN

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

CLASS_COLORS = {
    "MA": (255, 50, 50),
    "HE": (180, 0, 180),
    "EX": (255, 165, 0),
    "SE": (0, 200, 200),
}


class LesionSegService:
    def __init__(self, device: str = "auto"):
        config = load_config("config.yaml")
        cfg = config.get("lesion_seg", {})

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.img_size = cfg.get("img_size", 1024)
        self.default_alpha = cfg.get("default_alpha", 0.5)
        self.threshold = cfg.get("threshold", 0.5)
        weights_dir = Path(cfg.get("weights_dir", "weights/lesion_seg"))

        self.models: dict[str, VNet2d] = {}
        for cls in LESION_CLASSES:
            weight_path = weights_dir / f"{cls}.pth"
            if not weight_path.exists():
                raise FileNotFoundError(f"权重文件不存在: {weight_path}")
            model = VNet2d(in_channels=3, num_classes=1)
            ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.to(self.device)
            model.eval()
            self.models[cls] = model

        self.class_names = LESION_CLASSES
        self.num_classes = len(LESION_CLASSES)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        img = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        return tensor.unsqueeze(0)

    def _infer_single(self, model: VNet2d, tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            try:
                logits = model(tensor.to(self.device))
            except RuntimeError:
                model.to("cpu")
                self.device = torch.device("cpu")
                logits = model(tensor.to("cpu"))
            prob = torch.sigmoid(logits)
        return (prob.squeeze().cpu().numpy() > self.threshold).astype(np.uint8)

    def predict(self, image: Image.Image) -> dict:
        tensor = self._preprocess(image)
        w, h = image.size
        masks = {}
        statistics = {}

        for cls in LESION_CLASSES:
            mask = self._infer_single(self.models[cls], tensor)
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks[cls] = mask_resized
            pixel_count = int(mask_resized.sum())
            total_pixels = w * h
            statistics[cls] = {
                "pixel_count": pixel_count,
                "ratio": round(pixel_count / total_pixels, 6),
                "name_cn": LESION_NAMES_CN[cls],
            }

        return {
            "masks": masks,
            "statistics": statistics,
            "class_names": LESION_CLASSES,
            "image_size": {"width": w, "height": h},
        }

    def predict_with_overlay(self, image: Image.Image, alpha: float | None = None) -> dict:
        if alpha is None:
            alpha = self.default_alpha

        result = self.predict(image)
        overlay = np.array(image).copy()

        for cls in LESION_CLASSES:
            mask = result["masks"][cls]
            color = CLASS_COLORS[cls]
            region = mask > 0
            if region.any():
                for c in range(3):
                    overlay[:, :, c][region] = (
                        overlay[:, :, c][region] * (1 - alpha) + color[c] * alpha
                    ).astype(np.uint8)

        result["overlay_image"] = Image.fromarray(overlay)
        result["mask_image"] = self._build_color_mask(result["masks"])
        return result

    def _build_color_mask(self, masks: dict) -> Image.Image:
        h, w = list(masks.values())[0].shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in LESION_CLASSES:
            region = masks[cls] > 0
            if region.any():
                color_mask[region] = CLASS_COLORS[cls]

        mask_img = Image.fromarray(color_mask)
        draw = ImageDraw.Draw(mask_img)

        font_size = max(16, h // 40)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        padding = max(8, h // 100)
        box_size = max(12, h // 50)
        line_height = int(font_size * 1.5)
        x_start = padding
        y_start = h - padding - len(LESION_CLASSES) * line_height

        for i, cls in enumerate(LESION_CLASSES):
            y = y_start + i * line_height
            color = CLASS_COLORS[cls]
            draw.rectangle([x_start, y, x_start + box_size, y + box_size], fill=color)
            label = f"{cls} - {LESION_NAMES_CN[cls]}"
            draw.text((x_start + box_size + 8, y - 2), label, fill=(255, 255, 255), font=font)

        return mask_img

    def predict_from_path(self, image_path: str) -> dict:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {image_path}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"不支持的图片格式: {path.suffix}")
        image = Image.open(path).convert("RGB")
        return self.predict(image)

    async def predict_async(self, image: Image.Image) -> dict:
        return await asyncio.to_thread(self.predict, image)

    async def predict_with_overlay_async(
        self, image: Image.Image, alpha: float | None = None
    ) -> dict:
        return await asyncio.to_thread(self.predict_with_overlay, image, alpha)
