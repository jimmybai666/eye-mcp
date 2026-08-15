"""OCT 分割服务测试"""
import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eye_mcp.services.oct_seg_service import (
    OCTSegService,
    SUPPORTED_FORMATS,
    CLASS_COLORS,
)
from eye_mcp.models.oct_seg_unet import build_unet, NUM_CLASSES, CLASS_NAMES

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data" / "oct_seg"
TEST_RESULTS_DIR = TEST_DATA_DIR / "results"


# ========================= 模型单元测试 =========================


def test_build_unet():
    """测试 UNet 模型构建"""
    import torch

    model = build_unet(n_classes=NUM_CLASSES, pretrained=False)
    assert model is not None
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    assert output.shape == (1, NUM_CLASSES, 512, 512)


def test_class_names():
    """测试类别名称定义"""
    assert len(CLASS_NAMES) == 5
    assert CLASS_NAMES == ["background", "retina", "choroid", "irc", "mh"]


def test_class_colors():
    """测试可视化颜色定义"""
    assert len(CLASS_COLORS) == NUM_CLASSES
    assert CLASS_COLORS[0] == (0, 0, 0)


# ========================= 服务集成测试 =========================


class TestOCTSegService:
    """OCT 分割服务集成测试（需要权重文件）"""

    @pytest.fixture
    def service(self):
        """初始化服务实例"""
        return OCTSegService(device="cpu")

    @pytest.fixture
    def dummy_image(self):
        """创建测试用假图像"""
        return Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )

    def test_service_init(self, service):
        """测试服务初始化"""
        assert service.num_classes == 5
        assert service.class_names == CLASS_NAMES
        assert service.model is not None

    def test_predict(self, service, dummy_image):
        """测试推理流程"""
        result = service.predict(dummy_image)
        assert "mask" in result
        assert "probabilities" in result
        assert "class_names" in result
        assert "statistics" in result
        assert result["mask"].shape == (512, 512)
        assert result["probabilities"].shape[0] == NUM_CLASSES
        assert len(result["statistics"]) == NUM_CLASSES
        # 统计占比之和应为 1
        total_ratio = sum(s["ratio"] for s in result["statistics"].values())
        assert abs(total_ratio - 1.0) < 0.01

    def test_predict_with_overlay(self, service, dummy_image):
        """测试叠加可视化"""
        result = service.predict_with_overlay(dummy_image, alpha=0.4)
        assert "overlay_image" in result
        assert isinstance(result["overlay_image"], Image.Image)
        assert result["overlay_image"].size == (
            result["mask"].shape[1],
            result["mask"].shape[0],
        )

    def test_overlay_alpha_range(self, service, dummy_image):
        """测试不同透明度"""
        for alpha in [0.0, 0.5, 1.0]:
            result = service.predict_with_overlay(dummy_image, alpha=alpha)
            assert result["overlay_image"] is not None

    def test_predict_from_path(self, service, tmp_path):
        """测试从路径加载并推理"""
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        img_path = tmp_path / "test_oct.png"
        img.save(img_path)
        result = service.predict_from_path(str(img_path))
        assert result["mask"].shape == (256, 256)

    def test_predict_from_path_unsupported_format(self, service, tmp_path):
        """测试不支持的图片格式"""
        fake_path = tmp_path / "test.xyz"
        fake_path.write_text("not an image")
        with pytest.raises(ValueError, match="不支持的图片格式"):
            service.predict_from_path(str(fake_path))

    def test_predict_from_path_file_not_found(self, service):
        """测试文件不存在"""
        with pytest.raises(FileNotFoundError):
            service.predict_from_path("nonexistent.png")

    def test_supported_formats(self):
        """测试支持的格式列表"""
        assert ".png" in SUPPORTED_FORMATS
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".jpeg" in SUPPORTED_FORMATS
        assert ".bmp" in SUPPORTED_FORMATS
        assert ".webp" in SUPPORTED_FORMATS


# ========================= 异步测试 =========================


class TestOCTSegServiceAsync:
    """异步推理测试"""

    @pytest.fixture
    def service(self):
        return OCTSegService(device="cpu")

    @pytest.fixture
    def dummy_image(self):
        return Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )

    async def test_predict_async(self, service, dummy_image):
        """测试异步推理"""
        result = await service.predict_async(dummy_image)
        assert "mask" in result
        assert result["mask"].shape == (256, 256)

    async def test_predict_from_path_async(self, service, tmp_path):
        """测试异步路径推理"""
        img = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        )
        img_path = tmp_path / "async_test.png"
        img.save(img_path)
        result = await service.predict_from_path_async(str(img_path))
        assert result["mask"].shape == (128, 128)


# ========================= 真实 OCT 图片测试 =========================


class TestOCTSegWithRealData:
    """使用真实 OCT 图片的端到端测试

    请在 tests/services/test_data/oct_seg/ 目录下放入 OCT 测试图片
    """

    @pytest.fixture
    def service(self):
        return OCTSegService(device="cpu")

    def _get_test_images(self):
        """获取测试数据目录中的所有图片"""
        if not TEST_DATA_DIR.exists():
            return []
        images = []
        for suffix in SUPPORTED_FORMATS:
            images.extend(TEST_DATA_DIR.glob(f"*{suffix}"))
        return sorted(images)

    def test_real_images_exist(self):
        """检查是否有测试图片"""
        images = self._get_test_images()
        if not images:
            pytest.skip(
                f"未找到测试图片，请在 {TEST_DATA_DIR} 中放入 OCT 图片"
            )

    def test_real_image_predict(self, service):
        """测试真实图片推理"""
        images = self._get_test_images()
        if not images:
            pytest.skip("无测试图片")

        for img_path in images:
            result = service.predict_from_path(str(img_path))
            assert result["mask"].shape[0] > 0
            assert result["mask"].shape[1] > 0
            assert len(result["statistics"]) == NUM_CLASSES
            # 各类别占比之和 = 1
            total = sum(s["ratio"] for s in result["statistics"].values())
            assert abs(total - 1.0) < 0.01, f"{img_path.name}: ratio sum = {total}"

    def test_real_image_overlay(self, service):
        """测试真实图片可视化叠加，并保存结果"""
        images = self._get_test_images()
        if not images:
            pytest.skip("无测试图片")

        TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            result = service.predict_with_overlay(image)
            assert result["overlay_image"].mode == "RGB"
            assert result["overlay_image"].size == (
                result["mask"].shape[1],
                result["mask"].shape[0],
            )

            # 保存 overlay 和彩色 mask 到 results 目录
            stem = img_path.stem
            result["overlay_image"].save(TEST_RESULTS_DIR / f"{stem}_overlay.png")

            # 将 mask 转为彩色图（0~4 灰度值肉眼不可见，需映射颜色）
            mask = result["mask"]
            color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for class_idx, color in enumerate(CLASS_COLORS):
                color_mask[mask == class_idx] = color
            Image.fromarray(color_mask).save(TEST_RESULTS_DIR / f"{stem}_mask.png")

    async def test_real_image_async(self, service):
        """测试真实图片异步推理"""
        images = self._get_test_images()
        if not images:
            pytest.skip("无测试图片")

        # 并发处理所有测试图片
        tasks = [
            service.predict_from_path_async(str(p)) for p in images
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == len(images)
        for r in results:
            assert "mask" in r
            assert "statistics" in r
