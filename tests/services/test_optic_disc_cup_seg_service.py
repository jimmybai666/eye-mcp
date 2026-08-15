"""视杯视盘分割服务测试"""
import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eye_mcp.services.optic_disc_cup_seg_service import (
    OpticDiscCupSegService,
    SUPPORTED_FORMATS,
    CLASS_COLORS,
)
from eye_mcp.models.optic_disc_cup_seg import NUM_CLASSES, CLASS_NAMES

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data" / "optic_disc_cup_seg"
TEST_RESULTS_DIR = TEST_DATA_DIR / "results"


# ========================= 模型单元测试 =========================


def test_class_names():
    """测试类别名称定义"""
    assert len(CLASS_NAMES) == 3
    assert CLASS_NAMES == ["background", "optic_disc", "optic_cup"]


def test_class_colors():
    """测试可视化颜色定义"""
    assert len(CLASS_COLORS) == NUM_CLASSES
    assert CLASS_COLORS[0] == (0, 0, 0)       # background 黑色
    assert CLASS_COLORS[1] == (51, 102, 255)  # disc 蓝色
    assert CLASS_COLORS[2] == (0, 153, 51)    # cup 深绿色


# ========================= 服务集成测试 =========================


class TestOpticDiscCupSegService:
    """视杯视盘分割服务集成测试（需要权重文件）"""

    @pytest.fixture
    def service(self):
        """初始化服务实例"""
        return OpticDiscCupSegService(device="cpu")

    @pytest.fixture
    def dummy_image(self):
        """创建测试用假图像"""
        return Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )

    def test_service_init(self, service):
        """测试服务初始化"""
        assert service.num_classes == 3
        assert service.class_names == CLASS_NAMES
        assert service.model is not None
        assert service.processor is not None

    def test_predict(self, service, dummy_image):
        """测试推理流程"""
        result = service.predict(dummy_image)
        assert "mask" in result
        assert "class_names" in result
        assert "statistics" in result
        assert "cdr" in result
        assert "image_size" in result
        assert result["mask"].shape == (512, 512)
        assert len(result["statistics"]) == NUM_CLASSES
        # CDR 范围合理
        assert 0 <= result["cdr"] <= 1.0
        # 统计占比之和应为 1
        total_ratio = sum(s["ratio"] for s in result["statistics"].values())
        assert abs(total_ratio - 1.0) < 0.01

    def test_predict_with_overlay(self, service, dummy_image):
        """测试叠加可视化"""
        result = service.predict_with_overlay(dummy_image, alpha=0.4)
        assert "overlay_image" in result
        assert isinstance(result["overlay_image"], Image.Image)
        assert result["overlay_image"].mode == "RGB"

    def test_predict_from_path(self, service, tmp_path):
        """测试从路径加载并推理"""
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 384, 3), dtype=np.uint8)
        )
        img_path = tmp_path / "test_fundus.png"
        img.save(img_path)
        result = service.predict_from_path(str(img_path))
        assert result["mask"].shape == (256, 384)
        assert result["image_size"] == {"width": 384, "height": 256}

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


# ========================= 异步测试 =========================


class TestOpticDiscCupSegServiceAsync:
    """异步推理测试"""

    @pytest.fixture
    def service(self):
        return OpticDiscCupSegService(device="cpu")

    @pytest.fixture
    def dummy_image(self):
        return Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )

    async def test_predict_async(self, service, dummy_image):
        """测试异步推理"""
        result = await service.predict_async(dummy_image)
        assert "mask" in result
        assert "cdr" in result

    async def test_predict_with_overlay_async(self, service, dummy_image):
        """测试异步 overlay"""
        result = await service.predict_with_overlay_async(dummy_image)
        assert "overlay_image" in result


# ========================= 真实图片测试 =========================


class TestOpticDiscCupSegWithRealData:
    """使用真实眼底图片的端到端测试

    请在 tests/services/test_data/optic_disc_cup_seg/ 目录下放入眼底测试图片
    """

    @pytest.fixture
    def service(self):
        return OpticDiscCupSegService(device="cpu")

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
                f"未找到测试图片，请在 {TEST_DATA_DIR} 中放入眼底图片"
            )

    def test_real_image_predict(self, service):
        """测试真实图片推理"""
        images = self._get_test_images()
        if not images:
            pytest.skip("无测试图片")

        for img_path in images:
            result = service.predict_from_path(str(img_path))
            assert result["mask"].shape[0] > 0
            assert 0 <= result["cdr"] <= 1.0
            total = sum(s["ratio"] for s in result["statistics"].values())
            assert abs(total - 1.0) < 0.01, f"{img_path.name}: ratio sum = {total}"

    def test_real_image_overlay(self, service):
        """测试真实图片可视化，并保存结果"""
        images = self._get_test_images()
        if not images:
            pytest.skip("无测试图片")

        TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            result = service.predict_with_overlay(image)
            assert result["overlay_image"].mode == "RGB"

            # 保存 overlay 和彩色 mask
            stem = img_path.stem
            result["overlay_image"].save(TEST_RESULTS_DIR / f"{stem}_overlay.png")

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

        tasks = [
            service.predict_from_path_async(str(p)) for p in images
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == len(images)
        for r in results:
            assert "mask" in r
            assert "cdr" in r
