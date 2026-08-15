"""Fovea 中央凹定位服务测试"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eye_mcp.services.fovea_location_service import (
    FoveaLocationService,
    SUPPORTED_FORMATS,
    CROSS_COLOR,
)
from eye_mcp.models.fovea_location_net import (
    FoveaHeatmapNet,
    get_fovea_coords_from_heatmap,
)

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data" / "fovea_location"
TEST_RESULTS_DIR = TEST_DATA_DIR / "results"


# ========================= 模型单元测试 =========================


def test_fovea_heatmap_net():
    """测试模型构建和前向传播"""
    import torch

    model = FoveaHeatmapNet(pretrained=False)
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    # 输出为单通道热力图，尺寸为输入的 1/4
    assert output.shape == (1, 1, 128, 128)
    # Sigmoid 输出范围 0~1
    assert output.min() >= 0
    assert output.max() <= 1


def test_get_fovea_coords_from_heatmap():
    """测试从热力图提取坐标"""
    import torch

    # 构造一个峰值在中心的热力图
    heatmap = torch.zeros(1, 1, 128, 128)
    heatmap[0, 0, 64, 64] = 1.0  # 中心位置

    coords = get_fovea_coords_from_heatmap(heatmap)
    assert coords.shape == (1, 2)
    # 坐标应接近 (0.5, 0.5)
    assert abs(coords[0, 0].item() - 0.5) < 0.01
    assert abs(coords[0, 1].item() - 0.5) < 0.01


def test_cross_color():
    """测试十字颜色为深绿色"""
    assert CROSS_COLOR == (0, 153, 51)


# ========================= 服务集成测试 =========================


class TestFoveaLocationService:
    """Fovea 定位服务集成测试（需要权重文件）"""

    @pytest.fixture
    def service(self):
        """初始化服务实例"""
        return FoveaLocationService(device="cpu")

    @pytest.fixture
    def dummy_image(self):
        """创建测试用假图像"""
        return Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )

    def test_service_init(self, service):
        """测试服务初始化"""
        assert service.model is not None
        assert service.img_size == 512

    def test_predict(self, service, dummy_image):
        """测试推理流程"""
        result = service.predict(dummy_image)
        assert "x" in result
        assert "y" in result
        assert "pixel_x" in result
        assert "pixel_y" in result
        assert "image_size" in result
        # 坐标范围 0~1
        assert 0 <= result["x"] <= 1
        assert 0 <= result["y"] <= 1
        # 像素坐标在图像范围内
        assert 0 <= result["pixel_x"] <= 512
        assert 0 <= result["pixel_y"] <= 512

    def test_predict_with_overlay(self, service, dummy_image):
        """测试叠加十字可视化"""
        result = service.predict_with_overlay(dummy_image, line_width=5)
        assert "overlay_image" in result
        assert isinstance(result["overlay_image"], Image.Image)
        assert result["overlay_image"].mode == "RGB"
        assert result["overlay_image"].size == dummy_image.size

    def test_overlay_line_width(self, service, dummy_image):
        """测试不同线宽"""
        for width in [3, 5, 8]:
            result = service.predict_with_overlay(dummy_image, line_width=width)
            assert result["overlay_image"] is not None

    def test_predict_from_path(self, service, tmp_path):
        """测试从路径加载并推理"""
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 384, 3), dtype=np.uint8)
        )
        img_path = tmp_path / "test_fundus.png"
        img.save(img_path)
        result = service.predict_from_path(str(img_path))
        assert result["image_size"] == {"width": 384, "height": 256}
        assert 0 <= result["pixel_x"] <= 384
        assert 0 <= result["pixel_y"] <= 256

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


class TestFoveaLocationServiceAsync:
    """异步推理测试"""

    @pytest.fixture
    def service(self):
        return FoveaLocationService(device="cpu")

    @pytest.fixture
    def dummy_image(self):
        return Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )

    async def test_predict_async(self, service, dummy_image):
        """测试异步推理"""
        result = await service.predict_async(dummy_image)
        assert "x" in result
        assert "pixel_x" in result

    async def test_predict_with_overlay_async(self, service, dummy_image):
        """测试异步 overlay"""
        result = await service.predict_with_overlay_async(dummy_image)
        assert "overlay_image" in result


# ========================= 真实图片测试 =========================


class TestFoveaLocationWithRealData:
    """使用真实眼底图片的端到端测试

    请在 tests/services/test_data/fovea_location/ 目录下放入眼底测试图片
    """

    @pytest.fixture
    def service(self):
        return FoveaLocationService(device="cpu")

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
            assert 0 <= result["x"] <= 1
            assert 0 <= result["y"] <= 1
            assert result["pixel_x"] >= 0
            assert result["pixel_y"] >= 0

    def test_real_image_overlay(self, service):
        """测试真实图片可视化，并保存结果"""
        images = self._get_test_images()
        if not images:
            pytest.skip("无测试图片")

        TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            result = service.predict_with_overlay(image, line_width=5)
            assert result["overlay_image"].mode == "RGB"

            # 保存可视化结果
            stem = img_path.stem
            result["overlay_image"].save(TEST_RESULTS_DIR / f"{stem}_fovea.png")

    async def test_real_image_async(self, service):
        """测试真实图片异步推理"""
        import asyncio

        images = self._get_test_images()
        if not images:
            pytest.skip("无测试图片")

        tasks = [
            service.predict_from_path_async(str(p)) for p in images
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == len(images)
        for r in results:
            assert "x" in r
            assert "pixel_x" in r
