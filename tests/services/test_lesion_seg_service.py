"""眼底病灶分割服务测试"""
import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eye_mcp.services.lesion_seg_service import (
    LesionSegService,
    SUPPORTED_FORMATS,
    CLASS_COLORS,
)
from eye_mcp.models.lesion_seg_vnet2d import LESION_CLASSES, LESION_NAMES_CN

TEST_DATA_DIR = Path(__file__).parent / "test_data" / "lesion_seg"
TEST_RESULTS_DIR = TEST_DATA_DIR / "results"


# ========================= 模型单元测试 =========================


def test_lesion_classes():
    assert len(LESION_CLASSES) == 4
    assert LESION_CLASSES == ["MA", "HE", "EX", "SE"]


def test_class_colors():
    for cls in LESION_CLASSES:
        assert cls in CLASS_COLORS


def test_lesion_names_cn():
    for cls in LESION_CLASSES:
        assert cls in LESION_NAMES_CN


# ========================= 服务集成测试 =========================


class TestLesionSegService:
    @pytest.fixture
    def service(self):
        return LesionSegService(device="cpu")

    @pytest.fixture
    def dummy_image(self):
        return Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )

    def test_service_init(self, service):
        assert service.num_classes == 4
        assert service.class_names == LESION_CLASSES
        assert len(service.models) == 4

    def test_predict(self, service, dummy_image):
        result = service.predict(dummy_image)
        assert "masks" in result
        assert "statistics" in result
        assert "class_names" in result
        assert "image_size" in result
        for cls in LESION_CLASSES:
            assert cls in result["masks"]
            assert result["masks"][cls].shape == (512, 512)
            assert cls in result["statistics"]
            assert "pixel_count" in result["statistics"][cls]
            assert "ratio" in result["statistics"][cls]

    def test_predict_with_overlay(self, service, dummy_image):
        result = service.predict_with_overlay(dummy_image, alpha=0.5)
        assert "overlay_image" in result
        assert isinstance(result["overlay_image"], Image.Image)
        assert result["overlay_image"].mode == "RGB"

    def test_predict_from_path(self, service, tmp_path):
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 384, 3), dtype=np.uint8)
        )
        img_path = tmp_path / "test_fundus.png"
        img.save(img_path)
        result = service.predict_from_path(str(img_path))
        assert result["image_size"] == {"width": 384, "height": 256}

    def test_predict_from_path_unsupported_format(self, service, tmp_path):
        fake_path = tmp_path / "test.xyz"
        fake_path.write_text("not an image")
        with pytest.raises(ValueError, match="不支持的图片格式"):
            service.predict_from_path(str(fake_path))

    def test_predict_from_path_file_not_found(self, service):
        with pytest.raises(FileNotFoundError):
            service.predict_from_path("nonexistent.png")


# ========================= 异步测试 =========================


class TestLesionSegServiceAsync:
    @pytest.fixture
    def service(self):
        return LesionSegService(device="cpu")

    @pytest.fixture
    def dummy_image(self):
        return Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )

    async def test_predict_async(self, service, dummy_image):
        result = await service.predict_async(dummy_image)
        assert "masks" in result
        for cls in LESION_CLASSES:
            assert cls in result["masks"]


# ========================= 真实图片测试 =========================


class TestLesionSegWithRealData:
    """使用真实眼底图片的端到端测试

    请在 tests/services/test_data/lesion_seg/ 目录下放入眼底测试图片
    """

    @pytest.fixture
    def service(self):
        return LesionSegService(device="cpu")

    def _get_test_images(self):
        if not TEST_DATA_DIR.exists():
            return []
        images = []
        for suffix in SUPPORTED_FORMATS:
            images.extend(TEST_DATA_DIR.glob(f"*{suffix}"))
        return sorted(images)

    def test_real_images_exist(self):
        images = self._get_test_images()
        if not images:
            pytest.skip(f"未找到测试图片，请在 {TEST_DATA_DIR} 中放入眼底图片")

    def test_real_image_predict(self, service):
        images = self._get_test_images()
        if not images:
            pytest.skip("无测试图片")

        for img_path in images:
            result = service.predict_from_path(str(img_path))
            for cls in LESION_CLASSES:
                assert result["masks"][cls].shape[0] > 0

    def test_real_image_overlay(self, service):
        images = self._get_test_images()
        if not images:
            pytest.skip("无测试图片")

        TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            result = service.predict_with_overlay(image)
            assert result["overlay_image"].mode == "RGB"

            stem = img_path.stem
            result["overlay_image"].save(TEST_RESULTS_DIR / f"{stem}_lesion_overlay.png")
            result["mask_image"].save(TEST_RESULTS_DIR / f"{stem}_lesion_mask.png")
