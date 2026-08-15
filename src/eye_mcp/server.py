"""
Eye MCP 主服务
基于 MCP (Model Context Protocol) 提供 OCT 图像分割等 AI 工具
"""
import asyncio
import base64
import io
import json
from typing import Annotated

import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image

from eye_mcp.services.oct_seg_service import OCTSegService
from eye_mcp.services.fovea_location_service import FoveaLocationService
from eye_mcp.services.optic_disc_cup_seg_service import OpticDiscCupSegService
from eye_mcp.services.vessel_seg_service import VesselSegService
from eye_mcp.services.lesion_seg_service import LesionSegService

# 创建 MCP 服务实例
mcp = FastMCP("eye-mcp")

# 延迟加载服务（首次调用时初始化）
_oct_seg_service: OCTSegService | None = None
_fovea_location_service: FoveaLocationService | None = None
_optic_disc_cup_seg_service: OpticDiscCupSegService | None = None
_vessel_seg_service: VesselSegService | None = None
_lesion_seg_service: LesionSegService | None = None


def _get_oct_seg_service() -> OCTSegService:
    """懒加载 OCT 分割服务"""
    global _oct_seg_service
    if _oct_seg_service is None:
        _oct_seg_service = OCTSegService()
    return _oct_seg_service


def _get_fovea_location_service() -> FoveaLocationService:
    """懒加载 Fovea 定位服务"""
    global _fovea_location_service
    if _fovea_location_service is None:
        _fovea_location_service = FoveaLocationService()
    return _fovea_location_service


def _get_optic_disc_cup_seg_service() -> OpticDiscCupSegService:
    """懒加载视杯视盘分割服务"""
    global _optic_disc_cup_seg_service
    if _optic_disc_cup_seg_service is None:
        _optic_disc_cup_seg_service = OpticDiscCupSegService()
    return _optic_disc_cup_seg_service


def _get_vessel_seg_service() -> VesselSegService:
    """懒加载血管分割服务"""
    global _vessel_seg_service
    if _vessel_seg_service is None:
        _vessel_seg_service = VesselSegService()
    return _vessel_seg_service


def _get_lesion_seg_service() -> LesionSegService:
    """懒加载病灶分割服务"""
    global _lesion_seg_service
    if _lesion_seg_service is None:
        _lesion_seg_service = LesionSegService()
    return _lesion_seg_service


@mcp.tool(name="oct_segment", description="对OCT眼底图像进行语义分割，识别视网膜/脉络膜/囊肿/黄斑裂孔等结构，返回像素统计、分割mask和可视化叠加图")
async def oct_segment(
    image_base64: Annotated[str, "Base64 编码的 OCT 眼底图像，支持 PNG/JPG/BMP/WEBP 格式"],
    alpha: Annotated[float, "分割区域叠加到原图上的透明度，0为完全透明，1为完全不透明，推荐0.4"] = 0.4,
) -> str:
    """
    对 OCT 图像进行语义分割，返回完整的分析结果

    功能说明:
    将 OCT 眼底图像自动分割为 5 个解剖区域，同时返回:
    1. 各区域像素占比统计
    2. 分割 mask（类别索引图，PNG格式，Base64编码）
    3. 原图叠加分割颜色的可视化图（PNG格式，Base64编码）

    分割类别:
    - background (0): 背景
    - retina (1): 视网膜 - 蓝色标记
    - choroid (2): 脉络膜 - 橙色标记
    - irc (3): 视网膜内囊肿 - 绿色标记
    - mh (4): 黄斑裂孔 - 粉红色标记

    使用场景:
    - 辅助医生分析 OCT 图像中各组织结构
    - 定量评估视网膜/脉络膜等区域面积比例
    - 筛查视网膜内囊肿(irc)和黄斑裂孔(mh)等病变
    """
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    service = _get_oct_seg_service()

    # 推理放入线程池，不阻塞事件循环
    result = await asyncio.to_thread(service.predict_with_overlay, image, alpha)

    # 编码 mask 为彩色 PNG（类别索引映射为颜色）
    from eye_mcp.services.oct_seg_service import CLASS_COLORS as OCT_COLORS

    mask = result["mask"]
    mh, mw = mask.shape
    color_mask = np.zeros((mh, mw, 3), dtype=np.uint8)
    for class_idx in range(1, len(OCT_COLORS)):
        region = mask == class_idx
        if region.any():
            color_mask[region] = OCT_COLORS[class_idx]

    mask_image = Image.fromarray(color_mask)
    mask_buf = io.BytesIO()
    mask_image.save(mask_buf, format="PNG")
    mask_base64 = base64.b64encode(mask_buf.getvalue()).decode("utf-8")

    # 编码 overlay 为 PNG
    overlay_buf = io.BytesIO()
    result["overlay_image"].save(overlay_buf, format="PNG")
    overlay_base64 = base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

    response = {
        "class_names": result["class_names"],
        "class_mapping": {str(i): name for i, name in enumerate(result["class_names"])},
        "statistics": result["statistics"],
        "image_size": {"width": image.width, "height": image.height},
        "mask_base64": mask_base64,
        "overlay_base64": overlay_base64,
    }

    return json.dumps(response, ensure_ascii=False, indent=2)


@mcp.tool(name="fovea_locate", description="定位眼底图像中的中央凹(fovea)位置，返回坐标和十字标记可视化图")
async def fovea_locate(
    image_base64: Annotated[str, "Base64 编码的眼底图像，支持 PNG/JPG/BMP/WEBP 格式"],
    line_width: Annotated[int, "十字标记线宽(像素)，越大越醒目，推荐5"] = 5,
) -> str:
    """
    定位眼底图像中的中央凹(fovea)位置

    功能说明:
    使用热力图回归模型精确定位中央凹坐标，同时返回:
    1. 归一化坐标 (x, y)，范围 0~1
    2. 像素坐标 (pixel_x, pixel_y)，基于原图尺寸
    3. 原图叠加深绿色十字标记的可视化图（PNG格式，Base64编码）

    使用场景:
    - 辅助医生快速定位中央凹位置
    - 用于后续的黄斑区域分析
    - 眼底图像质量评估（中央凹是否在视野中心）
    """
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    service = _get_fovea_location_service()

    # 推理放入线程池，不阻塞事件循环
    result = await asyncio.to_thread(service.predict_with_overlay, image, line_width)

    # 编码 overlay 为 PNG
    overlay_buf = io.BytesIO()
    result["overlay_image"].save(overlay_buf, format="PNG")
    overlay_base64 = base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

    response = {
        "x": result["x"],
        "y": result["y"],
        "pixel_x": result["pixel_x"],
        "pixel_y": result["pixel_y"],
        "image_size": result["image_size"],
        "overlay_base64": overlay_base64,
    }

    return json.dumps(response, ensure_ascii=False, indent=2)


@mcp.tool(name="optic_disc_cup_seg", description="分割眼底图像中的视盘和视杯区域，计算杯盘比(CDR)，返回分割mask、统计数据和可视化叠加图")
async def optic_disc_cup_seg(
    image_base64: Annotated[str, "Base64 编码的眼底图像，支持 PNG/JPG/BMP/WEBP 格式"],
    alpha: Annotated[float, "分割区域叠加到原图上的透明度，0为完全透明，1为完全不透明，推荐0.4"] = 0.4,
) -> str:
    """
    对眼底图像进行视杯视盘语义分割

    功能说明:
    使用 SegFormer 模型分割眼底图像中的视盘和视杯区域，同时返回:
    1. 各区域像素占比统计
    2. 杯盘比 CDR (cup-to-disc ratio) - 青光眼筛查核心指标
    3. 分割 mask（类别索引图，PNG格式，Base64编码）
    4. 原图叠加分割颜色的可视化图（PNG格式，Base64编码）

    分割类别:
    - background (0): 背景
    - optic_disc (1): 视盘 - 蓝色标记
    - optic_cup (2): 视杯 - 深绿色标记

    使用场景:
    - 青光眼筛查
    - 视盘/视杯面积定量评估
    - 眼底结构变化随访对比

    模型来源: pamixsun/segformer_for_optic_disc_cup_segmentation (Apache-2.0)
    """
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    service = _get_optic_disc_cup_seg_service()

    # 推理放入线程池，不阻塞事件循环
    result = await asyncio.to_thread(service.predict_with_overlay, image, alpha)

    # 编码 mask 为彩色 PNG
    from eye_mcp.services.optic_disc_cup_seg_service import CLASS_COLORS as DISC_COLORS

    mask = result["mask"]
    mh2, mw2 = mask.shape
    color_mask2 = np.zeros((mh2, mw2, 3), dtype=np.uint8)
    for class_idx in range(1, len(DISC_COLORS)):
        region = mask == class_idx
        if region.any():
            color_mask2[region] = DISC_COLORS[class_idx]

    mask_image = Image.fromarray(color_mask2)
    mask_buf = io.BytesIO()
    mask_image.save(mask_buf, format="PNG")
    mask_base64 = base64.b64encode(mask_buf.getvalue()).decode("utf-8")

    # 编码 overlay 为 PNG
    overlay_buf = io.BytesIO()
    result["overlay_image"].save(overlay_buf, format="PNG")
    overlay_base64 = base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

    response = {
        "class_names": result["class_names"],
        "class_mapping": {str(i): name for i, name in enumerate(result["class_names"])},
        "statistics": result["statistics"],
        "cdr": result["cdr"],
        "image_size": result["image_size"],
        "mask_base64": mask_base64,
        "overlay_base64": overlay_base64,
    }

    return json.dumps(response, ensure_ascii=False, indent=2)


@mcp.tool(name="vessel_seg", description="分割眼底图像中的视网膜血管，区分动脉(红)和静脉(蓝)，返回各类血管mask、像素统计和可视化叠加图")
async def vessel_seg(
    image_base64: Annotated[str, "Base64 编码的眼底图像，支持 PNG/JPG/BMP/WEBP 格式"],
    alpha: Annotated[float, "血管颜色叠加到原图上的透明度，0为完全透明，1为完全不透明，推荐0.5"] = 0.5,
) -> str:
    """
    对眼底图像进行视网膜动静脉血管分割

    功能说明:
    使用 RIP-AV 模型分割眼底图像中的血管，区分动脉和静脉，同时返回:
    1. 动脉/静脉/全血管各自的像素统计
    2. 原图叠加血管颜色的可视化图（PNG格式，Base64编码）

    分割类别:
    - artery: 动脉 - 红色标记
    - vein: 静脉 - 蓝色标记
    - vessel: 全血管 - 浅绿色标记

    使用场景:
    - 视网膜血管形态分析
    - 动静脉比值计算
    - 糖尿病视网膜病变评估
    - 高血压视网膜病变筛查

    模型来源: RIP-AV (Wei Dai, MIT License)
    """
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    service = _get_vessel_seg_service()

    # 推理放入线程池，不阻塞事件循环
    result = await asyncio.to_thread(service.predict_with_overlay, image, alpha)

    # 编码 overlay 为 PNG
    overlay_buf = io.BytesIO()
    result["overlay_image"].save(overlay_buf, format="PNG")
    overlay_base64 = base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

    response = {
        "class_names": result["class_names"],
        "statistics": result["statistics"],
        "image_size": result["image_size"],
        "overlay_base64": overlay_base64,
    }

    return json.dumps(response, ensure_ascii=False, indent=2)


@mcp.tool(name="lesion_seg", description="对眼底图像进行4类病灶分割(微血管瘤/出血/硬性渗出/软性渗出)，返回各类病灶mask、像素统计和可视化叠加图")
async def lesion_seg(
    image_base64: Annotated[str, "Base64 编码的眼底图像，支持 PNG/JPG/BMP/WEBP 格式"],
    alpha: Annotated[float, "病灶颜色叠加到原图上的透明度，0为完全透明，1为完全不透明，推荐0.5"] = 0.5,
) -> str:
    """
    对眼底图像进行多类病灶语义分割

    功能说明:
    使用 4 个独立 VNet2d 模型分别检测各类眼底病灶，同时返回:
    1. 每类病灶的像素统计（面积占比）
    2. 各病灶 mask（二值图，PNG格式，Base64编码）
    3. 原图叠加所有病灶颜色的可视化图（PNG格式，Base64编码）

    分割类别:
    - MA: 微血管瘤 - 红色标记
    - HE: 出血 - 紫色标记
    - EX: 硬性渗出 - 橙色标记
    - SE: 软性渗出(棉絮斑) - 青色标记

    使用场景:
    - 糖尿病视网膜病变(DR)分级筛查
    - 眼底病变定量分析与随访对比
    """
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    service = _get_lesion_seg_service()

    result = await asyncio.to_thread(service.predict_with_overlay, image, alpha)

    # 编码带图例的彩色 mask
    mask_buf = io.BytesIO()
    result["mask_image"].save(mask_buf, format="PNG")
    mask_base64 = base64.b64encode(mask_buf.getvalue()).decode("utf-8")

    # 编码 overlay
    overlay_buf = io.BytesIO()
    result["overlay_image"].save(overlay_buf, format="PNG")
    overlay_base64 = base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

    response = {
        "class_names": result["class_names"],
        "statistics": result["statistics"],
        "image_size": result["image_size"],
        "mask_base64": mask_base64,
        "overlay_base64": overlay_base64,
    }

    return json.dumps(response, ensure_ascii=False, indent=2)
