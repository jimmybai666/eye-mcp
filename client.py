"""
Eye MCP 简易测试客户端

用法:
    uv run python client.py oct_segment tests/services/test_data/oct_seg/xxx.png
    uv run python client.py fovea_locate tests/services/test_data/fovea_location/xxx.jpg
    uv run python client.py optic_disc_cup_seg tests/services/test_data/optic_disc_cup_seg/xxx.jpg
    uv run python client.py vessel_seg tests/services/test_data/vessel_seg/xxx.jpg
    uv run python client.py lesion_seg tests/services/test_data/lesion_seg/xxx.jpg
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    if len(sys.argv) < 3:
        print("用法: python client.py <tool_name> <image_path> [alpha]")
        print("可用工具: oct_segment, fovea_locate, optic_disc_cup_seg, vessel_seg, lesion_seg")
        sys.exit(1)

    tool_name = sys.argv[1]
    image_path = Path(sys.argv[2])
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else None

    if not image_path.exists():
        print(f"文件不存在: {image_path}")
        sys.exit(1)

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    arguments = {"image_base64": image_base64}
    if alpha is not None:
        arguments["alpha"] = alpha
    if tool_name == "fovea_locate" and alpha is not None:
        arguments = {"image_base64": image_base64, "line_width": int(alpha)}

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "eye-mcp"],
    )

    print(f"连接 eye-mcp 服务...")
    print(f"工具: {tool_name}")
    print(f"图片: {image_path} ({image_path.stat().st_size / 1024:.1f} KB)")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"可用工具: {[t.name for t in tools.tools]}")

            print(f"\n调用 {tool_name}...")
            result = await session.call_tool(tool_name, arguments)

            response_text = result.content[0].text
            response = json.loads(response_text)

            print(f"\n{'='*50}")
            print(f"结果:")
            print(f"{'='*50}")

            if "statistics" in response:
                print("\n统计信息:")
                for key, val in response["statistics"].items():
                    if isinstance(val, dict):
                        ratio = val.get("ratio", 0)
                        name = val.get("name_cn", key)
                        print(f"  {name} ({key}): {ratio*100:.2f}%")
                    else:
                        print(f"  {key}: {val}")

            if "cdr" in response:
                print(f"\n杯盘比 CDR: {response['cdr']:.4f}")

            if "x" in response and "y" in response:
                print(f"\n中央凹位置: ({response['x']:.4f}, {response['y']:.4f})")
                print(f"像素坐标: ({response['pixel_x']}, {response['pixel_y']})")

            if "image_size" in response:
                print(f"\n图像尺寸: {response['image_size']}")

            output_dir = Path("client_output")
            output_dir.mkdir(exist_ok=True)
            stem = image_path.stem

            if "mask_base64" in response:
                mask_path = output_dir / f"{stem}_{tool_name}_mask.png"
                mask_path.write_bytes(base64.b64decode(response["mask_base64"]))
                print(f"\nMask 已保存: {mask_path}")

            if "overlay_base64" in response:
                overlay_path = output_dir / f"{stem}_{tool_name}_overlay.png"
                overlay_path.write_bytes(base64.b64decode(response["overlay_base64"]))
                print(f"Overlay 已保存: {overlay_path}")

            print("\n完成!")


if __name__ == "__main__":
    asyncio.run(main())
