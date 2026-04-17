#!/usr/bin/env python3
"""
示例：如何使用 ultralytics-auto-cut 进行模型转换
"""

from pathlib import Path
from onnx_cut import CutConfig, cut_ultralytics_onnx

def example_yolov8_640():
    """YOLOv8 640x640 模型转换示例"""
    print("=" * 60)
    print("示例 1: YOLOv8 640x640 模型")
    print("=" * 60)
    
    input_onnx = Path("yolov8n.onnx")  # 替换为你的模型路径
    output_onnx = Path("yolov8n.cut.onnx")
    
    if not input_onnx.exists():
        print(f"❌ 模型文件不存在: {input_onnx}")
        print("请先准备一个 YOLOv8 ONNX 模型")
        return
    
    cfg = CutConfig(
        model_type="auto",  # 自动识别
        imgsz=(640, 640),
        num_classes=80,
        strides=(8, 16, 32),
        simplify=True,
    )
    
    print(f"📥 输入: {input_onnx}")
    print(f"📤 输出: {output_onnx}")
    print(f"⚙️  配置: {cfg}")
    print()
    
    try:
        cut_ultralytics_onnx(input_onnx, output_onnx, cfg)
        print(f"✅ 转换成功！")
        print(f"📦 输出文件: {output_onnx}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")


def example_yolov5_custom():
    """YOLOv5 自定义配置示例"""
    print("\n" + "=" * 60)
    print("示例 2: YOLOv5 自定义配置")
    print("=" * 60)
    
    input_onnx = Path("yolov5s.onnx")
    output_onnx = Path("yolov5s.cut.onnx")
    
    if not input_onnx.exists():
        print(f"❌ 模型文件不存在: {input_onnx}")
        return
    
    cfg = CutConfig(
        model_type="yolov5",  # 强制指定类型
        imgsz=(416, 416),     # 自定义输入尺寸
        num_classes=20,       # 自定义类别数
        strides=(8, 16, 32),
        simplify=False,
    )
    
    print(f"📥 输入: {input_onnx}")
    print(f"📤 输出: {output_onnx}")
    print(f"⚙️  配置: {cfg}")
    print()
    
    try:
        cut_ultralytics_onnx(input_onnx, output_onnx, cfg)
        print(f"✅ 转换成功！")
    except Exception as e:
        print(f"❌ 转换失败: {e}")


def example_dry_run():
    """预览模式示例（不生成输出文件）"""
    print("\n" + "=" * 60)
    print("示例 3: 预览模式（dry-run）")
    print("=" * 60)
    
    input_onnx = Path("yolov8n.onnx")
    
    if not input_onnx.exists():
        print(f"❌ 模型文件不存在: {input_onnx}")
        return
    
    cfg = CutConfig(
        model_type="auto",
        imgsz=(640, 640),
        dry_run=True,  # 只预览，不生成文件
    )
    
    print(f"📥 输入: {input_onnx}")
    print(f"🔍 预览模式（不生成输出文件）")
    print()
    
    try:
        cut_ultralytics_onnx(input_onnx, Path("dummy.onnx"), cfg)
        print(f"✅ 预览完成！")
    except Exception as e:
        print(f"❌ 预览失败: {e}")


if __name__ == "__main__":
    print("🚀 Ultralytics Auto Cut - 使用示例")
    print()
    
    # 运行示例
    example_yolov8_640()
    # example_yolov5_custom()
    # example_dry_run()
    
    print("\n" + "=" * 60)
    print("💡 提示:")
    print("  - 使用 Web UI: ./start_webui.sh")
    print("  - 使用 CLI: python ultralytics_auto_cut.py --help")
    print("=" * 60)
