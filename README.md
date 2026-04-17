# ultralytics-auto-cut.axera

把 Ultralytics 系列 YOLO 模型（`.pt` / `.onnx`）自动导出/裁剪成 **按 stride 输出的检测头特征**，并在输出后追加 `Transpose(NCHW->NHWC)`，方便对接 AXERA 推理后处理代码。

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Web UI
./start_webui.sh

# 或使用 CLI
python ultralytics_auto_cut.py model.onnx -o output.onnx
```

📖 [查看快速启动指南](QUICKSTART.md) | [详细使用文档](USAGE.md)

## 支持的 cut 形态（以 COCO 80 类、640x640 为例）

- **YOLOv5**（3 个输出，按 stride 8/16/32）  
  - cut（NCHW）：`1x255x80x80`、`1x255x40x40`、`1x255x20x20`  
  - transpose 后（NHWC）：`1x80x80x255`、`1x40x40x255`、`1x20x20x255`  
  - 输出名：`stride_8` `stride_16` `stride_32`

- **YOLOv8 / YOLOv11**（6 个输出：每个 stride 为 `cls` + `bbox`）  
  - cut（NCHW）：`cls=1x80xHxW`，`bbox=1x64xHxW`  
  - transpose 后（NHWC）：`stride_<s>_cls=1xHxWx80`，`stride_<s>_bbox=1xHxWx64`

- **YOLO26**（6 个输出：每个 stride 为 `cls` + `bbox`，bbox 为 ltrb）  
  - cut（NCHW）：`cls=1x80xHxW`，`bbox=1x4xHxW`  
  - transpose 后（NHWC）：`stride_<s>_cls=1xHxWx80`，`stride_<s>_bbox=1xHxWx4`
  - 默认输出顺序匹配 `ax_yolo26_steps.cc`：`bbox, cls` 交错（`bbox8,cls8,bbox16,cls16,bbox32,cls32`）

## 使用

### Web UI（推荐）

**快速启动：**
```bash
./start_webui.sh
# 或
python web_ui.py --host 127.0.0.1 --port 18080
```

浏览器打开：`http://127.0.0.1:18080`

**功能特性：**
- 🎨 现代化深色主题界面
- 📊 实时 ONNX 结构可视化（原始模型 vs 裁剪后模型）
- ⚙️ 直观的参数配置面板
- 🔍 预览功能（识别模型类型和输出张量）
- 📥 一键转换和下载
- 🖱️ 交互式节点查看（点击节点查看详细信息）

### CLI 命令行

**1) 输入 ONNX**

```bash
python ultralytics_auto_cut.py /path/to/model.onnx -o model.cut.onnx --imgsz 640
```

**2) 输入 PT（自动导出 ONNX 再 cut）**

```bash
python ultralytics_auto_cut.py /path/to/model.pt -o model.cut.onnx --imgsz 640
```

**3) 仅查看识别结果（不写输出）**

```bash
python ultralytics_auto_cut.py /path/to/model.onnx --dry-run
```

## 说明

- `--model-type auto` 会基于 head tensor 的 shape 自动识别：`yolov5 / yolov8(yolov11) / yolo26`。
- 如遇到识别失败，可用 `--model-type` 强制指定，并用 `--imgsz`/通道参数修正。
