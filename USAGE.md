# 使用指南

## 快速开始

### 方式 1: Web UI（推荐新手）

1. 启动服务：
```bash
./start_webui.sh
```

2. 打开浏览器访问：`http://127.0.0.1:18080`

3. 操作步骤：
   - 选择模型文件（.pt 或 .onnx）
   - 调整参数（通常保持默认即可）
   - 点击"预览"查看识别结果
   - 点击"转换"生成裁剪模型
   - 点击"下载"获取输出文件

### 方式 2: CLI（推荐自动化）

基本用法：
```bash
python ultralytics_auto_cut.py model.onnx -o output.onnx
```

完整参数：
```bash
python ultralytics_auto_cut.py model.pt \
  -o output.onnx \
  --imgsz 640 \
  --model-type auto \
  --classes 80 \
  --strides 8,16,32 \
  --simplify
```

## 参数说明

### 基础参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `model` | 输入模型路径 | - | `yolov8n.onnx` |
| `-o, --output` | 输出模型路径 | `<input>.cut.onnx` | `output.onnx` |
| `--imgsz` | 输入图像尺寸 | `640` | `640` 或 `640,640` |
| `--model-type` | 模型类型 | `auto` | `yolov5/yolov8/yolov11/yolo26` |
| `--classes` | 类别数量 | `80` | `80` |
| `--strides` | 检测头步长 | `8,16,32` | `8,16,32,64` (P6) |

### 高级参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--v8-bbox-ch` | YOLOv8/v11 bbox 通道数 | `64` |
| `--yolo26-bbox-ch` | YOLO26 bbox 通道数 | `4` |
| `--decoupled-order` | 解耦头输出顺序 | `auto` |
| `--simplify` | 使用 onnxsim 简化 | `False` |
| `--dry-run` | 仅预览不生成文件 | `False` |

### PT 导出参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--export-opset` | ONNX opset 版本 | `19` |
| `--exported-onnx` | 导出的 ONNX 保存路径 | `<input>.exported.onnx` |

## 使用场景

### 场景 1: 标准 YOLOv8 模型（COCO 80 类）

```bash
# 从 PT 文件开始
python ultralytics_auto_cut.py yolov8n.pt -o yolov8n.cut.onnx

# 从 ONNX 文件开始
python ultralytics_auto_cut.py yolov8n.onnx -o yolov8n.cut.onnx
```

### 场景 2: 自定义类别数

```bash
python ultralytics_auto_cut.py custom_model.onnx \
  -o output.onnx \
  --classes 20 \
  --imgsz 416
```

### 场景 3: P6 模型（4 个检测头）

```bash
python ultralytics_auto_cut.py yolov8n-p6.onnx \
  -o output.onnx \
  --strides 8,16,32,64
```

### 场景 4: 仅预览识别结果

```bash
python ultralytics_auto_cut.py model.onnx --dry-run
```

输出示例：
```
[识别结果]
model_type: yolov8
outputs:
  stride_8_cls: (1, 80, 80, 80) <- /model.22/cv2.0/cv2.0.2/Conv_output_0
  stride_8_bbox: (1, 80, 80, 64) <- /model.22/cv3.0/cv3.0.2/Conv_output_0
  ...
```

### 场景 5: 强制指定模型类型

当自动识别失败时：
```bash
python ultralytics_auto_cut.py model.onnx \
  --model-type yolov5 \
  --classes 80 \
  --imgsz 640
```

## 输出说明

### YOLOv5 输出格式

3 个输出张量（NHWC 格式）：
- `stride_8`: `[1, 80, 80, 255]`
- `stride_16`: `[1, 40, 40, 255]`
- `stride_32`: `[1, 20, 20, 255]`

其中 `255 = (80 + 5) * 3`（80 类 + 5 个 bbox 参数，3 个 anchor）

### YOLOv8/YOLOv11 输出格式

6 个输出张量（NHWC 格式）：
- `stride_8_cls`: `[1, 80, 80, 80]`
- `stride_8_bbox`: `[1, 80, 80, 64]`
- `stride_16_cls`: `[1, 40, 40, 80]`
- `stride_16_bbox`: `[1, 40, 40, 64]`
- `stride_32_cls`: `[1, 20, 20, 80]`
- `stride_32_bbox`: `[1, 20, 20, 64]`

### YOLO26 输出格式

6 个输出张量（NHWC 格式，bbox-cls 顺序）：
- `stride_8_bbox`: `[1, 80, 80, 4]`
- `stride_8_cls`: `[1, 80, 80, 80]`
- `stride_16_bbox`: `[1, 40, 40, 4]`
- `stride_16_cls`: `[1, 40, 40, 80]`
- `stride_32_bbox`: `[1, 20, 20, 4]`
- `stride_32_cls`: `[1, 20, 20, 80]`

## 常见问题

### Q1: 如何知道我的模型是什么类型？

使用 `--dry-run` 预览：
```bash
python ultralytics_auto_cut.py model.onnx --dry-run
```

### Q2: 转换后的模型如何使用？

裁剪后的模型可以直接用于 AXERA 芯片推理，输出已经是 NHWC 格式，方便后处理。

### Q3: 支持哪些 YOLO 版本？

- YOLOv5 (u/n/s/m/l/x)
- YOLOv8 (n/s/m/l/x)
- YOLOv11 (n/s/m/l/x)
- YOLO26

### Q4: 如何处理自定义训练的模型？

只需指定正确的 `--classes` 参数即可：
```bash
python ultralytics_auto_cut.py custom.onnx --classes 10
```

### Q5: 转换失败怎么办？

1. 检查模型文件是否完整
2. 尝试使用 `--model-type` 强制指定类型
3. 检查 `--imgsz` 和 `--classes` 是否正确
4. 查看错误信息，可能需要调整其他参数

## 性能优化

### 使用 onnxsim 简化模型

```bash
python ultralytics_auto_cut.py model.onnx -o output.onnx --simplify
```

这会在裁剪后运行 onnxsim 进行图优化，可能减小模型大小和提升推理速度。

### 批量处理

创建脚本批量转换：
```bash
#!/bin/bash
for model in models/*.onnx; do
  python ultralytics_auto_cut.py "$model" -o "output/$(basename $model .onnx).cut.onnx"
done
```

## 开发和调试

### 查看详细日志

Python 代码中可以启用详细输出：
```python
from onnx_cut import cut_ultralytics_onnx, CutConfig

cfg = CutConfig(dry_run=True)  # 预览模式
cut_ultralytics_onnx("model.onnx", "output.onnx", cfg)
```

### 自定义配置

```python
from onnx_cut import CutConfig, cut_ultralytics_onnx

cfg = CutConfig(
    model_type="yolov8",
    imgsz=(640, 640),
    num_classes=80,
    v8_bbox_ch=64,
    strides=(8, 16, 32),
    decoupled_order="cls-bbox",
    simplify=True,
    dry_run=False,
)

cut_ultralytics_onnx("input.onnx", "output.onnx", cfg)
```

## 更多资源

- [GitHub 仓库](https://github.com/your-repo)
- [问题反馈](https://github.com/your-repo/issues)
- [AXERA 官方文档](https://www.axera-tech.com/)
