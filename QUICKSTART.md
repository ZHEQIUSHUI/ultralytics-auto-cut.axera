# 快速启动指南

## 🚀 5 分钟上手

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 启动 Web UI

```bash
./start_webui.sh
```

然后在浏览器打开：http://127.0.0.1:18080

### 3️⃣ 转换模型

**方式 A: 使用 Web UI**
1. 点击"选择文件"上传 .pt 或 .onnx 模型
2. 调整参数（通常保持默认）
3. 点击"转换"
4. 点击"下载"获取裁剪后的模型

**方式 B: 使用命令行**
```bash
python ultralytics_auto_cut.py your_model.onnx -o output.onnx
```

## 📋 常用命令

### 预览模型信息
```bash
python ultralytics_auto_cut.py model.onnx --dry-run
```

### 转换 PT 模型
```bash
python ultralytics_auto_cut.py model.pt -o output.onnx
```

### 自定义参数
```bash
python ultralytics_auto_cut.py model.onnx \
  -o output.onnx \
  --imgsz 640 \
  --classes 80 \
  --simplify
```

## 🎯 输出说明

转换后的模型会：
- ✅ 裁剪掉后处理节点
- ✅ 按 stride (8/16/32) 分离检测头
- ✅ 添加 NCHW->NHWC Transpose
- ✅ 输出张量命名规范（如 `stride_8_cls`）

## 📚 更多文档

- [详细使用指南](USAGE.md)
- [项目结构说明](PROJECT.md)
- [改进建议](TODO.md)

## ❓ 遇到问题？

1. 检查模型文件是否完整
2. 尝试使用 `--dry-run` 预览
3. 查看 [USAGE.md](USAGE.md) 常见问题部分
4. 提交 Issue 到 GitHub

## 🎉 完成！

现在你可以将裁剪后的模型部署到 AXERA 芯片上了！
