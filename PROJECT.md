# 项目结构

```
ultralytics-auto-cut.axera/
├── README.md                    # 项目说明文档
├── USAGE.md                     # 详细使用指南
├── TODO.md                      # 改进建议和待办事项
├── requirements.txt             # Python 依赖
├── .gitignore                   # Git 忽略文件
│
├── ultralytics_auto_cut.py      # CLI 主程序
├── onnx_cut.py                  # 核心裁剪逻辑
├── export_ultralytics.py        # PT 转 ONNX 导出
├── web_ui.py                    # Web UI 服务
│
├── start_webui.sh               # Web UI 启动脚本
├── example.py                   # 使用示例代码
│
└── static/                      # Web UI 静态资源
    ├── cytoscape.min.js         # 图形可视化库
    ├── dagre.min.js             # 布局算法库
    └── cytoscape-dagre.js       # Cytoscape Dagre 插件
```

## 核心模块说明

### 1. ultralytics_auto_cut.py
CLI 命令行工具主程序，提供：
- 参数解析
- PT/ONNX 输入处理
- 调用核心裁剪逻辑

### 2. onnx_cut.py
核心裁剪逻辑，包含：
- 模型类型自动识别
- 检测头定位
- 图裁剪和 Transpose 插入
- 支持 YOLOv5/v8/v11/v26

### 3. export_ultralytics.py
PT 模型导出为 ONNX：
- 使用 Ultralytics 官方导出接口
- 支持自定义 opset 和输入尺寸

### 4. web_ui.py
Web UI 服务，提供：
- FastAPI 后端
- 现代化前端界面
- 实时 ONNX 图可视化
- 文件上传/下载
- 参数配置面板

## 功能特性

### ✅ 已实现
- [x] 支持 YOLOv5/v8/v11/v26 模型
- [x] 自动模型类型识别
- [x] PT/ONNX 输入支持
- [x] 按 stride 裁剪检测头
- [x] 自动添加 NCHW->NHWC Transpose
- [x] CLI 命令行工具
- [x] Web UI 界面
- [x] 实时图可视化
- [x] 预览模式（dry-run）
- [x] onnxsim 简化支持
- [x] 自定义类别数/步长
- [x] P6 模型支持（4 个检测头）

### 🎨 UI 特性
- [x] 现代化深色主题
- [x] 渐变色设计
- [x] 响应式布局
- [x] 交互式节点查看
- [x] 实时状态反馈
- [x] 图标和视觉提示
- [x] 平滑动画效果

## 使用场景

1. **AXERA 芯片部署**：将 Ultralytics YOLO 模型转换为 AXERA 推理格式
2. **模型优化**：裁剪不需要的后处理节点，减小模型大小
3. **格式转换**：NCHW -> NHWC，方便后处理代码对接
4. **批量处理**：CLI 支持脚本化批量转换

## 技术栈

### 后端
- Python 3.8+
- ONNX (模型处理)
- Ultralytics (PT 导出)
- FastAPI (Web 服务)
- Uvicorn (ASGI 服务器)

### 前端
- 原生 HTML/CSS/JavaScript
- Cytoscape.js (图可视化)
- Dagre (图布局算法)

## 开发指南

### 环境配置
```bash
pip install -r requirements.txt
```

### 运行测试
```bash
# CLI 测试
python ultralytics_auto_cut.py model.onnx --dry-run

# Web UI 测试
./start_webui.sh
```

### 代码风格
- 使用 type hints
- 遵循 PEP 8
- 函数/类添加 docstring

## 性能指标

- 支持模型大小：< 1GB
- 转换速度：通常 < 5 秒
- Web UI 响应：< 100ms
- 图可视化：支持 < 1000 节点

## 兼容性

### 支持的模型
- YOLOv5 (所有变体)
- YOLOv8 (n/s/m/l/x)
- YOLOv11 (n/s/m/l/x)
- YOLO26

### 支持的输入
- ONNX 模型 (opset 11+)
- PyTorch 模型 (.pt)

### 输出格式
- ONNX (opset 11+)
- NHWC 张量布局
- 按 stride 分离的检测头

## 许可证

待定

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- GitHub: [项目地址]
- Email: [联系邮箱]
