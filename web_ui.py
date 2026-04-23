from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from onnx import shape_inference
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from export_ultralytics import export_pt_to_onnx
from onnx_cut import CutConfig, cut_ultralytics_onnx


@dataclass
class JobInfo:
    created_at: float
    input_name: str
    work_dir: str
    original_onnx_path: str
    cut_onnx_path: str


JOBS: dict[str, JobInfo] = {}
MAX_JOBS = 20


def _parse_imgsz(value: str) -> tuple[int, int]:
    value = value.strip()
    if "," in value:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("imgsz must be like '640' or '640,640'")
        return (int(parts[0]), int(parts[1]))
    size = int(value)
    return (size, size)


def _short_name(name: str, max_len: int = 28) -> str:
    if not name:
        return ""
    # Prefer the last path token for readability.
    token = name.split("/")[-1]
    if len(token) <= max_len:
        return token
    return token[: max_len - 1] + "…"


def _get_tensor_shape_map(model: onnx.ModelProto) -> dict[str, str]:
    shapes: dict[str, str] = {}

    def fmt_dim(d: onnx.TensorShapeProto.Dimension) -> str:
        if d.dim_value > 0:
            return str(int(d.dim_value))
        if d.dim_param:
            return str(d.dim_param)
        return "?"

    def add_vi(vi: onnx.ValueInfoProto) -> None:
        if not vi.type.HasField("tensor_type"):
            return
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return
        dims = [fmt_dim(d) for d in tt.shape.dim]
        if dims:
            shapes[vi.name] = "x".join(dims)

    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        add_vi(vi)
    return shapes


def _onnx_to_cytoscape_elements(
    model: onnx.ModelProto,
    include_initializers: bool = False,
    tensor_shapes: dict[str, str] | None = None,
) -> dict[str, Any]:
    initializer_names = {i.name for i in model.graph.initializer}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # Build stable node ids
    node_ids: list[str] = []
    for idx, node in enumerate(model.graph.node):
        node_id = node.name.strip() if node.name else f"{node.op_type}_{idx}"
        # Ensure uniqueness if needed.
        if node_id in node_ids:
            node_id = f"{node_id}__{idx}"
        node_ids.append(node_id)

    tensor_producer: dict[str, str] = {}

    def tshape(name: str) -> str:
        if not tensor_shapes:
            return ""
        return tensor_shapes.get(name, "")

    def node_io(node: onnx.NodeProto) -> dict[str, Any]:
        ins = [{"name": n, "shape": tshape(n)} for n in node.input if n]
        outs = [{"name": n, "shape": tshape(n)} for n in node.output if n]
        return {"inputs": ins, "outputs": outs}

    def add_node(node_id: str, label: str, kind: str, op_type: str | None = None) -> None:
        nodes.append(
            {
                "data": {
                    "id": node_id,
                    "label": label,
                    "kind": kind,
                    "op_type": op_type or kind,
                }
            }
        )

    # Graph inputs (non-initializer)
    for inp in model.graph.input:
        if inp.name in initializer_names and not include_initializers:
            continue
        nid = f"input::{inp.name}"
        shape_str = tshape(inp.name)
        label = f"Input\\n{_short_name(inp.name)}"
        if shape_str:
            label += f"\\n{shape_str}"
        add_node(nid, label, "input")
        tensor_producer[inp.name] = nid

    # Optional initializer nodes
    if include_initializers:
        for init in model.graph.initializer:
            nid = f"init::{init.name}"
            label = f"Init\\n{_short_name(init.name)}"
            add_node(nid, label, "init")
            tensor_producer[init.name] = nid

    # Operator nodes
    for idx, node in enumerate(model.graph.node):
        nid = node_ids[idx]
        out_shapes = [tshape(o) for o in node.output if o and tshape(o)]
        out_line = out_shapes[0] if out_shapes else ""
        if out_line and len(out_shapes) > 1:
            out_line = f"{out_line} (+{len(out_shapes) - 1})"

        label = f"{node.op_type}"
        if out_line:
            label += f"\\n{out_line}"
        else:
            label += f"\\n?"

        entry = {
            "data": {
                "id": nid,
                "label": label,
                "kind": "op",
                "op_type": node.op_type,
                "name": node.name or "",
                "io": node_io(node),
            }
        }
        nodes.append(entry)
        for out in node.output:
            if out:
                tensor_producer[out] = nid

    edge_idx = 0

    def add_edge(src: str, dst: str, label: str = "") -> None:
        nonlocal edge_idx
        edge_idx += 1
        edges.append({"data": {"id": f"e{edge_idx}", "source": src, "target": dst, "label": label}})

    # Edges between nodes
    id_by_node = {i: node_ids[i] for i in range(len(node_ids))}
    for idx, node in enumerate(model.graph.node):
        dst = id_by_node[idx]
        for inp in node.input:
            if not inp:
                continue
            src = tensor_producer.get(inp)
            if src:
                add_edge(src, dst)

    # Graph outputs
    for out in model.graph.output:
        nid = f"output::{out.name}"
        shape_str = tshape(out.name)
        label = f"Output\\n{_short_name(out.name)}"
        if shape_str:
            label += f"\\n{shape_str}"
        add_node(nid, label, "output")
        src = tensor_producer.get(out.name)
        if src:
            add_edge(src, nid)

    return {
        "elements": {"nodes": nodes, "edges": edges},
        "stats": {"nodes": len(nodes), "edges": len(edges)},
    }


def _load_graph_json(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path))
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    shapes = _get_tensor_shape_map(inferred)
    return _onnx_to_cytoscape_elements(inferred, include_initializers=False, tensor_shapes=shapes)


def _purge_old_jobs() -> None:
    if len(JOBS) <= MAX_JOBS:
        return
    items = sorted(JOBS.items(), key=lambda kv: kv[1].created_at)
    for job_id, info in items[: max(0, len(items) - MAX_JOBS)]:
        try:
            shutil.rmtree(info.work_dir, ignore_errors=True)
        except Exception:
            pass
        JOBS.pop(job_id, None)


HTML = """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Ultralytics Auto Cut - AXERA</title>
    <style>
      * { box-sizing: border-box; }
      :root {
        --bg: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        --panel: rgba(30, 41, 59, 0.6);
        --panel-hover: rgba(51, 65, 85, 0.8);
        --text: #f1f5f9;
        --text-muted: #94a3b8;
        --accent: #3b82f6;
        --accent-hover: #2563eb;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --border: rgba(148, 163, 184, 0.2);
        --shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
      }
      html, body {
        height: 100%;
        margin: 0;
        background: var(--bg);
        background-attachment: fixed;
        color: var(--text);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        line-height: 1.6;
      }
      .app {
        display: flex;
        height: 100vh;
        gap: 16px;
        padding: 16px;
      }
      .col {
        flex: 1;
        background: var(--panel);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        min-width: 320px;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
      }
      .col:hover { border-color: rgba(148, 163, 184, 0.4); }
      .col.collapsed {
        flex: 0 0 48px;
        min-width: 48px;
      }
      .col.collapsed .header h3,
      .col.collapsed .hint,
      .col.collapsed .graph,
      .col.collapsed .details {
        display: none;
      }
      .mid { flex: 0 0 460px; }
      .header {
        padding: 20px 24px;
        border-bottom: 1px solid var(--border);
        background: rgba(15, 23, 42, 0.5);
        position: relative;
      }
      .toggle-btn {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid var(--accent);
        color: var(--accent);
        width: 28px;
        height: 28px;
        border-radius: 6px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        transition: all 0.2s ease;
        padding: 0;
      }
      .toggle-btn:hover {
        background: rgba(59, 130, 246, 0.3);
        transform: translateY(-50%) scale(1.1);
      }
      .toggle-left { right: 12px; }
      .toggle-right { left: 12px; }
      .collapsed .toggle-btn {
        top: 12px;
        transform: none;
      }
      .collapsed .toggle-left { right: 10px; }
      .collapsed .toggle-right { left: 10px; }
      .header h3 {
        margin: 0 0 4px 0;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: -0.02em;
      }
      .hint {
        color: var(--text-muted);
        font-size: 12px;
        font-weight: 500;
      }
      .graph {
        flex: 1;
        position: relative;
        background: rgba(15, 23, 42, 0.3);
        overflow: hidden;
      }
      .graph iframe { 
        position: absolute; 
        inset: 0;
        border: none;
        background: #0f172a;
        width: 100%;
        height: 100%;
      }
      .details {
        border-top: 1px solid var(--border);
        padding: 12px 16px;
        font-size: 11px;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        color: var(--text-muted);
        background: rgba(15, 23, 42, 0.7);
        min-height: 80px;
        max-height: 140px;
        overflow-y: auto;
        white-space: pre-wrap;
        line-height: 1.5;
      }
      .panel {
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        overflow-y: auto;
      }
      .section {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
      }
      .section-title {
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 12px 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
      }
      .row {
        display: grid;
        grid-template-columns: 130px 1fr;
        gap: 12px;
        align-items: center;
        margin-bottom: 12px;
      }
      .row:last-child { margin-bottom: 0; }
      label {
        font-size: 12px;
        font-weight: 500;
        color: var(--text-muted);
      }
      input, select {
        width: 100%;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: rgba(15, 23, 42, 0.6);
        color: var(--text);
        outline: none;
        transition: all 0.2s ease;
        font-size: 13px;
      }
      input:focus, select:focus {
        border-color: var(--accent);
        background: rgba(15, 23, 42, 0.8);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
      }
      input[type=file] {
        padding: 8px;
        cursor: pointer;
      }
      input[type=checkbox] {
        width: auto;
        height: 18px;
        cursor: pointer;
        accent-color: var(--accent);
      }
      button {
        padding: 12px 20px;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      .actions {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
        margin-top: 8px;
      }
      #previewBtn {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
      }
      #previewBtn:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
      }
      #convertBtn {
        background: linear-gradient(135deg, var(--accent) 0%, #1d4ed8 100%);
        color: white;
      }
      #convertBtn:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
      }
      #downloadBtn {
        background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
        color: white;
      }
      #downloadBtn:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
      }
      .status {
        font-size: 12px;
        font-family: 'SF Mono', Monaco, monospace;
        color: var(--text-muted);
        white-space: pre-wrap;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px;
        min-height: 90px;
        line-height: 1.6;
      }
      .footer {
        font-size: 11px;
        color: var(--text-muted);
        text-align: center;
        padding: 12px;
        border-top: 1px solid var(--border);
        background: rgba(15, 23, 42, 0.4);
      }
      .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      .badge-info { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
      .badge-success { background: rgba(16, 185, 129, 0.2); color: #34d399; }
      a { color: var(--accent); text-decoration: none; transition: color 0.2s; }
      a:hover { color: var(--accent-hover); }
      ::-webkit-scrollbar { width: 8px; height: 8px; }
      ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.3); }
      ::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.3); border-radius: 4px; }
      ::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.5); }
    </style>
  </head>
  <body>
    <div class="app">
      <div class="col" id="leftCol">
        <div class="header">
          <h3>📥 原始模型</h3>
          <div class="hint">导出/输入 ONNX · <span id="origStats" class="badge badge-info">-</span></div>
          <button class="toggle-btn toggle-left" id="toggleLeft" title="收起/展开">◀</button>
        </div>
        <div class="graph" id="origGraph"></div>
        <div class="details" id="origDetails">上传模型后显示</div>
      </div>

      <div class="col mid">
        <div class="header">
          <h3>⚙️ 参数配置</h3>
          <div class="hint">YOLO 模型自动裁剪工具</div>
        </div>
        <div class="panel">
          <div class="section">
            <div class="row">
              <label>模型文件</label>
              <input id="modelFile" type="file" accept=".onnx,.pt" />
            </div>
          </div>

          <div class="section">
            <div class="section-title">基础配置</div>
            <div class="row">
              <label>imgsz</label>
              <input id="imgsz" value="640" placeholder="640 or 640,640" />
            </div>
            <div class="row">
              <label>model_type</label>
              <select id="modelType">
                <option value="auto" selected>auto</option>
                <option value="yolov5">yolov5</option>
                <option value="yolov8">yolov8</option>
                <option value="yolov11">yolov11</option>
                <option value="yolo26">yolo26</option>
              </select>
            </div>
            <div class="row">
              <label>classes</label>
              <input id="classes" value="80" type="number" />
            </div>
            <div class="row">
              <label>strides</label>
              <input id="strides" value="8,16,32" placeholder="8,16,32 or 8,16,32,64" />
            </div>
          </div>

          <div class="section">
            <div class="section-title">高级配置</div>
            <div class="row">
              <label>v8_bbox_ch</label>
              <input id="v8BboxCh" value="64" type="number" />
            </div>
            <div class="row">
              <label>yolo26_bbox_ch</label>
              <input id="y26BboxCh" value="4" type="number" />
            </div>
            <div class="row">
              <label>decoupled_order</label>
              <select id="decoupledOrder">
                <option value="" selected>auto</option>
                <option value="cls-bbox">cls-bbox</option>
                <option value="bbox-cls">bbox-cls</option>
              </select>
            </div>
            <div class="row">
              <label>simplify</label>
              <input id="simplify" type="checkbox" />
            </div>
          </div>

          <div class="actions">
            <button id="previewBtn">预览</button>
            <button id="convertBtn">转换</button>
            <button id="downloadBtn" disabled>下载</button>
          </div>

          <div class="status" id="statusBox">💡 请选择 .pt 或 .onnx 模型文件开始</div>
        </div>
        <div class="footer">
          使用 Netron 可视化 · 点击查看完整模型结构
        </div>
      </div>

      <div class="col" id="rightCol">
        <div class="header">
          <h3>📤 裁剪模型</h3>
          <div class="hint">Cut + Transpose · <span id="cutStats" class="badge badge-success">-</span></div>
          <button class="toggle-btn toggle-right" id="toggleRight" title="收起/展开">▶</button>
        </div>
        <div class="graph" id="cutGraph"></div>
        <div class="details" id="cutDetails">转换后显示</div>
      </div>
    </div>

    <script src="/static/cytoscape.min.js"></script>
    <script src="/static/dagre.min.js"></script>
    <script src="/static/cytoscape-dagre.js"></script>
    <script>
      let currentDownloadUrl = null;
      let currentJobId = null;

      function setDownloadEnabled(enabled) {
        document.getElementById('downloadBtn').disabled = !enabled;
      }

      function setStatus(msg, icon) {
        if (!icon) icon = '💡';
        document.getElementById('statusBox').textContent = icon + ' ' + msg;
      }

      function showNetron(containerId, modelUrl) {
        const container = document.getElementById(containerId);
        const iframe = document.createElement('iframe');
        iframe.src = 'https://netron.app/?url=' + encodeURIComponent(window.location.origin + modelUrl);
        iframe.style.cssText = 'width:100%;height:100%;border:none;';
        container.innerHTML = '';
        container.appendChild(iframe);
      }

      async function buildFormData() {
        const fd = new FormData();
        const f = document.getElementById('modelFile').files[0];
        if (!f) { 
          setStatus('请先选择一个 .pt 或 .onnx 文件', '⚠️'); 
          return null; 
        }
        fd.append('model_file', f);
        fd.append('imgsz', document.getElementById('imgsz').value);
        fd.append('model_type', document.getElementById('modelType').value);
        fd.append('classes', document.getElementById('classes').value);
        fd.append('v8_bbox_ch', document.getElementById('v8BboxCh').value);
        fd.append('yolo26_bbox_ch', document.getElementById('y26BboxCh').value);
        fd.append('decoupled_order', document.getElementById('decoupledOrder').value);
        fd.append('strides', document.getElementById('strides').value);
        fd.append('simplify', document.getElementById('simplify').checked ? 'true' : 'false');
        return fd;
      }

      async function preview() {
        console.log('Preview button clicked');
        const fd = await buildFormData();
        if (!fd) return;
        
        const btn = document.getElementById('previewBtn');
        btn.disabled = true;
        setStatus('预览中…（识别模型类型和输出张量）', '🔍');
        
        try {
          console.log('Sending preview request...');
          const resp = await fetch('/api/preview', { method: 'POST', body: fd });
          console.log('Response status:', resp.status);
          const data = await resp.json();
          console.log('Response data:', data);
          
          if (!resp.ok) { 
            setStatus('预览失败：' + (data.detail || JSON.stringify(data)), '❌'); 
            return; 
          }
          
          const msg = '[预览完成]' + String.fromCharCode(10) + 
                      'model_type: ' + data.model_type + String.fromCharCode(10) +
                      'outputs:' + String.fromCharCode(10) + '  ' + data.outputs.join(String.fromCharCode(10) + '  ');
          setStatus(msg, '✅');
          document.getElementById('origStats').textContent = data.model_type;
          document.getElementById('origDetails').textContent = '模型类型: ' + data.model_type + String.fromCharCode(10) + '输出数量: ' + data.outputs.length;
          
          if (data.original_onnx_url) {
            showNetron('origGraph', data.original_onnx_url);
          }
        } catch (e) {
          console.error('Preview error:', e);
          setStatus('异常：' + e.message, '❌');
        } finally {
          btn.disabled = false;
        }
      }

      async function convert() {
        const fd = await buildFormData();
        if (!fd) return;

        setDownloadEnabled(false);
        currentDownloadUrl = null;

        const btn = document.getElementById('convertBtn');
        btn.disabled = true;
        setStatus('处理中…（上传/导出/裁剪中）', '⚙️');

        try {
          const resp = await fetch('/api/convert', { method: 'POST', body: fd });
          const text = await resp.text();
          let data = null;
          try { data = JSON.parse(text); } catch (e) { data = { detail: text || String(e) }; }
          
          if (!resp.ok) {
            setStatus('失败：' + (data.detail || JSON.stringify(data)), '❌');
            return;
          }
          
          const msg = '[转换完成]' + String.fromCharCode(10) + 
                      'model_type: ' + data.model_type + String.fromCharCode(10) +
                      'outputs:' + String.fromCharCode(10) + '  ' + data.outputs.join(String.fromCharCode(10) + '  ');
          setStatus(msg, '✅');
          
          currentJobId = data.job_id;
          currentDownloadUrl = data.download_url;
          setDownloadEnabled(true);
          
          document.getElementById('origStats').textContent = data.model_type;
          document.getElementById('cutStats').textContent = 'NHWC';
          document.getElementById('origDetails').textContent = '模型类型: ' + data.model_type + String.fromCharCode(10) + '输出数量: ' + data.outputs.length;
          document.getElementById('cutDetails').textContent = '裁剪完成' + String.fromCharCode(10) + '输出数量: ' + data.outputs.length + String.fromCharCode(10) + '格式: NHWC';
          
          if (data.original_onnx_url) {
            showNetron('origGraph', data.original_onnx_url);
          }
          
          if (data.cut_onnx_url) {
            showNetron('cutGraph', data.cut_onnx_url);
          }
        } catch (e) {
          setStatus('异常：' + e.message, '❌');
          console.error('Convert error:', e);
        } finally {
          btn.disabled = false;
        }
      }

      document.getElementById('previewBtn').addEventListener('click', function() {
        console.log('Preview button event triggered');
        preview();
      });
      document.getElementById('convertBtn').addEventListener('click', function() {
        console.log('Convert button event triggered');
        convert();
      });
      document.getElementById('downloadBtn').addEventListener('click', function() {
        console.log('Download button event triggered');
        if (!currentDownloadUrl) { 
          setStatus('请先转换生成模型', '⚠️'); 
          return; 
        }
        window.location.href = currentDownloadUrl;
      });

      // Toggle collapse for left and right panels
      document.getElementById('toggleLeft').addEventListener('click', function() {
        const col = document.getElementById('leftCol');
        const btn = document.getElementById('toggleLeft');
        col.classList.toggle('collapsed');
        btn.textContent = col.classList.contains('collapsed') ? '▶' : '◀';
      });

      document.getElementById('toggleRight').addEventListener('click', function() {
        const col = document.getElementById('rightCol');
        const btn = document.getElementById('toggleRight');
        col.classList.toggle('collapsed');
        btn.textContent = col.classList.contains('collapsed') ? '◀' : '▶';
      });
    </script>
  </body>
</html>
"""


STATIC_DIR = Path(__file__).with_name("static")
app = FastAPI()

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


def _parse_strides(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("strides must be like '8,16,32'")
    return tuple(int(p) for p in parts)


@app.post("/api/preview")
async def api_preview(
    model_file: UploadFile = File(...),
    imgsz: str = Form("640"),
    model_type: str = Form("auto"),
    classes: str = Form("80"),
    v8_bbox_ch: str = Form("64"),
    yolo26_bbox_ch: str = Form("4"),
    decoupled_order: str = Form(""),
    strides: str = Form("8,16,32"),
) -> JSONResponse:
    """Dry-run: detect plan and return output tensor info without writing any file."""
    try:
        imgsz_tuple = _parse_imgsz(imgsz)
        num_classes = int(classes)
        v8_bbox = int(v8_bbox_ch)
        y26_bbox = int(yolo26_bbox_ch)
        order = decoupled_order.strip() or None
        strides_tuple = _parse_strides(strides)
    except Exception as e:
        return JSONResponse({"detail": f"Invalid parameters: {e}"}, status_code=400)

    suffix = Path(model_file.filename or "model").suffix.lower()
    if suffix not in (".onnx", ".pt"):
        return JSONResponse({"detail": "Only .onnx and .pt are supported"}, status_code=400)

    tmpdir = Path(tempfile.mkdtemp(prefix="ultra-cut-preview-"))
    uploaded_path = tmpdir / f"input{suffix}"
    try:
        with uploaded_path.open("wb") as f:
            f.write(await model_file.read())

        original_onnx = uploaded_path
        if suffix == ".pt":
            original_onnx = tmpdir / "exported.onnx"
            try:
                export_pt_to_onnx(uploaded_path, original_onnx, imgsz_tuple, opset=19)
            except Exception as e:
                return JSONResponse({"detail": f"Export .pt -> .onnx failed: {e}"}, status_code=500)

        cfg = CutConfig(
            model_type=model_type,
            imgsz=imgsz_tuple,
            num_classes=num_classes,
            v8_bbox_ch=v8_bbox,
            yolo26_bbox_ch=y26_bbox,
            decoupled_order=order,
            strides=strides_tuple,
            dry_run=True,
        )
        try:
            from onnx_cut import _detect_plan
            model_proto = onnx.load(str(original_onnx))
            plan = _detect_plan(model_proto, cfg)
            outputs = [f"{name}: {src.shape} -> {shape}" for (src, name, shape) in plan.outputs]
            
            # 保存到 JOBS 以便访问
            job_id = uuid.uuid4().hex
            JOBS[job_id] = JobInfo(
                created_at=time.time(),
                input_name=model_file.filename or "model",
                work_dir=str(tmpdir),
                original_onnx_path=str(original_onnx),
                cut_onnx_path="",
            )
            _purge_old_jobs()
            
            return JSONResponse({
                "model_type": plan.model_type, 
                "outputs": outputs,
                "original_onnx_url": f"/api/view_onnx/{job_id}/original"
            })
        except RuntimeError as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            error_msg = str(e)
            if "No candidates to pick from" in error_msg:
                error_msg = (
                    "无法识别模型结构。可能原因：\n"
                    "1. 模型不是标准的 YOLOv5/v8/v11/v26\n"
                    "2. 输入尺寸(imgsz)不正确\n"
                    "3. 类别数(classes)不正确\n"
                    "4. 模型已经被裁剪过\n\n"
                    "建议：\n"
                    "- 检查 imgsz 参数（如 640, 416, 1280 等）\n"
                    "- 检查 classes 参数（COCO=80）\n"
                    "- 尝试手动指定 model_type"
                )
            return JSONResponse({"detail": error_msg}, status_code=400)
        except Exception as e:
            import traceback
            traceback.print_exc()
            shutil.rmtree(tmpdir, ignore_errors=True)
            return JSONResponse({"detail": f"处理失败: {str(e)}"}, status_code=500)
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.post("/api/convert")
async def api_convert(
    model_file: UploadFile = File(...),
    imgsz: str = Form("640"),
    model_type: str = Form("auto"),
    classes: str = Form("80"),
    v8_bbox_ch: str = Form("64"),
    yolo26_bbox_ch: str = Form("4"),
    decoupled_order: str = Form(""),
    strides: str = Form("8,16,32"),
    simplify: str = Form("false"),
) -> JSONResponse:
    try:
        imgsz_tuple = _parse_imgsz(imgsz)
        num_classes = int(classes)
        v8_bbox = int(v8_bbox_ch)
        y26_bbox = int(yolo26_bbox_ch)
        order = decoupled_order.strip() or None
        strides_tuple = _parse_strides(strides)
        do_simplify = simplify.strip().lower() in ("true", "1", "yes")
    except Exception as e:
        return JSONResponse({"detail": f"Invalid parameters: {e}"}, status_code=400)

    suffix = Path(model_file.filename or "model").suffix.lower()
    if suffix not in (".onnx", ".pt"):
        return JSONResponse({"detail": "Only .onnx and .pt are supported"}, status_code=400)

    tmpdir = Path(tempfile.mkdtemp(prefix="ultra-cut-"))
    uploaded_path = tmpdir / f"input{suffix}"
    with uploaded_path.open("wb") as f:
        f.write(await model_file.read())

    original_onnx = uploaded_path
    if suffix == ".pt":
        original_onnx = tmpdir / "exported.onnx"
        try:
            export_pt_to_onnx(uploaded_path, original_onnx, imgsz_tuple, opset=19)
        except Exception as e:
            return JSONResponse({"detail": f"Export .pt -> .onnx failed: {e}"}, status_code=500)

    cfg = CutConfig(
        model_type=model_type,
        imgsz=imgsz_tuple,
        num_classes=num_classes,
        v8_bbox_ch=v8_bbox,
        yolo26_bbox_ch=y26_bbox,
        decoupled_order=order,
        strides=strides_tuple,
        simplify=do_simplify,
        dry_run=False,
    )

    cut_path = tmpdir / "cut.onnx"
    try:
        cut_ultralytics_onnx(original_onnx, cut_path, cfg)
    except Exception as e:
        return JSONResponse({"detail": f"Cut failed: {e}"}, status_code=500)

    try:
        original_graph = _load_graph_json(original_onnx)
        cut_graph = _load_graph_json(cut_path)
    except Exception as e:
        return JSONResponse({"detail": f"Graph visualization build failed: {e}"}, status_code=500)

    # Use a quick dry-run to report outputs/model_type.
    try:
        from onnx_cut import _detect_plan
        plan = _detect_plan(onnx.load(str(original_onnx)), cfg)
        outputs = [f"{name}: {src.shape} -> {shape}" for (src, name, shape) in plan.outputs]
        detected_type = plan.model_type
    except RuntimeError as e:
        error_msg = str(e)
        if "No candidates to pick from" in error_msg:
            error_msg = (
                "无法识别模型结构。可能原因：\n"
                "1. 模型不是标准的 YOLOv5/v8/v11/v26\n"
                "2. 输入尺寸(imgsz)不正确\n"
                "3. 类别数(classes)不正确\n"
                "4. 模型已经被裁剪过"
            )
        return JSONResponse({"detail": error_msg}, status_code=400)
    except Exception as e:
        outputs = []
        detected_type = model_type

    job_id = uuid.uuid4().hex
    JOBS[job_id] = JobInfo(
        created_at=time.time(),
        input_name=model_file.filename or "model",
        work_dir=str(tmpdir),
        original_onnx_path=str(original_onnx),
        cut_onnx_path=str(cut_path),
    )
    _purge_old_jobs()

    return JSONResponse(
        {
            "job_id": job_id,
            "model_type": detected_type,
            "outputs": outputs,
            "original_graph": original_graph,
            "cut_graph": cut_graph,
            "download_url": f"/api/download/{job_id}",
            "original_onnx_url": f"/api/view_onnx/{job_id}/original",
            "cut_onnx_url": f"/api/view_onnx/{job_id}/cut",
        }
    )


@app.get("/api/view_onnx/{job_id}/{model_type}")
def api_view_onnx(job_id: str, model_type: str) -> FileResponse:
    """Serve ONNX file for Netron visualization"""
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if model_type == "original":
        path = info.original_onnx_path
    elif model_type == "cut":
        path = info.cut_onnx_path
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type")
    
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(path=path, media_type="application/octet-stream")


@app.get("/api/download/{job_id}")
def api_download(job_id: str) -> FileResponse:
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found (expired or invalid id)")
    filename = f"{Path(info.input_name).stem}.cut.onnx"
    return FileResponse(path=info.cut_onnx_path, filename=filename, media_type="application/octet-stream")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Web UI for ultralytics-auto-cut")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
