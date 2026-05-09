"""Build synthetic Ultralytics-style ONNX graphs for unit tests.

The graphs aren't trained, but they reproduce the shape topology that
``onnx_cut._detect_plan`` relies on: per-stride Conv outputs whose shapes
match the model-type's signature (yolov5 single-tensor / yolov8/v11 cls+bbox /
yolo26 cls+bbox with 4-ch ltrb bbox).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _conv_branch(
    *,
    in_name: str,
    out_name: str,
    in_ch: int,
    out_ch: int,
    h_in: int,
    w_in: int,
    h_out: int,
    w_out: int,
    node_name: str,
    nodes: list,
    inits: list,
) -> None:
    """Add a 1x1 Conv that reshapes spatial dims via stride to (out_ch, h_out, w_out)."""
    sh = max(1, h_in // h_out)
    sw = max(1, w_in // w_out)
    w = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * 0.01
    inits.append(numpy_helper.from_array(w, name=f"{node_name}_W"))
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[in_name, f"{node_name}_W"],
            outputs=[out_name],
            strides=[sh, sw],
            pads=[0, 0, 0, 0],
            kernel_shape=[1, 1],
            name=node_name,
        )
    )


def _finalize(graph_name: str, nodes, inits, inp_vi, outs_vi) -> onnx.ModelProto:
    g = helper.make_graph(nodes, graph_name, [inp_vi], outs_vi, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 7
    onnx.checker.check_model(m)
    return m


def make_yolov5(
    *,
    imgsz: int = 32,
    num_classes: int = 80,
    strides: Iterable[int] = (8, 16, 32),
) -> onnx.ModelProto:
    """One head per stride, channels = (num_classes + 5) * 3."""
    h, w = imgsz, imgsz
    ch = (num_classes + 5) * 3
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, h, w])
    nodes: list = []
    inits: list = []
    outs_vi: list = []
    for s in strides:
        ho, wo = h // s, w // s
        out_name = f"head_{s}"
        _conv_branch(
            in_name="images",
            out_name=out_name,
            in_ch=3,
            out_ch=ch,
            h_in=h,
            w_in=w,
            h_out=ho,
            w_out=wo,
            node_name=f"conv_head_{s}",
            nodes=nodes,
            inits=inits,
        )
        outs_vi.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, ch, ho, wo]))
    return _finalize("toy_yolov5", nodes, inits, inp, outs_vi)


def _make_decoupled(
    *,
    imgsz: int,
    num_classes: int,
    bbox_ch: int,
    strides: Iterable[int],
    name: str,
) -> onnx.ModelProto:
    h, w = imgsz, imgsz
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, h, w])
    nodes: list = []
    inits: list = []
    outs_vi: list = []
    for s in strides:
        ho, wo = h // s, w // s
        cls_name = f"cls_{s}"
        bbox_name = f"bbox_{s}"
        _conv_branch(
            in_name="images",
            out_name=cls_name,
            in_ch=3,
            out_ch=num_classes,
            h_in=h, w_in=w, h_out=ho, w_out=wo,
            node_name=f"conv_cls_{s}",
            nodes=nodes,
            inits=inits,
        )
        _conv_branch(
            in_name="images",
            out_name=bbox_name,
            in_ch=3,
            out_ch=bbox_ch,
            h_in=h, w_in=w, h_out=ho, w_out=wo,
            node_name=f"conv_bbox_{s}",
            nodes=nodes,
            inits=inits,
        )
        outs_vi.append(helper.make_tensor_value_info(cls_name, TensorProto.FLOAT, [1, num_classes, ho, wo]))
        outs_vi.append(helper.make_tensor_value_info(bbox_name, TensorProto.FLOAT, [1, bbox_ch, ho, wo]))
    return _finalize(name, nodes, inits, inp, outs_vi)


def make_yolov8(*, imgsz: int = 32, num_classes: int = 80, bbox_ch: int = 64,
                strides: Iterable[int] = (8, 16, 32)) -> onnx.ModelProto:
    return _make_decoupled(
        imgsz=imgsz, num_classes=num_classes, bbox_ch=bbox_ch,
        strides=strides, name="toy_yolov8",
    )


def make_yolo26(*, imgsz: int = 32, num_classes: int = 80, bbox_ch: int = 4,
                strides: Iterable[int] = (8, 16, 32)) -> onnx.ModelProto:
    return _make_decoupled(
        imgsz=imgsz, num_classes=num_classes, bbox_ch=bbox_ch,
        strides=strides, name="toy_yolo26",
    )
