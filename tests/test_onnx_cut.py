from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from onnx_cut import (
    AutoDetectError,
    CutConfig,
    CutOutput,
    _detect_plan,
    cut_ultralytics_onnx,
)
from tests.synthetic_models import (
    make_yolov5,
    make_yolov8,
    make_yolo26,
    make_yolov8_seg,
)


# ---------------------------------------------------------------------------
# _detect_plan
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_rng():
    np.random.seed(0)


@pytest.mark.parametrize(
    "factory,expected_type",
    [
        (lambda: make_yolov5(), "yolov5"),
        (lambda: make_yolov8(), "yolov8"),
        (lambda: make_yolo26(), "yolo26"),
    ],
)
def test_auto_detect_identifies_model_type(factory, expected_type):
    model = factory()
    plan = _detect_plan(model, CutConfig(model_type="auto"))
    assert plan.model_type == expected_type


def test_yolov5_plan_has_one_output_per_stride():
    model = make_yolov5()
    plan = _detect_plan(model, CutConfig(model_type="auto"))
    assert len(plan.outputs) == 3
    for out in plan.outputs:
        assert isinstance(out, CutOutput)
        assert len(out.sources) == 1
        assert out.shape[-1] == (80 + 5) * 3  # NHWC channels last


def test_yolov8_split_emits_two_outputs_per_stride():
    model = make_yolov8()
    plan = _detect_plan(model, CutConfig(model_type="auto"))
    assert len(plan.outputs) == 6
    names = [o.name for o in plan.outputs]
    assert names == [
        "stride_8_cls", "stride_8_bbox",
        "stride_16_cls", "stride_16_bbox",
        "stride_32_cls", "stride_32_bbox",
    ]
    for out in plan.outputs:
        assert len(out.sources) == 1


def test_yolov8_merge_stride_emits_one_output_per_stride():
    model = make_yolov8()
    plan = _detect_plan(model, CutConfig(model_type="auto", merge_stride=True))
    assert len(plan.outputs) == 3
    for out in plan.outputs:
        assert len(out.sources) == 2
        # bbox first, cls second per the user-spec.
        assert out.sources[0].shape[1] == 64  # bbox_ch
        assert out.sources[1].shape[1] == 80  # cls_ch (num_classes)
        assert out.shape[-1] == 64 + 80
    assert [o.name for o in plan.outputs] == ["stride_8", "stride_16", "stride_32"]


def test_yolo26_default_order_is_bbox_cls():
    model = make_yolo26()
    plan = _detect_plan(model, CutConfig(model_type="auto"))
    names = [o.name for o in plan.outputs]
    assert names == [
        "stride_8_bbox", "stride_8_cls",
        "stride_16_bbox", "stride_16_cls",
        "stride_32_bbox", "stride_32_cls",
    ]


def test_yolo26_merge_stride_concats_bbox_then_cls():
    model = make_yolo26()
    plan = _detect_plan(model, CutConfig(model_type="auto", merge_stride=True))
    assert len(plan.outputs) == 3
    for out in plan.outputs:
        assert len(out.sources) == 2
        assert out.sources[0].shape[1] == 4   # ltrb
        assert out.sources[1].shape[1] == 80
        assert out.shape[-1] == 4 + 80


def test_yolov5_merge_stride_is_a_noop():
    """yolov5 already emits a single tensor per stride; merge_stride must not change shape."""
    model = make_yolov5()
    plan_split = _detect_plan(model, CutConfig(model_type="auto", merge_stride=False))
    plan_merge = _detect_plan(model, CutConfig(model_type="auto", merge_stride=True))
    assert [o.name for o in plan_split.outputs] == [o.name for o in plan_merge.outputs]
    assert [o.shape for o in plan_split.outputs] == [o.shape for o in plan_merge.outputs]


def test_auto_detect_failure_when_strides_dont_fit():
    # imgsz=32 + strides 3,5 -> hws {3:(10,10), 5:(6,6)}; no such tensors in the model.
    model = make_yolov8()
    cfg = CutConfig(model_type="auto", strides=(3, 5))
    with pytest.raises(AutoDetectError):
        _detect_plan(model, cfg)


def test_auto_detect_failure_when_classes_mismatch():
    """Coarse pick may match a type but per-stride lookup fails - still raises AutoDetectError."""
    model = make_yolov8()  # has 80 classes
    cfg = CutConfig(model_type="auto", num_classes=20)
    with pytest.raises(AutoDetectError):
        _detect_plan(model, cfg)


def test_explicit_model_type_does_not_raise_auto_detect_error():
    """Forcing model_type bypasses auto-detect; mismatch surfaces as plain RuntimeError."""
    model = make_yolov8()
    cfg = CutConfig(model_type="yolov8", num_classes=20)
    with pytest.raises(RuntimeError) as ei:
        _detect_plan(model, cfg)
    assert not isinstance(ei.value, AutoDetectError)


# ---------------------------------------------------------------------------
# instance segmentation (yolov8/v11-seg): mask-coef branch + proto
# ---------------------------------------------------------------------------


def test_yolov8_seg_auto_detect_and_ten_outputs():
    model = make_yolov8_seg()
    plan = _detect_plan(model, CutConfig(model_type="auto"))
    assert plan.model_type == "yolov8-seg"
    names = [o.name for o in plan.outputs]
    assert names == [
        "stride_8_cls", "stride_8_bbox",
        "stride_16_cls", "stride_16_bbox",
        "stride_32_cls", "stride_32_bbox",
        "stride_8_mask", "stride_16_mask", "stride_32_mask",
        "proto",
    ]
    for out in plan.outputs:
        assert len(out.sources) == 1
    masks = [o for o in plan.outputs if o.name.endswith("_mask")]
    assert all(o.shape[-1] == 32 for o in masks)  # NHWC mask channels
    proto = next(o for o in plan.outputs if o.name == "proto")
    assert proto.shape == (1, 8, 8, 32)  # imgsz 32 / 4 = 8, NHWC


def test_seg_proto_prefers_post_activation_graph_output():
    """proto must come from the post-act graph output (proto_out), never the raw Conv."""
    model = make_yolov8_seg()
    plan = _detect_plan(model, CutConfig(model_type="auto"))
    proto = next(o for o in plan.outputs if o.name == "proto")
    assert proto.sources[0].name == "proto_out"


def test_seg_off_keeps_detect_only():
    model = make_yolov8_seg()
    plan = _detect_plan(model, CutConfig(model_type="auto", seg="off"))
    assert plan.model_type == "yolov8"
    assert len(plan.outputs) == 6
    assert all("mask" not in o.name and o.name != "proto" for o in plan.outputs)


def test_plain_yolov8_under_auto_is_not_seg():
    """A detect-only model under seg=auto must not falsely trigger seg heads."""
    model = make_yolov8()
    plan = _detect_plan(model, CutConfig(model_type="auto"))
    assert plan.model_type == "yolov8"
    assert len(plan.outputs) == 6


def test_seg_on_without_branch_raises():
    """--seg on but a plain detect model has no mask/proto branch -> RuntimeError."""
    model = make_yolov8()
    cfg = CutConfig(model_type="yolov8", seg="on")
    with pytest.raises(RuntimeError):
        _detect_plan(model, cfg)


# ---------------------------------------------------------------------------
# cut_ultralytics_onnx end-to-end (writes file, reload, inspect outputs)
# ---------------------------------------------------------------------------


def _saved_outputs(tmp_path: Path, model: onnx.ModelProto, cfg: CutConfig) -> tuple[onnx.ModelProto, list[tuple[str, list[int]]]]:
    src = tmp_path / "in.onnx"
    dst = tmp_path / "out.onnx"
    onnx.save(model, str(src))
    cut_ultralytics_onnx(src, dst, cfg)
    cut = onnx.load(str(dst))
    out_shapes = [
        (o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
        for o in cut.graph.output
    ]
    return cut, out_shapes


def test_cut_yolov8_split_produces_six_nhwc_outputs(tmp_path):
    cut, outs = _saved_outputs(tmp_path, make_yolov8(), CutConfig(model_type="yolov8"))
    assert len(outs) == 6
    op_kinds = {n.op_type for n in cut.graph.node}
    assert "Transpose" in op_kinds
    assert "Concat" not in op_kinds  # split mode never adds Concat
    # All outputs are NHWC: last dim is channel count
    for name, shape in outs:
        assert shape[0] == 1
        if name.endswith("_cls"):
            assert shape[-1] == 80
        elif name.endswith("_bbox"):
            assert shape[-1] == 64


def test_cut_yolov8_merge_produces_three_nhwc_outputs_with_concat(tmp_path):
    cut, outs = _saved_outputs(tmp_path, make_yolov8(), CutConfig(model_type="yolov8", merge_stride=True))
    assert len(outs) == 3
    op_kinds = [n.op_type for n in cut.graph.node]
    assert op_kinds.count("Concat") == 3
    assert op_kinds.count("Transpose") == 3
    expected_channel = 64 + 80
    for name, shape in outs:
        assert name in ("stride_8", "stride_16", "stride_32")
        assert shape[0] == 1
        assert shape[-1] == expected_channel


def test_cut_yolo26_merge_concats_bbox_then_cls(tmp_path):
    cut, outs = _saved_outputs(tmp_path, make_yolo26(), CutConfig(model_type="yolo26", merge_stride=True))
    assert len(outs) == 3
    # The Concat node consumes (bbox, cls) in that order — verify input order.
    concats = [n for n in cut.graph.node if n.op_type == "Concat"]
    assert len(concats) == 3
    for n in concats:
        bbox_in, cls_in = n.input  # exact order
        assert "bbox" in bbox_in
        assert "cls" in cls_in
    for _, shape in outs:
        assert shape[-1] == 4 + 80


def test_cut_dry_run_does_not_write_file(tmp_path, capsys):
    src = tmp_path / "in.onnx"
    dst = tmp_path / "out.onnx"
    onnx.save(make_yolov8(), str(src))
    cut_ultralytics_onnx(src, dst, CutConfig(model_type="yolov8", dry_run=True))
    assert not dst.exists()
    captured = capsys.readouterr()
    assert "model_type=yolov8" in captured.out
    assert "stride_8_cls" in captured.out


def test_cut_pruned_graph_drops_unused_outputs(tmp_path):
    """The pruner should keep only nodes feeding the cut tensors and the new Transpose nodes."""
    cut, _ = _saved_outputs(tmp_path, make_yolov8(), CutConfig(model_type="yolov8"))
    # Original toy graph has 6 Convs; pruned should keep all 6 because they all feed outputs.
    convs = [n for n in cut.graph.node if n.op_type == "Conv"]
    assert len(convs) == 6
    # Nothing else lying around.
    extra_ops = [n.op_type for n in cut.graph.node if n.op_type not in {"Conv", "Transpose"}]
    assert extra_ops == []


def test_cut_yolov8_seg_produces_ten_nhwc_outputs(tmp_path):
    cut, outs = _saved_outputs(tmp_path, make_yolov8_seg(), CutConfig(model_type="auto"))
    assert len(outs) == 10
    d = dict(outs)
    assert d["proto"] == [1, 8, 8, 32]  # NHWC, imgsz 32 / 4
    for name, shape in outs:
        assert shape[0] == 1
        if name.endswith("_mask"):
            assert shape[-1] == 32
        elif name.endswith("_cls"):
            assert shape[-1] == 80
        elif name.endswith("_bbox"):
            assert shape[-1] == 64
    # proto's Transpose feeds from the Relu (post-act), so the Relu survives pruning.
    assert "Relu" in {n.op_type for n in cut.graph.node}
