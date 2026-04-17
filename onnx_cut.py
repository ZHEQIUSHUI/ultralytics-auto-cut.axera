from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import onnx
from onnx import TensorProto, shape_inference


class TensorInfo(NamedTuple):
    name: str
    elem_type: int
    shape: tuple[int | None, ...]


@dataclass(frozen=True)
class CutConfig:
    model_type: str = "auto"  # auto|yolov5|yolov8|yolov11|yolo26
    imgsz: tuple[int, int] = (640, 640)
    num_classes: int = 80
    v8_bbox_ch: int = 64
    yolo26_bbox_ch: int = 4
    decoupled_order: str | None = None  # cls-bbox|bbox-cls
    strides: tuple[int, ...] = (8, 16, 32)
    simplify: bool = False
    dry_run: bool = False


class CutPlan(NamedTuple):
    model_type: str
    # list of (source_tensor, output_tensor_name, output_shape)
    outputs: tuple[tuple[TensorInfo, str, tuple[int, int, int, int]], ...]


def _get_value_info(model: onnx.ModelProto) -> dict[str, TensorInfo]:
    out: dict[str, TensorInfo] = {}

    def add(vi: onnx.ValueInfoProto) -> None:
        if not vi.type.HasField("tensor_type"):
            return
        tt = vi.type.tensor_type
        elem_type = int(tt.elem_type)
        if not tt.HasField("shape"):
            return
        dims: list[int | None] = []
        for d in tt.shape.dim:
            if d.dim_value > 0:
                dims.append(int(d.dim_value))
            else:
                dims.append(None)
        out[vi.name] = TensorInfo(vi.name, elem_type, tuple(dims))

    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        add(vi)
    return out


def _infer_img_hw(model: onnx.ModelProto, fallback: tuple[int, int]) -> tuple[int, int]:
    initializer_names = {i.name for i in model.graph.initializer}
    image_input = None
    for i in model.graph.input:
        if i.name not in initializer_names:
            image_input = i
            break
    if image_input is None:
        return fallback

    tt = image_input.type.tensor_type
    if not tt.HasField("shape"):
        return fallback
    dims = [d.dim_value for d in tt.shape.dim]
    if len(dims) != 4 or any(v <= 0 for v in dims):
        return fallback

    # Heuristic for NCHW vs NHWC.
    if dims[1] == 3 and dims[2] > 3 and dims[3] > 3:
        return (int(dims[2]), int(dims[3]))
    if dims[3] == 3 and dims[1] > 3 and dims[2] > 3:
        return (int(dims[1]), int(dims[2]))
    return (int(dims[-2]), int(dims[-1]))


def _build_producer_map(model: onnx.ModelProto) -> dict[str, tuple[int, onnx.NodeProto]]:
    producer: dict[str, tuple[int, onnx.NodeProto]] = {}
    for idx, node in enumerate(model.graph.node):
        for o in node.output:
            producer[o] = (idx, node)
    return producer


def _build_consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumer: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for i in node.input:
            consumer.setdefault(i, []).append(node)
    return consumer


def _pick_best_tensor(
    candidates: Iterable[TensorInfo],
    producer: dict[str, tuple[int, onnx.NodeProto]],
    consumer: dict[str, list[onnx.NodeProto]],
) -> TensorInfo:
    cands = list(candidates)
    if not cands:
        raise RuntimeError("No candidates to pick from")

    conv_cands: list[tuple[int, TensorInfo]] = []
    for t in cands:
        p = producer.get(t.name)
        if not p:
            continue
        idx, node = p
        if node.op_type == "Conv":
            conv_cands.append((idx, t))

    def consumer_types(t: TensorInfo) -> set[str]:
        return {n.op_type for n in consumer.get(t.name, [])}

    def has_headish_consumer(t: TensorInfo) -> bool:
        # Concat is common when assembling multi-branch heads; Reshape is also used, but can appear elsewhere.
        cts = consumer_types(t)
        return ("Concat" in cts) or ("Reshape" in cts)

    if conv_cands:
        headish_conv = [(idx, t) for idx, t in conv_cands if has_headish_consumer(t)]
        if headish_conv:
            # Prefer Concat-consuming tensors, then latest.
            def key(it: tuple[int, TensorInfo]) -> tuple[int, int]:
                idx, t = it
                return (1 if "Concat" in consumer_types(t) else 0, idx)

            headish_conv.sort(key=key, reverse=True)
            return headish_conv[0][1]
        conv_cands.sort(key=lambda it: it[0], reverse=True)
        return conv_cands[0][1]

    # Fallback: prefer tensors consumed by Reshape, else latest producer.
    headish_any = [t for t in cands if has_headish_consumer(t)]
    if headish_any:
        headish_any.sort(
            key=lambda t: (
                1 if "Concat" in consumer_types(t) else 0,
                producer.get(t.name, (-1, None))[0],
            ),
            reverse=True,
        )
        return headish_any[0]

    cands.sort(key=lambda t: producer.get(t.name, (-1, None))[0], reverse=True)  # type: ignore[index]
    return cands[0]


def _detect_plan(model: onnx.ModelProto, cfg: CutConfig) -> CutPlan:
    inferred = shape_inference.infer_shapes(model)
    vi = _get_value_info(inferred)
    producer = _build_producer_map(model)
    consumer = _build_consumer_map(model)

    input_h, input_w = _infer_img_hw(inferred, cfg.imgsz)
    hws = {s: (input_h // s, input_w // s) for s in cfg.strides}

    def tensors_with_shape(shape: tuple[int, int, int, int]) -> list[TensorInfo]:
        out: list[TensorInfo] = []
        for t in vi.values():
            if t.elem_type == 0:
                continue
            if len(t.shape) != 4:
                continue
            if tuple(t.shape) == tuple(shape):
                out.append(t)
        return out

    def pick(shape: tuple[int, int, int, int]) -> TensorInfo:
        return _pick_best_tensor(tensors_with_shape(shape), producer, consumer)

    def pick_pair(cls_shape: tuple[int, int, int, int], bbox_shape: tuple[int, int, int, int]) -> tuple[TensorInfo, TensorInfo]:
        cls = pick(cls_shape)
        cls_prod = producer.get(cls.name)
        cls_idx = cls_prod[0] if cls_prod else -1
        cls_node_name = cls_prod[1].name if cls_prod else ""

        bbox_candidates = tensors_with_shape(bbox_shape)
        if not bbox_candidates:
            raise RuntimeError(f"Missing bbox tensor with shape {bbox_shape}")

        def prefix_tokens(a: str, b: str) -> int:
            if not a or not b:
                return 0
            ap = a.split("/")
            bp = b.split("/")
            n = 0
            for x, y in zip(ap, bp):
                if x == y:
                    n += 1
                else:
                    break
            return n

        best: TensorInfo | None = None
        best_score: float = float("-inf")
        for t in bbox_candidates:
            prod = producer.get(t.name)
            if not prod:
                continue
            idx, node = prod
            if node.op_type != "Conv":
                continue
            ctypes = {n.op_type for n in consumer.get(t.name, [])}

            # Prefer tensors that look like head branches and are close to the cls head node.
            score = 0.0
            score += (1000.0 if "Concat" in ctypes else 0.0)
            score += (100.0 if "Reshape" in ctypes else 0.0)
            score += prefix_tokens(cls_node_name, node.name) * 10.0
            if cls_idx >= 0:
                score -= abs(idx - cls_idx) * 1.0
            score += idx * 1e-3  # slight preference for later nodes

            if score > best_score:
                best_score = score
                best = t

        if best is not None:
            return (cls, best)

        # Fallback to global heuristic (may be less robust).
        bbox = _pick_best_tensor(bbox_candidates, producer, consumer)
        return (cls, bbox)

    forced = cfg.model_type
    if forced == "auto":
        yolov5_ch = (cfg.num_classes + 5) * 3
        yolov5_shapes = [(1, yolov5_ch, h, w) for h, w in hws.values()]
        y26_shapes = [(1, cfg.yolo26_bbox_ch, h, w) for h, w in hws.values()]
        v8_shapes = [(1, cfg.v8_bbox_ch, h, w) for h, w in hws.values()]

        def any_exists(shapes: list[tuple[int, int, int, int]]) -> bool:
            return any(tensors_with_shape(s) for s in shapes)

        if any_exists(yolov5_shapes):
            forced = "yolov5"
        elif any_exists(y26_shapes):
            forced = "yolo26"
        elif any_exists(v8_shapes):
            forced = "yolov8"
        else:
            all_4d = sorted(
                {str(t.shape) for t in vi.values() if len(t.shape) == 4},
                key=lambda s: s,
            )
            hint = "\n  ".join(all_4d) if all_4d else "(none found)"
            raise RuntimeError(
                f"Failed to auto-detect model type.\n"
                f"Expected yolov5 ch={yolov5_ch}, yolo26 bbox_ch={cfg.yolo26_bbox_ch}, "
                f"yolov8 bbox_ch={cfg.v8_bbox_ch} at strides={list(cfg.strides)} imgsz={cfg.imgsz}.\n"
                f"All 4-D tensor shapes found:\n  {hint}\n"
                f"Try --model-type / --strides / --imgsz / --classes to override."
            )

    if forced == "yolov5":
        yolov5_ch = (cfg.num_classes + 5) * 3
        outs: list[tuple[TensorInfo, str, tuple[int, int, int, int]]] = []
        for stride in cfg.strides:
            h, w = hws[stride]
            src = pick((1, yolov5_ch, h, w))
            outs.append((src, f"stride_{stride}", (1, h, w, yolov5_ch)))
        return CutPlan("yolov5", tuple(outs))

    if forced in ("yolov8", "yolov11"):
        order = cfg.decoupled_order or "cls-bbox"
        outs = []
        for stride in cfg.strides:
            h, w = hws[stride]
            cls, bbox = pick_pair((1, cfg.num_classes, h, w), (1, cfg.v8_bbox_ch, h, w))
            if order == "cls-bbox":
                outs.extend(
                    [
                        (cls, f"stride_{stride}_cls", (1, h, w, cfg.num_classes)),
                        (bbox, f"stride_{stride}_bbox", (1, h, w, cfg.v8_bbox_ch)),
                    ]
                )
            else:
                outs.extend(
                    [
                        (bbox, f"stride_{stride}_bbox", (1, h, w, cfg.v8_bbox_ch)),
                        (cls, f"stride_{stride}_cls", (1, h, w, cfg.num_classes)),
                    ]
                )
        return CutPlan(forced, tuple(outs))

    if forced == "yolo26":
        order = cfg.decoupled_order or "bbox-cls"
        outs = []
        for stride in cfg.strides:
            h, w = hws[stride]
            cls, bbox = pick_pair((1, cfg.num_classes, h, w), (1, cfg.yolo26_bbox_ch, h, w))
            if order == "cls-bbox":
                outs.extend(
                    [
                        (cls, f"stride_{stride}_cls", (1, h, w, cfg.num_classes)),
                        (bbox, f"stride_{stride}_bbox", (1, h, w, cfg.yolo26_bbox_ch)),
                    ]
                )
            else:
                outs.extend(
                    [
                        (bbox, f"stride_{stride}_bbox", (1, h, w, cfg.yolo26_bbox_ch)),
                        (cls, f"stride_{stride}_cls", (1, h, w, cfg.num_classes)),
                    ]
                )
        return CutPlan("yolo26", tuple(outs))

    raise RuntimeError(f"Unsupported model type: {forced}")


def _prune_to_outputs(model: onnx.ModelProto, output_tensors: list[TensorInfo]) -> onnx.ModelProto:
    producer = _build_producer_map(model)
    initializer_names = {i.name for i in model.graph.initializer}
    input_names = {i.name for i in model.graph.input if i.name not in initializer_names}

    required_nodes: set[int] = set()
    required_tensors: set[str] = {t.name for t in output_tensors}
    stack = list(required_tensors)

    while stack:
        tname = stack.pop()
        if tname in input_names or tname in initializer_names:
            continue
        prod = producer.get(tname)
        if prod is None:
            continue
        idx, node = prod
        if idx in required_nodes:
            continue
        required_nodes.add(idx)
        for inp in node.input:
            if inp and inp not in required_tensors:
                required_tensors.add(inp)
                stack.append(inp)

    kept_nodes = [n for i, n in enumerate(model.graph.node) if i in required_nodes]

    # Keep initializers referenced by kept nodes.
    kept_init_names: set[str] = set()
    for node in kept_nodes:
        for inp in node.input:
            if inp in initializer_names:
                kept_init_names.add(inp)

    kept_initializers = [i for i in model.graph.initializer if i.name in kept_init_names]
    kept_inputs = [i for i in model.graph.input if (i.name in required_tensors and i.name in input_names)]

    outputs_vi = [
        onnx.helper.make_tensor_value_info(
            t.name,
            t.elem_type if t.elem_type else TensorProto.FLOAT,
            list(t.shape),
        )
        for t in output_tensors
    ]

    graph = onnx.helper.make_graph(
        nodes=kept_nodes,
        name=model.graph.name or "cut_graph",
        inputs=kept_inputs,
        outputs=outputs_vi,
        initializer=kept_initializers,
    )
    pruned = onnx.helper.make_model(graph, opset_imports=list(model.opset_import))
    pruned.ir_version = model.ir_version
    pruned.producer_name = model.producer_name
    pruned.producer_version = model.producer_version
    pruned.domain = model.domain
    pruned.model_version = model.model_version
    pruned.doc_string = model.doc_string
    if model.metadata_props:
        pruned.metadata_props.extend(model.metadata_props)
    return pruned


def _try_simplify(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        import onnxsim  # type: ignore
        simplified, ok = onnxsim.simplify(model)
        if ok:
            return simplified
        print("[simplify] onnxsim returned ok=False, keeping original")
    except ImportError:
        print("[simplify] onnxsim not installed (pip install onnxsim), skipping")
    except Exception as e:
        print(f"[simplify] failed: {e}, keeping original")
    return model


def cut_ultralytics_onnx(in_path: Path, out_path: Path, cfg: CutConfig) -> None:
    model = onnx.load(str(in_path))
    plan = _detect_plan(model, cfg)

    if cfg.dry_run:
        print(f"[dry-run] model_type={plan.model_type}")
        for src, out_name, out_shape in plan.outputs:
            print(f"[dry-run] {out_name}: from {src.name} {src.shape} -> {out_shape}")
        return

    # 1) prune to raw head tensors (pre-transpose)
    raw_outputs = [src for (src, _, _) in plan.outputs]
    pruned = _prune_to_outputs(model, raw_outputs)

    # 2) optional simplify before adding transpose
    if cfg.simplify:
        pruned = _try_simplify(pruned)

    # 3) add transpose nodes, set new outputs
    inferred = shape_inference.infer_shapes(pruned)
    vi = _get_value_info(inferred)

    transpose_nodes: list[onnx.NodeProto] = []
    final_outputs_vi: list[onnx.ValueInfoProto] = []
    for src, out_name, out_shape in plan.outputs:
        src_info = vi.get(src.name, src)
        node = onnx.helper.make_node(
            "Transpose",
            inputs=[src.name],
            outputs=[out_name],
            perm=[0, 2, 3, 1],
            name=f"{out_name}_transpose",
        )
        transpose_nodes.append(node)
        final_outputs_vi.append(
            onnx.helper.make_tensor_value_info(
                out_name,
                src_info.elem_type if src_info.elem_type else TensorProto.FLOAT,
                list(out_shape),
            )
        )

    pruned.graph.node.extend(transpose_nodes)
    del pruned.graph.output[:]
    pruned.graph.output.extend(final_outputs_vi)

    onnx.checker.check_model(pruned)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(pruned, str(out_path))
