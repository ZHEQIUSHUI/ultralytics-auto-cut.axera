from __future__ import annotations

import argparse
from pathlib import Path

from export_ultralytics import export_pt_to_onnx
from onnx_cut import CutConfig, cut_ultralytics_onnx


def _parse_imgsz(value: str) -> tuple[int, int]:
    if "," in value:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("imgsz must be like '640' or '640,640'")
        h, w = (int(parts[0]), int(parts[1]))
        return (h, w)
    size = int(value)
    return (size, size)


def _parse_strides(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("strides must be like '8,16,32' or '8,16,32,64'")
    return tuple(int(p) for p in parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ultralytics-auto-cut",
        description="Export/cut Ultralytics-style YOLO ONNX models into stride heads + add NHWC Transpose outputs.",
    )
    p.add_argument("model", type=Path, help="Input model (.onnx or .pt)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path (default: <input_stem>.cut.onnx in CWD)",
    )
    p.add_argument(
        "--imgsz",
        type=_parse_imgsz,
        default=(640, 640),
        help="Export/infer input size. '640' or '640,640' (default: 640)",
    )
    p.add_argument(
        "--model-type",
        choices=["auto", "yolov5", "yolov8", "yolov11", "yolo26"],
        default="auto",
        help="Override model type detection (default: auto)",
    )
    p.add_argument("--classes", type=int, default=80, help="Number of classes (default: 80)")
    p.add_argument(
        "--v8-bbox-ch",
        type=int,
        default=64,
        help="YOLOv8/YOLOv11 bbox channels (default: 64 = 4*reg_max)",
    )
    p.add_argument(
        "--yolo26-bbox-ch",
        type=int,
        default=4,
        help="YOLO26 bbox channels (default: 4 = ltrb)",
    )
    p.add_argument(
        "--decoupled-order",
        choices=["cls-bbox", "bbox-cls"],
        default=None,
        help="For decoupled heads, output order per stride. Default: yolov8/yolov11=cls-bbox, yolo26=bbox-cls",
    )
    p.add_argument(
        "--strides",
        type=_parse_strides,
        default=(8, 16, 32),
        help="Detection head strides, comma-separated (default: 8,16,32; P6 models: 8,16,32,64)",
    )
    p.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnxsim simplification after pruning (requires: pip install onnxsim)",
    )
    p.add_argument(
        "--merge-stride",
        action="store_true",
        help="For decoupled heads (yolov8/v11/yolo26), Concat(bbox, cls) along channel "
             "per stride and emit one merged output. Ignored for yolov5.",
    )
    p.add_argument(
        "--export-opset",
        type=int,
        default=19,
        help="When exporting .pt to ONNX, use this opset (default: 19)",
    )
    p.add_argument(
        "--exported-onnx",
        type=Path,
        default=None,
        help="When input is .pt, write exported ONNX here (default: <input_stem>.exported.onnx in CWD)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only detect and print cut tensors; do not write output model",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    output_path = args.output
    if output_path is None:
        output_path = Path.cwd() / f"{args.model.stem}.cut.onnx"

    in_path = args.model
    if in_path.suffix.lower() == ".pt":
        exported_path = args.exported_onnx
        if exported_path is None:
            exported_path = Path.cwd() / f"{in_path.stem}.exported.onnx"
        export_pt_to_onnx(
            pt_path=in_path,
            onnx_path=exported_path,
            imgsz=args.imgsz,
            opset=args.export_opset,
        )
        in_path = exported_path
    elif in_path.suffix.lower() != ".onnx":
        raise SystemExit("Only .onnx and .pt are supported")

    cfg = CutConfig(
        model_type=args.model_type,
        imgsz=args.imgsz,
        num_classes=args.classes,
        v8_bbox_ch=args.v8_bbox_ch,
        yolo26_bbox_ch=args.yolo26_bbox_ch,
        decoupled_order=args.decoupled_order,
        strides=args.strides,
        simplify=args.simplify,
        merge_stride=args.merge_stride,
        dry_run=args.dry_run,
    )
    cut_ultralytics_onnx(in_path, output_path, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
