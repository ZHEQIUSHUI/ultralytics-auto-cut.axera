from __future__ import annotations

from pathlib import Path
import shutil


def export_pt_to_onnx(pt_path: Path, onnx_path: Path, imgsz: tuple[int, int], opset: int = 19) -> None:
    pt_path = Path(pt_path)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ultralytics python package is required to export .pt -> .onnx. "
            "Install with: pip install ultralytics"
        ) from e

    model = YOLO(str(pt_path))
    # NOTE: We prefer 'nms=False' since we will cut to raw heads anyway.
    model.export(
        format="onnx",
        imgsz=list(imgsz),
        opset=opset,
        dynamic=False,
        simplify=False,
        nms=False,
        half=False,
        int8=False,
        optimize=False,
    )

    # Ultralytics writes next to the .pt by default; locate the newest onnx.
    # Common pattern: <stem>.onnx
    candidate = pt_path.with_suffix(".onnx")
    if candidate.exists():
        if candidate.resolve() != onnx_path.resolve():
            shutil.copy2(candidate, onnx_path)
        return

    # Fallback: search for any onnx in pt dir with same stem.
    matches = sorted(pt_path.parent.glob(f"{pt_path.stem}*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise RuntimeError("Ultralytics export did not produce an .onnx file")
    if matches[0].resolve() != onnx_path.resolve():
        shutil.copy2(matches[0], onnx_path)
