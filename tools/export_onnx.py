"""Export a trained YOLO26-sem checkpoint to ONNX (opset 11 by default).

Example (from model_get/yolo26-sem/):

    python tools/export_onnx.py \
      --weights runs/correct_files_yolo26n_sem/weights/best.pt \
      --output runs/correct_files_yolo26n_sem/weights/best.onnx

The semantic head switches to export mode, so the graph takes a float32 NCHW
image tensor and returns a uint8 class map with shape (N, H, W) -- not logits.
The class map is already argmax-reduced and upsampled to the input resolution.

Requires:
    pip install onnx

Optional:
    pip install onnxruntime   # enables numerical verification
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output .onnx path; defaults to the checkpoint path with .onnx",
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument(
        "--device",
        default="cpu",
        help="export device, e.g. cpu or 0 (CPU is the most portable)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="export dynamic batch and spatial axes",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the Ultralytics graph simplifier after export",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare ONNX Runtime output with PyTorch (if installed)",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def verify_onnx(
    output: Path,
    weights: Path,
    height: int,
    width: int,
    batch: int,
) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "ONNX verification requires the 'onnx' package. "
            "Install it with: pip install onnx"
        ) from exc

    onnx.checker.check_model(str(output))
    print("[verify] onnx.checker: OK")

    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed; skipped numerical check")
        return

    session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    print(
        f"[verify] input  {input_meta.name}: "
        f"{input_meta.shape} {input_meta.type}"
    )
    print(
        f"[verify] output {output_meta.name}: "
        f"{output_meta.shape} {output_meta.type}"
    )

    dummy = torch.randn(batch, 3, height, width, dtype=torch.float32)
    onnx_result = session.run(None, {input_meta.name: dummy.numpy()})[0]

    # Reproduce the exporter's inference mode on the PyTorch side.
    pt = YOLO(str(weights))
    torch_model = pt.model.float().eval()
    head = torch_model.model[-1]
    head.export = True
    head.format = "onnx"
    with torch.no_grad():
        expected = torch_model(dummy)
    if isinstance(expected, (list, tuple)):
        expected = expected[0]
    expected = expected.cpu().numpy()

    if tuple(onnx_result.shape) != tuple(expected.shape):
        raise SystemExit(
            f"[verify] shape mismatch: onnx={onnx_result.shape}, "
            f"torch={expected.shape}"
        )

    agreement = float((onnx_result == expected).mean())
    print(f"[verify] class-map agreement: {agreement:.6f}")
    if agreement < 0.999:
        raise SystemExit("[verify] ONNX output differs from PyTorch")


def main() -> None:
    args = parse_args()
    weights = resolve(args.weights)
    output = resolve(args.output) if args.output else weights.with_suffix(".onnx")

    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint not found: {weights}")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")

    if importlib.util.find_spec("onnx") is None:
        raise SystemExit(
            "ONNX export requires the 'onnx' package. "
            "Install it with: pip install onnx"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    print(f"checkpoint : {weights}")
    print(f"output     : {output}")
    print(f"input      : {args.batch}x3x{args.height}x{args.width}")
    print(f"opset      : {args.opset}")
    print(f"dynamic    : {args.dynamic}")
    print(f"simplify   : {args.simplify}")

    exported = Path(
        model.export(
            format="onnx",
            imgsz=[args.height, args.width],
            batch=args.batch,
            opset=args.opset,
            device=args.device,
            dynamic=args.dynamic,
            simplify=args.simplify,
        )
    )
    if exported.resolve() != output.resolve():
        shutil.move(str(exported), str(output))

    print(f"file size  : {output.stat().st_size / 1024**2:.2f} MiB")

    if args.verify:
        verify_onnx(output, weights, args.height, args.width, args.batch)

    print(f"exported -> {output}")


if __name__ == "__main__":
    main()
