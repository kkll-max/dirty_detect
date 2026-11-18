"""在训练好的权重上跑 `model.val()`，打印逐类 IoU。
    python tools/val.py --config configs/train.yaml                 # 主 val
    python tools/val.py --config configs/train.yaml --split test    # AI 光斑专项集
    python tools/val.py --config configs/train.yaml --weights runs/dataset_6cls/weights/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="六类分割逐类 IoU 验证")
    parser.add_argument("--config", type=Path,
                        default=PKG_ROOT / "configs/train.yaml")
    parser.add_argument("--weights", type=Path, default=None,
                        help="默认取 <project>/<name>/weights/best.pt")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--split", default="val", choices=("val", "test", "train"))
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--name", type=str, default=None, help="输出目录名")
    parser.add_argument("--plots", action="store_true", help="额外存混淆矩阵等图")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PKG_ROOT / path)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} 里不是映射结构")
    return data


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(PKG_ROOT))
    from tools.crop_dataset import install
    from tools.train import check_crop_imgsz, validate_data

    config_path = resolve(args.config)
    cfg = load_yaml(config_path)
    data_path = resolve(args.data or cfg.get("data", "configs/data_dataset1.yaml"))

    data, root = validate_data(data_path)

    imgsz = int(args.imgsz or cfg.get("imgsz", 640))
    check_crop_imgsz(data, {"imgsz": imgsz})

    project = resolve(cfg.get("project", "runs"))
    run_name = cfg.get("name", "glare_3cls")
    weights = resolve(args.weights) if args.weights else project / run_name / "weights/best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在: {weights}（训练还没跑完？）")

    # 数据路径换成绝对路径：Ultralytics 对相对路径会去 DATASETS_DIR 下找
    resolved = dict(data)
    resolved["path"] = str(root)
    resolved_path = project / "_data_val.yaml"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, allow_unicode=True, sort_keys=False)

    install()

    from ultralytics import YOLO

    out_name = args.name or f"{run_name}_val_{args.split}"
    print(f"weights: {weights}")
    print(f"split  : {args.split}  imgsz={imgsz}  batch={args.batch}")
    print(f"output : {project / out_name}\n")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(resolved_path),
        split=args.split,
        imgsz=imgsz,
        rect=True,
        batch=args.batch,
        device=args.device or cfg.get("device", 0),
        project=str(project),
        name=out_name,
        exist_ok=True,
        plots=args.plots,
    )

    print_per_class(metrics)


def print_per_class(metrics) -> None:
    """逐类 IoU / 像素准确率 / 像素占比。"""
    names = getattr(metrics, "names", None) or {}
    iou = getattr(metrics, "per_class_iou", None)
    if iou is None:
        print("这个任务没有 per_class_iou，跳过逐类表")
        return
    acc = getattr(metrics, "per_class_pixel_accuracy", None)
    pixels = getattr(metrics, "nt_per_class", None)
    total = float(pixels.sum()) if pixels is not None and pixels.sum() else 0.0

    print(f"\n{'class':<10} {'IoU':>8} {'pix_acc':>9} {'像素占比':>9}")
    for c in range(len(iou)):
        label = names.get(c, str(c)) if isinstance(names, dict) else names[c]
        share = f"{pixels[c] / total:.2%}" if total else "-"
        pa = f"{acc[c]:.4f}" if acc is not None else "-"
        print(f"{label:<10} {iou[c]:8.4f} {pa:>9} {share:>9}")

    print(f"\nmIoU       : {metrics.miou:.4f}")
    print(f"pixel_acc  : {metrics.pixel_accuracy:.4f}")
    print("注意：val/test 的类别先验不同时，mIoU 不能直接横向比")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}")
        raise SystemExit(1)
