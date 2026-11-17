"""把多类别 mask 转成 labelme 的 JSON，用于在预标注基础上人工修改。

流程：

    masks/*.png  --[本脚本]-->  annotations/*.json  --[labelme 修改]-->
    annotations/*.json  --[labelme2mask.py]-->  masks/*.png

每个类别值（0 和 255 除外）分别取外轮廓，按 labels.txt 的行号还原成对应标签名：
`0=clean` 是背景不画，`1=dirty`、`2=glare`、`3=water1`……都会写成 polygon。
轮廓点会用 approxPolyDP 抽稀，否则一张 640x384 的掩膜边界动辄几百个点，
labelme 里拖不动。

labels.txt 的查找顺序：`--labels` > `<data-dir>/labels.txt` >
`<masks 的上一级>/labels.txt` > `<masks 的上两级>/labels.txt`。
都不存在时退回旧行为：只把值 1 当成 `--label`（默认 dirty）。

全 clean 的 mask 也会写一个空 shapes 的 JSON，这样 labelme 里每张图都有入口，
不会出现"这张没有 JSON 所以跳过"的情况（不想写就加 --skip-empty）。

举例：

    # 按 split 转换：labels.txt 会自动从 masks 的上两级找到
    python tools/mask2labelme.py \
        --masks data/dataset2/masks/test --images data/dataset2/images/test \
        -o data/dataset2/annotations/test

    # 指定标签表和抽稀强度（epsilon 越大点越少，越小越贴合）
    python tools/mask2labelme.py \
        --masks data/dataset2/masks/test --images data/dataset2/images/test \
        -o data/dataset2/annotations/test \
        --labels data/dataset2/labels.txt --epsilon 0.001 --min-area 30

    # 只处理单一类别（旧用法，值 1 全部写成 dirty）
    python tools/mask2labelme.py --data-dir data/matrix --label dirty

    # predict.py --save-mask 产出的 mask 带 _pred 后缀（100_rgb_pred.png），
    # 原图是 100_rgb.png。默认会先找同名，再尝试去掉 _pred；JSON 也按
    # 100_rgb.json 命名，方便 labelme2mask.py 回写成训练用的同名 mask。

然后：

    labelme --labels data/dataset2/labels.txt --output data/dataset2/annotations \
            data/dataset2/images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PKG_ROOT = Path(__file__).resolve().parents[1]
LABELME_VERSION = "5.0.1"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
IGNORE_VALUE = 255
IGNORE_NAMES = {"ignore", "ignored", "void"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多类别 mask -> labelme JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="数据集根目录，含 images/ 和 masks/；给了就不用下面两个")
    parser.add_argument("--masks", type=Path, default=None, help="mask 目录")
    parser.add_argument("--images", type=Path, default=None,
                        help="原图目录，用来取尺寸和写 imagePath")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="标注输出目录，默认 <data-dir>/annotations")
    parser.add_argument("--labels", type=Path, default=None,
                        help="类别表（行号=类别 ID）；不给就自动找 <data-dir>/labels.txt "
                             "或 masks 上层的 labels.txt")
    parser.add_argument("--label", default=None,
                        help="强制单类别模式：只把值 1 写成该标签（旧用法，例如 dirty）。"
                             "不传时按 labels.txt 逐类转换")
    parser.add_argument("--min-area", type=float, default=16.0,
                        help="小于该像素数的连通块丢弃，默认 16")
    parser.add_argument("--epsilon", type=float, default=0.002,
                        help="多边形抽稀系数（占周长的比例），默认 0.002；0 表示不抽稀")
    parser.add_argument("--strip-suffix", default="_pred",
                        help="原图找不到时，尝试去掉 mask 文件名的该后缀再找；"
                             "默认 _pred（predict.py --save-mask 的命名）。传空字符串关闭")
    parser.add_argument("--skip-empty", action="store_true",
                        help="全 clean 的 mask 不写 JSON")
    parser.add_argument("--overwrite", action="store_true", help="JSON 已存在时覆盖")
    return parser.parse_args()


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.data_dir is not None:
        base = args.data_dir
        masks = args.masks or (base / "masks")
        images = args.images or (base / "images")
        out = args.output or (base / "annotations")
    else:
        if args.masks is None or args.images is None:
            raise ValueError("要么给 --data-dir，要么同时给 --masks 和 --images")
        masks = args.masks
        images = args.images
        out = args.output or (masks.parent / "annotations")
    return masks.resolve(), images.resolve(), out.resolve()


def find_image(images: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def resolve_stem(stem: str, images: Path, strip_suffix: str) -> tuple[str, Path | None]:
    """先按原文件名找原图，找不到再尝试去掉 strip_suffix。

    predict.py --save-mask 写的是 `<stem>_pred.png`，原图是 `<stem>.png`。
    JSON 也用解析后的 stem 命名，否则 labelme2mask 回写出来的 mask 会对不上原图。
    同名原图存在时优先用它，避免误伤本身就叫 `*_pred` 的图。
    """
    candidates = [stem]
    if strip_suffix and stem.endswith(strip_suffix) and len(stem) > len(strip_suffix):
        candidates.append(stem[: -len(strip_suffix)])
    for candidate in candidates:
        image = find_image(images, candidate)
        if image is not None:
            return candidate, image
    return stem, None


def resolve_labels_file(args: argparse.Namespace, masks_dir: Path) -> Path | None:
    """找类别表：显式 --labels 优先，其次 data-dir / masks 上两层里的 labels.txt。"""
    if args.labels is not None:
        if not args.labels.is_file():
            raise FileNotFoundError(f"labels 文件不存在: {args.labels}")
        return args.labels
    candidates = []
    if args.data_dir is not None:
        candidates.append(args.data_dir / "labels.txt")
    candidates.append(masks_dir.parent / "labels.txt")
    candidates.append(masks_dir.parent.parent / "labels.txt")
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_id_to_name(path: Path) -> dict[int, str]:
    """读类别表，返回 {类别 ID: 标签名}。

    规则和 labelme2mask.py 一致：行号从 0 开始，`ignore/ignored/void` 固定 255、
    不占类别 ID，空行和 `#` 开头的注释行跳过。
    """
    mapping: dict[int, str] = {}
    next_id = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if name.lower() in IGNORE_NAMES:
            mapping[IGNORE_VALUE] = name
            continue
        mapping[next_id] = name
        next_id += 1
    if not mapping:
        raise ValueError(f"labels 文件是空的: {path}")
    return mapping


def mask_to_polygons(mask: np.ndarray, class_id: int, label: str,
                     min_area: float, epsilon: float):
    binary = (mask == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    dropped = 0
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            dropped += 1
            continue
        if epsilon > 0:
            contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            dropped += 1
            continue
        shapes.append(
            {
                "label": label,
                "points": [[float(x), float(y)] for x, y in points],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        )
    return shapes, dropped


def main() -> None:
    args = parse_args()
    masks_dir, images_dir, out_dir = resolve_dirs(args)

    if not masks_dir.is_dir():
        raise FileNotFoundError(f"mask 目录不存在: {masks_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    all_masks = sorted(
        p for p in masks_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

    def is_aux_prelabel(path: Path) -> bool:
        """同名正式 mask 存在时，`*_pred.png` 只是预标注，跳过。

        dataset1 的 masks/ 里 GT 和 `_pred` 混放；不跳的话 --overwrite 会把
        先前由 GT 生成的 JSON 覆盖成预标注结果。
        只有 `_pred` 一类文件时（如 dataset2 的预标注目录）照常处理。
        """
        suffix = args.strip_suffix
        if not suffix or not path.stem.endswith(suffix):
            return False
        base = path.stem[: -len(suffix)]
        return (masks_dir / f"{base}.png").is_file()

    mask_files = [p for p in all_masks if not is_aux_prelabel(p)]
    aux_skipped = len(all_masks) - len(mask_files)
    if not mask_files:
        raise FileNotFoundError(f"没有 mask: {masks_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    labels_file = resolve_labels_file(args, masks_dir)
    if args.label is not None:
        classes = [(1, args.label)]
        label_source = f"--label {args.label}（强制单类别，只转换值 1）"
    elif labels_file is not None:
        id_to_name = load_id_to_name(labels_file)
        classes = [(cid, name) for cid, name in sorted(id_to_name.items())
                   if cid not in (0, IGNORE_VALUE)]
        if not classes:
            raise ValueError(f"labels 里除 clean / ignore 外没有类别: {labels_file}")
        label_source = str(labels_file)
    else:
        classes = [(1, "dirty")]
        label_source = "未找到 labels.txt，退回旧行为：只把值 1 写成 dirty"

    print(f"masks   : {masks_dir}")
    print(f"images  : {images_dir}")
    print(f"output  : {out_dir}")
    print(f"labels  : {label_source}")
    print("classes : " + "  ".join(f"{cid}={name}" for cid, name in classes))
    print(f"min_area={args.min_area}  epsilon={args.epsilon}")
    if aux_skipped:
        print(f"忽略 {aux_skipped} 个有同名正式 mask 的 *_pred.png 预标注")
    print()

    written = skipped = empty = missing_img = 0
    dropped_total = 0
    unknown_values: set[int] = set()
    unknown_masks = 0
    shape_counts = {name: 0 for _, name in classes}

    for index, mask_path in enumerate(mask_files, start=1):
        stem, image_path = resolve_stem(mask_path.stem, images_dir, args.strip_suffix)
        json_path = out_dir / f"{stem}.json"
        if json_path.exists() and not args.overwrite:
            skipped += 1
            continue

        if image_path is None:
            print(f"  [skip] 找不到原图: {mask_path.stem}")
            missing_img += 1
            continue

        mask = np.asarray(Image.open(mask_path))
        if mask.ndim != 2:
            raise ValueError(f"mask 不是单通道: {mask_path} shape={mask.shape}")
        height, width = mask.shape

        present = {int(v) for v in np.unique(mask)}
        unknown = sorted(present - {0, IGNORE_VALUE} - {cid for cid, _ in classes})
        if unknown:
            unknown_values.update(unknown)
            unknown_masks += 1
            print(f"  [warn] {mask_path.name} 有未定义类别 {unknown}，已跳过")

        shapes, dropped = [], 0
        for cid, label in classes:
            if cid not in present:
                continue
            part, part_dropped = mask_to_polygons(
                mask, cid, label, args.min_area, args.epsilon
            )
            shapes.extend(part)
            dropped += part_dropped
        dropped_total += dropped
        if not shapes:
            empty += 1
            if args.skip_empty:
                continue

        # labelme 的 imagePath 是相对 JSON 所在目录的；
        # 跨盘符时 relpath 会抛 ValueError，退回绝对路径
        try:
            rel = os.path.relpath(image_path, out_dir)
        except ValueError:
            rel = str(image_path)
        rel = rel.replace("\\", "/")

        payload = {
            "version": LABELME_VERSION,
            "flags": {},
            "shapes": shapes,
            "imagePath": rel,
            "imageData": None,
            "imageHeight": int(height),
            "imageWidth": int(width),
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        written += 1
        for shape in shapes:
            shape_counts[shape["label"]] = shape_counts.get(shape["label"], 0) + 1

        if index % 50 == 0 or index == len(mask_files):
            print(f"  {index}/{len(mask_files)}")

    print(
        f"\n写出 {written} 个 JSON，跳过已存在 {skipped}，"
        f"空 mask {empty}，缺原图 {missing_img}"
    )
    if any(shape_counts.values()):
        print("各类多边形: " + "  ".join(f"{name} {count}"
                                        for name, count in shape_counts.items()))
    if dropped_total:
        print(f"丢弃过小/退化多边形 {dropped_total} 个")
    if unknown_masks:
        print(f"[warn] {unknown_masks} 张 mask 含未定义类别 {sorted(unknown_values)}，已跳过；"
              f"如果这些值也是类别，补进 labels.txt 再跑")
    print(f"输出 -> {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
