"""抽取校准图片，并按训练预处理转成 RDK（hb_mapper）用的 NCHW 二进制。

两件事
------
1. 从 `--source`（默认 dataset3 的 train）随机抽取 `--ratio`（默认 10%）的图片，
   复制到 `--images-out`（默认 data/cal_img），种子固定 42。
2. 把这些图片按**和训练完全一致**的几何预处理转成 `.raw`：
   长边缩到 640（INTER_LINEAR，和不放大一致）→ 按 5:3 居中裁到 384×640
   → RGB → NCHW 1×3×384×640 float32，**不除 255**。

为什么要自己裁，而不是直接 resize 到 640×384
-----------------------------------------------
训练用的是 `tools/crop_dataset.py`：先长边缩到 640，再居中裁 5:3。
dataset3 的 train 里有三种原图：

| 原图 | 训练时 | 本脚本 |
|---|---|---|
| 640×448 | 上下各去 32 行 | 同样 |
| 640×384 | 原样 | 同样 |
| 1280×960（wood） | 缩到 640×480 再上下各去 48 行 | 同样 |

直接 `cv2.resize((640, 384))` 会把 4:3 的 wood 压扁，量化校准看到的
输入分布就偏了。所以这里复刻 `aspect_crop_box` 的一刀。

量化口径
--------
- 数据：RGB、0..255、float32、NCHW，即 hb_mapper 的 `cal_data_type: float32`；
- 归一化放进模型配置：`norm_type: data_scale`、`scale_value: 0.00392156862745098`，
  和 Ultralytics 推理时的 `im /= 255` 对齐；
- 输出目录里**只放 `.raw`**（manifest CSV 写在目录外），hb_mapper 会把
  目录里的每个文件都当成校准数据。

用法（在 some_classes/ 下执行）：

    python tools/make_calib.py                      # 抽 10% + 转 raw
    python tools/make_calib.py --ratio 0.05 --seed 0
    python tools/make_calib.py --images data/cal_img   # 只转换已有目录
    python tools/make_calib.py --check              # 只检查已有 .raw
    python tools/make_calib.py --force              # 允许覆盖非空的 cal_img
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_IMGSZ = 640
DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抽取校准图并转成 NCHW float32 raw（训练口径裁切）")
    parser.add_argument("--source", type=Path,
                        default=PKG_ROOT / "data/dataset3/images/train",
                        help="采样来源目录，默认 dataset3 的 train")
    parser.add_argument("--images", type=Path, default=None,
                        help="跳过采样，直接转换这个目录里的图片")
    parser.add_argument("--images-out", type=Path,
                        default=PKG_ROOT / "data/cal_img",
                        help="抽样图片的存放目录")
    parser.add_argument("--out", type=Path,
                        default=PKG_ROOT / "data/cal_nchw",
                        help="*.raw 输出目录")
    parser.add_argument("--ratio", type=float, default=0.20,
                        help="抽样比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="训练时的 imgsz（长边缩放目标），默认 640")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--limit", type=int, default=0,
                        help="只转前 N 张，0 表示全部")
    parser.add_argument("--force", action="store_true",
                        help="cal_img / cal_nchw 非空时清空后重建")
    parser.add_argument("--check", action="store_true",
                        help="不转换，只校验已有 .raw")
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"图片目录不存在: {root}")
    return sorted(path for path in root.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def prepare_dir(directory: Path, force: bool, what: str) -> None:
    if directory.exists() and any(directory.iterdir()):
        if not force:
            raise FileExistsError(f"{what} 已存在且非空: {directory}（确认重建加 --force）")
        print(f"[注意] 清空已有目录 {directory}")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def sample_images(source: Path, images_out: Path, ratio: float,
                  seed: int, force: bool) -> list[Path]:
    images = list_images(source)
    if not images:
        raise RuntimeError(f"{source} 下没有图片")
    count = max(1, round(len(images) * ratio))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(images, count))
    prepare_dir(images_out, force, what="cal_img")
    for path in chosen:
        shutil.copy2(path, images_out / path.name)
    print(f"抽样   : {source} -> {len(chosen)}/{len(images)} 张 "
          f"(ratio={ratio}, seed={seed})")
    meta = {
        "source": str(source),
        "ratio": ratio,
        "seed": seed,
        "count": count,
        "files": [p.name for p in chosen],
    }
    meta_path = images_out.with_name(f"{images_out.name}_selection.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"抽样清单: {meta_path}")
    return chosen


def training_resize_crop(image: np.ndarray, imgsz: int,
                         target_h: int, target_w: int) -> np.ndarray:
    """复刻训练预处理：长边缩到 imgsz，再按目标宽高比居中裁切。

    缩放部分和 ultralytics `BaseDataset.load_image(rect_mode=True,
    resize_short=False)` 一致；裁切部分和 `tools/crop_dataset.aspect_crop_box`
    一致。
    """
    h0, w0 = image.shape[:2]
    ratio = imgsz / max(h0, w0)
    if ratio != 1:
        w = min(math.ceil(w0 * ratio), imgsz)
        h = min(math.ceil(h0 * ratio), imgsz)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    height, width = image.shape[:2]
    target_ratio = target_w / target_h
    source_ratio = width / height
    if abs(source_ratio - target_ratio) < 1e-6:
        top, left, new_h, new_w = 0, 0, height, width
    elif source_ratio > target_ratio:      # 太宽，裁左右
        new_w = round(height * target_ratio)
        left = (width - new_w) // 2
        top, new_h = 0, height
    else:                                  # 太扁，裁上下
        new_h = round(width / target_ratio)
        top = (height - new_h) // 2
        left, new_w = 0, width
    image = image[top:top + new_h, left:left + new_w]

    if (new_h, new_w) != (target_h, target_w):
        # 正常数据走不到这里；真出现了也按 LetterBox 的兜底做法处理
        image = cv2.resize(image, (target_w, target_h),
                           interpolation=cv2.INTER_LINEAR)
    return image


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"读不了图片: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_one(src: Path, dst: Path, imgsz: int,
                height: int, width: int) -> dict:
    image = load_rgb(src)
    original = (image.shape[1], image.shape[0])
    image = training_resize_crop(image, imgsz, height, width)
    array = np.ascontiguousarray(image.transpose(2, 0, 1)[np.newaxis, ...])
    array = array.astype(np.float32)       # 不除 255，scale 交给 hb_mapper
    expected = 1 * 3 * height * width * 4
    if array.nbytes != expected:
        raise AssertionError(f"{src}: {array.nbytes} 字节，应为 {expected}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(dst)
    return {
        "stem": src.stem,
        "source": str(src),
        "raw": str(dst),
        "original": f"{original[0]}x{original[1]}",
        "width": width,
        "height": height,
        "bytes": array.nbytes,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def expected_bytes(height: int, width: int) -> int:
    return 3 * height * width * 4


def check_dir(directory: Path, height: int, width: int) -> int:
    files = sorted(directory.glob("*.raw"))
    if not files:
        print(f"no .raw files in {directory}")
        return 1
    want = expected_bytes(height, width)
    bad = 0
    mins, maxs = [], []
    for path in files:
        size = path.stat().st_size
        if size != want:
            print(f"  BAD  {path.name}: {size} bytes, expected {want}")
            bad += 1
            continue
        sample = np.fromfile(path, dtype=np.float32, count=4096)
        mins.append(float(sample.min()))
        maxs.append(float(sample.max()))
    print(f"checked {len(files)} files, {bad} bad, expected {want} bytes each")
    if mins:
        print(f"sampled value range: min={min(mins):.1f} max={max(maxs):.1f}")
        if min(mins) < 0 or max(maxs) > 255:
            print("  warning: values outside 0..255; check the RGB/scale pipeline")
    return 1 if bad else 0


def main() -> int:
    args = parse_args()

    if args.check:
        return check_dir(args.out, args.height, args.width)

    if args.images is not None:
        images = list_images(args.images)
    elif args.source is not None:
        images = sample_images(args.source.resolve(), args.images_out.resolve(),
                               args.ratio, args.seed, args.force)
    else:
        raise ValueError("要么给 --source 采样，要么给 --images 直接转换")

    if args.limit:
        images = images[: args.limit]
    if not images:
        raise RuntimeError("没有可转换的图片")

    prepare_dir(args.out.resolve(), args.force, what="cal_nchw")
    manifest_path = args.out.with_name(f"{args.out.name}_manifest.csv")
    rows: list[dict] = []
    failures = 0

    print(f"output : {args.out.resolve()}")
    print(f"target : NCHW 1x3x{args.height}x{args.width} float32 RGB 0..255")
    print(f"images : {len(images)}")

    for index, src in enumerate(images, 1):
        dst = args.out.resolve() / f"{src.stem}.raw"
        try:
            rows.append(convert_one(src, dst, args.imgsz,
                                    args.height, args.width))
        except Exception as exc:  # noqa: BLE001 - 单张失败不中断
            print(f"[{index}/{len(images)}] FAILED {src}: {exc}",
                  file=sys.stderr)
            failures += 1
            continue
        if index % 10 == 0 or index == len(images):
            print(f"[{index}/{len(images)}] ok")

    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print()
    print(f"converted : {len(rows)}")
    print(f"failed    : {failures}")
    print(f"bytes/each: {expected_bytes(args.height, args.width)}")
    print(f"manifest  : {manifest_path}")

    if rows:
        print()
        print("yaml block:")
        print("  input_parameters:")
        print('    input_name: "images"')
        print("    input_type_train: 'rgb'")
        print("    input_layout_train: 'NCHW'")
        print(f"    input_shape: '1x3x{args.height}x{args.width}'")
        print("    input_type_rt: 'nv12'")
        print("    norm_type: 'data_scale'")
        print("    scale_value: '0.00392156862745098'")
        print("  calibration_parameters:")
        print(f"    cal_data_dir: '{args.out.as_posix()}'")
        print("    cal_data_type: 'float32'")

    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, NotADirectoryError,
            RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
