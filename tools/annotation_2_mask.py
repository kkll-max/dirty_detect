"""把 labelme JSON 转成多类别 mask（像素值 = labels.txt 里的类别 ID）。

labels.txt 每行一个类名，行号即类别 ID；`ignore` 固定映射到 255。
`clean` 必须是第一个非 ignore 标签，未标注背景填 0。没有 labels.txt 时用内置
6 类 clean / dirt_dot / glare / water / grass_residue / mud_film。

用法（在 code_clean/ 下执行）：

    # -i 原图、-o 标注、-m 输出；-i/-m 可省略，按 annotations 同级目录推断
    python tools/annotation_2_mask.py -i dataset5_2/images/train \
        -o dataset5_2/annotations/train -m dataset5_2/masks/train --preview

    # 批量：--data-dir 指向含多个子集的根目录（子目录内有 images/ 和 annotations/）
    python tools/annotation_2_mask.py --data-dir x5/data_get --preview

支持 polygon / rectangle / circle；其他形状跳过并提示。后画的 shape 覆盖先画的，
所以可以在脏类上压 clean 抠洞。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SUPPORTED = {"polygon", "rectangle", "circle"}
IGNORE_VALUE = 255
IGNORE_NAMES = {"ignore", "ignored", "void"}
# 当前 6 类（方案 A），顺序和 dataset5_2/labels.txt 一致；找不到 labels.txt 时用它。
DEFAULT_LABELS = ("clean", "dirt_dot", "glare", "water", "grass_residue", "mud_film")

# 预览配色和 prelabel.py / 部署侧 SEG_PALETTE_RGB 一致，方便对照
PREVIEW_ALPHA = 0.5
CLASS_RGB = {
    "dirt_dot": (165, 42, 42),       # 红褐 #a52a2a
    "glare": (255, 255, 0),          # 明黄 #ffff00
    "water": (0, 0, 255),            # 蓝   #0000ff
    "grass_residue": (0, 200, 0),    # 绿   #00c800
    "mud_film": (180, 120, 40),      # 黄褐 #b47828
}
FALLBACK_RGB = ((0, 200, 0), (255, 160, 0), (255, 64, 64), (160, 0, 255), (0, 255, 255))

# 方案 A 的旧标签合并；只在 labels.txt 里确实有目标类时生效
LABEL_ALIASES = {"water_film": "water", "water_drop": "water"}


@dataclass(frozen=True)
class Job:
    """一个待处理的数据集目录：annotations / images / masks 三件套。"""

    name: str
    json_dir: Path
    images_dir: Path | None
    out_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="labelme JSON -> 多类别 mask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--images", type=Path,
                        help="原图目录；用于尺寸校验和预览。默认按 annotations 同级推断")
    parser.add_argument("-o", "--annotations", type=Path, help="labelme JSON 目录")
    parser.add_argument("-m", "--masks", "--output", type=Path,
                        help="mask 输出目录，默认 <annotations 上一级>/masks")
    parser.add_argument("--data-dir", type=Path,
                        help="数据集根目录（含 annotations/），或含多个子集的父目录")
    parser.add_argument("--labels", type=Path,
                        help="标签表；默认在数据集根找 labels.txt，找不到用内置 6 类")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 mask")
    parser.add_argument("--preview", nargs="?", const="auto",
                        help="输出半透明预览；单数据集默认 <输出目录>_preview，"
                             "批量默认 <子集>/preview，也可显式给目录")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 个 JSON，0=全部")
    return parser.parse_args()


def swap_component(path: Path, old: str, new: str) -> Path | None:
    """把最后一个名为 old 的路径组件换成 new；找不到返回 None。"""
    parts = list(path.parts)
    for index in reversed(range(len(parts))):
        if parts[index].lower() == old:
            parts[index] = new
            return Path(*parts)
    return None


def infer_images_dir(json_dir: Path) -> Path | None:
    """按 annotations 的同级目录推断 images/。"""
    candidate = swap_component(json_dir, "annotations", "images")
    if candidate is not None and candidate.is_dir():
        return candidate
    sibling = json_dir.parent / "images"
    return sibling if sibling.is_dir() else None


def infer_mask_dir(json_dir: Path) -> Path:
    """按 annotations 的同级目录推断 masks/；不是 annotations 布局时退到上一级。"""
    return swap_component(json_dir, "annotations", "masks") or (json_dir.parent / "masks")


def make_job(annotations: Path, images: Path | None, out: Path,
             name: str | None = None) -> Job:
    return Job(
        name or annotations.name or "dataset",
        annotations.resolve(),
        images.resolve() if images is not None and images.is_dir() else None,
        out.resolve(),
    )


def resolve_jobs(args: argparse.Namespace) -> list[Job]:
    """解析待处理的数据集。

    `-o/--annotations` 处理单个数据集；`--data-dir` 兼容数据集根目录和
    “父目录下每个子目录各有 annotations/”的批量布局。
    """
    if args.annotations is not None:
        if not args.annotations.is_dir():
            raise FileNotFoundError(f"标注目录不存在: {args.annotations}")
        if args.images is not None and not args.images.is_dir():
            raise FileNotFoundError(f"原图目录不存在: {args.images}")
        return [make_job(
            args.annotations,
            args.images or infer_images_dir(args.annotations),
            args.masks or infer_mask_dir(args.annotations),
        )]

    base = args.data_dir
    if base is None:
        raise ValueError("要么给 -o/--annotations，要么给 --data-dir")
    if not base.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {base}")

    if (base / "annotations").is_dir():
        return [make_job(base / "annotations", args.images or base / "images",
                         args.masks or base / "masks", base.name)]

    if any(base.glob("*.json")):
        # --data-dir 直接指到了 annotations/<split>（或任意 JSON 目录）
        return [make_job(base, args.images or infer_images_dir(base) or base / "images",
                         args.masks or infer_mask_dir(base))]

    if args.images is not None:
        raise ValueError("批量模式下不能指定 --images：每个子集用自己的 images/")

    children = [
        child for child in sorted(base.iterdir())
        if (child / "annotations").is_dir()
        and any((child / "annotations").glob("*.json"))
    ]
    if not children:
        raise FileNotFoundError(
            f"--data-dir 下既没有 annotations/ 也没有 JSON: {base}\n"
            f"批量模式要求子目录含 annotations/ 和 images/，例如 "
            f"{base.name}/<子集>/annotations；"
            f"数据集按 split 存放时，直接把 -o 指到 split，例如 "
            f"-o data/dataset2/annotations/test；"
            f"或者用 -i / -o / -m 分别指定。"
        )

    return [
        make_job(child / "annotations",
                 child / "images" if (child / "images").is_dir() else None,
                 args.masks / child.name if args.masks else child / "masks",
                 child.name)
        for child in children
    ]


def resolve_preview_dir(args: argparse.Namespace, job: Job,
                        multiple: bool) -> Path | None:
    """批量时预览落到每个子集的 <子集>/preview；单个数据集保持旧命名。"""
    if args.preview is None:
        return None
    if args.preview == "auto":
        name = "preview" if multiple else f"{job.out_dir.name}_preview"
        return (job.out_dir.parent / name).resolve()
    base = Path(args.preview)
    return (base / job.name if multiple else base).resolve()


def resolve_label_file(args: argparse.Namespace, json_dir: Path) -> Path | None:
    """显式 --labels 优先；否则在数据集根和 annotations 上级找 labels.txt。"""
    if args.labels is not None:
        if not args.labels.is_file():
            raise FileNotFoundError(f"labels 文件不存在: {args.labels}")
        return args.labels.resolve()

    candidates = [json_dir.parent / "labels.txt"]
    # annotations/<split> 布局：labels.txt 在数据集根，往上多找一层
    if json_dir.parent.name.lower() == "annotations":
        candidates.append(json_dir.parent.parent / "labels.txt")
    candidates.append(json_dir / "labels.txt")
    if args.data_dir is not None:
        candidates.append(args.data_dir / "labels.txt")
    return next((path.resolve() for path in candidates if path.is_file()), None)


def load_label_map(path: Path | None) -> tuple[dict[str, int], str]:
    """读取标签表，返回 {label: class_id}。

    `ignore` 固定 255，不占类别 ID；其余标签按出现顺序从 0 开始编号。
    背景像素固定为 0，所以 clean 必须是第一个非 ignore 标签。
    """
    if path is None:
        labels, source = list(DEFAULT_LABELS), "内置默认"
    else:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        labels = [line.strip() for line in lines
                  if line.strip() and not line.strip().startswith("#")]
        source = str(path)
    if not labels:
        raise ValueError("labels 文件是空的")

    mapping: dict[str, int] = {}
    next_id = 0
    for name in labels:
        if name in mapping:
            raise ValueError(f"labels 里有重复标签: {name}")
        if name.lower() in IGNORE_NAMES:
            mapping[name] = IGNORE_VALUE
        else:
            if next_id >= IGNORE_VALUE:
                raise ValueError("类别太多，255 必须保留给 ignore")
            mapping[name] = next_id
            next_id += 1

    if "clean" not in mapping:
        raise ValueError("labels 里必须有 clean，未标注背景才能填 0")
    if mapping["clean"] != 0:
        raise ValueError("clean 必须是第一个非 ignore 标签，否则背景会被当成别的类别")
    return mapping, source


def find_image(images: Path | None, stem: str) -> Path | None:
    if images is None:
        return None
    for suffix in IMAGE_SUFFIXES:
        candidate = images / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def preview_colors(id_to_name: dict[int, str]) -> dict[int, np.ndarray]:
    """每个类别一个预览色：固定色优先，其余按出现顺序取后备色；clean 不叠色。"""
    colors: dict[int, np.ndarray] = {}
    fallback = 0
    for class_id, name in sorted(id_to_name.items()):
        if class_id == 0:
            continue
        rgb = CLASS_RGB.get(str(name).lower())
        if rgb is None:
            rgb = FALLBACK_RGB[fallback % len(FALLBACK_RGB)]
            fallback += 1
        colors[class_id] = np.array(rgb, dtype=np.uint8)
    return colors


def render_preview(image_path: Path, mask: np.ndarray,
                   colors: dict[int, np.ndarray]) -> np.ndarray:
    """把 mask 半透明叠到原图上，返回 RGB 数组。

    细小的 glare 光斑只靠填充看不出来，所以沿外轮廓再描 1 像素纯色线。
    """
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.shape[:2] != mask.shape[:2]:
        rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]),
                         interpolation=cv2.INTER_LINEAR)

    # 用 LUT 一次完成整幅混合，等价于逐类叠加（每像素只属于一个类）
    color_lut = np.zeros((256, 3), dtype=np.float32)
    alpha_lut = np.zeros(256, dtype=np.float32)
    for class_id, color in colors.items():
        color_lut[class_id] = color
        alpha_lut[class_id] = PREVIEW_ALPHA
    alpha = alpha_lut[mask][..., None]
    out = (rgb * (1 - alpha) + color_lut[mask] * alpha).astype(np.uint8)

    for class_id, color in colors.items():
        region = mask == class_id
        if not region.any():
            continue
        contours, _ = cv2.findContours(region.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(v) for v in color), 1)
    return out


def save_preview(image_path: Path | None, mask: np.ndarray,
                 colors: dict[int, np.ndarray], preview_dir: Path, stem: str) -> bool:
    """写出 <stem>_preview.png；没有原图时返回 False。"""
    if image_path is None:
        return False
    Image.fromarray(render_preview(image_path, mask, colors)).save(
        preview_dir / f"{stem}_preview.png")
    return True


def shape_to_contour(shape: dict) -> np.ndarray | None:
    """把 labelme 的 shape 转成 cv2.fillPoly 能用的 int32 点集。"""
    kind = shape.get("shape_type", "polygon")
    points = shape.get("points") or []
    if kind == "polygon":
        if len(points) < 3:
            return None
        poly = points
    elif kind == "rectangle":
        if len(points) < 2:
            return None
        (x1, y1), (x2, y2) = points[0], points[1]
        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif kind == "circle":
        if len(points) < 2:
            return None
        (cx, cy), (ex, ey) = points[0], points[1]
        radius = float(np.hypot(ex - cx, ey - cy))
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        poly = np.stack(
            [cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1
        )
    else:
        return None
    return np.round(np.asarray(poly, dtype=np.float64)).astype(np.int32)


def canonical_label(label: object, label_map: dict[str, int],
                    warn: Counter) -> str | None:
    """把旧标签别名解析到 labels.txt 里的类名；找不到时记未知标签。"""
    if isinstance(label, str) and label in label_map:
        return label
    if isinstance(label, str):
        target = LABEL_ALIASES.get(label)
        if target is not None and target in label_map:
            warn[f"别名:{label}->{target}"] += 1
            return target
    warn[f"未知标签:{label}"] += 1
    return None


def build_mask(data: dict, label_map: dict[str, int],
               warn: Counter) -> np.ndarray:
    height = int(data["imageHeight"])
    width = int(data["imageWidth"])
    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data.get("shapes", []):
        shape_type = shape.get("shape_type", "polygon")
        label = canonical_label(shape.get("label"), label_map, warn)
        if label is None:
            continue
        if shape_type not in SUPPORTED:
            warn[f"不支持的形状:{shape_type}"] += 1
            continue
        contour = shape_to_contour(shape)
        if contour is None:
            warn[f"退化形状:{shape_type}"] += 1
            continue
        cv2.fillPoly(mask, [contour], int(label_map[label]))
    return mask


def print_header(job: Job, label_source: str, id_to_name: dict[int, str],
                 class_ids: list[int], has_ignore: bool,
                 colors: dict[int, np.ndarray], preview_dir: Path | None) -> None:
    print(f"=== {job.name} ===")
    print(f"json    : {job.json_dir}")
    print(f"images  : {job.images_dir or '未指定（跳过尺寸校验和预览）'}")
    print(f"output  : {job.out_dir}")
    print(f"labels  : {label_source}")
    print("          " + "  ".join(f"{id_to_name[cid]}={cid}" for cid in class_ids)
          + ("  ignore=255" if has_ignore else ""))
    if preview_dir is not None:
        print(f"preview : {preview_dir}")
        print("          配色 " + "  ".join(
            f"{id_to_name[cid]}={tuple(int(v) for v in color)}"
            for cid, color in colors.items()))
    print()


def process_job(job: Job, args: argparse.Namespace, multiple: bool) -> Counter:
    if not job.json_dir.is_dir():
        raise FileNotFoundError(f"JSON 目录不存在: {job.json_dir}")

    json_files = sorted(p for p in job.json_dir.glob("*.json")
                        if p.name != "prelabel_summary.json")
    if args.limit > 0:
        json_files = json_files[:args.limit]
    if not json_files:
        splits = sorted(p.name for p in job.json_dir.iterdir()
                        if p.is_dir() and any(p.glob("*.json")))
        hint = (f"；这下面是分 split 的，把 --data-dir 指到其中一个，例如 "
                f"{job.json_dir.name}/{splits[0]}") if splits else ""
        raise FileNotFoundError(f"没有 JSON: {job.json_dir}{hint}")

    label_map, label_source = load_label_map(resolve_label_file(args, job.json_dir))
    class_ids = sorted(cid for cid in set(label_map.values()) if cid != IGNORE_VALUE)
    id_to_name = {cid: name for name, cid in label_map.items() if cid != IGNORE_VALUE}
    has_ignore = IGNORE_VALUE in label_map.values()

    out_dir = job.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = resolve_preview_dir(args, job, multiple)
    colors = preview_colors(id_to_name)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print_header(job, label_source, id_to_name, class_ids, has_ignore,
                 colors, preview_dir)

    warn: Counter = Counter()
    counts: Counter = Counter()
    ratios: dict[int, list[float]] = {cid: [] for cid in class_ids}
    ignore_ratios: list[float] = []

    for index, json_path in enumerate(json_files, start=1):
        stem = json_path.stem
        mask_path = out_dir / f"{stem}.png"

        if mask_path.exists() and not args.overwrite:
            counts["skipped"] += 1
            # 已有 mask 不重写，但 --preview 仍然画出来，方便检查磁盘上的结果
            if preview_dir is not None:
                image_path = find_image(job.images_dir, stem)
                if image_path is None:
                    counts["preview_missing"] += 1
                else:
                    with Image.open(mask_path) as existing:
                        existing_mask = np.asarray(existing)
                    save_preview(image_path, existing_mask, colors, preview_dir, stem)
                    counts["previewed"] += 1
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "imageHeight" not in data or "imageWidth" not in data:
                raise ValueError("JSON 缺少 imageHeight/imageWidth")
        except (json.JSONDecodeError, ValueError) as exc:
            counts["failed"] += 1
            print(f"  [fail] {json_path.name}: {exc}")
            continue

        mask = build_mask(data, label_map, warn)
        image_path = find_image(job.images_dir, stem)
        if image_path is not None:
            with Image.open(image_path) as image:
                if image.size != (mask.shape[1], mask.shape[0]):
                    warn[
                        f"尺寸不一致:{image.size} vs {mask.shape[::-1]}"
                    ] += 1

        Image.fromarray(mask).save(mask_path)
        counts["written"] += 1
        if preview_dir is not None:
            if save_preview(image_path, mask, colors, preview_dir, stem):
                counts["previewed"] += 1
            else:
                counts["preview_missing"] += 1

        # 一次 bincount 拿到全部类别像素数，避免逐类整图比较
        pixels = np.bincount(mask.ravel(), minlength=IGNORE_VALUE + 1)
        for cid in class_ids:
            ratios[cid].append(int(pixels[cid]) / mask.size)
        if has_ignore:
            ignore_ratios.append(int(pixels[IGNORE_VALUE]) / mask.size)

        if index % 50 == 0 or index == len(json_files):
            print(f"  {index}/{len(json_files)}")

    written = counts["written"]
    print(f"\n写出 {written} 张 mask，跳过 {counts['skipped']}，失败 {counts['failed']}")
    if written:
        for cid in class_ids:
            values = np.asarray(ratios[cid])
            print(f"  {id_to_name[cid]:<12s} id={cid}: "
                  f"出现 {int((values > 0).sum())}/{written} 张, "
                  f"mean={values.mean():.4f} max={values.max():.4f}")
        if has_ignore:
            values = np.asarray(ignore_ratios)
            print(f"  {'ignore':<12s} id=255: "
                  f"出现 {int((values > 0).sum())}/{written} 张, "
                  f"mean={values.mean():.4f} max={values.max():.4f}")
        non_clean = [cid for cid in class_ids if cid != 0]
        if non_clean:
            dirty = np.zeros(written, dtype=bool)
            for cid in non_clean:
                dirty |= np.asarray(ratios[cid]) > 0
            print(f"  全 clean: {written - int(dirty.sum())}/{written} 张")
    if warn:
        print("\n提示:")
        for key, count in warn.most_common():
            print(f"  {key}  x{count}")
    if preview_dir is not None:
        print(f"预览 {counts['previewed']} 张 -> {preview_dir}")
        if counts["preview_missing"]:
            print(f"[warn] {counts['preview_missing']} 张找不到原图，没生成预览")
    print(f"输出 -> {out_dir}")
    return counts


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(args)
    multiple = len(jobs) > 1
    totals: Counter = Counter()
    failures: list[tuple[str, str]] = []

    for job in jobs:
        try:
            totals.update(process_job(job, args, multiple))
        except (FileNotFoundError, ValueError) as exc:
            failures.append((job.name, str(exc)))
            print(f"[fail] {job.name}: {exc}", file=sys.stderr)
            print()

    if multiple:
        print(f"总计: {len(jobs)} 个数据集, 写出 {totals['written']} 张 mask, "
              f"跳过 {totals['skipped']}, 失败 {totals['failed']}, "
              f"预览 {totals['previewed']} 张")
    if failures:
        names = ", ".join(name for name, _ in failures)
        raise ValueError(f"{len(failures)} 个数据集处理失败: {names}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
