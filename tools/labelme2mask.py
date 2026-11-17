"""把 labelme 的 JSON 转成多类别 mask。

和 mask2labelme.py 配套，形成"预标注 -> labelme 修改 -> 回写 mask"的闭环。

labels.txt 每行一个类别名，**行号就是类别 ID**（当前 6 类方案 A）：

    clean          -> 0
    dirt_dot       -> 1
    glare          -> 2
    water          -> 3
    grass_residue  -> 4
    mud_film       -> 5
    ignore         -> 255（跳过 loss，不是训练类别）

约定：
- `clean` 必须是第一个非 ignore 标签，背景默认填 0；
- `ignore` 映射到 255，不占类别 ID；
- 不在 labels 里的标签会被跳过并提示；
- 绘制顺序就是叠放顺序，后画的覆盖先画的，所以可以在脏类上压 clean 抠洞。

没有 `--labels` 时，优先找 `<annotations 的上一级>/labels.txt`（annotations 是
split 目录时再往上一层找），最后找 `<data-dir>/labels.txt`；都没有就用内置的
当前 6 类 `clean / dirt_dot / glare / water / grass_residue / mud_film`。

支持的 shape_type：polygon / rectangle / circle。其他类型（line、point、
linestrip）会跳过并提示。

举例：

    # 常用写法：-i 原图、-o 原标注（JSON）、-m mask 输出
    python tools/labelme2mask.py -i data/guangban2/images \
        -o data/guangban2/annotations -m data/guangban2/masks

    # -i / -m 可省略：images 和 masks 按 annotations 的同级目录推断
    python tools/labelme2mask.py -o data/guangban2/annotations

    # 指定标签表（默认内置就是当前 6 类）
    python tools/labelme2mask.py -i data/glare/images -o data/glare/annotations \
        -m data/glare/masks_final --labels data/glare/labels.txt

    # 生成 mask 的同时把结果半透明叠到原图上，肉眼检查标注对不对；
    # 不写预览目录时落在 <mask 输出目录>_preview
    python tools/labelme2mask.py -o data/dataset2/annotations/test \
        -m data/dataset2/masks/test --preview
    python tools/labelme2mask.py -i data/dataset2/images/test \
        -o data/dataset2/annotations/test -m data/dataset2/masks/test \
        --preview runs/preview/dataset2_test

    # 旧写法（仍兼容）：--data-dir 指向数据集根目录
    python tools/labelme2mask.py --data-dir data/matrix

    # 批量：--data-dir 指向包含多个子数据集的根目录（每个子目录都有
    # images/ 和 annotations/），逐个写 <子目录>/masks 和 <子目录>/preview
    python tools/labelme2mask.py --data-dir x5/data_get --preview

兼容旧标签：labels.txt 里有 `water` 时，`water_film` / `water_drop` 会自动
并入 `water`（方案 A），并单独计入提示，不会当成未知标签丢掉。
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

PKG_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SUPPORTED = {"polygon", "rectangle", "circle"}
IGNORE_VALUE = 255
IGNORE_NAMES = {"ignore", "ignored", "void"}
# 当前 6 类（方案 A），顺序和 data/dataset5_1/labels.txt 一致；
# 找不到 labels.txt 时用它。clean 必须第一个。
DEFAULT_LABELS = ("clean", "dirt_dot", "glare", "water", "grass_residue", "mud_film")

# 预览配色和 predict.py 保持一致，方便两边对照
# 6 类调色板，和 stream_server.c / min_dirty_infer 的 SEG_PALETTE_RGB 同色
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
LABEL_ALIASES = {
    "water_film": "water",
    "water_drop": "water",
}


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
    parser.add_argument("-i", "--images", type=Path, default=None,
                        help="原图目录；用于校验尺寸和生成预览。不给时按 "
                             "annotations 的同级 images/ 推断")
    parser.add_argument("-o", "--annotations", type=Path, default=None,
                        help="原标注目录（labelme JSON）")
    parser.add_argument("--json-dir", type=Path, default=None,
                        help="已弃用，等同 -o/--annotations")
    parser.add_argument("-m", "--masks", "--output", type=Path, default=None,
                        help="mask 输出目录，默认 <annotations 上一级>/masks")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="旧写法：数据集根目录，含 annotations/ 和 images/；"
                             "也可以指向含多个此类子目录的父目录，批量处理")
    parser.add_argument("--labels", type=Path, default=None,
                        help="标签表，每行一个类别名，行号即类别 ID；默认在 annotations "
                             "上一级 / data-dir 根目录找 labels.txt，找不到用内置 6 类"
                             "（clean/dirt_dot/glare/water/grass_residue/mud_film）")
    parser.add_argument("--overwrite", action="store_true", help="mask 已存在时覆盖")
    parser.add_argument("--preview", nargs="?", const="auto", default=None,
                        help="把生成的 mask 半透明叠到原图上存成 <stem>_preview.png；"
                             "单数据集不带值写到 <输出目录>_preview，"
                             "批量模式写到 <子集>/preview；也可以显式给一个目录")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def replace_component(path: Path, old: str, new: str) -> Path | None:
    """把最后一个名为 old 的路径组件换成 new；找不到返回 None。

    `.../dataset2/annotations/test` 把 annotations 换成 images
    -> `.../dataset2/images/test`
    """
    parts = list(path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == old:
            return Path(*parts[:index], new, *parts[index + 1:])
    return None


def infer_images_dir(json_dir: Path) -> Path | None:
    """按 annotations 的同级目录推断 images/。"""
    candidate = replace_component(json_dir, "annotations", "images")
    if candidate is not None and candidate.is_dir():
        return candidate
    sibling = json_dir.parent / "images"
    return sibling if sibling.is_dir() else None


def infer_mask_dir(json_dir: Path) -> Path:
    """按 annotations 的同级目录推断 masks/；不是 annotations 布局时退到上一级。"""
    candidate = replace_component(json_dir, "annotations", "masks")
    return candidate if candidate is not None else (json_dir.parent / "masks")


def resolve_jobs(args: argparse.Namespace) -> list[Job]:
    """解析出所有待处理的数据集。

    优先级：`-o/--annotations`（新写法，单数据集）> `--data-dir`（旧写法）。

    `--data-dir` 本身没有 annotations/ 和 JSON、但直接子目录里有 `annotations/`
    时，按子目录批量处理：

        x5/data_get/
          clean_63/{images,annotations}  -> clean_63/masks + clean_63/preview
          green_154/{images,annotations} -> green_154/masks + green_154/preview
    """
    explicit = args.annotations or args.json_dir
    if explicit is not None:
        if not explicit.is_dir():
            raise FileNotFoundError(f"标注目录不存在: {explicit}")
        if args.images is not None and not args.images.is_dir():
            raise FileNotFoundError(f"原图目录不存在: {args.images}")
        images = args.images or infer_images_dir(explicit)
        out = args.masks or infer_mask_dir(explicit)
        return [Job(explicit.name or "dataset", explicit.resolve(),
                    images.resolve() if images else None, out.resolve())]

    if args.data_dir is None:
        raise ValueError("要么给 -o/--annotations（原标注目录），要么给 --data-dir")

    base = args.data_dir
    if (base / "annotations").is_dir():
        json_dir = base / "annotations"
        images = args.images or (base / "images")
        out = args.masks or (base / "masks")
        return [Job(base.name or "dataset", json_dir.resolve(),
                    images.resolve(), out.resolve())]

    if any(base.glob("*.json")):
        # --data-dir 直接指到了 annotations/<split>（或任意 JSON 目录）：
        # 就地当 JSON 目录，images / masks 按同名 split 推出来
        json_dir = base
        images = args.images or infer_images_dir(base) or (base / "images")
        out = args.masks or infer_mask_dir(base)
        return [Job(base.name or "dataset", json_dir.resolve(),
                    images.resolve() if images else None, out.resolve())]

    if args.images is not None:
        raise ValueError("批量模式下不能指定 --images：每个子数据集用自己的 images/")

    children = [
        child for child in sorted(base.iterdir())
        if child.is_dir() and (child / "annotations").is_dir()
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

    jobs = []
    for child in children:
        images = child / "images"
        out = (args.masks / child.name) if args.masks else (child / "masks")
        jobs.append(Job(child.name, (child / "annotations").resolve(),
                        images.resolve() if images.is_dir() else None,
                        out.resolve()))
    return jobs


def resolve_preview_dir(args: argparse.Namespace, job: Job,
                        multiple: bool) -> Path | None:
    """批量时预览落到每个子集的 <子集>/preview；单个数据集保持旧命名。"""
    if args.preview is None:
        return None
    if args.preview == "auto":
        if multiple:
            return (job.out_dir.parent / "preview").resolve()
        return (job.out_dir.parent / f"{job.out_dir.name}_preview").resolve()
    base = Path(args.preview)
    if multiple:
        base = base / job.name
    return base.resolve()


def resolve_label_file(args: argparse.Namespace, json_dir: Path) -> Path | None:
    """显式 --labels 优先；否则在 data-dir、子数据集和 annotations 上一级找。"""
    if args.labels is not None:
        path = args.labels
        if not path.is_file():
            raise FileNotFoundError(f"labels 文件不存在: {path}")
        return path.resolve()

    candidates = [json_dir.parent / "labels.txt"]
    # annotations/<split> 布局：labels.txt 在数据集根，往上多找一层
    if json_dir.parent.name.lower() == "annotations":
        candidates.append(json_dir.parent.parent / "labels.txt")
    # 兼容直接把 --json-dir 指到 annotations 的情况
    candidates.append(json_dir / "labels.txt")
    if args.data_dir is not None:
        # 批量模式下 --data-dir 是子数据集的公共根目录，作为最后的兜底
        candidates.append(args.data_dir / "labels.txt")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def load_label_map(path: Path | None) -> tuple[dict[str, int], str]:
    """读取标签表，返回 {label: class_id}。

    `ignore` 固定 255，不占类别 ID；其余标签按出现顺序从 0 开始编号。
    背景像素固定为 0，所以 clean 必须是第一个非 ignore 标签。
    """
    if path is None:
        labels = list(DEFAULT_LABELS)
        source = "内置默认"
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
    """每个类别一个预览色：dirty / glare 固定，其余按出现顺序取后备色。"""
    colors: dict[int, np.ndarray] = {}
    fallback = 0
    for class_id, name in sorted(id_to_name.items()):
        if class_id == 0 or class_id == IGNORE_VALUE:
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

    细小的 glare 光斑只靠半透明填充看不出来，所以沿外轮廓再描 1 像素的纯色线。
    原图和 mask 尺寸不一致时（正常不该发生），把原图缩到 mask 尺寸再叠。
    """
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.shape[:2] != mask.shape[:2]:
        rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    out = rgb.copy()
    for class_id, color in sorted(colors.items()):
        region = mask == class_id
        if not region.any():
            continue
        base = out[region].astype(np.float32)
        out[region] = (base * (1 - PREVIEW_ALPHA)
                       + color.astype(np.float32) * PREVIEW_ALPHA).astype(np.uint8)
        contours, _ = cv2.findContours(region.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(v) for v in color), 1)
    return out


def shape_to_contour(shape: dict) -> np.ndarray | None:
    """把 labelme 的 shape 转成 cv2.fillPoly 能用的 int32 点集。"""
    shape_type = shape.get("shape_type", "polygon")
    points = shape.get("points") or []
    if shape_type == "polygon":
        if len(points) < 3:
            return None
        return np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
    if shape_type == "rectangle":
        if len(points) < 2:
            return None
        (x1, y1), (x2, y2) = points[0], points[1]
        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return np.round(np.asarray(poly, dtype=np.float64)).astype(np.int32)
    if shape_type == "circle":
        if len(points) < 2:
            return None
        (cx, cy), (ex, ey) = points[0], points[1]
        radius = float(np.hypot(ex - cx, ey - cy))
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        poly = np.stack(
            [cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1
        )
        return np.round(poly).astype(np.int32)
    return None


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


def process_job(job: Job, args: argparse.Namespace, multiple: bool) -> Counter:
    json_dir, images_dir, out_dir = job.json_dir, job.images_dir, job.out_dir

    if not json_dir.is_dir():
        raise FileNotFoundError(f"JSON 目录不存在: {json_dir}")

    json_files = sorted(json_dir.glob("*.json"))
    json_files = [p for p in json_files if p.name != "prelabel_summary.json"]
    if args.limit > 0:
        json_files = json_files[: args.limit]
    if not json_files:
        splits = sorted(p.name for p in json_dir.iterdir()
                        if p.is_dir() and any(p.glob("*.json")))
        hint = (f"；这下面是分 split 的，把 --data-dir 指到其中一个，例如 "
                f"{json_dir.name}/{splits[0]}") if splits else ""
        raise FileNotFoundError(f"没有 JSON: {json_dir}{hint}")

    label_path = resolve_label_file(args, json_dir)
    label_map, label_source = load_label_map(label_path)
    class_ids = sorted(cid for cid in set(label_map.values()) if cid != IGNORE_VALUE)
    id_to_name = {cid: name for name, cid in label_map.items() if cid != IGNORE_VALUE}
    has_ignore = IGNORE_VALUE in label_map.values()

    out_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = resolve_preview_dir(args, job, multiple)
    colors = preview_colors(id_to_name)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {job.name} ===")
    print(f"json    : {json_dir}")
    print(f"images  : {images_dir if images_dir else '未指定（跳过尺寸校验和预览）'}")
    print(f"output  : {out_dir}")
    print(f"labels  : {label_source}")
    print("          " + "  ".join(f"{id_to_name[cid]}={cid}" for cid in class_ids)
          + ("  ignore=255" if has_ignore else ""))
    if preview_dir is not None:
        print(f"preview : {preview_dir}")
        print("          配色 " + "  ".join(
            f"{id_to_name[cid]}={tuple(int(v) for v in color)}"
            for cid, color in colors.items())
        )
    print()

    warn: Counter = Counter()
    written = skipped = failed = 0
    previewed = preview_missing = 0
    class_ratios: dict[int, list[float]] = {cid: [] for cid in class_ids}
    ignore_ratios: list[float] = []

    for index, json_path in enumerate(json_files, start=1):
        stem = json_path.stem
        mask_path = out_dir / f"{stem}.png"
        if mask_path.exists() and not args.overwrite:
            skipped += 1
            # 已有 mask 不重写，但 --preview 仍然画出来，方便检查磁盘上的结果
            if preview_dir is not None:
                skipped_image = find_image(images_dir, stem)
                if skipped_image is None:
                    preview_missing += 1
                else:
                    with Image.open(mask_path) as existing:
                        skipped_mask = np.asarray(existing)
                    Image.fromarray(
                        render_preview(skipped_image, skipped_mask, colors)
                    ).save(preview_dir / f"{stem}_preview.png")
                    previewed += 1
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "imageHeight" not in data or "imageWidth" not in data:
                raise ValueError("JSON 缺少 imageHeight/imageWidth")
        except (json.JSONDecodeError, ValueError) as exc:
            failed += 1
            print(f"  [fail] {json_path.name}: {exc}")
            continue

        mask = build_mask(data, label_map, warn)

        image_path = find_image(images_dir, stem)
        if image_path is not None:
            with Image.open(image_path) as image:
                if image.size != (mask.shape[1], mask.shape[0]):
                    warn[
                        f"尺寸不一致:{image.size} vs {mask.shape[::-1]}"
                    ] += 1

        Image.fromarray(mask, mode="L").save(mask_path)
        written += 1
        if preview_dir is not None:
            if image_path is None:
                preview_missing += 1
            else:
                Image.fromarray(render_preview(image_path, mask, colors)).save(
                    preview_dir / f"{stem}_preview.png"
                )
                previewed += 1
        for cid in class_ids:
            class_ratios[cid].append(float((mask == cid).mean()))
        if has_ignore:
            ignore_ratios.append(float((mask == IGNORE_VALUE).mean()))

        if index % 50 == 0 or index == len(json_files):
            print(f"  {index}/{len(json_files)}")

    print(f"\n写出 {written} 张 mask，跳过 {skipped}，失败 {failed}")
    if written:
        for cid in class_ids:
            values = np.array(class_ratios[cid])
            present = int((values > 0).sum())
            print(f"  {id_to_name[cid]:<12s} id={cid}: "
                  f"出现 {present}/{written} 张, "
                  f"mean={values.mean():.4f} max={values.max():.4f}")
        if has_ignore:
            values = np.array(ignore_ratios)
            print(f"  {'ignore':<12s} id=255: "
                  f"出现 {int((values > 0).sum())}/{written} 张, "
                  f"mean={values.mean():.4f} max={values.max():.4f}")
        non_clean = [cid for cid in class_ids if cid != 0]
        if non_clean:
            all_clean = sum(
                1 for i in range(written)
                if all(class_ratios[cid][i] == 0 for cid in non_clean)
            )
            print(f"  全 clean: {all_clean}/{written} 张")
    if warn:
        print("\n提示:")
        for key, count in warn.most_common():
            print(f"  {key}  x{count}")
    if preview_dir is not None:
        print(f"预览 {previewed} 张 -> {preview_dir}")
        if preview_missing:
            print(f"[warn] {preview_missing} 张找不到原图，没生成预览")
    print(f"输出 -> {out_dir}")

    return Counter(
        written=written, skipped=skipped, failed=failed,
        previewed=previewed, preview_missing=preview_missing,
    )


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
