r"""用预标注模型给一批图片生成 labelme JSON。

预处理和部署口径一致：先按 5:3 居中裁到 384×640，再入网；预测 mask 再放回
原图坐标。裁切框以外的像素模型没看过，不会生成 shape，需要人工补。

输出是可直接用 labelme 打开修正的 JSON：

    python tools/prelabel.py -i path/images -o path/annotations

    # 只跑前 20 张，CPU
    python tools/prelabel.py -i data/dataset4/yu_data/images/val ^
        -o runs/prelabel/val --limit 20 --device cpu

    # 同时保存整帧预测 mask（裁切框外为 255）
    python tools/prelabel.py -i path/images -o path/annotations --save-mask

    # 一次给出图片、预测 mask、JSON 和预览图四个目录
    python tools/prelabel.py -i path/images -m path/masks ^
        -o path/annotations -p path/preview

    # 不写 JSON，只看预览
    python tools/prelabel.py -i path/images -p path/preview

默认权重 runs/prelabel_mix6/weights/best.pt（yu_data + x5/data_get 混合六类模型）。
类别名优先取模型 names；如果模型没有 names，用 --labels 指定 labels.txt。
旧的七类模型会输出 water_film / water_drop，与当前 labels.txt 不一致，
只能显式用 -w 指定并自行转换。
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
DEFAULT_WEIGHTS = PKG_ROOT / "runs/dataset4_2_6cls/weights/best.pt"
DEFAULT_CROP_HW = (384, 640)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
LABELME_VERSION = "5.0.1"
IGNORE = 255

# 6 类调色板，和 stream_server.c / min_dirty_infer 的 SEG_PALETTE_RGB 同色，
# 检测图按 50% 叠加，网页图例也用这套。prelabel / labelme2mask / predict 三边一致。
PREVIEW_ALPHA = 0.5
CLASS_RGB = {
    "dirt_dot": (165, 42, 42),       # 红褐 #a52a2a
    "glare": (255, 255, 0),          # 明黄 #ffff00
    "water": (0, 0, 255),            # 蓝   #0000ff
    "grass_residue": (0, 200, 0),    # 绿   #00c800
    "mud_film": (180, 120, 40),      # 黄褐 #b47828
}
FALLBACK_RGB = ((0, 200, 0), (255, 160, 0), (255, 64, 64), (160, 0, 255), (0, 255, 255))
CROP_BOX_RGB = np.array([0, 255, 255], dtype=np.uint8)   # 青，只标裁切框


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用预标注模型生成 labelme JSON / 预测 mask / 预览图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--images", "--source", dest="images", type=Path,
                        required=True, help="单张图片或图片目录（递归）")
    parser.add_argument("-o", "--annotations", "--output", dest="output", type=Path,
                        default=None,
                        help="labelme JSON 输出目录；不给就不写 JSON")
    parser.add_argument("-w", "--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"模型权重，默认 {DEFAULT_WEIGHTS}")
    parser.add_argument("--labels", type=Path, default=None,
                        help="labels.txt；模型有 names 时可省略")
    parser.add_argument("--crop-hw", default=",".join(map(str, DEFAULT_CROP_HW)),
                        help="裁切目标 H,W，默认 384,640")
    parser.add_argument("--full-frame", action="store_true",
                        help="不裁切，整图入网（和部署口径不一致）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="入网长边，默认 640（= crop 的宽）")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--min-area", type=float, default=16.0,
                        help="小于该像素数的连通块丢弃，默认 16")
    parser.add_argument("--epsilon", type=float, default=0.002,
                        help="多边形抽稀系数，默认 0.002；0 表示不抽稀")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 张，0 = 全部")
    parser.add_argument("--overwrite", action="store_true",
                        help="JSON 已存在时覆盖")
    parser.add_argument("--skip-empty", action="store_true",
                        help="全 clean 的图不写 JSON")
    parser.add_argument("--save-mask", action="store_true",
                        help="同时保存整帧预测 mask（框外 255）")
    parser.add_argument("-m", "--mask-dir", "--masks", dest="mask_dir", type=Path,
                        default=None,
                        help="预测 mask 输出目录；指定即保存（框外 255），"
                             "不给 --save-mask 时默认 <annotations>_masks")
    parser.add_argument("-p", "--preview", "--preview-output", dest="preview",
                        type=Path, default=None,
                        help="预览图输出目录：左原图、右 mask 叠加的两栏对照，"
                             "存 <stem>_preview.png")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PKG_ROOT / path)


def collect_images(source: Path, exclude: tuple[Path, ...] = ()) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不是支持的图片格式: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"源不存在: {source}")

    skip = tuple(p.resolve() for p in exclude if p is not None)

    def blocked(path: Path) -> bool:
        return any(path.is_relative_to(item) for item in skip)

    files = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        and not p.name.startswith(".") and not blocked(p)
    )
    if not files:
        raise FileNotFoundError(f"目录下没有图片: {source}")
    return files


def load_id_to_name(path: Path) -> dict[int, str]:
    """labels.txt 行号 = 类别 ID；ignore/ignored/void 固定 255。"""
    mapping: dict[int, str] = {}
    next_id = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if name.lower() in {"ignore", "ignored", "void"}:
            mapping[IGNORE] = name
            continue
        mapping[next_id] = name
        next_id += 1
    if not mapping:
        raise ValueError(f"labels 文件是空的: {path}")
    return mapping


def class_names(args: argparse.Namespace, model_names: dict | None) -> dict[int, str]:
    """优先用模型的 names；给了 --labels 时校验两边一致。"""
    from_model = {int(cid): str(name).lower()
                  for cid, name in (model_names or {}).items()}
    if args.labels is None:
        if not from_model:
            raise ValueError("模型没有 names，请用 --labels 指定 labels.txt")
        return from_model

    if not args.labels.is_file():
        raise FileNotFoundError(f"labels 文件不存在: {args.labels}")
    from_file = load_id_to_name(args.labels)
    if from_model:
        mismatch = [
            (cid, from_model.get(cid), name)
            for cid, name in from_file.items()
            if cid != IGNORE and from_model.get(cid) != name.lower()
        ]
        if mismatch:
            raise ValueError(f"--labels 和模型 names 不一致: {mismatch[:3]}")
    return {cid: name for cid, name in from_file.items() if cid != IGNORE}


def aspect_crop_box(height: int, width: int, target_h: int, target_w: int) -> tuple[int, int, int, int]:
    target_ratio = target_w / target_h
    source_ratio = width / height
    if abs(source_ratio - target_ratio) < 1e-6:
        return 0, 0, height, width
    if source_ratio > target_ratio:
        new_w = round(height * target_ratio)
        return 0, (width - new_w) // 2, height, new_w
    new_h = round(width / target_ratio)
    return (height - new_h) // 2, 0, new_h, width


def crop_image(rgb: np.ndarray, crop_hw: tuple[int, int] | None):
    height, width = rgb.shape[:2]
    if crop_hw is None:
        return rgb, (0, 0, height, width)
    top, left, new_h, new_w = aspect_crop_box(height, width, *crop_hw)
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"裁切窗口非法: {height}×{width} -> {new_h}×{new_w}")
    return rgb[top:top + new_h, left:left + new_w], (top, left, new_h, new_w)


def fit_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.NEAREST)
    )


def mask_to_shapes(mask: np.ndarray, classes: dict[int, str],
                   min_area: float, epsilon: float):
    """把整帧类别 mask 转成 labelme polygon shapes。"""
    shapes = []
    dropped = 0
    for cid, label in sorted(classes.items()):
        if cid in (0, IGNORE):
            continue
        binary = (mask == cid).astype(np.uint8)
        if not binary.any():
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                dropped += 1
                continue
            if epsilon > 0:
                contour = cv2.approxPolyDP(
                    contour, epsilon * cv2.arcLength(contour, True), True
                )
            points = contour.reshape(-1, 2)
            if len(points) < 3:
                dropped += 1
                continue
            shapes.append({
                "label": label,
                "points": [[float(x), float(y)] for x, y in points],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            })
    return shapes, dropped


def preview_colors(classes: dict[int, str]) -> dict[int, np.ndarray]:
    """每个类别一个预览色；已知类固定，其余按出现顺序取后备色。"""
    colors: dict[int, np.ndarray] = {}
    fallback = 0
    for cid, name in sorted(classes.items()):
        if cid in (0, IGNORE):
            continue
        rgb = CLASS_RGB.get(name.lower())
        if rgb is None:
            rgb = FALLBACK_RGB[fallback % len(FALLBACK_RGB)]
            fallback += 1
        colors[cid] = np.array(rgb, dtype=np.uint8)
    return colors


def draw_crop_box(rgb: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    """用青色框标出模型实际看过的区域；框就是整图时不画。"""
    if box is None:
        return rgb
    top, left, height, width = box
    if (top, left, height, width) == (0, 0, rgb.shape[0], rgb.shape[1]):
        return rgb
    out = rgb.copy()
    bottom, right = top + height, left + width
    out[top:top + 2, left:right] = CROP_BOX_RGB
    out[bottom - 2:bottom, left:right] = CROP_BOX_RGB
    out[top:bottom, left:left + 2] = CROP_BOX_RGB
    out[top:bottom, right - 2:right] = CROP_BOX_RGB
    return out


def blend_mask(rgb: np.ndarray, mask: np.ndarray,
               colors: dict[int, np.ndarray]) -> np.ndarray:
    """把预测 mask 半透明叠到图上。

    细小的 glare 光斑只靠半透明填充看不出来，所以沿外轮廓再描 1 像素纯色线。
    框外的 255（ignore）不着色。
    """
    out = rgb.copy()
    for cid, color in sorted(colors.items()):
        region = mask == cid
        if not region.any():
            continue
        base = out[region].astype(np.float32)
        out[region] = (base * (1 - PREVIEW_ALPHA)
                       + color.astype(np.float32) * PREVIEW_ALPHA).astype(np.uint8)
        contours, _ = cv2.findContours(region.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(v) for v in color), 1)
    return out


def compose_preview(left: np.ndarray, right: np.ndarray,
                    gap: int = 2, fill: int = 255) -> np.ndarray:
    """左右两栏横向拼接，中间留白线分隔。"""
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + gap + right.shape[1]
    out = np.full((height, width, 3), fill, dtype=np.uint8)
    out[:left.shape[0], :left.shape[1]] = left
    out[:right.shape[0], left.shape[1] + gap:] = right
    return out


def render_preview(rgb: np.ndarray, mask: np.ndarray,
                   colors: dict[int, np.ndarray],
                   box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """左栏原图、右栏预测叠加，方便逐像素对照。"""
    original = draw_crop_box(rgb, box)
    overlay = draw_crop_box(blend_mask(rgb, mask, colors), box)
    return compose_preview(original, overlay)


def image_relpath(image_path: Path, out_dir: Path) -> str:
    try:
        rel = os.path.relpath(image_path, out_dir)
    except ValueError:
        rel = str(image_path)
    return rel.replace("\\", "/")


def main() -> None:
    args = parse_args()

    weights = resolve(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在: {weights}")

    source = resolve(args.images)
    out_dir = resolve(args.output) if args.output is not None else None
    save_mask = args.save_mask or args.mask_dir is not None
    mask_dir = None
    if save_mask:
        if args.mask_dir is not None:
            mask_dir = resolve(args.mask_dir)
        elif out_dir is not None:
            mask_dir = out_dir.parent / f"{out_dir.name}_masks"
        else:
            raise ValueError("没有 -o 时，--save-mask 需要用 -m 指定 mask 目录")
    preview_dir = resolve(args.preview) if args.preview is not None else None
    if preview_dir is not None and preview_dir == mask_dir:
        raise ValueError("-p 预览目录不能和 -m mask 目录相同")
    if out_dir is None and mask_dir is None and preview_dir is None:
        raise ValueError("至少指定一个输出：-o 写 JSON、-m 写预测 mask、-p 写预览图")

    crop_hw: tuple[int, int] | None = None
    if not args.full_frame:
        parts = [int(v) for v in args.crop_hw.replace(" ", "").split(",") if v]
        if len(parts) != 2:
            raise ValueError(f"--crop-hw 要写成 H,W，收到: {args.crop_hw}")
        crop_hw = (parts[0], parts[1])
        if args.imgsz != crop_hw[1]:
            print(f"[提示] imgsz={args.imgsz} 与 crop 宽 {crop_hw[1]} 不一致，"
                  f"入网时会被缩放，和部署对不上")

    images_root = source if source.is_dir() else source.parent
    images = collect_images(source, exclude=(out_dir, mask_dir, preview_dir))
    if args.limit > 0:
        images = images[:args.limit]

    sys.path.insert(0, str(PKG_ROOT))
    from ultralytics import YOLO

    model = YOLO(str(weights))
    classes = class_names(args, getattr(model, "names", None))
    colors = preview_colors(classes)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"weights : {weights}")
    print(f"images  : {source}")
    print(f"output  : {out_dir if out_dir is not None else '不写 JSON'}")
    print(f"masks   : {mask_dir if mask_dir is not None else '不保存'}")
    print(f"preview : {preview_dir if preview_dir is not None else '不保存'}")
    print(f"crop    : {'整图（--full-frame）' if crop_hw is None else f'{crop_hw[0]}×{crop_hw[1]}'}"
          f"  imgsz={args.imgsz}  batch={args.batch}")
    print("classes : " + "  ".join(f"{cid}={name}" for cid, name in sorted(classes.items())))
    print(f"min_area={args.min_area}  epsilon={args.epsilon}\n")

    written = skipped = empty = mask_written = previewed = 0
    dropped_total = 0
    unknown_values: set[int] = set()
    shape_counts: dict[str, int] = {name: 0 for cid, name in classes.items() if cid}
    bad: list[dict] = []

    step = max(1, args.batch)
    for start in range(0, len(images), step):
        chunk = images[start:start + step]
        pending = []
        for path in chunk:
            rel = path.relative_to(images_root) if path.is_relative_to(images_root) else Path(path.name)
            json_path = (out_dir / rel.parent / f"{rel.stem}.json"
                         if out_dir is not None else None)
            if json_path is not None and json_path.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                with Image.open(path) as handle:
                    rgb = np.asarray(handle.convert("RGB"))
                crop, box = crop_image(rgb, crop_hw)
            except Exception as exc:  # noqa: BLE001 - 单张坏图不中断整批
                bad.append({"file": rel.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
                print(f"  [skip] {rel.as_posix():<36} 读图 {type(exc).__name__}: {exc}")
                continue
            pending.append((path, rel, json_path, rgb, crop, box))

        if not pending:
            continue

        inputs = [np.ascontiguousarray(crop[:, :, ::-1]) for _, _, _, _, crop, _ in pending]
        results = model.predict(source=inputs, imgsz=args.imgsz, rect=True,
                                batch=step, device=args.device, verbose=False)

        for (path, rel, json_path, rgb, crop, box), result in zip(pending, results):
            data = getattr(result, "semantic_mask", None)
            if data is None:
                bad.append({"file": rel.as_posix(), "error": "没有 semantic_mask 输出"})
                print(f"  [skip] {rel.as_posix():<36} 推理 没有 semantic_mask 输出")
                continue

            tensor = data.data if hasattr(data, "data") else data
            mask = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)
            if mask.ndim == 3:
                mask = mask[0]
            crop_h, crop_w = crop.shape[:2]
            labels = np.clip(fit_mask(mask, crop_h, crop_w), 0, IGNORE).astype(np.uint8)

            height, width = rgb.shape[:2]
            top, left, box_h, box_w = box
            full_mask = np.full((height, width), IGNORE, dtype=np.uint8)
            full_mask[top:top + box_h, left:left + box_w] = labels

            unknown = sorted({int(v) for v in np.unique(full_mask)}
                             - set(classes) - {0, IGNORE})
            if unknown:
                unknown_values.update(unknown)
                print(f"  [warn] {rel.as_posix()} 有未定义类别 {unknown}，已跳过")

            shapes, dropped = mask_to_shapes(full_mask, classes, args.min_area, args.epsilon)
            dropped_total += dropped
            if not shapes:
                empty += 1

            if json_path is not None and (shapes or not args.skip_empty):
                payload = {
                    "version": LABELME_VERSION,
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": image_relpath(path, out_dir),
                    "imageData": None,
                    "imageHeight": int(height),
                    "imageWidth": int(width),
                }
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                written += 1
                for shape in shapes:
                    shape_counts[shape["label"]] = shape_counts.get(shape["label"], 0) + 1

            if mask_dir is not None:
                mask_path = mask_dir / rel.parent / f"{rel.stem}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(full_mask).save(mask_path)
                mask_written += 1

            if preview_dir is not None:
                preview_path = preview_dir / rel.parent / f"{rel.stem}_preview.png"
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(render_preview(rgb, full_mask, colors, box)).save(preview_path)
                previewed += 1

            print(f"  {rel.as_posix():<40} shapes={len(shapes)}")

    summary = f"\n图片 {len(images)} 张"
    if out_dir is not None:
        summary += f"，写出 JSON {written}，跳过已存在 {skipped}，空标注 {empty}"
    summary += f"，坏文件 {len(bad)}，丢弃小块 {dropped_total}"
    print(summary)
    if any(shape_counts.values()):
        print("逐类 shape: " + "  ".join(f"{name}={n}" for name, n in shape_counts.items() if n))
    if unknown_values:
        print(f"未定义类别值: {sorted(unknown_values)}（检查 --labels 和模型是否一致）")
    if mask_dir is not None:
        print(f"预测 mask -> {mask_dir}（{mask_written} 张）")
    if preview_dir is not None:
        print(f"预览图 -> {preview_dir}（{previewed} 张）")
    if bad:
        print("\n坏文件:")
        for item in bad:
            print(f"  {item['file']:<46} {item['error']}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}")
        raise SystemExit(1)
