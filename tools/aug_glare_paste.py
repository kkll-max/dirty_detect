"""把 guangban2 里已标注的 glare patch 合成到 clean 帧上，生成离线增强样本。
    python tools/aug_glare_paste.py --dry-run
    python tools/aug_glare_paste.py --preview
    python tools/aug_glare_paste.py --max-samples 100 --force
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from tools.labelme2mask import (  # noqa: E402
    build_mask,
    load_label_map,
    preview_colors,
    render_preview,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class Patch:
    """从 donor 上抠下来的一块眩光。"""

    donor: str
    rgb: np.ndarray        # 含 padding 的 BGR patch
    core: np.ndarray       # bool，眩光核心像素
    alpha: np.ndarray      # float32 0..1，边缘羽化
    area_px: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="glare patch 离线合成：guangban2 的眩光贴到 clean 帧",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--images", type=Path,
                        default=PKG_ROOT / "data/guangban2/images",
                        help="donor/receiver 原图目录")
    parser.add_argument("-o", "--annotations", type=Path,
                        default=PKG_ROOT / "data/guangban2/annotations",
                        help="donor/receiver 的 labelme JSON 目录")
    parser.add_argument("--out", type=Path,
                        default=PKG_ROOT / "data/guangban2_syn",
                        help="输出根目录（images/masks/annotations 三件套）")
    parser.add_argument("--labels", type=Path, default=None,
                        help="标签表；默认按 annotations 布局自动找，找不到用内置 6 类")
    parser.add_argument("--receivers", choices=("clean", "all"), default="clean",
                        help="clean=只用空 JSON 的帧；all=任意帧（原标注保留）")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="最多合成多少张；0 = 用尽所有 receiver（每帧一次）")
    parser.add_argument("--min-area", type=int, default=80,
                        help="小于该像素数的眩光连通域不抠，默认 80")
    parser.add_argument("--pad", type=int, default=6,
                        help="patch 外扩像素，默认 6")
    parser.add_argument("--feather", type=int, default=4,
                        help="边缘羽化宽度（像素），默认 4")
    parser.add_argument("--soften", type=float, default=2.0,
                        help="对光源 patch 做高斯软化（sigma 像素），避免 screen 后"
                             "出现硬边白块，默认 2")
    parser.add_argument("--gain-jitter", type=float, default=0.1,
                        help="亮度增益随机幅度，1±该值，默认 0.1")
    parser.add_argument("--gate", type=float, default=180.0,
                        help="亮度门限：只有灰度高于该值的像素算作'光'，既是合成权重，"
                             "也决定贴过去后 mask 标哪块（避免把 donor 的楼/树 ghost 过去），"
                             "默认 180")
    parser.add_argument("--gate-soft", type=float, default=35.0,
                        help="门限的软过渡宽度，默认 35")
    parser.add_argument("--min-delta", type=float, default=12.0,
                        help="mask 只标合成后灰度提升 >= 该值的像素（画面必须真的变亮），"
                             "默认 12")
    parser.add_argument("--tries", type=int, default=8,
                        help="每张图随机试多少个位置，取亮度余量最大的，默认 8")
    parser.add_argument("--min-mask-px", type=int, default=40,
                        help="可见眩光小于该像素数就跳过这张，默认 40")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true",
                        help="额外输出 <out>_preview/，mask 50% 叠在合成图上")
    parser.add_argument("--force", action="store_true",
                        help="输出目录非空时先清空")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计 donor/receiver，不写文件")
    return parser.parse_args()


def image_stems(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def extract_patches(stem: str, image_path: Path, mask: np.ndarray,
                    glare_id: int, args: argparse.Namespace) -> list[Patch]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"读不了原图: {image_path}")
    # 只取"看得见的光"：glare 标注里经常还包着树枝、楼面色块，
    # 直接把整块搬过去会 ghost，所以按亮度再切一刀
    lum = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    glare = ((mask == glare_id) & (lum >= args.gate)).astype(np.uint8)
    if not glare.any():
        return []
    height, width = glare.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(glare, 8)
    patches: list[Patch] = []
    for index in range(1, num):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < args.min_area:
            continue
        x, y, w, h = (int(stats[index, cv2.CC_STAT_LEFT]),
                      int(stats[index, cv2.CC_STAT_TOP]),
                      int(stats[index, cv2.CC_STAT_WIDTH]),
                      int(stats[index, cv2.CC_STAT_HEIGHT]))
        pad = max(args.pad, args.feather + 2)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        core = labels[y0:y1, x0:x1] == index
        rgb = image[y0:y1, x0:x1].copy()
        # 核心内 alpha=1，核心外按距离在 feather 像素内渐隐
        dist_out = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(1.0 - dist_out / max(1.0, float(args.feather)), 0.0, 1.0)
        patches.append(Patch(stem, rgb, core,
                             alpha.astype(np.float32), area))
    return patches


def choose_placement(receiver: np.ndarray, patch: Patch, rng: random.Random,
                     tries: int) -> tuple[int, int]:
    """在画面上半部随机试几个位置，挑"最暗"的那个贴。

    眩光只能在有余量的地方显出来：把光贴在本来就 250+ 的天空上，screen
    混出来几乎不变，mask 却标着 glare——标签和画面自相矛盾。所以按
    core 区域的亮度余量（255-lum）打分，取最好的一个。
    """
    height, width = receiver.shape[:2]
    ph, pw = patch.core.shape
    if ph > height or pw > width:
        raise ValueError(f"patch {pw}×{ph} 比接收帧 {width}×{height} 还大")
    # 眩光只在画面上半部：patch 矮时限制 y，patch 本身很高时退回全高
    if ph <= height // 2:
        max_y = max(0, min(height - ph, int(height * 0.35)))
    else:
        max_y = height - ph
    lum = cv2.cvtColor(receiver, cv2.COLOR_BGR2GRAY).astype(np.float32)
    best: tuple[float, int, int] | None = None
    for _ in range(max(1, tries)):
        y = rng.randint(0, max_y) if max_y > 0 else 0
        x = rng.randint(0, max(0, width - pw))
        headroom = float(np.mean(255.0 - lum[y:y + ph, x:x + pw][patch.core]))
        if best is None or headroom > best[0]:
            best = (headroom, x, y)
    assert best is not None
    return best[1], best[2]


def place_patch(receiver: np.ndarray, mask: np.ndarray, patch: Patch,
                glare_id: int, gain: float, gate: float, gate_soft: float,
                soften: float, min_delta: float, min_mask_px: int,
                rng: random.Random, tries: int) -> tuple[int, int, int, int, int, float]:
    x, y = choose_placement(receiver, patch, rng, tries)
    ph, pw = patch.core.shape

    before = cv2.cvtColor(receiver[y:y + ph, x:x + pw],
                          cv2.COLOR_BGR2GRAY).astype(np.float32)
    dst = receiver[y:y + ph, x:x + pw].astype(np.float32)
    # 只取 patch 里的"光"：灰度高于门限的部分才参与合成，
    # 否则 donor 画面里的楼/树会被 screen 成一圈半透明 ghost
    lum = cv2.cvtColor(patch.rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    weight = np.clip((lum - gate) / max(1e-3, gate_soft), 0.0, 1.0)[..., None]
    src = np.clip(patch.rgb.astype(np.float32) * weight * gain, 0.0, 255.0)
    if soften > 0:
        src = cv2.GaussianBlur(src, (0, 0), soften)
    screened = 255.0 - (255.0 - dst) * (255.0 - src) / 255.0
    alpha = patch.alpha[..., None]
    blended = dst * (1.0 - alpha) + screened * alpha
    receiver[y:y + ph, x:x + pw] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    # mask 只标"画面里真的变亮了"的部分：亮度提升不够的像素不标，
    # 免得教模型把本来就亮的区域当 glare
    after = cv2.cvtColor(receiver[y:y + ph, x:x + pw],
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
    delta = after - before
    visible = patch.core & (delta >= min_delta)
    visible_px = int(visible.sum())
    if visible_px < max(1, min_mask_px):
        raise ValueError(f"合成后可见眩光只有 {visible_px} px（< {min_mask_px}），跳过")

    mask[y:y + ph, x:x + pw][visible] = glare_id
    mean_delta = float(delta[visible].mean())
    return x, y, pw, ph, visible_px, mean_delta


def core_to_polygons(core: np.ndarray, x: int, y: int) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2) + (x, y)
        if len(approx) >= 3:
            polygons.append([[float(px), float(py)] for px, py in approx])
    return polygons


def main() -> None:
    args = parse_args()
    images_dir = args.images.resolve()
    ann_dir = args.annotations.resolve()
    out = args.out.resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"原图目录不存在: {images_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {ann_dir}")

    label_path = args.labels
    if label_path is None:
        for candidate in (ann_dir.parent / "labels.txt", ann_dir / "labels.txt"):
            if candidate.is_file():
                label_path = candidate
                break
    label_map, label_source = load_label_map(label_path)
    glare_id = label_map.get("glare")
    if glare_id is None:
        raise ValueError("labels 里没有 glare")
    id_to_name = {cid: name for name, cid in label_map.items() if cid != 255}

    images = image_stems(images_dir)
    jsons = {p.stem: p for p in sorted(ann_dir.glob("*.json"))
             if p.name != "prelabel_summary.json"}
    usable = sorted(set(images) & set(jsons))
    if not usable:
        raise SystemExit("没有图片 + JSON 都存在的样本")

    warn: Counter = Counter()
    donors: list[Patch] = []
    receivers: list[str] = []
    for stem in usable:
        data = json.loads(jsons[stem].read_text(encoding="utf-8"))
        shapes = data.get("shapes", [])
        has_glare = any(shape.get("label") == "glare" for shape in shapes)
        if has_glare:
            mask = build_mask(data, label_map, warn)
            donors.extend(extract_patches(stem, images[stem], mask, glare_id, args))
        if args.receivers == "all" or not shapes:
            receivers.append(stem)

    if not donors:
        raise SystemExit("没有可用的 glare patch（检查 JSON 里的 glare shape）")
    if not receivers:
        raise SystemExit("没有可用的 receiver（默认只收空 JSON 的 clean 帧）")
    print(f"labels   : {label_source}")
    print(f"donor    : {len({p.donor for p in donors})} 帧 / {len(donors)} 块 patch "
          f"(area>= {args.min_area})")
    print(f"receiver : {len(receivers)} 帧（{args.receivers}）")

    rng = random.Random(args.seed)
    rng.shuffle(receivers)
    count = len(receivers) if args.max_samples <= 0 else min(args.max_samples, len(receivers))
    receivers = receivers[:count]
    print(f"计划合成 : {count} 张 -> {out}")
    if args.dry_run:
        return

    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"输出目录非空: {out}（加 --force 清空）")
        shutil.rmtree(out)
    for sub in ("images", "masks", "annotations"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    preview_dir = None
    if args.preview:
        preview_dir = out.parent / f"{out.name}_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        colors = preview_colors(id_to_name)

    items: dict[str, dict] = {}
    class_pixels = Counter()
    total_pixels = 0

    for index, receiver_stem in enumerate(receivers, start=1):
        stem = f"syn_{index:04d}"
        receiver = cv2.imread(str(images[receiver_stem]), cv2.IMREAD_COLOR)
        if receiver is None:
            print(f"  [skip] 读不了 {images[receiver_stem]}")
            continue
        height, width = receiver.shape[:2]

        receiver_data = json.loads(jsons[receiver_stem].read_text(encoding="utf-8"))
        mask = build_mask(receiver_data, label_map, warn)
        if mask.shape != (height, width):
            raise ValueError(
                f"{receiver_stem}: 图片 {width}×{height} 与 JSON "
                f"{mask.shape[1]}×{mask.shape[0]} 不一致")

        patch = rng.choice(donors)
        gain = 1.0 + rng.uniform(-args.gain_jitter, args.gain_jitter)
        try:
            x, y, w, h, mask_px, mean_delta = place_patch(
                receiver, mask, patch, glare_id, gain, args.gate,
                args.gate_soft, args.soften, args.min_delta,
                args.min_mask_px, rng, args.tries)
        except ValueError as exc:
            print(f"  [skip] {receiver_stem}: {exc}")
            continue

        image_name = f"{stem}.png"
        cv2.imwrite(str(out / "images" / image_name), receiver)
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), mask)

        shapes = [dict(shape) for shape in receiver_data.get("shapes", [])]
        visible = mask[y:y + h, x:x + w] == glare_id
        for polygon in core_to_polygons(visible, x, y):
            shapes.append({
                "label": id_to_name[glare_id],
                "points": polygon,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })
        data = {
            "version": "5.0.1",
            "flags": dict(receiver_data.get("flags", {})),
            "shapes": shapes,
            "imagePath": f"..\\images\\{image_name}",
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        (out / "annotations" / f"{stem}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        for cid in id_to_name:
            class_pixels[cid] += int((mask == cid).sum())
        total_pixels += int(mask.size)

        items[stem] = {
            "donor": patch.donor,
            "receiver": receiver_stem,
            "patch_area": patch.area_px,
            "mask_area": mask_px,
            "mean_delta": round(mean_delta, 2),
            "x": x, "y": y, "w": w, "h": h,
            "gain": round(gain, 4),
        }

        if preview_dir is not None:
            Image.fromarray(
                render_preview(out / "images" / image_name, mask, colors)
            ).save(preview_dir / f"{stem}_preview.png")

        if index % 50 == 0 or index == len(receivers):
            print(f"  {index}/{len(receivers)}")

    manifest = {
        "policy": (f"glare patch 离线合成：{len(donors)} 块 patch 贴到 {args.receivers} "
                   f"接收帧，screen 混合 + 距离羽化；seed={args.seed}"),
        "labels": str(label_path) if label_path else "内置 6 类",
        "receivers": args.receivers,
        "feather": args.feather,
        "pad": args.pad,
        "min_area": args.min_area,
        "gain_jitter": args.gain_jitter,
        "gate": args.gate,
        "gate_soft": args.gate_soft,
        "soften": args.soften,
        "min_delta": args.min_delta,
        "tries": args.tries,
        "min_mask_px": args.min_mask_px,
        "items": items,
    }
    (out / "manifest_synth.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if total_pixels:
        print(f"\n合成 {len(items)} 张，像素占比: " + "  ".join(
            f"{id_to_name[cid]}={class_pixels[cid] / total_pixels:.2%}"
            for cid in sorted(id_to_name) if cid != 0))
    donor_counts = Counter(item["donor"] for item in items.values())
    print(f"复用最多的 donor: {donor_counts.most_common(3)}")
    if warn:
        print("标注提示:", dict(warn.most_common(5)))
    if preview_dir is not None:
        print(f"预览 -> {preview_dir}")
    print("输出 ->", out)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
