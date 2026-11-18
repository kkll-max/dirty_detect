"""dataset 六类语义分割训练入口（YOLO26n-sem）。

数据保持原始分辨率，384×640 的居中裁切在 Dataset 内完成
（`tools/crop_dataset.py`），不生成派生数据目录。

用法（在 `code_clean/` 下执行）：

    python tools/train.py --config configs/train.yaml --check
    python tools/train.py --config configs/train.yaml --check --smoke
    python tools/train.py --config configs/train.yaml
    python tools/train.py --config configs/train.yaml --batch 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from packaging.version import Version
from PIL import Image

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PKG_ROOT / "configs/train.yaml"
DEFAULT_DATA = "configs/data.yaml"
DEFAULT_NAME = "dataset_6cls"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MIN_ULTRALYTICS = "8.4.52"
AUX_MASK_SUFFIX = "_pred"
CONTROL_KEYS = {"model", "data", "project", "name"}  # 由脚本单独解析


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="dataset5_2 六类语义分割训练入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", nargs="?", const="", default=None, metavar="LAST_PT")
    parser.add_argument("--check", action="store_true", help="只校验数据和裁切，不训练")
    parser.add_argument("--smoke", action="store_true", help="真构建一个 batch 看形状")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PKG_ROOT / path


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} 里不是映射结构")
    return data


def resolve_model(value: str) -> str:
    local = resolve(value)
    return str(local) if local.exists() else value


def config_name(value: object) -> str:
    """name 必须是字符串；YAML 会把 `4_2` 解析成整数 42。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"name 必须是字符串，当前解析成 {value!r}（{type(value).__name__}）。"
            "YAML 会把 4_2 当成数字，请写成 name: \"4_2\"。"
        )
    return value.strip()


def mask_dir_for(image_dir: Path, masks_dir: str) -> Path:
    parts = image_dir.parts
    if "images" not in parts:
        raise ValueError(f"图片目录必须含 'images' 这一段: {image_dir}")
    index = len(parts) - 1 - parts[::-1].index("images")
    return Path(*parts[:index], masks_dir, *parts[index + 1:])


def class_names(data: dict) -> dict[int, str]:
    names = data.get("names")
    if not isinstance(names, dict) or not names:
        raise ValueError("数据 yaml 必须定义非空的 names 映射")
    try:
        mapping = {int(key): str(value) for key, value in names.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"names 的键必须是整数类别 ID: {names}") from exc
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError(f"names 的类别 ID 必须从 0 连续编号，当前为 {sorted(mapping)}")
    return mapping


def drop_stale_cache(mask_dir: Path, split: str, expected: int) -> None:
    """删掉样本数对不上的 *.cache。

    Ultralytics 的缓存只按自己的 hash 校验；用 fraction 做过短跑后，可能留下
    只含少量样本的缓存，正式训练就会少读图片。条数不等于图片数时直接删掉重建。
    """
    for candidate in (mask_dir.parent / f"{split}.cache", mask_dir / f"{split}.cache"):
        if not candidate.is_file():
            continue
        try:
            count = len(np.load(candidate, allow_pickle=True).item().get("labels", []))
        except Exception:
            count = -1
        if count != expected:
            candidate.unlink()
            print(f"       删除不完整的缓存 {candidate.name}（{count} 条 != {expected} 张），会重建")


def validate_data(data_path: Path) -> tuple[dict, Path]:
    from tools.crop_dataset import aspect_crop_box

    data = load_yaml(data_path)
    root = Path(data.get("path", data_path.parent))
    root = root if root.is_absolute() else (PKG_ROOT / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")

    masks_dir = str(data.get("masks_dir", "masks"))
    id_to_name = class_names(data)
    num_classes = len(id_to_name)
    if "nc" in data and int(data["nc"]) != num_classes:
        raise ValueError(f"nc={data['nc']} 与 names 长度 {num_classes} 不一致")

    crop = data.get("crop_hw")
    crop_hw = (int(crop[0]), int(crop[1])) if crop else None
    print(f"root    : {root}")
    print(f"classes : {num_classes} "
          + "  ".join(f"{cid}={name}" for cid, name in id_to_name.items()))
    print(f"crop    : {crop_hw if crop_hw else '不裁'}")

    allowed = set(range(num_classes)) | {255}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if not value:
            if split in ("train", "val"):
                raise ValueError(f"数据 yaml 缺少 '{split}'")
            print(f"{split:5s}: 未配置")
            continue
        image_dir = Path(value if isinstance(value, str) else value[0])
        image_dir = image_dir if image_dir.is_absolute() else root / image_dir
        if not image_dir.is_dir():
            raise FileNotFoundError(f"{split}: 图片目录不存在: {image_dir}")
        mask_dir = mask_dir_for(image_dir, masks_dir)
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"{split}: mask 目录不存在: {mask_dir}")

        image_paths = {p.stem: p for p in image_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        images = set(image_paths)
        all_masks = {p.stem for p in mask_dir.glob("*.png")}
        aux_masks = {stem for stem in all_masks if stem.endswith(AUX_MASK_SUFFIX)}
        masks = all_masks - aux_masks
        if not images:
            raise RuntimeError(f"{split}: {image_dir} 下没有图片")
        drop_stale_cache(mask_dir, split, len(images))
        missing = sorted(images - masks)
        extra = sorted(masks - images)
        if missing:
            raise FileNotFoundError(f"{split}: 缺 {len(missing)} 个 mask，如 {missing[:3]}")
        if extra:
            raise RuntimeError(f"{split}: {len(extra)} 个 mask 没有对应图片，如 {extra[:3]}")

        # 图片和 mask 必须同尺寸，否则裁切窗口会错位
        size_bad, value_bad, ignored = [], [], 0
        class_images = [0] * num_classes
        class_pixels = [0] * num_classes
        total_pixels = 0
        shapes = {}
        for stem in sorted(images):
            with Image.open(image_paths[stem]) as handle:
                isize = handle.size
            with Image.open(mask_dir / f"{stem}.png") as handle:
                msize = handle.size
                mask = np.asarray(handle)
            if isize != msize:
                size_bad.append(stem)
                continue
            values, counts = np.unique(mask, return_counts=True)
            values = [int(v) for v in values]
            if not set(values) <= allowed:
                value_bad.append(f"{stem}:{values}")
                continue
            total_pixels += int(mask.size)
            for value, count in zip(values, counts):
                if value < num_classes:
                    class_pixels[value] += int(count)
                    class_images[value] += 1
            if 255 in values:
                ignored += 1
            key = (isize[0], isize[1])
            if crop_hw:
                box = aspect_crop_box(isize[1], isize[0], *crop_hw)
                key = (box[3], box[2])
            shapes[key] = shapes.get(key, 0) + 1
        if size_bad:
            raise ValueError(f"{split}: {len(size_bad)} 组图文尺寸不一致，如 {size_bad[:3]}")
        if value_bad:
            raise ValueError(
                f"{split}: {len(value_bad)} 张 mask 取值超出 0..{num_classes - 1} 或 255，"
                f"如 {value_bad[:3]}"
            )

        detail = ", ".join(f"{w}×{h}:{n}"
                           for (w, h), n in sorted(shapes.items(), key=lambda kv: -kv[1]))
        target = f"-> 入网 {crop_hw[0]}×{crop_hw[1]} (H×W)" if crop_hw else ""
        print(f"{split:5s}: {len(images)} 张, 裁切窗口 {detail} (W×H) {target}")
        stats = []
        for cid, name in id_to_name.items():
            share = class_pixels[cid] / total_pixels if total_pixels else 0.0
            stats.append(f"{name}={class_images[cid]}张/{share:.2%}px")
        print("       " + "  ".join(stats))
        if ignored:
            print(f"       ignore: {ignored} 张含 255")
        if aux_masks:
            print(f"       忽略 {len(aux_masks)} 个预标注 *_pred.png")

    return data, root


def write_resolved_data(data: dict, root: Path) -> Path:
    """Ultralytics 对相对路径会去 DATASETS_DIR 下找，这里落一份绝对路径快照。"""
    path = PKG_ROOT / "runs/_data_resolved.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({**data, "path": str(root)}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def build_overrides(cfg: dict, args: argparse.Namespace, data_path: Path) -> dict:
    # 除控制键外，配置里的训练参数原样透传；新增参数不用再改这里。
    overrides = {key: value for key, value in cfg.items() if key not in CONTROL_KEYS}
    overrides["task"] = "semantic"
    overrides["model"] = resolve_model(args.model or cfg.get("model", "yolo26n-sem.pt"))
    overrides["data"] = str(data_path)
    overrides["project"] = str(resolve(args.project or cfg.get("project", "runs")))
    overrides["name"] = config_name(
        args.name if args.name is not None else cfg.get("name", DEFAULT_NAME)
    )
    if "exist_ok" not in overrides:
        overrides["exist_ok"] = stale_run_dir(Path(overrides["project"]), overrides["name"])
    for key in ("epochs", "imgsz", "batch", "workers", "device"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    return overrides


def check_crop_imgsz(data: dict, overrides: dict) -> None:
    """crop 只在 `imgsz == crop_hw[1]` 时是纯裁切，否则训练口径和部署不一致。"""
    crop = data.get("crop_hw")
    if not crop:
        return
    target_h, target_w = int(crop[0]), int(crop[1])
    imgsz = overrides.get("imgsz", target_w)
    imgsz = max(imgsz) if isinstance(imgsz, (list, tuple)) else int(imgsz)
    if imgsz != target_w:
        raise ValueError(
            f"imgsz={imgsz} 与 crop_hw 的宽 {target_w} 不一致。crop 只有在入网宽度等于 "
            f"{target_w} 时才是纯裁切（{target_w}×{target_h}）；改成 {imgsz} 会把源图先降采样，"
            f"裁出来的高也不再是 32 的倍数，LetterBox 会补边，跟部署对不上。"
            f"要改分辨率就连 crop_hw 一起改，别只覆盖 --imgsz。"
        )


def stale_run_dir(project: Path, name: str) -> bool:
    """上一次跑崩了留下空目录时复用，跑成功过就让它自增一个新目录。"""
    run_dir = project / name
    if not run_dir.exists():
        return False
    if any(run_dir.glob("weights/*.pt")) or (run_dir / "results.csv").exists():
        return False
    print(f"[注意] {run_dir} 是上次失败的残留（没有权重也没有 results.csv），本次复用该目录")
    return True


def check_ultralytics() -> None:
    import ultralytics

    print(f"ultralytics: {ultralytics.__version__}")
    if Version(ultralytics.__version__) < Version(MIN_ULTRALYTICS):
        raise RuntimeError(f"YOLO26-sem 需要 ultralytics >= {MIN_ULTRALYTICS}")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(PKG_ROOT))
    from tools.crop_dataset import install

    config_path = resolve(args.config)
    cfg = load_yaml(config_path)
    data_path = resolve(args.data or cfg.get("data", DEFAULT_DATA))
    data, root = validate_data(data_path)

    install()  # 先替换 Dataset，再让 Ultralytics 建数据
    resolved_data = write_resolved_data(data, root)
    check_ultralytics()

    if args.resume is not None:
        resume(args, cfg, resolved_data)
        return

    overrides = build_overrides(cfg, args, resolved_data)
    check_crop_imgsz(data, overrides)
    print(f"config : {config_path}")
    print(f"model  : {overrides['model']}")
    print(f"epochs : {overrides.get('epochs')}  imgsz={overrides.get('imgsz')}  "
          f"rect={overrides.get('rect')}  batch={overrides.get('batch')}  "
          f"device={overrides.get('device')}")
    print(f"output : {overrides['project']}/{overrides['name']}")

    if args.check:
        if args.smoke:
            smoke(overrides, resolved_data)
        print("校验通过，未开始训练")
        return

    from ultralytics import YOLO

    print("开始训练 ...")
    YOLO(overrides.pop("model")).train(**overrides)


def resume(args: argparse.Namespace, cfg: dict, data_path: Path) -> None:
    project = resolve(args.project or cfg.get("project", "runs"))
    name = config_name(
        args.name if args.name is not None else cfg.get("name", DEFAULT_NAME)
    )
    last = resolve(args.resume) if args.resume else project / name / "weights/last.pt"
    if not last.is_file():
        raise FileNotFoundError(f"续训权重不存在: {last}")
    print(f"resume : {last}")
    if args.check:
        print("校验通过，未开始训练")
        return

    from ultralytics import YOLO

    print("开始续训 ...")
    YOLO(str(last)).train(resume=True, data=str(data_path))


def smoke(overrides: dict, data_path: Path) -> None:
    """真建一个 batch，确认裁切后的形状是 384×640。"""
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_path))
    base = {
        "task": "semantic", "imgsz": int(overrides.get("imgsz", 640)),
        "rect": bool(overrides.get("rect", True)), "cache": False,
        "single_cls": False, "classes": None, "fraction": 1.0, "split": "val",
    }

    def build(mode: str):
        cfg = get_cfg(DEFAULT_CFG, {**base, "mosaic": float(overrides.get("mosaic", 0.0)),
                                    "close_mosaic": 0})
        return build_yolo_dataset(cfg, data["train"], 2, data, mode=mode)

    dataset = build("val")
    print(f"dataset class: {type(dataset).__name__}  crop_hw={getattr(dataset, 'crop_hw', None)}")
    sample = dataset[0]
    print(f"smoke(val)  : img={tuple(sample['img'].shape)}  "
          f"mask={tuple(sample['semantic_mask'].shape)}  "
          f"classes={sorted(int(v) for v in np.unique(sample['semantic_mask'].numpy()))}")

    # train 模式下 RandomPerspective 的 translate/scale 会用 255 填掩膜边界，
    # 那是 ignore 索引（loss 会跳过），不是数据脏。这里只看形状。
    sample = build("train")[0]
    print(f"smoke(train): img={tuple(sample['img'].shape)}  "
          f"mask={tuple(sample['semantic_mask'].shape)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}")
        raise SystemExit(1)
