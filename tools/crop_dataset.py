"""
在训练时的 Dataset 里把图裁到 640×384，**不生成任何派生数据目录。
在 data yaml 里写一行 `crop_hw: [384, 640]`，然后在构造模型前调一次 `install()`：
    from tools.crop_dataset import install
    install()
    model = YOLO(weights)
"""

from __future__ import annotations

import numpy as np
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import SemanticDataset

#: 默认裁切目标 (H, W)，与部署输入一致
DEFAULT_CROP_HW = (384, 640)


def aspect_crop_box(height: int, width: int, target_h: int, target_w: int) -> tuple[int, int, int, int]:
    """按目标宽高比算居中裁切窗口，返回 (top, left, height, width)。

    ``height`` / ``width`` 是**源**的像素数，目标宽高来自 ``(target_h, target_w)``。
    """
    target_ratio = target_w / target_h   # 宽/高
    source_ratio = width / height
    if abs(source_ratio - target_ratio) < 1e-6:
        return 0, 0, height, width
    if source_ratio > target_ratio:      # 太宽，裁左右
        new_w = round(height * target_ratio)
        left = (width - new_w) // 2
        return 0, left, height, new_w
    new_h = round(width / target_ratio)  # 太扁，裁上下
    top = (height - new_h) // 2
    return top, 0, new_h, width


def aspect_crop(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """HWC 图像和 HW mask 共用这一个函数——两者必须同一刀。"""
    height, width = array.shape[:2]
    top, left, new_h, new_w = aspect_crop_box(height, width, target_h, target_w)
    if (new_h, new_w) == (height, width):
        return array
    return array[top:top + new_h, left:left + new_w]


class CropSemanticDataset(SemanticDataset):
    """按目标宽高比居中裁切的语义分割 dataset。"""

    def __init__(self, *args, data: dict, **kwargs):
        crop = data.get("crop_hw")
        self.crop_hw = (int(crop[0]), int(crop[1])) if crop else None
        super().__init__(*args, data=data, **kwargs)

    def _crop(self, array: np.ndarray) -> np.ndarray:
        if self.crop_hw is None:
            return array
        return aspect_crop(array, *self.crop_hw)

    def load_image(self, i: int, rect_mode: bool = True):
        im, _, _ = BaseDataset.load_image(self, i, rect_mode=rect_mode, resize_short=False)
        im = self._crop(im)
        height, width = im.shape[:2]
        return im, (height, width), (height, width)

    def load_mask(self, index: int, image_shape=None) -> np.ndarray:
        return self._crop(super().load_mask(index, image_shape))

    def set_rectangle(self) -> None:
        if self.crop_hw is not None:
            for label in self.labels:
                shape = label.get("shape")
                if shape is not None:
                    label["shape"] = (aspect_crop_box(shape[0], shape[1], *self.crop_hw)[2:])
        super().set_rectangle()


def install() -> type:
    """把 Ultralytics 的语义分割 dataset 换成裁切版，返回替换后的类。"""
    from ultralytics.data import build as build_module

    build_module.SemanticDataset = CropSemanticDataset
    assert build_module.SemanticDataset is CropSemanticDataset, "dataset 替换失败"
    return CropSemanticDataset
