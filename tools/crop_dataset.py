"""在训练时的 Dataset 里把图裁到 640×384，**不生成任何派生数据目录**。

存档（`dataset1/`、`dataset2/`）保持采集原始分辨率，crop 发生在 `load_image` /
`load_mask` 里，模型看到的就是部署口径。

规则只有一条：**按目标宽高比居中裁切**。四个来源自然收敛到同一个尺寸：

| 来源 | 原生 | rect 下先缩到 | 裁切后 |
|---|---|---|---|
| caiji_1029 | 640×448 | 640×448（长边已是 640） | 384×640 |
| guangban | 640×384 | 640×384 | 384×640 |
| wood（dataset1） | 1280×960 | 640×480 | 384×640 |
| other1（dataset1） | 1920×1200 | 640×400 | 384×640 |

caiji 那一行就是部署链路的真实动作：640×448 上下各去 32 行，宽度不动
（`2class_model_get/scrips/README.md`「真机出图 640×448」）。

**为什么不直接从 1280×960 裁出 640×384**：那要丢掉 80% 的画面
（`2class_model_get/tools/README.md` §9.1 方案 C），2× 放大补不回来。裁到 5:3
只丢 20%，且无变形。

用法
----
在 data yaml 里写一行 `crop_hw: [384, 640]`，然后在构造模型前调一次 `install()`：

    from tools.crop_dataset import install
    install()
    model = YOLO(weights)

`install()` 把 `ultralytics.data.build.SemanticDataset` 换成 `CropSemanticDataset`。
训练集和验证集都由 `build_yolo_dataset` 构造，所以一处替换两边都生效。yaml 里
没有 `crop_hw` 时它退化成原生 `SemanticDataset`，行为完全不变。

为什么不是 monkeypatch 得更深
------------------------------
Ultralytics 没有官方的 dataset 注入点（`build_yolo_dataset` 里的类名是写死的），
所以只能替换模块级名字。这里只动 `data.build` 这一个引用，并在 `install()`
里立刻断言替换成功，出了版本差异会当场报错而不是静默失效。
"""

from __future__ import annotations

import numpy as np
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
    """按目标宽高比居中裁切的语义分割 dataset。

    必须定义在**模块层**：Windows 的 DataLoader worker 用 spawn 启动，dataset
    对象要 pickle 传给子进程，pickle 按 `模块名.类名` 查类。写在 `install()`
    里就是 `<locals>.CropSemanticDataset`，子进程会报
    `AttributeError: Can't get local object`。
    """

    def __init__(self, *args, data: dict, **kwargs):
        crop = data.get("crop_hw")
        self.crop_hw = (int(crop[0]), int(crop[1])) if crop else None
        super().__init__(*args, data=data, **kwargs)

    def _crop(self, array: np.ndarray) -> np.ndarray:
        if self.crop_hw is None:
            return array
        return aspect_crop(array, *self.crop_hw)

    def load_image(self, i: int, rect_mode: bool = True):
        # 故意绕开 SemanticDataset.load_image：它在 augment 时会把短边拉到 imgsz，
        # 640×448 会上采样到 640×896，再裁就被拉变形了。这里只按长边缩到 imgsz。
        from ultralytics.data.base import BaseDataset

        im, _, _ = BaseDataset.load_image(self, i, rect_mode=rect_mode, resize_short=False)
        im = self._crop(im)
        height, width = im.shape[:2]
        return im, (height, width), (height, width)

    def load_mask(self, index: int, image_shape=None) -> np.ndarray:
        return self._crop(super().load_mask(index, image_shape))

    def set_rectangle(self) -> None:
        # 分批用的宽高比必须按裁切后的算，否则 batch_shape 会回到 448×640，
        # LetterBox 再把 padding 加回来，crop 就白做了。
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
