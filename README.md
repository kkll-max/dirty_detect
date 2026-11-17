# 割草机镜头脏污检测 — 六类语义分割

基于 Ultralytics **YOLO26n-sem** 的单帧语义分割：输入 RGB 图像，按部署口径居中裁到
**384×640**，输出 6 类逐像素 mask。仓库包含训练/验证/推理/导出代码、dataset5_2 的
配置与训练结果；**数据集本身（约 820 MB）不纳入 Git**。

> 状态（2026-11-17）：dataset5_2 训练与逐类验证已跑通，val mIoU **0.7603**。
> 指标来自 image-level 随机划分，val 偏乐观，只用于训练监控和同口径对比，不能作为
> 最终泛化基准（见第 8 节）。

## 1. 任务与标签

| 索引 | 类别 | 含义 | 预览色 |
|---|---|---|---|
| 0 | `clean` | 干净镜面（背景） | 不叠加 |
| 1 | `dirt_dot` | 点状泥污 / 脏点 | `#a52a2a` |
| 2 | `glare` | 强光 / 眩光（阳光直射镜头） | `#ffff00` |
| 3 | `water` | 水渍 / 水滴 / 水膜 | `#0000ff` |
| 4 | `grass_residue` | 草屑残留 | `#00c800` |
| 5 | `mud_film` | 片状泥膜 | `#b47828` |

`glare` 不是污渍，但必须单独建类：阳光直射产生的高亮区域在灰度上很容易被误判成
脏污，显式给负类才能把"强光"和"脏"分开。其余五类按压在镜头上的材质区分，不是
透明度等级。预览配色与部署侧 `SEG_PALETTE_RGB` 一致。

## 2. 仓库内容

```text
.
├── configs/
│   ├── data_dataset5_2.yaml     # 数据路径、crop_hw、类别名
│   └── train_dataset5_2.yaml    # 模型、优化器、增强、早停、输出
├── tools/
│   ├── train.py                 # 训练入口（--check / --smoke / --resume）
│   ├── val.py                   # 逐类 IoU 验证
│   ├── prelabel.py              # 模型生成 labelme JSON / mask / 预览
│   ├── annotation_2_mask.py     # labelme JSON → 多类别 mask
│   ├── aug_glare_paste.py       # glare 离线合成（需要 guangban2 源数据）
│   ├── crop_dataset.py          # 训练时按部署口径裁切（模块，无 CLI）
│   ├── export_onnx.py           # 导出 ONNX
│   └── make_calib.py            # 生成量化校准集（NCHW float32 .raw）
├── runs/dataset5_2_6cls/        # 已提交的训练结果
│   ├── weights/best.pt          # val mIoU 最优（FP16，3.3 MB）
│   ├── weights/last.pt
│   ├── results.csv/.png
│   ├── confusion_matrix*.png
│   ├── iou_bar_chart.png
│   ├── val_batch*_{labels,pred}.jpg
│   └── args.yaml
├── requirements.txt
└── .gitignore                   # 忽略 dataset5_2、缓存和临时数据 yaml
```

不包含的内容：

- `dataset5_2/`（约 820 MB），需要单独放到仓库根目录；
- `runs/dataset4_2_6cls/weights/best.pt`（训练起点权重）；
- `make_dataset5_1.py`、`make_dataset5_2.py`、`analyze_model.py`、`predict.py`
  等数据集构建和分析脚本，在 `some_classes` 工程中。

## 3. 环境

本地验证环境：Python 3.12.15、torch 2.6.0+cu124、ultralytics 8.4.183、
numpy 2.5.4、opencv-python 4.11.0、Pillow 12.3.0、PyYAML 6.0.3。

```bat
conda create -n yolo python=3.12 -y
conda activate yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

关键约束：**`ultralytics==8.4.183`**。语义分割接口（`task: semantic`、
`SemanticDataset`、`masks_dir`）是 8.4.x 才有的，换版本可能直接不兼容。

## 4. 数据

### 4.1 目录

数据集不纳入 Git，放到仓库根目录：

```text
dataset5_2/
├── labels.txt
├── images/{train,val}/<stem>.png
├── masks/{train,val}/<stem>.png          # 单通道，像素值 = 类别 id（0..5，255=忽略）
└── annotations/{train,val}/<stem>.json   # labelme JSON
```

`labels.txt` 一行一个类名，行号即类别 id：

```text
clean
dirt_dot
glare
water
grass_residue
mud_film
```

mask 可以由 labelme JSON 现场生成（`-i` / `-m` 省略时按 `annotations` 的同级目录
推断，`--preview` 输出半透明叠加图）：

```bat
python tools\annotation_2_mask.py -i dataset5_2\images\train ^
  -o dataset5_2\annotations\train -m dataset5_2\masks\train --preview
```

### 4.2 dataset5_2 的组成

| 来源 | train | val | 合计 |
|---|---:|---:|---:|
| `dataset4_2` 原 train | 1033 | 165 | 1198 |
| `dataset4_2` 原 val | 165 | 134 | 299 |
| `guangban2` | 204 | 51 | 255 |
| `guangban2_syn`（离线合成） | 122 | 0 | 122 |
| **合计** | **1524** | **350** | **1874** |

- 真实部分的 train/val 与 `dataset5_1` 逐张相同、划分相同（seed 42），两套数据的
  **val 完全一致**，指标可以直接对比。
- 合成样本共生成 180 张，纳入 122 张：33 张的 receiver 落在 `dataset5_1` 的 val、
  25 张的 donor 落在 val，为避免场景泄漏全部过滤。
- 合成样本只进 train，stem 带前缀 `gsyn_`，`manifest_dataset5_2.json` 记录每张的
  receiver / donor / mask_area。

类别统计（同一张图可含多类，图片级数字不能相加）：

| split | 图片数 | 全 clean | dirt_dot | glare | water | grass_residue | mud_film |
|---|---:|---:|---:|---:|---:|---:|---:|
| train 图片级 | 1524 | 382 | 295 | 316 | 319 | 285 | 200 |
| val 图片级 | 350 | 118 | 91 | 61 | 118 | 95 | 62 |
| train 像素级 | — | 84.77% | 2.84% | 1.55% | 1.64% | 8.03% | 1.17% |
| val 像素级 | — | 82.74% | 2.62% | 2.10% | 1.97% | 8.43% | 2.13% |

合成样本是"clean 帧 + 真实眩光 patch"，单帧眩光面积很小（mask 中位约 620 px，
全图中位约 0.25%）。它补的是**图片级眩光检出和小面积眩光形态**，不是像素占比。

### 4.3 数据 yaml

```yaml
path: dataset5_2
train: images/train
val: images/val
masks_dir: masks
crop_hw: [384, 640]
nc: 6
names:
  0: clean
  1: dirt_dot
  2: glare
  3: water
  4: grass_residue
  5: mud_film
```

`crop_hw` 由 `tools/crop_dataset.py` 在 Dataset 内执行：先把长边缩到 640，再按 5:3
居中裁切，不生成派生数据目录。训练、验证、导出和量化校准必须用同一刀。

## 5. 训练

在仓库根目录执行：

```powershell
# 只校验配置、图片/mask 对齐和裁切
python tools\train.py --config configs\train_dataset5_2.yaml --check

# 真构造一个 batch，确认入网形状是 384×640
python tools\train.py --config configs\train_dataset5_2.yaml --check --smoke

# 正式训练，输出 runs\dataset5_2_6cls
python tools\train.py --config configs\train_dataset5_2.yaml
```

`configs/train_dataset5_2.yaml` 里的
`model: runs/dataset4_2_6cls/weights/best.pt` 是训练起点，**仓库没有收录这个权重**。
要复现训练，任选一种：

```powershell
# 1) 从 some_classes 复制 dataset4_2 的 best.pt 到配置指定的位置
# 2) 或覆盖 model
python tools\train.py --config configs\train_dataset5_2.yaml `
  --model "D:\Dirt detection\some_classes\runs\dataset4_2_6cls\weights\best.pt"
# 3) 或从公开的 yolo26n-sem 预训练权重起步
python tools\train.py --config configs\train_dataset5_2.yaml --model yolo26n-sem.pt
```

仓库里的 `runs/dataset5_2_6cls/weights/best.pt` 是训练结果，不是训练起点。

### 训练策略

- 模型：`YOLO26n-sem`，1,626,132 参数，4.4 GFLOPs，6 类输出。
- 损失：Ultralytics 语义分割默认的 **CE + Dice**；模型选择只看 **mIoU**。
- 优化器：AdamW，`lr0=2e-4`、`lrf=0.01`、余弦退火、`weight_decay=5e-4`、
  warmup 1 epoch。
- 训练：50 epoch、`patience=15`、`batch=8`、`workers=4`、AMP、seed 42。
- 入网：`imgsz=640`、`rect=true`、`crop_hw=[384,640]`。`imgsz` 必须等于
  `crop_hw[1]`，否则源图会先降采样，训练和部署口径不一致。

### 数据增强

`rect: true` 时 Ultralytics 会强制关闭 mosaic / mixup / cutmix，配置里显式写 0；
`copy_paste` 和 `erasing` 对语义分割是空操作。只保留保标签的增强：

| 参数 | 值 | 原因 |
|---|---:|---|
| `translate` | 0.15 | 太阳在画面里的位置变化大 |
| `scale` | 0.3 | 0.7×–1.3×；再大会把条纹放大成死白块 |
| `fliplr` | 0.5 | 左右翻转不改变物理含义 |
| `flipud` / `degrees` / `shear` / `perspective` | 0 | 相机水平安装，旋转/透视会造出物理上不存在的分布 |
| `hsv_h` | 0.0 | 色相扰动会把白色日轮染成彩色 |
| `hsv_s` / `hsv_v` | 0.2 / 0.2 | 只模拟曝光波动；默认 0.7/0.4 会把天空提亮成"像 glare"，mask 却不动 |

## 6. 评估结果

### 本次 run（best epoch 32，共训 47 epoch）

| 指标 | 数值 |
|---|---:|
| mIoU | **0.7603** |
| pixel acc | 0.9633 |
| best epoch 的 val CE / Dice | 0.1928 / 0.4258 |
| 训练耗时 | 约 32.4 min（41.3 s/epoch） |

val 逐类 IoU：

| clean | dirt_dot | glare | water | grass_residue | mud_film |
|---:|---:|---:|---:|---:|---:|
| 0.9642 | 0.7373 | 0.8280 | **0.4865** | 0.8958 | 0.6501 |

`water` 是最弱项（IoU 0.49），主要错在边界和薄水膜；`mud_film`（0.65）次之；
`clean` 已接近饱和。

### 与 dataset5_1 的对比

两者 val 逐张相同（350 张），在 `some_classes` 中用同一份数据分别跑两个 `best.pt`：

| 指标 | dataset5_1 | dataset5_2 |
|---|---:|---:|
| mIoU | 0.7583 | **0.7603** |
| glare IoU | 0.8121 | **0.8283** |
| glare recall / precision | 0.853 / 0.945 | **0.868 / 0.947** |
| 仅 guangban2 子集的 glare IoU | 0.387 | **0.475** |
| 全 clean 帧出现 glare 假阳 | 4 / 118 帧 | **2 / 118 帧** |

合成 glare 提升了眩光召回和 clean 帧的区分度；mIoU 的 +0.002 在单 seed 噪声内，
其余类别的变化不稳健。结论应表述为"眩光子项改善"，而不是"整体显著提升"。

## 7. 验证、推理与导出

```powershell
# 逐类 IoU（自动按 384×640 裁切，和训练口径一致）
python tools\val.py --config configs\train_dataset5_2.yaml --plots

# 推理并生成预览 / labelme JSON / 预测 mask
python tools\prelabel.py -i "D:\some\images" -p runs\preview `
  -w runs\dataset5_2_6cls\weights\best.pt

# 导出 ONNX（默认 1×3×384×640、opset 11）
python tools\export_onnx.py --weights runs\dataset5_2_6cls\weights\best.pt `
  --output runs\dataset5_2_6cls\weights\best.onnx

# 生成 RKNN / hb_mapper 量化校准集
python tools\make_calib.py --source dataset5_2\images\train --ratio 0.1 --force
```

说明：

- **不要直接用 `yolo val` 命令行**：它不会走 `install()` 裁切，数字和训练时不是
  一回事。
- `prelabel.py` 默认权重指向 `runs/dataset4_2_6cls/weights/best.pt`（未收录），
  所以必须显式传 `-w`。
- ONNX 语义分割头会切到 export 模式：输出是 `(N, H, W)` 的 uint8 类别图，不是
  logits。
- `make_calib.py` 默认从 dataset3 抽样；本仓库没有 dataset3，用 `--source` 指到
  dataset5_2，并保持与训练相同的裁切口径。

## 8. 重要说明与局限

1. **image-level 随机划分**：同一采集段的相邻帧可能跨 train/val，场景高度相似，
   val 指标偏乐观，只用于早停、调参和同口径横向对比，不能写进最终报告。
2. **没有 test split**。正式评估前应按视频 / session / 采集批次重新划分，或先做
   近重复去重（dHash / embedding 聚类）。
3. **合成样本只进 train**，val 全是真实帧；不要用 val 指标调合成参数。
4. **182 张空 JSON 按 clean 处理，未逐张复核**，可能漏标 glare，影响 clean 帧
   假阳统计。
5. **CE 与 Dice 可以反向变化**：两者统计口径不同，同一阶段 CE 升、Dice 降是正常
   现象。模型选择只看 mIoU，不要拿 loss 曲线跨 run 比质量。
6. **`imgsz` 必须等于 `crop_hw[1]`**（当前 640）。只改 `--imgsz` 会让裁切变成
   "先缩放再裁切"，训练/部署口径不一致。
7. **`best.pt` / `last.pt` 是 FP16 且已剥离优化器状态**（`optimizer=None`、
   `epoch=-1`），只用于推理和导出，不能恢复训练。
8. **数据集不在 Git 中**：`dataset5_2/` 约 820 MB，`.gitignore` 已忽略；需要单独
   分发或从 `some_classes` 重新构建。
9. `aug_glare_paste.py`、`make_calib.py` 的部分默认路径依赖 `some_classes` 的源
   数据，在本仓库单独运行时需要显式传路径。

## 9. 后续工作

- 补真实眩光：按时间段、太阳高度角、天气和镜头朝向补 `guangban2` 的 JSON，再重跑
  `make_dataset5_1.py` → `aug_glare_paste.py` → `make_dataset5_2.py`。
- 换划分口径：按采集段 / session 分组划分，把 val 变成可信评估集。
- 难类专项：`water` 和 `mud_film` 是主要误差来源，补薄水膜、逆光水膜、干涸泥膜
  边界等样本。
- 部署链路：ONNX → RKNN / TensorRT，实测 FP16/INT8 延迟和掉点。
- 阶段二时序校验：单帧分割只做候选提取，后续接运动门控、单应配准和驻留积分，
  利用"污渍固定在镜头上、场景在动"的先验压假阳。
