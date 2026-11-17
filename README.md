# dataset5_2 — 六类镜头脏污语义分割训练包

本目录是割草机镜头脏污检测的**单帧语义分割**训练包：输入 RGB 图像，按部署口径
居中裁到 **384×640**，输出 6 类逐像素 mask。包含训练/验证代码、`dataset5_2`
数据集，以及该数据集训练出的 `best.pt` / `last.pt`。

> 状态（2026-11-17）：**dataset5_2 训练与逐类验证已跑通**。指标来自 image-level
> 随机划分，val 偏乐观，只作为训练监控和横向对比，不能作为最终泛化基准
> （见第 9 节）。

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
脏污，显式给负类才能把"强光"和"脏"分开。其余五类是按压在镜头上的材质区分的，
不是透明度等级。

配色与部署侧 `stream_server.c` / `min_dirty_infer` 的 `SEG_PALETTE_RGB` 一致，
`prelabel.py`、`labelme2mask.py`、`mask2labelme.py` 共用同一套。

## 2. 目录结构

```text
code_clean/
├── README.md
├── README0.md              # Plan1 工程的 README，仅作格式参考，不属于本项目
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data_dataset5_2.yaml    # 数据路径、crop_hw、类别名
│   └── train_dataset5_2.yaml   # 模型、优化器、增强、早停、输出
├── dataset5_2/             # 约 820 MB
│   ├── labels.txt          # 一行一个类名，行号即类别 id
│   ├── images/{train,val}/ # 1524 / 350 张
│   ├── masks/{train,val}/  # 与图片同名 PNG，像素值 0..5
│   ├── annotations/{train,val}/  # labelme JSON
│   └── README.md           # 数据来源、合成方法、类别统计
├── runs/
│   └── dataset5_2_6cls/
│       ├── weights/best.pt     # 按 val mIoU 选出的最优权重（3.3 MB）
│       ├── weights/last.pt     # 最后一个 epoch
│       ├── results.csv/.png
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│       ├── iou_bar_chart.png
│       ├── val_batch*_{labels,pred}.jpg
│       ├── labels.jpg
│       ├── train_batch*.jpg
│       └── args.yaml           # 本次 run 实际生效的完整参数
└── tools/
    ├── train.py            # 训练入口（--check / --smoke / --resume）
    ├── val.py              # 逐类 IoU 验证
    ├── prelabel.py         # 模型直接生成 labelme JSON / mask / 预览
    ├── labelme2mask.py     # labelme JSON → mask
    ├── mask2labelme.py     # mask → labelme JSON（预标注）
    ├── aug_glare_paste.py  # glare 离线合成（本包缺源数据，仅保留代码）
    ├── crop_dataset.py     # 训练时按部署口径裁切（模块，无 CLI）
    ├── export_onnx.py      # 导出 ONNX
    ├── make_calib.py       # 生成量化校准集（NCHW float32 .raw）
    └── README.md           # 每个脚本的参数和用法
```

说明：

- 数据集直接放在根目录 `dataset5_2/`，**不是** `data/dataset5_2`；`data_*.yaml`
  里的 `path` 已按本包布局写好。
- `make_dataset5_1.py`、`make_dataset5_2.py`、`analyze_model.py`、`predict.py`
  不在本包内；要重新构建数据集或做逐场景分析，回到 `some_classes`。
- `aug_glare_paste.py` 依赖 `some_classes/data/guangban2`，在本包内无法直接运行。

## 3. 环境

使用现有 `yolo` conda 环境（Python 3.12，torch 2.6 + CUDA 12.4）：

```powershell
C:\Users\HP\.conda\envs\yolo\python.exe --version
C:\Users\HP\.conda\envs\yolo\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

本地验证版本：

| 包 | 版本 |
|---|---|
| Python | 3.12.15 |
| torch | 2.6.0+cu124 |
| ultralytics | 8.4.183 |
| numpy | 2.5.4 |
| opencv-python | 4.11.0 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |

也可以新建环境：

```bat
conda create -n yolo python=3.12 -y
conda activate yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

关键约束：**`ultralytics==8.4.183`**。语义分割接口（`task: semantic`、
`SemanticDataset`、`masks_dir`）是 8.4.x 才有的，换版本可能直接不兼容。

## 4. 数据格式

### 4.1 目录与文件

```text
dataset5_2/
├── labels.txt
├── images/{train,val}/<stem>.png
├── masks/{train,val}/<stem>.png          # 单通道，像素值 = 类别 id
└── annotations/{train,val}/<stem>.json   # labelme，可重新导出 mask
```

- 图片与 mask 同名、同尺寸；`train.py` 启动时会校验缺 mask、尺寸不一致和非法取值。
- mask 取值 `0..5`，`255` 表示忽略像素。
- `labels.txt`：

  ```text
  clean
  dirt_dot
  glare
  water
  grass_residue
  mud_film
  ```

- mask 可以由 labelme JSON 现场生成：

  ```bat
  python tools\labelme2mask.py -i dataset5_2\images\train ^
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
- 合成样本共生成 180 张，纳入 122 张：33 张的 receiver（clean 底色）落在
  `dataset5_1` 的 val、25 张的 donor（眩光来源）落在 val，为避免场景泄漏全部过滤。
- 合成样本 stem 带前缀 `gsyn_`，`manifest_dataset5_2.json` 记录每张的
  receiver / donor / mask_area，可回查来路。

类别统计（同一张图可含多类，所以图片级数字不能相加）：

| split | 图片数 | 全 clean | dirt_dot | glare | water | grass_residue | mud_film |
|---|---:|---:|---:|---:|---:|---:|---:|
| train 图片级 | 1524 | 382 | 295 | 316 | 319 | 285 | 200 |
| val 图片级 | 350 | 118 | 91 | 61 | 118 | 95 | 62 |
| train 像素级 | — | 84.77% | 2.84% | 1.55% | 1.64% | 8.03% | 1.17% |
| val 像素级 | — | 82.74% | 2.62% | 2.10% | 1.97% | 8.43% | 2.13% |

（像素级里的 clean 是剩余占比；`dirt_dot` 等为占全图有效像素的比例。）

合成样本是"clean 帧 + 真实眩光 patch"，单帧眩光面积很小（mask 中位约 620 px，
全图中位约 0.25%）。它补的是**图片级眩光检出和小面积眩光形态**，不是像素占比；
train 的 glare 像素占比反而从 1.64% 降到 1.55%。

### 4.3 数据 yaml

`configs/data_dataset5_2.yaml` 的关键字段：

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

`crop_hw` 由 `tools/crop_dataset.py` 在 Dataset 内执行：先把长边缩到 640，再按
5:3 居中裁切，**不生成派生数据目录**。训练、验证、导出和量化校准都必须用同一刀。

## 5. 训练

从本目录运行：

```powershell
cd "D:\Dirt detection\code_clean"

# 只校验配置、图片/mask 对齐和裁切，不训练
C:\Users\HP\.conda\envs\yolo\python.exe tools\train.py `
  --config configs\train_dataset5_2.yaml --check

# 真构造一个 batch，确认入网形状是 384×640
... tools\train.py --config configs\train_dataset5_2.yaml --check --smoke

# 正式训练，输出 runs\dataset5_2_6cls
... tools\train.py --config configs\train_dataset5_2.yaml
```

`train.py` 支持 `--resume <last.pt>`，但本包里的 `last.pt` 是训练结束后的最终
checkpoint（优化器状态已剥离，`epoch=-1`），不能用来续训；只有在训练中途崩溃、
checkpoint 尚未 strip 时才有意义。

**关于初始权重**：`configs/train_dataset5_2.yaml` 里的
`model: runs/dataset4_2_6cls/weights/best.pt` 是在 `some_classes` 中训练
dataset5_2 时的起点，**本包没有收录这个权重**。要复现训练，任选一种：

```powershell
# 1) 从 some_classes 复制 dataset4_2 的 best.pt 到 configs 指定的位置
# 2) 或直接覆盖 model
... tools\train.py --config configs\train_dataset5_2.yaml `
  --model "D:\Dirt detection\some_classes\runs\dataset4_2_6cls\weights\best.pt"
# 3) 或从公开的 yolo26n-sem 预训练权重起步（本地没有时会尝试下载）
... tools\train.py --config configs\train_dataset5_2.yaml --model yolo26n-sem.pt
```

本包里的 `runs/dataset5_2_6cls/weights/best.pt` 是**训练结果**，不是训练起点。

### 训练策略

- 模型：`YOLO26n-sem`，73 层，1,626,132 参数，4.4 GFLOPs，6 类输出。
- 损失：Ultralytics 语义分割默认的 **CE + Dice**；CE 按有效像素求均值，Dice
  按类求 soft Dice 后取平均。
- 优化器：AdamW，`lr0=2e-4`，`lrf=0.01`，余弦退火，`weight_decay=5e-4`，
  warmup 1 epoch。
- 训练：50 epoch、`patience=15`、`batch=8`、`workers=4`、AMP 开启、seed 42。
- 入网：`imgsz=640`、`rect=true`、`crop_hw=[384,640]`。`imgsz` **必须等于
  `crop_hw[1]`**，否则源图会先降采样，裁切口径和部署对不上；`train.py` 会直接报错。
- 监控：语义分割的 `fitness` 就是 **mIoU**，早停和 `best.pt` 都按它选。

### 数据增强

`rect: true` 时 ultralytics 会强制关闭 mosaic / mixup / cutmix，配置里再显式写 0：

```yaml
mosaic: 0
mixup: 0
cutmix: 0
close_mosaic: 0
copy_paste: 0      # 语义分割没有 copy-paste 实现，本来就是空操作
erasing: 0         # 语义分割没接 RandomErasing
```

只保留保标签的成像链路模拟：

| 参数 | 值 | 原因 |
|---|---:|---|
| `translate` | 0.15 | 太阳在画面里的位置变化大 |
| `scale` | 0.3 | 0.7×–1.3×；再大条纹会被放大成死白块 |
| `fliplr` | 0.5 | 左右翻转不改变物理含义 |
| `flipud` | 0.0 | 相机水平安装，太阳不会跑到画面下方 |
| `degrees` / `shear` / `perspective` | 0 | 旋转/透视会造出物理上不存在的分布 |
| `hsv_h` | 0.0 | 强色相扰动会把白色日轮染成彩色 |
| `hsv_s` / `hsv_v` | 0.2 / 0.2 | 只模拟曝光波动；默认 0.7/0.4 会把天空提亮成"像 glare"，mask 却不动，纯属标签噪声 |

## 6. 输出

每次训练写入 `project/name`（默认 `runs/dataset5_2_6cls/`）：

| 文件 | 内容 |
|---|---|
| `weights/best.pt` | 按 val mIoU 最优的 checkpoint |
| `weights/last.pt` | 最后一个 epoch |
| `results.csv` | 每 epoch 的 train/val CE、Dice、aux loss、mIoU、pixel acc、lr、耗时 |
| `results.png` | 上述曲线的汇总图 |
| `confusion_matrix.png` | 像素级混淆矩阵 |
| `confusion_matrix_normalized.png` | 按行归一化 |
| `iou_bar_chart.png` | 逐类 IoU 柱状图 |
| `val_batch*_labels.jpg` / `_pred.jpg` | 验证集真值 / 预测对照 |
| `train_batch*.jpg` | 训练 batch 可视化 |
| `labels.jpg` | 训练集标签分布 |
| `args.yaml` | 本次运行实际生效的全部参数（含数据绝对路径） |

指标说明：

- `metrics/mIoU`：6 类 IoU 的算术平均，**训练主监控指标**；
- `metrics/pixel_acc`：逐像素准确率，被 clean（约 82–85% 像素）主导，不能单独看；
- 逐类 IoU / pixel acc / 像素占比用 `tools/val.py` 打印；
- `val/ce_loss` 与 `val/dice_loss` 统计口径不同（CE 按像素、Dice 按类平均），
  可以反向变化，不要拿来跨 run 比质量。

### 逐类验证

```powershell
... tools\val.py --config configs\train_dataset5_2.yaml --plots
```

`val.py` 会先 `install()` 裁切 Dataset，再调 `model.val()`；**不要直接用
`yolo val` 命令行**，它不会裁切，数字和训练时不是一回事。

### 本次 run 结果（best epoch 32，共训 47 epoch）

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

`water` 是最弱项（IoU 0.49，pixel acc 0.59），主要错在边界和薄水膜；
`mud_film`（0.65）次之。`clean` 已接近饱和。

### 与 dataset5_1 的对比

两者 val 逐张相同（350 张），在 `some_classes` 中用同一份数据分别跑两个
`best.pt`：

| 指标 | dataset5_1 | dataset5_2 |
|---|---:|---:|
| mIoU | 0.7583 | **0.7603** |
| glare IoU | 0.8121 | **0.8283** |
| glare recall / precision | 0.853 / 0.945 | **0.868 / 0.947** |
| 仅 guangban2 子集的 glare IoU | 0.387 | **0.475** |
| 全 clean 帧出现 glare 假阳 | 4 / 118 帧 | **2 / 118 帧** |

合成 glare 提升了眩光召回和 clean 帧的区分度；mIoU 的 +0.002 在单 seed 噪声内，
其余类别的变化不稳健（`dirt_dot` / `mud_film` 在 best 轮次的波动换 `last.pt`
复核后不再成立）。结论应表述为"眩光子项改善"，而不是"整体显著提升"。

## 7. 推理与导出

本包没有单独的 `predict.py`。用 `prelabel.py` 可以直接出预测 mask、labelme JSON
和两栏预览图：

```powershell
# 只看预览（左原图 / 右叠加，青色框是实际推理的裁切区域）
... tools\prelabel.py -i "D:\some\images" -p runs\preview `
  -w runs\dataset5_2_6cls\weights\best.pt

# 同时写 labelme JSON + 整帧预测 mask
... tools\prelabel.py -i "D:\some\images" -o runs\annotations `
  -m runs\masks -w runs\dataset5_2_6cls\weights\best.pt
```

- 默认按 `--crop-hw 384,640` 裁切后推理，再把 mask 放回原图坐标；裁切框外为
  `255`（忽略），不生成 shape。
- 默认权重路径指向 `runs/dataset4_2_6cls/weights/best.pt`（本包未收录），
  所以**必须显式传 `-w`**。
- `--min-area 16` 过滤小连通块，`--epsilon 0.002` 控制多边形抽稀。

导出 ONNX：

```powershell
... tools\export_onnx.py --weights runs\dataset5_2_6cls\weights\best.pt `
  --output runs\dataset5_2_6cls\weights\best.onnx
```

- 默认输入 `1×3×384×640`、opset 11、固定 batch；`--dynamic` 导出动态轴。
- 语义分割头会切到 export 模式：输出是 **`(N, H, W)` 的 uint8 类别图**（已 argmax
  并上采样到输入分辨率），不是 logits。
- `--verify` 默认开启，需要 `onnxruntime`。

量化校准（RDK `hb_mapper`）：

```powershell
# 脚本默认从 dataset3/train 抽 20%；本包没有 dataset3，需用 --source 指到 dataset5_2
... tools\make_calib.py --source dataset5_2\images\train --ratio 0.1 --force
```

输出 RGB、0..255、float32、NCHW 的 `.raw`；归一化放在模型配置里
（`data_scale = 1/255`），和 Ultralytics 推理时的 `im /= 255` 对齐。

## 8. 模型大小与推理速度

| 项目 | 数值 |
|---|---:|
| `best.pt` 文件大小 | 3.3 MiB |
| 参数量 | 1.63 M |
| GFLOPs（384×640） | 4.4 |
| 纯权重 FP32（估算） | 约 6.2 MiB |
| 纯权重 FP16（估算） | 约 3.1 MiB |

`val.py` 在 RTX 4060 Laptop、batch=16、384×640 下的逐图平均耗时：

| 阶段 | 耗时 |
|---|---:|
| preprocess | 5.3 ms |
| inference | 3.3 ms |
| postprocess | 0.2 ms |

注意：

- 这是 PyTorch + CUDA 的桌面端数字，**不是嵌入式实测**；部署需另测
  ONNX Runtime / RKNN 的 FP16/INT8 延迟。
- 上表是 `val` 路径的逐图均值，包含 dataloader 之外的前后处理；实时性以部署侧
  端到端（取图 → 裁切 → 推理 → 后处理 → 告警）为准。
- `best.pt` / `last.pt` 里的权重是 **FP16**，优化器状态已剥离
  （`optimizer=None`、`epoch=-1`），文件才只有 3.3 MB；它们只用于推理和导出，
  不能恢复训练。只部署时导出 ONNX。

## 9. 数据划分的重要警告

1. **image-level 随机划分**：同一采集段的相邻帧可能跨 train/val（`guangban2`
   尤其明显），场景内容高度相似。val 指标偏乐观，只用于早停、调参和同口径
   横向对比，**不能写进最终报告**。
2. **没有 test split**。`dataset5_2` 只配了 train/val；正式评估前应按
   视频 / session / 采集批次重新划分，或先做近重复去重（dHash / embedding 聚类）。
3. **dataset5_2 的 val 与 dataset5_1 逐张相同**，两者可以直接比；但和
   `dataset4_2` 的 val 不同，`0.5479` 之类的旧数字不能直接拿来比。要留基线就用
   同一份 data yaml 重跑旧权重。
4. **合成样本只进 train**，val 全是真实帧。不要用 val 指标去调合成参数，否则
   等于在验证集上选增强策略；合成样本也不要手工塞进 val。
5. **182 张空 JSON 按 clean 处理，未逐张复核**，可能漏标 glare。这会让
   "clean 帧假阳"的统计偏乐观或偏悲观，取决于漏标位置。
6. **CE 与 Dice 可以反向变化**：两者统计口径不同，同一阶段 CE 上升、Dice 下降
   （或相反）都是正常现象，dataset5_1 后段就出现过。模型选择只看 mIoU，不要拿
   loss 曲线跨 run 比质量。
7. **`imgsz` 必须等于 `crop_hw[1]`**（当前 640）。只改 `--imgsz` 会让裁切变成
   "先缩放再裁切"，训练/部署口径不一致；要改分辨率就连 `crop_hw` 一起改。
8. `dataset5_2/README.md` 里的构建命令（`make_dataset5_2.py`、
   `aug_glare_paste.py --preview`）依赖 `some_classes` 的源数据，在本包内无法执行；
   本包的 `dataset5_2/` 是已经构建好的成品。

## 10. 后续扩展

- **补真实眩光**：按时间段、太阳高度角、天气和镜头朝向补 `guangban2` 的 JSON，
  再重跑 `make_dataset5_1.py` → `aug_glare_paste.py` → `make_dataset5_2.py`。
  合成只能补形态，不能替代真实分布。
- **换划分口径**：按采集段 / session 分组划分，或先聚近重复帧再分层；这是把
  val 变成可信评估集的前提。
- **合成策略**：增加眩光形态、强度、入射角和与脏污叠加的样本；同时控制合成
  比例，避免模型只记住"上半部小黄斑"这一种形态。
- **难类专项**：`water`（0.49）和 `mud_film`（0.65）是当前主要误差来源，可补
  薄水膜、逆光水膜、干涸泥膜边界等样本，或加边界损失。
- **部署链路**：ONNX → RKNN / TensorRT，实测 FP16/INT8 延迟和掉点；用
  `make_calib.py` 生成与训练同口径的校准集。
- **阶段二时序校验**：单帧分割只做候选提取，后续接运动门控、单应配准和驻留
  积分，利用"污渍固定在镜头上、场景在动"的先验压假阳。
- **补分析工具**：需要逐图 / 逐场景报告时，从 `some_classes` 取回
  `analyze_model.py`；需要批量预测可视化时取回 `predict.py`。

## 附：上传 Gitee

本包的数据和权重体积较大，默认 `.gitignore` 会忽略 `runs/`、`*.pt` 和 `*.cache`，
**不会**忽略根目录的 `dataset5_2/`：

```bat
git init -b main
git add .
git status          # 确认 dataset5_2 是否要提交
git commit -m "dataset5_2 六类语义分割训练包"
git remote add gitee https://gitee.com/<用户名>/<仓库名>.git
git push -u gitee main
```

建议：

- `dataset5_2` 约 820 MB，直接进 Git 会让仓库很重，也不适合普通 Gitee 仓库；
  优先只提交**代码 + 配置 + README**，数据和权重走网盘 / Release / Git LFS。
- 若确实要提交权重，`*.pt` 在 `.gitignore` 里，需要 `git add -f
  runs/dataset5_2_6cls/weights/best.pt`，或调整忽略规则。
- 提交前跑一遍 `python tools\train.py --config configs\train_dataset5_2.yaml --check`，
  确认数据路径在本机可用。
