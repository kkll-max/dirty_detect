# 割草机镜头脏污检测 — 六类语义分割

基于 Ultralytics **YOLO26n-sem** 的单帧语义分割

## 1. 任务与标签

| 索引 | 类别 | 含义 | 预览色 |
|---|---|---|---|
| 0 | `clean` | 干净镜面（背景） | 不叠加 |
| 1 | `dirt_dot` | 点状泥污 / 脏点 | `#a52a2a` |
| 2 | `glare` | 强光 / 眩光（阳光直射镜头） | `#ffff00` |
| 3 | `water` | 水渍 / 水滴 / 水膜 | `#0000ff` |
| 4 | `grass_residue` | 草屑残留 | `#00c800` |
| 5 | `mud_film` | 片状泥膜 | `#b47828` |

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


## 4. 训练

在仓库根目录执行：

```powershell
# 只校验配置、图片/mask 对齐和裁切
python tools\train.py --config configs\train.yaml --check

# 真构造一个 batch，确认入网形状是 384×640
python tools\train.py --config configs\train.yaml --check --smoke

# 正式训练，输出 runs\dataset5_2_6cls
python tools\train.py --config configs\train.yaml
```

## 5. 验证、推理与导出

```powershell
# 逐类 IoU（自动按 384×640 裁切，和训练口径一致）
python tools\val.py --config configs\train_dataset5_2.yaml --plots

# 推理并生成预览 / labelme JSON / 预测 mask
python tools\prelabel.py -i "D:\some\images" -p runs\preview `
  -w runs\dataset5_2_6cls\weights\best.pt

# 导出 ONNX（默认 1×3×384×640、opset 11）
python tools\export_onnx.py --weights runs\dataset5_2_6cls\weights\best.pt `
  --output runs\dataset5_2_6cls\weights\best.onnx

# 生成 hb_mapper 量化校准集
python tools\make_calib.py --source dataset5_2\images\train --ratio 0.1 --force
```
