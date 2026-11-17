# Plan1 — Stage-1 镜头脏污分割训练框架

本目录是阶段一（单帧语义分割）的独立训练工程：输入 RGB 图像，输出 4 类逐像素 mask。

> 状态（2026-10-19）：**训练、评测、推理和 benchmark 均已跑通**。当前指标受随机单帧划分的泄漏影响，属于 MVP 结果，不能作为最终泛化基准（见第 9 节）。

## 1. 任务与标签

| 索引 | 类别 | 含义 |
|---|---|---|
| 0 | `clean` | 干净 |
| 1 | `transparent` | 透明污渍 |
| 2 | `semi-transparent` | 半透明污渍 |
| 3 | `opaque` | 不透明污渍 |

WoodScape soiling 标签的颜色约定：黑=clean、绿=transparent、蓝=semi-transparent、红=opaque。
注意这四类是**透明度等级**，不是草屑/蜗牛/泥土等材质类型。

## 2. 目录结构

```text
Plan1/
├── README.md
├── configs/
│   └── train.yaml          # 数据、模型、优化器、早停、输出配置
├── datasets/               # 已由 tools/data_split.py 生成
│   ├── train/
│   │   ├── rgbImages/      # 3400 张
│   │   └── gtLabels/       # 同名索引 mask (0-3)
│   ├── val/                # 600 张
│   └── test/               # 1000 张（官方 test）
├── src/
│   ├── dataset.py          # 读取 {split}/rgbImages + gtLabels，resize/归一化/翻转
│   ├── model.py            # MobileNetV3-Large + DeepLabV3，替换主/辅助分类头
│   ├── engine.py           # CE+Dice+aux loss、mIoU/dirty IoU、训练循环、早停、checkpoint
│   └── utils.py            # YAML/JSON、随机种子
├── images/                 # 待推理图像
├── tools/
│   ├── data_split.py       # 已运行：生成 datasets/（当前为随机单帧划分）
│   ├── train.py            # 训练入口
│   ├── infer.py            # 使用 best.pt 推理 images/ 或指定图像/目录
│   └── benchmark.py        # checkpoint 大小、参数量、推理延迟/吞吐
└── runs/
    └── soiling_deeplab/
        ├── best.pt         # 最优 checkpoint（含优化器状态）
        ├── last.pt
        ├── history.csv
        ├── metrics.json
        ├── config.yaml
        └── predictions/    # infer.py 输出
```

## 3. 环境

使用现有 `yolo` conda 环境（Python 3.12，torch 2.6 + CUDA 12.4，torchvision 0.21）：

```powershell
C:\Users\HP\.conda\envs\yolo\python.exe --version
C:\Users\HP\.conda\envs\yolo\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

依赖：`torch`、`torchvision`、`Pillow`、`numpy`、`PyYAML`、`tqdm`。
**不依赖** `segmentation_models_pytorch`，模型直接使用 torchvision 实现。

## 4. 数据格式

`datasets/{train,val,test}/` 下各有两个目录：

```text
rgbImages/0001_FV.png
gtLabels/0001_FV.png    # 单通道 PNG，像素值 0/1/2/3
```

两个目录的文件名必须一一对应；`src/dataset.py` 会在启动时校验，缺 mask 会直接报错。
图像默认缩放到 **512×384**（宽×高），mask 使用最近邻插值。

## 5. 训练

从 `Plan1` 目录运行：

```powershell
cd "D:\Dirt detection\Plan1"
C:\Users\HP\.conda\envs\yolo\python.exe tools\train.py --config configs\train.yaml
```

常用覆盖参数：

```powershell
# 小规模冒烟测试（不会自动执行，仅供手动使用）
C:\Users\HP\.conda\envs\yolo\python.exe tools\train.py `
  --config configs\train.yaml --epochs 1 --batch-size 4 --num-workers 0 `
  --out-dir runs\smoke

# 修改数据路径 / 输出目录
... --data-root "D:\path\to\datasets" --out-dir "D:\path\to\runs\exp1"
```

### 训练策略

- 模型：`deeplabv3_mobilenet_v3_large`（约 11M 参数），ImageNet 预训练，4 类输出。
- 损失：`CrossEntropy(weight=类频率倒数) + 0.5×Dice + 0.4×aux CE`。
- 优化器：AdamW，余弦退火；默认 30 epoch、batch 8、lr 3e-4。
- 混合精度：CUDA 下默认开启。
- 验证监控：`dirty_iou`（类别 1–3 对 0 的集合 IoU），连续 8 个 epoch 无提升则早停。
- 数据增强：仅随机水平翻转；后续可加入亮度/模糊/合成污渍。

## 6. 输出

每次运行写入 `output.dir`（默认 `runs/soiling_deeplab/`）：

| 文件 | 内容 |
|---|---|
| `best.pt` | 按 `monitor` 最优的 checkpoint（含 model/optimizer/epoch/metrics/config） |
| `last.pt` | 最后一个 epoch 的 checkpoint |
| `history.csv` | 每个 epoch 的 loss、lr、耗时、val 指标 |
| `metrics.json` | 完整配置、类别频率/权重、history、最优验证指标、test 指标与混淆矩阵 |
| `config.yaml` | 本次运行实际生效的配置快照 |

指标说明：

- `pixel_acc`：逐像素准确率；
- `miou`：4 类平均 IoU；
- `dirty_iou`：把 1/2/3 合并为“脏”后的 IoU，**训练主监控指标**；
- `dirty_recall` / `dirty_precision`：脏像素召回/精确率；
- `per_class_iou`：clean/transparent/semi-transparent/opaque 各自 IoU。

### 本次 run 结果（30 epoch，best epoch 27）

| 指标 | 验证集（best） | 测试集 |
|---|---:|---:|
| dirty IoU | 0.9725 | **0.9628** |
| mIoU | 0.8798 | **0.8661** |
| pixel acc | 0.9485 | 0.9457 |
| dirty recall / precision | 0.9896 / 0.9826 | 0.9859 / 0.9762 |

测试集逐类 IoU：

| clean | transparent | semi-transparent | opaque |
|---:|---:|---:|---:|
| 0.9495 | 0.7936 | 0.7967 | 0.9247 |

训练约 29 分钟（平均 58 s/epoch）。dirty/clean 已接近饱和，透明类仍是主要误差来源。

## 7. 推理

默认读取 `runs/soiling_deeplab/best.pt`，对 `images/` 下所有图像推理：

```powershell
cd "D:\Dirt detection\Plan1"
C:\Users\HP\.conda\envs\yolo\python.exe tools\infer.py `
  --weights runs\soiling_deeplab\best.pt `
  --source images `
  --out-dir runs\soiling_deeplab\predictions
```

也可以传单张图像或任意目录：

```powershell
... tools\infer.py --weights runs\soiling_deeplab\best.pt --source "D:\some\frame.png"
```

输出（保持输入子目录结构）：

| 文件 | 内容 |
|---|---|
| `*_pred.png` | 三联图：原图 \| 彩色 mask \| 叠加图 |
| `*_mask.png` | 单通道索引 mask（0=clean, 1=transparent, 2=semi, 3=opaque） |
| `summary.json` | 每张图的 `dirty_coverage` 与逐类覆盖率 |

可调参数：`--height/--width`（默认 384/512）、`--alpha`（叠加透明度，默认 0.45）、`--device auto|cuda|cpu`。

## 8. 模型大小与推理速度

```powershell
cd "D:\Dirt detection\Plan1"

# GPU FP32
C:\Users\HP\.conda\envs\yolo\python.exe tools\benchmark.py `
  --weights runs\soiling_deeplab\best.pt --device cuda

# GPU FP16 autocast
... tools\benchmark.py --weights runs\soiling_deeplab\best.pt --device cuda --amp

# CPU
... tools\benchmark.py --weights runs\soiling_deeplab\best.pt --device cpu
```

实测环境：RTX 4060 Laptop，batch=1，384×512，仅模型前向（不含图像读取、resize、归一化和后处理）：

| 项目 | 数值 |
|---|---:|
| `best.pt` 文件大小 | 126.52 MiB |
| 参数量 | 11.02 M |
| 纯模型权重（FP32） | 42.16 MiB |
| GPU FP32 | 6.51 ms / 153.5 FPS |
| GPU FP16 autocast | 8.61 ms / 116.1 FPS |
| CPU FP32 | 67.67 ms / 14.8 FPS |

注意：

- `best.pt` 包含模型权重、AdamW 优化器状态、epoch、指标和配置，因此约为纯权重的 3 倍。部署时只保存 `state_dict`：

  ```python
  import torch
  ckpt = torch.load("runs/soiling_deeplab/best.pt", map_location="cpu")
  torch.save(ckpt["model"], "runs/soiling_deeplab/model_only.pt")
  ```

  纯权重 FP32 约 42 MiB，转 FP16 约 21 MiB。
- 本次测试中 FP16 autocast 反而略慢，小模型/小 batch 下转换开销可能占主导。
- 嵌入式目标需另行测 ONNX Runtime / TensorRT / RKNN 的 FP16/INT8 延迟。

## 9. 数据划分的重要警告

当前 `tools/data_split.py` 是**按单帧随机打散**（train 85% / val 15%），脚本自身也打印了警告：

> 随机单帧划分存在序列泄露，val 指标仅用于训练监控，最终评估只用 test 集合。

WoodScape soiling 的官方 split 含近重复帧，且文件名前缀是帧 ID、不是序列 ID，因此：

1. `val` 指标会偏高，只用于早停/调参，不能写进最终报告；
2. 最终结论以 `test` 为准，但 `test` 与 train 同样可能存在跨 split 近重复，指标仍偏乐观；
3. 正式实验前应按视频/session/采集批次重新划分，或先做近重复去重（dHash/embedding 聚类）；
4. 若重新生成数据，保持本 README 的目录约定，训练代码无需修改。

## 10. 后续扩展

- **残差通道**：`gray - GaussianBlur(gray)` 作为第 4 通道输入，验证对暗污渍/失焦的增益；
- **更强模型**：替换 `src/model.py` 为 MobileNetV3 + U-Net（需安装 `segmentation_models_pytorch`）；
- **推理/可视化**：`tools/infer.py` 已完成单图/目录推理；可继续补视频推理和批量指标评测；
- **部署导出**：ONNX → TensorRT / RKNN / OpenVINO，实测嵌入式 FP16/INT8 延迟；
- **阶段二**：训练产物接入时序校验（运动门控、单应配准、驻留积分）。
