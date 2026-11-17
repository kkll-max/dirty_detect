# tools 使用说明

所有命令默认在 `some_classes` 根目录执行。除 `labelme` 外，脚本都依赖项目环境中的
`ultralytics` / `opencv` / `torch`；用系统 Python 直接跑可能报 `ModuleNotFoundError`。

## 工具速查

| 脚本 | 用途 |
|---|---|
| `predict.py` | 推理并生成对比图 / 预测 mask |
| `mask2labelme.py` | mask → labelme JSON（预标注） |
| `labelme2mask.py` | labelme JSON → mask |
| `make_dataset3.py` | dataset2 + wood → dataset3 |
| `make_dataset4.py` | 构建 dataset4 六类（方案 A）标注脚手架 |
| `make_yu_split.py` | dataset4 val+test → yu_data train/val |
| `make_prelabel_mix.py` | yu_data + x5/data_get → prelabel_mix |
| `make_dataset4_1.py` | dataset4/train + prelabel_mix → dataset4_1 |
| `make_dataset4_2.py` | dataset4_1 + x5/nitu1 + x5/green4 → dataset4_2 |
| `make_dataset5_1.py` | dataset4_2 + guangban2（glare 补充）→ dataset5_1 |
| `aug_glare_paste.py` | 离线合成：guangban2 的 glare patch → clean 帧 → guangban2_syn |
| `make_dataset5_2.py` | dataset5_1 + guangban2_syn（train-only）→ dataset5_2 |
| `dataset_stats.py` | 类别均衡度统计（图片/像素/实例/场景） |
| `crop_dataset.py` | 训练时按部署口径裁切（模块，无 CLI） |
| `train.py` | 训练入口 |
| `val.py` | 逐类 IoU 验证 |
| `analyze_model.py` | 逐类/逐图/逐场景效果分析报告 |
| `export_onnx.py` | 导出 ONNX |
| `make_calib.py` | 生成量化校准集 |
| `extract_originals.py` | 从并列对比图提取原图 |
| `chou.py` | 随机抽样复制图片（路径写在脚本里） |

## 1. 推理与可视化

### predict.py

```bat
python .\tools\predict.py -w .\runs\water_4cls\weights\best.pt -s .\path\images ^
  --mask-output .\path\masks --preview-output .\path\preview
```

- 默认按部署口径裁到 `384×640`（`--crop-hw 384,640`）；`--full-frame` 只看整图效果。
- `--mask-output` 指定后自动保存预测 mask；`--mask-binary` 只存 `dirty` 二值，供只认
  `dirty=1` 的 `mask2labelme.py` 使用。
- 其他常用：`--limit N`、`--overwrite`、`--batch`、`--device`。

## 2. 标注

### prelabel.py（模型直接生成 labelme JSON）

```bat
# 只出 JSON
python .\tools\prelabel.py -i .\path\images -o .\path\annotations

# 同时出预测 mask 和预览图
python .\tools\prelabel.py -i .\path\images -m .\path\masks ^
  -o .\path\annotations -p .\path\preview

# 不写 JSON，只出两栏预览
python .\tools\prelabel.py -i .\path\images -p .\path\preview
```

- `-o` 可省略；不给就只推理不写 JSON。`-o/-m/-p` 至少要给一个。
- 默认权重 `runs/prelabel_mix6/weights/best.pt`（yu_data + x5 混合六类），先按 `384×640` 裁切再推理。
- 输出可直接用 labelme 打开修正的 JSON；裁切框外不生成 shape。
- `-m/--mask-dir` 保存整帧预测 mask（框外 255）；不给时 `--save-mask` 落到 `<annotations>_masks/`。
- `-p/--preview` 存 `<stem>_preview.png`：左栏原图、右栏 mask 叠加，两栏对照；
  青色框标出实际推理区域。预览配色和 `labelme2mask.py` / `predict.py` 一致。
- 配色统一用 6 类调色板（和部署侧 `stream_server.c` / `min_dirty_infer` 的
  `SEG_PALETTE_RGB` 同色），按 50% 叠加，网页图例同色：
  clean 不叠加；dirt_dot `#a52a2a`、glare `#ffff00`、water `#0000ff`、
  grass_residue `#00c800`、mud_film `#b47828`。
- 常用：`--limit 20`、`--device cpu`、`--overwrite`、`--skip-empty`。
- `--min-area 16` 过滤小连通块，`--epsilon 0.002` 控制多边形抽稀。

### mask2labelme.py（mask → JSON ）

```bat
python .\tools\mask2labelme.py --masks .\path\masks --images .\path\images -o .\path\annotations
```

- 需要 `opencv`。
- 常用：`--labels .\path\labels.txt`、`--min-area 16`、`--epsilon 0.002`、
  `--strip-suffix _pred`、`--skip-empty`、`--overwrite`。

### labelme（人工修正）

```bat
labelme --labels .\path\labels.txt --output .\path\annotations .\path\images
```

### labelme2mask.py（JSON → mask）

```bat
# 常用：-i 原图、-o 原标注（labelme JSON）、-m mask 输出
python .\tools\labelme2mask.py -i .\data\guangban2\images ^
  -o .\data\guangban2\annotations -m .\data\guangban2\masks

# -i / -m 不写时按 annotations 的同级目录推断
python .\tools\labelme2mask.py -o .\data\guangban2\annotations

# 同时出预览（mask 50% 叠到原图上，配色和 predict.py 一致）
python .\tools\labelme2mask.py -i .\data\guangban2\images ^
  -o .\data\guangban2\annotations -m .\data\guangban2\masks --preview

# 旧写法（仍兼容）：--data-dir 指向数据集根目录
python .\tools\labelme2mask.py --data-dir .\data\wood --preview

# x5/data_get 下有多个子集，批量写 <子集>/masks 和 <子集>/preview
python .\tools\labelme2mask.py --data-dir .\x5\data_get --preview
```

- 需要 `opencv`。
- 类别按 `labels.txt`（行号即 ID）；找不到 labels.txt 时用内置 6 类
  `clean / dirt_dot / glare / water / grass_residue / mud_film`。`-i` 只用于校验
  尺寸和生成预览，缺了也能出 mask。
- `--data-dir` 可以指向单个数据集（含 `images/`、`annotations/`），也可以指向包含多个
  此类子目录的父目录；批量模式按子目录分别输出 `masks/` 和 `preview/`。
- `--labels` 默认在 `annotations` 上一级、`annotations` 自身和 `--data-dir` 根目录
  找 `labels.txt`。
- 方案 A 下 `water_film` / `water_drop` 会自动并入 `water`，并在汇总里单独提示。
- `--limit`、`--overwrite` 可选；`--preview` 不写路径时输出到
  `<mask 输出目录>_preview`，批量模式输出到 `<子集>/preview`。
- `-m` 的旧名字是 `--output`，仍可用；注意 `-o` 现在指**原标注目录**，
  不再是 mask 输出目录。

## 3. 数据集构建

### make_dataset3.py

```bat
# dataset2 + wood，wood 按 8:1:1 分层随机划分，seed 42
python .\tools\make_dataset3.py

# 改比例 / 种子 / 输出目录
python .\tools\make_dataset3.py --split 7:1:2 --seed 0 -o .\data\dataset3_v2 --force
```

### make_dataset4.py

```bat
# 预览，不复制文件
python .\tools\make_dataset4.py --dry-run

# 构建
python .\tools\make_dataset4.py
```

输出是 **6 类（方案 A）标注脚手架**，不是可直接训练的 GT：

- `images/`、`annotations_legacy/`、`masks_legacy/`、`annotations_prelabel/` 由
  dataset2 + green_grass 生成。
- `annotations/`、`masks/` 为空，等待按 6 类重新标注。
- 方案 A 把 `water_film` + `water_drop` 合并为 `water`；旧 `dirty`/`water1`
  只按映射改名成预标注，不能当 6 类 GT。
- green_grass 按时间戳分簇后落位；默认 cluster 0→train、1→val、2→test。
- 常用：`--copy-preds`、`--keep-unannotated`、`--no-remap-prelabel`、
  `--green-cluster-map train=0,val=1,test=2`、`--force`。

### make_yu_split.py

```bat
# 合并 dataset4 val+test 的已标注样本，按最稀有类分层成 train/val
python .\tools\make_yu_split.py --source-splits val,test --val-ratio 0.2 --seed 42
```

- 只用图片和 JSON 都存在的样本；缺 JSON 的图、无对应图片的 JSON 会跳过。
- mask 从当前 JSON 重新生成，不直接复用可能过期的源 mask。
- 输出 `data/dataset4/yu_data/{images,masks,annotations}/{train,val}`。
- 常用：`--out`、`--force`；是 image-level 划分，val 可能和 train 同采集段。

### make_prelabel_mix.py

```bat
# 预览来源和数量
python .\tools\make_prelabel_mix.py --dry-run

# 生成 data/prelabel_mix（已存在时加 --force）
python .\tools\make_prelabel_mix.py --force
```

- 合并 `data/dataset4/yu_data` 和 `x5/data_get` 的 `clean_63` / `green_154` / `water_104`。
- 划分策略：`yu_data/train` + 全部 x5 → train；`yu_data/val` → val。x5 是连续帧，
  不随机拆进 val，避免同场景泄漏。
- x5 图片加 `x5clean_` / `x5green_` / `x5water_` 前缀；所有 mask 从当前 JSON 重生成。
- 输出 `data/prelabel_mix/{images,annotations,masks}/{train,val}` 和
  `manifest_prelabel_mix.json`；配套 `configs/data_prelabel_mix.yaml`、
  `configs/train_prelabel_mix.yaml`。
- 常用：`--x5-sources`、`--labels`、`--output`、`--dry-run`、`--force`。

### make_dataset4_1.py

```bat
# 预览分组和划分，不复制文件
python .\tools\make_dataset4_1.py --dry-run

# 构建 data/dataset4_1（已存在时加 --force）
python .\tools\make_dataset4_1.py --force
```

- 合并 `data/dataset4/train` 和 `data/prelabel_mix/{train,val}`，输出
  `data/dataset4_1/{images,annotations,masks}/{train,val}`（train 1067 / val 267）。
- 按类别分层随机划分：每张图按所含脏类打标签，全 clean 图单独一档；每类 val
  目标约 20%，贪心选图让各类占比贴近目标。image-level 随机划分，同一采集段的
  相邻帧可能跨 train/val，val 偏乐观。
- 只取图片 + mask 都存在的样本；mask 直接复用磁盘文件并校验尺寸/取值；JSON 的
  `imagePath` 重写为新目录相对路径，缺 JSON 时写空 JSON。
- dataset4 的 annotation 修正后，先重跑 `labelme2mask.py --overwrite` 再构建。
- 输出 `manifest_dataset4_1.json` 记录每张图的 source / source_split / group / classes。
- 配套 `configs/data_dataset4_1.yaml`、`configs/train_dataset4_1.yaml`（从
  `runs/dataset3_water4/weights/best.pt` 起步）。
- 常用：`--val-ratio`、`--seed`、`--dataset4-root`、`--prelabel-mix-root`、`--force`。

### make_dataset4_2.py

```bat
# 预览来源，不写文件
python .\tools\make_dataset4_2.py --dry-run

# 构建 data/dataset4_2（已存在时加 --force）
python .\tools\make_dataset4_2.py --force
```

- 把 `data/dataset4_1` 和 `x5/data_get/nitu1`（99 张）、
  `x5/data_get/green4`（64 张）全量混合，输出 `data/dataset4_2`
  （train 1198 / val 299）。
- 划分和 `make_dataset4_1.py` 同一套：按类别分层随机（seed 42），
  全 clean 图单独占一档；`--val-ratio`、`--seed` 可调。
- `--keep-base-split` 可退回旧口径：dataset4_1 的 train/val 原样保留，
  x5 新子集按 `--x5-split`（默认 train）整体落位。
- mask 用 `labelme2mask.build_mask` 从 JSON 现场生成，源目录不需要已有 mask。
- 新样本加前缀 `x5nitu1_` / `x5green4_`，manifest 里记 source / group。
- 配套 `configs/data_dataset4_2.yaml`、`configs/train_dataset4_2.yaml`（从
  `runs/dataset4_1_6cls/weights/best.pt` 起步）。
- 常用：`--val-ratio`、`--seed`、`--keep-base-split`、`--x5-sources`、
  `--base-root`、`--labels`、`--force`。

### make_dataset5_1.py

```bat
# 预览来源和划分，不写文件
python .\tools\make_dataset5_1.py --dry-run

# 构建 data/dataset5_1（已存在时加 --force）
python .\tools\make_dataset5_1.py --force

# 保留 dataset4_2 的 train/val，guangban2 全部进 train
python .\tools\make_dataset5_1.py --keep-base-split --force

# 把 guangban2 的空 JSON 也当 clean 纳入（需确认无眩光）
python .\tools\make_dataset5_1.py --include-empty --force
```

- 把 `data/dataset4_2`（1497 张）和 `data/guangban2` 合并成 `data/dataset5_1`
  （train 1256 / val 314）。
- `guangban2` 有 255 张图，默认只取 **73 张有 shape 的**；182 张空 JSON 跳过。
  当前 `data/dataset5_1` 是加 `--include-empty` 构建的，空 JSON 全部按 clean 纳入
  （train 1402 / val 350）。
- 划分和 `make_dataset4_1.py` / `4_2` 同一套：按类别分层随机（seed 42），
  全 clean 图单独占一档；`--val-ratio`、`--seed` 可调。
- `guangban2` 的 mask 用 `labelme2mask.build_mask` 从 JSON 现场生成，
  stem 加前缀 `gb2_`；构建时会打印它覆盖的采集段数（60s 间隔切分）。
- 源 `guangban2` 的 JSON 改了以后再跑一遍即可同步。
- 配套 `configs/data_dataset5_1.yaml`、`configs/train_dataset5_1.yaml`（从
  `runs/dataset4_2_6cls/weights/best.pt` 起步）。
- 常用：`--val-ratio`、`--seed`、`--keep-base-split`、`--extra-split`、
  `--include-empty`、`--base-root`、`--extra-root`、`--extra-prefix`、
  `--labels`、`--force`。

### aug_glare_paste.py（glare 离线合成）

```bat
# 只看 donor / receiver 数量
python .\tools\aug_glare_paste.py --dry-run

# 合成到 data/guangban2_syn，并输出 mask 叠加预览
python .\tools\aug_glare_paste.py --preview --force

# 限定数量 / 调整亮度门限
python .\tools\aug_glare_paste.py --max-samples 100 --gate 190 --force
```

- donor：`guangban2` 里标了 glare 的帧，按连通域拆成 patch；receiver：空 JSON
  的 clean 帧（`--receivers all` 可放开）。
- 合成用 screen 混合（加光，不是 alpha 贴图）：暗背景几乎不变，眩光中心自然
  过曝；边缘按距离变换羽化。
- 只搬"光"不搬场景：patch 按亮度门限（`--gate`）过滤，避免把 donor 的楼 / 树
  ghost 到 receiver 上。
- 贴之前按接收区域亮度余量选位置（`--tries`），贴完只把"实际变亮 >=
  `--min-delta`"的像素写进 mask——保证 mask 标的位置画面里真的更亮。
- 输出 `data/guangban2_syn/{images,masks,annotations}` + `manifest_synth.json`
  （每张的 donor / receiver / 位置 / mask_area），stem 为 `syn_0001`；
  `--preview` 额外写 `data/guangban2_syn_preview/`。
- 合成样本**只用于 train**，不要放进 val。
- 常用：`--receivers`、`--max-samples`、`--min-area`、`--feather`、`--gate`、
  `--gate-soft`、`--soften`、`--min-delta`、`--seed`、`--preview`、`--force`。

### make_dataset5_2.py

```bat
# 看能纳入多少合成样本（会按 train/val 过滤）
python .\tools\make_dataset5_2.py --dry-run

# 构建 data/dataset5_2
python .\tools\make_dataset5_2.py --force
```

- `dataset5_2 = dataset5_1 + guangban2_syn`，**dataset5_1 的 train/val 原样复制**，
  合成样本只追加到 train；val 全是真实数据，和 dataset5_1 的 val 完全相同。
- 默认过滤泄漏：合成样本的 receiver / donor 都必须在 dataset5_1 的 train。
  当前 180 张里 122 张可用（33 张 receiver 在 val、25 张 donor 在 val）。
- 合成样本 stem 加前缀 `gsyn_`，manifest 里额外记录 receiver / donor /
  mask_area，方便回查。
- mask 直接用 `aug_glare_paste.py` 生成的那份（只标画面真正变亮的像素），
  JSON polygon 是它的矢量近似（边界差 1 像素内）。
- 配套 `configs/data_dataset5_2.yaml`、`configs/train_dataset5_2.yaml`（从
  `runs/dataset4_2_6cls/weights/best.pt` 起步）。
- 合成数据增多后重跑 `aug_glare_paste.py` + `make_dataset5_2.py --force` 即可。

## 4. 类别均衡度统计

```bat
python .\tools\dataset_stats.py --root .\data\dataset4 ^
  --out .\data\dataset4\balance_report.md ^
  --json-out .\data\dataset4\balance_report.json
```

- 默认按部署口径裁到 `384×640`，分 split / 来源 / 场景统计图片、像素、实例、场景四级。
- 有 data yaml 时可直接读 `path` / `splits` / `masks_dir` / `crop_hw` / `names`：

```bat
python .\tools\dataset_stats.py --data .\configs\dataset4.yaml --out .\data\dataset4\balance_report.md
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--no-crop` | 按整图统计 |
| `--crop-hw 384,640` | 修改裁切尺寸 |
| `--splits train,val` | 只统计指定 split |
| `--scene-gap 60` | 场景聚类的时间戳间隔（秒） |
| `--groups groups.json` | 分组统计：`{split: {stem: {background, lighting}}}` |
| `--min-*` | 调整各类门槛 |
| `--strict` | 缺 mask / 尺寸不一致 / 非法 mask 值 / 未知标签时返回非 0 |

默认忽略 `*_pred.png`；`manifest_dataset4.json` 存在时自动读取 source / split。

## 5. 训练与验证

### train.py

```bat
# 只校验数据和裁切
python .\tools\train.py --config .\configs\train_dataset3.yaml --check

# 构造一个 batch 看形状
python .\tools\train.py --config .\configs\train_dataset3.yaml --smoke

# 正式训练
python .\tools\train.py --config .\configs\train_dataset3.yaml

# 混合预标注模型（yu_data + x5/data_get）
python .\tools\train.py --config .\configs\train_prelabel_mix.yaml

# dataset4_1（dataset4/train + prelabel_mix 混合划分）
python .\tools\train.py --config .\configs\train_dataset4_1.yaml

# dataset4_2（dataset4_1 + x5/nitu1 + x5/green4）
python .\tools\train.py --config .\configs\train_dataset4_2.yaml

# dataset5_1（dataset4_2 + guangban2，glare 补充）
python .\tools\train.py --config .\configs\train_dataset5_1.yaml
```

常用覆盖：`--epochs`、`--imgsz`、`--batch`、`--device`、`--workers`、
`--project`、`--name`、`--resume`。

### val.py

```bat
python .\tools\val.py --config .\configs\train_dataset3.yaml --split test --plots

# 混合预标注模型逐类 IoU
python .\tools\val.py --config .\configs\train_prelabel_mix.yaml --plots

# dataset4_1
python .\tools\val.py --config .\configs\train_dataset4_1.yaml --plots

# dataset5_1（glare 补充第一批）
python .\tools\val.py --config .\configs\train_dataset5_1.yaml --plots
```

- `--split {val,test,train}`，默认 val。
- `--weights` 默认取 `<project>/<name>/weights/best.pt`。

### analyze_model.py

```bat
# 默认 configs/train_dataset4_1.yaml，读 runs/dataset4_1_6cls/weights/best.pt
python .\tools\analyze_model.py

# 指定权重 / 输出 / 只跑前 50 张
python .\tools\analyze_model.py --weights .\runs\xxx\weights\best.pt ^
  --out .\runs\xxx\analysis_val.md --limit 50
```

- 和 `val.py` 的区别：`val.py` 只出逐类 IoU；这里补图片级漏检/误报、全 clean 帧
  脏像素分布、逐场景 IoU、真值像素去向、面积偏差。
- 图片用 `cv2` 读成 BGR 直接喂 `model.predict`；predictor 内部把 BGR 翻成 RGB，
  自己再翻一次会让细类整体崩掉。
- 在原图分辨率上按 `crop_hw` 裁切后推理，不先缩到短边 640，所以 IoU 和 `val.py`
  会差 0.01 量级。

## 6. 导出与量化

### export_onnx.py

```bat
python .\tools\export_onnx.py --weights .\runs\water_4cls\weights\best.pt
```

- 默认输入 `384×640`、`opset 11`、固定 batch；`--dynamic` 导出动态轴。
- `--simplify` 可选；`--verify` 默认开启（需要 onnxruntime）。
- 默认输出到权重同目录的 `.onnx`，可用 `--output` 指定。

### make_calib.py

```bat
# 从 dataset3 train 抽 10% 到 data/cal_img，再按训练口径转 NCHW raw
python .\tools\make_calib.py

# 只校验已有 .raw
python .\tools\make_calib.py --check

# 只转换已有目录 / 换比例
python .\tools\make_calib.py --images .\data\cal_img
python .\tools\make_calib.py --ratio 0.05 --seed 0 --force
```

默认输出：`data/cal_img`（抽样图）、`data/cal_nchw`（`*.raw`）。

## 7. 图像预处理

### extract_originals.py

```bat
# 默认处理 x5/dirty_imgs/05_water，输出到 <输入目录>\yuantu
python .\tools\extract_originals.py

python .\tools\extract_originals.py -i .\x5\dirty_imgs\05_water -o .\data\water_yuantu

python .\tools\extract_originals.py -i .\x5\dirty_imgs\11 -o .\data\11_yuantu
```

- 两栏图（1284）取左半，三栏图（1920）取左 1/3，栏数可自动识别。
- 常用：`--panels 2|3`、`--gap`（去掉分隔带）、`--auto`、`--format png|jpg`、
  `--limit`、`--overwrite`。

### chou.py

无命令行参数，路径和数量写在脚本里；**直接运行就会复制文件**：

```bat
python .\tools\chou.py
```

当前配置：从 `Prompt_D_Ghosting` 随机抽 100 张（seed 42）到 `data/glare/D`。
改路径或数量请编辑脚本顶部的 `SOURCE_FOLDER` / `TARGET_FOLDER` / `SAMPLE_COUNT`。

## 附：crop_dataset.py

不是独立命令，供训练时安装裁切 Dataset：

```python
from tools.crop_dataset import install

install()
model = YOLO(weights)
```

data yaml 里写 `crop_hw: [384, 640]` 生效；没有该字段时保持原生 `SemanticDataset`。
