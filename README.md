# HRNet-Footprint-Trainer

在**自己的**高分辨率正射影像上训练一个建筑物轮廓分割模型，输出可对大图做滑窗推理的权重。

> English: Training pipeline for building-footprint segmentation on high-resolution orthophotos. HRNet-W48 + FCNHead on MMSegmentation, with sliding-window dataset generation, a NoData-aware ignore-index transform, and sliding-window inference over arbitrarily large GeoTIFFs.

三个脚本，一条链路：**大影像 + 矢量标签 → 切片数据集 → HRNet 训练 → 大图滑窗推理**。

最适合的场景是：通用遥感模型在你的影像上（不同季节、不同传感器、不同城市肌理）边界不准，而你手上有本地测绘的建筑物矢量。那就别微调别人的权重了，直接训一个。

## 目录

- [和现成模型比，为什么要自己训](#和现成模型比为什么要自己训)
- [链路概览](#链路概览)
- [环境安装](#环境安装)
- [完整流程](#完整流程)
  - [1. 准备数据集](#1-准备数据集)
  - [2. 切片](#2-切片)
  - [3. 训练](#3-训练)
  - [4. 推理](#4-推理)
- [关键设计](#关键设计)
- [预训练权重](#预训练权重)
- [项目结构](#项目结构)
- [已知限制](#已知限制)
- [许可](#许可)

## 和现成模型比，为什么要自己训

现成的建筑物提取模型基本都在 Inria、SpaceNet 这类公开集上训的。它们在郊区独立房屋上表现很好，到高密度城区就开始出问题——楼挨得太近，粘连成一片；阴影方向不一样，边界系统性偏移；屋顶材质（彩钢瓦、绿化屋面）差异大，直接漏检。

这套流程的所有设计取舍都指向一件事：**让模型吃你自己的地物分布**。所以切片阶段有前景占比过滤（剔除全空和全满的样本），有 NoData 掩码（黑边不参与 loss），有 50% 的重叠滑窗（消除拼缝）。

## 链路概览

```text
原始大影像 (.tif)  +  建筑物矢量 (.shp)
        │
        ▼  prepare_dataset_multiprocess.py   （滑窗切片 + 前景占比过滤 + NoData 剔除）
   dataset/{train,val}/{images,masks}  +  *_list.txt
        │
        ▼  train.py                          （HRNet-W48 + FCNHead，MMSegmentation）
   work_dirs/hrnet_w48_footprint/iter_*.pth
        │
        ▼  predict.py                        （重叠滑窗推理 + 无缝拼接）
   test/prediction_result.tif  +  _vis.tif
```

## 环境安装

**推荐用 conda**，主要为了 GDAL / GEOS 的二进制依赖：

```bash
conda create -n footprint python=3.10 -y
conda activate footprint
conda install -c conda-forge geopandas rasterio -y
pip install -r requirements.txt
```

PyTorch 建议按自己的 CUDA 版本从[官网](https://pytorch.org/get-started/locally/)装，`requirements.txt` 里的 `torch>=2.0` 只是兜底。

Windows 上不想手动配环境，可以跑 `prepare/setup_geo_env.ps1`（PowerShell，需要 conda 先在 PATH 里）。

> **mmcv 编译不动？** 换成 `pip install mmcv-lite>=2.1.0`。`train.py` 里有一行 `mmcv.__version__ = '2.1.0'`
> 的 hack，就是为了绕过 MMSegmentation 对 mmcv 版本的强校验，让 lite 版能跑起来。

## 完整流程

### 1. 准备数据集

需要两类原始数据，**坐标系必须一致**：

- 高分辨率正射影像：`.tif` / `.tiff`，建议 GSD ≤ 0.2 m
- 建筑物轮廓矢量：`.shp`，一个图层一个要素即可，不用区分属性

把路径填进 `prepare/DATA.TXT`：

```bash
cp prepare/DATA.example.TXT prepare/DATA.TXT
```

```python
    datasets = [
        # (TIF_PATH, SHP_PATH)
        (r"/data/area_a/AreaA.tif", r"/data/area_a/buildings.shp"),
        (r"/data/area_b/AreaB.tif", r"/data/area_b/buildings.shp"),
    ]
```

也可以 `python prepare/prepare_dataset_multiprocess.py --config /path/to/my_list.txt` 指定别的配置文件。

### 2. 切片

```bash
python prepare/prepare_dataset_multiprocess.py
```

默认参数（在脚本顶部，可直接改）：

| 参数 | 默认值 | 含义 |
| :--- | :--- | :--- |
| `CROP_SIZE` | 1024 | 切片边长（像素） |
| `STRIDE` | 512 | 滑窗步长，即 50% 重叠 |
| `TRAIN_RATIO` | 0.8 | 训练集比例 |
| `MIN_FG_RATIO` | 0.01 | 建筑像素低于 1% 的切片丢弃 |
| `MAX_FG_RATIO` | 0.98 | 建筑像素高于 98% 的切片丢弃 |
| `NUM_PROCESSES` | min(cpu, 32) | 多进程数 |

产出：

```text
dataset/
├── train/{images,masks,masks_vis}/<name>_<x>_<y>.png
├── val/{images,masks,masks_vis}/
├── train_list.txt        # "images/xx.png masks/xx.png"
└── val_list.txt
```

**增量式**：重跑时已存在的切片会被跳过，新增数据源只需往 `DATA.TXT` 里追加一行。所以补数据不用重头再来。

### 3. 训练

```bash
python train.py
```

`train.py` 里用 Python dict 直接定义了完整配置（模型、数据集、数据增强、优化器、学习率策略），没有额外的 config 文件要维护。想调参就改脚本里的数字。

默认配置要点：

- 骨干 `HRNet`（`hrnet_w48`），解码器 `FCNHead`，`num_classes=2`（背景 / 建筑）
- 损失 = `CrossEntropyLoss` + `DiceLoss`，各权重 1.0
- 优化器 `SGD`，学习率 `PolyLR` 衰减
- 输入归一化用 ImageNet 均值方差，`bgr_to_rgb=True`

日志与权重落在 `work_dirs/hrnet_w48_footprint/`。

### 4. 推理

```bash
python predict.py \
  --input test/mosaic.tif \
  --checkpoint work_dirs/hrnet_w48_footprint/iter_40000.pth \
  --output test/prediction_result.tif \
  --device cuda:0
```

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--input` | `test/sample.tif` | 待预测的大影像 |
| `--checkpoint` | `work_dirs/hrnet_w48_footprint/iter_40000.pth` | 权重 |
| `--output` | `test/prediction_result.tif` | 输出掩膜 |
| `--crop-size` | 1024 | 滑窗边长，与训练一致 |
| `--stride` | 512 | 滑窗步长 |
| `--device` | `cuda:0` | 无 GPU 用 `cpu` |

产出两个文件：

- `prediction_result.tif` — 二值掩膜，0 背景 / 1 建筑，保留原影像的地理参考
- `prediction_result_vis.tif` — 白建筑黑背景的可视化版，直接拖进 QGIS 就能看

## 关键设计

**`MaskNoData` 变换（`train.py`）** — 加载每张切片时，把纯黑 `(0,0,0)` 和纯白 `(255,255,255)` 区域在标签里的对应像素置为 `255`。`255` 是 MMSegmentation 的 `ignore_index`，算 loss 时这些位置被完全跳过。这样模型不会因为"把影像黑边预测成背景"而被惩罚，注意力都集中在有效区域。这是这套流程里最见效的一个改动。

**切片级的双重过滤** — 前景占比 `[0.01, 0.98]` 之外的全丢。全空样本让模型学会"这里没有建筑"但没信息量；全满样本同理。丢掉它们能显著降低训练时间而不掉精度。

**重叠滑窗推理** — `predict.py` 用重叠窗口预测再拼接，既绕开大图无法整张进显存的问题，也用重叠区的加权消除了拼缝处的断层。

**增量式数据集构建** — `prepare_dataset_multiprocess.py` 会扫描已有切片并跳过，多进程并行处理新增部分。补数据是常态，这一步省的时间最多。

## 预训练权重

如果不想自己训，可以直接拿训好的 HRNet-W48 权重：

```bash
python scripts/download_models.py         # 约 542 MB，带 SHA256 校验
python scripts/download_models.py --check # 只校验本地文件
```

权重会落到 `work_dirs/hrnet_w48_footprint/iter_40000.pth`。

这份权重同时用于 [Footprint-Creater](https://github.com/UVE1048/Footprint-Creater)（带 Web 界面的提取工具），所以只托管在那一侧的 Release 里，避免同一个 542 MB 文件存两份。

```text
SHA256  73db3f97998fc62cbbcb83897925d1c6ffaef2e1dadd78b5b7f90267dab5cf18
```

`Footprint-Creater` 里还有个更轻的 HRNet-W32 权重（约 231 MB），如果你的显存吃紧，可以在 `train.py` 里把 `hrnet_w48` 换成 `hrnet_w32` 自己训一个。

## 项目结构

```text
.
├── train.py                              # 模型/数据集/训练流程定义
├── predict.py                            # 大图滑窗推理
├── prepare/
│   ├── prepare_dataset_multiprocess.py   # 滑窗切片 + 过滤 + NoData 处理
│   ├── check.py                          # 切片标注校验（YOLO 格式，可选）
│   ├── setup_geo_env.ps1                 # Windows conda 环境脚本
│   └── DATA.example.TXT                  # 数据集清单模板
├── scripts/
│   └── download_models.py                # 拉取预训练权重
├── requirements.txt
└── .gitignore                            # 排除 dataset/ work_dirs/ test/ *.pth
```

运行后自动生成：`dataset/`（切片）、`work_dirs/`（日志与权重）、`test/`（推理结果）。

## 已知限制

- **标签质量决定上限**。矢量和影像套合不准（常见于坐标系或时相不一致）时，模型学到的是系统性的边界偏移，加数据也没用。训之前先抽几处叠加检查。
- **类别只有建筑/背景两类**。要区分屋面类型、层数得改 `METAINFO` 的 `classes` 并重做标签。
- **`prepare/check.py` 是 YOLO 标注格式的校验脚本**，和主链路（语义分割）不是同一条线，按需使用。
- **显存与切片尺寸强相关**。1024 切片 + W48 大约要 8 GB；在 8 GB 卡上想再大就换 W32 或降 `CROP_SIZE`。

## 许可

[MIT](LICENSE)
