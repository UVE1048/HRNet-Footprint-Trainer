# HRNet-Footprint-Trainer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MMSegmentation](https://img.shields.io/badge/MMSegmentation-1.2%2B-orange.svg)](https://github.com/open-mmlab/mmsegmentation)
[![Stars](https://img.shields.io/github/stars/UVE1048/HRNet-Footprint-Trainer?style=social)](https://github.com/UVE1048/HRNet-Footprint-Trainer/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/UVE1048/HRNet-Footprint-Trainer)](https://github.com/UVE1048/HRNet-Footprint-Trainer/commits/main)
[![CI](https://github.com/UVE1048/HRNet-Footprint-Trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/UVE1048/HRNet-Footprint-Trainer/actions/workflows/ci.yml)

**在你自己的正射影像上训练建筑物轮廓分割模型，输出能对大图做滑窗推理的权重。**

Train HRNet building-footprint segmentation on your own orthophotos: sliding-window dataset generation, NoData-aware training, large-image sliding-window inference.

```text
原始大影像 (.tif) + 建筑物矢量 (.shp)
        │
        ▼  prepare_dataset_multiprocess.py   滑窗切片 + 前景占比过滤 + NoData 剔除
   dataset/{train,val}/{images,masks}  +  *_list.txt
        │
        ▼  train.py                          HRNet-W48 + FCNHead，MMSegmentation
   work_dirs/hrnet_w48_footprint/iter_*.pth
        │
        ▼  predict.py                        重叠滑窗推理 + 无缝拼接
   test/prediction_result.tif  +  _vis.tif
```

---

**也叫**：建筑物分割训练 · HRNet 训练脚本 · 遥感分割训练模板 · 建筑物提取模型训练 · building segmentation training · HRNet segmentation pipeline

**什么时候该用**：现成模型的精度在你的影像上不够，而你手上有本地测绘的建筑物矢量可以当标签。
**什么时候不该用**：完全没有标注数据（先用现成权重，见 Footprint-Creater）；只需推理不训练。

## 这是什么

三个脚本，一条链路：**大影像 + 矢量标签 → 切片数据集 → HRNet 训练 → 大图滑窗推理**。

适合这种情况：通用遥感模型在你的影像上边界不准 —— 季节不同、传感器不同、城市肌理不同 —— 而你手上有本地测绘的建筑物矢量。那就别微调别人的权重，直接训一个。

**关键词**：建筑物轮廓提取 · 建筑物分割 · 语义分割训练 · 遥感影像分割 · 正射影像 · HRNet · HRNet-W48 · MMSegmentation · OpenMMLab · 滑窗切片 · 重叠推理 · 数据集生成 · NoData 处理 · 遥感深度学习
**Keywords**: building footprint segmentation · building extraction · HRNet training · MMSegmentation · OpenMMLab · remote sensing segmentation · sliding window inference · tiling · large GeoTIFF inference · NoData masking · dataset preparation · PyTorch

## 为什么值得自己训

现成的建筑物提取模型基本都在 Inria、SpaceNet 这类公开集上训的。它们在郊区独立房屋上表现很好，到高密度城区就开始出问题：

- 楼挨得太近，**粘连成一片**
- 阴影方向不一样，**边界系统性偏移**
- 屋顶材质差异大（彩钢瓦、绿化屋面），**直接漏检**

这套流程的所有取舍都指向一件事：**让模型吃你自己的地物分布**。所以切片阶段有前景占比过滤（剔除全空/全满样本），有 NoData 掩码让黑边不参与 loss，有 50% 重叠滑窗消除拼缝。

## 环境安装

推荐用 conda，主要为了 GDAL / GEOS 的二进制依赖：

```bash
conda create -n footprint python=3.10 -y
conda activate footprint
conda install -c conda-forge geopandas rasterio -y
pip install -r requirements.txt
```

PyTorch 建议按自己的 CUDA 版本从[官网](https://pytorch.org/get-started/locally/)安装，`requirements.txt` 里的 `torch>=2.0` 只是兜底。

Windows 不想手动配环境可以跑 `prepare/setup_geo_env.ps1`（需要 conda 已在 PATH）。

> **mmcv 编译不动？** 换 `pip install mmcv-lite>=2.1.0`。`train.py` 里那行 `mmcv.__version__ = '2.1.0'`
> 的 hack 就是为了绕过 MMSegmentation 对 mmcv 版本的强校验，让 lite 版能跑起来。

## 完整流程

### 1. 准备数据集

需要两类原始数据，**坐标系必须一致**：

- 高分辨率正射影像：`.tif` / `.tiff`，建议 GSD ≤ 0.2 m
- 建筑物轮廓矢量：`.shp`，一个图层即可，不区分属性

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

也可以用 `python prepare/prepare_dataset_multiprocess.py --config /path/to/my_list.txt` 指定别的配置。

### 2. 切片

```bash
python prepare/prepare_dataset_multiprocess.py
```

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

**增量式**：重跑时已存在的切片会被跳过，新增数据源只需往 `DATA.TXT` 追加一行。补数据不用重头再来。

### 3. 训练

```bash
python train.py
```

`train.py` 用 Python dict 直接定义完整配置（模型、数据集、数据增强、优化器、学习率策略），没有额外的 config 文件要维护。

默认配置：

- 骨干 `HRNet`（`hrnet_w48`），解码器 `FCNHead`，`num_classes=2`（背景 / 建筑）
- 损失 = `CrossEntropyLoss` + `DiceLoss`，各权重 1.0
- 优化器 `SGD` + `PolyLR` 衰减
- 归一化用 ImageNet 均值方差，`bgr_to_rgb=True`

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
| `--input` | `test/sample.tif` | 待预测大影像 |
| `--checkpoint` | `work_dirs/hrnet_w48_footprint/iter_40000.pth` | 权重 |
| `--output` | `test/prediction_result.tif` | 输出掩膜 |
| `--crop-size` | 1024 | 滑窗边长，与训练一致 |
| `--stride` | 512 | 滑窗步长 |
| `--device` | `cuda:0` | 无 GPU 用 `cpu` |

产出两个文件：`prediction_result.tif`（二值掩码，保留地理参考）和 `prediction_result_vis.tif`（白建筑黑背景，可直接拖进 QGIS）。

## 关键设计

**`MaskNoData` 变换** —— 加载切片时把纯黑 `(0,0,0)` 与纯白 `(255,255,255)` 区域在标签中置为 `255`。`255` 是 MMSegmentation 的 `ignore_index`，算 loss 时完全跳过。这样模型不会因为"把影像黑边预测成背景"被惩罚。这是全套流程里最见效的一个改动。

**切片级双重过滤** —— 前景占比 `[0.01, 0.98]` 之外全丢。全空样本让模型学会"这里没建筑"但没信息量，全满样本同理。丢掉它们显著降低训练时间而不掉精度。

**重叠滑窗推理** —— 用重叠窗口预测再拼接，既绕开大图无法整张进显存的问题，也用重叠区加权消除拼缝断层。

**增量式数据集构建** —— 扫描已有切片并跳过，多进程处理新增部分。补数据是常态，这一步省的时间最多。

## 预训练权重

不想自己训可以直接拿训好的 HRNet-W48 权重：

```bash
python scripts/download_models.py         # 约 542 MB，带 SHA256 校验
python scripts/download_models.py --check # 只校验本地文件
```

权重落到 `work_dirs/hrnet_w48_footprint/iter_40000.pth`。同一份权重也用于 [Footprint-Creater](https://github.com/UVE1048/Footprint-Creater)（带 Web 界面的提取工具），所以只托管在那一侧的 Release，避免同一个 542 MB 文件存两份。

```text
SHA256  73db3f97998fc62cbbcb83897925d1c6ffaef2e1dadd78b5b7f90267dab5cf18
```

## 与替代方案对比

| 方案 | 用自己数据训练 | 大图推理 | 标签格式 | 上手成本 |
| :--- | :--- | :--- | :--- | :--- |
| **HRNet-Footprint-Trainer** | ✅ 滑窗切片全自动 | ✅ 重叠滑窗拼接 | 栅格 + SHP 矢量 | 填个 TXT 就能跑 |
| MMSegmentation 官方示例 | 需自己写 dataset 类 | ❌ 只给切片推理 | 依赖 ISPRS 格式 | 中等 |
| U-Net 类教程仓库 | ✅ | 通常缺大图方案 | 多样 | 需自己拼 |
| 直接用现成模型推理 | ❌ | 部分支持 | — | 低但精度受限 |

第一列是主要差异：多数教程给你一个切片级的训练脚本，本文这个仓库把"从 SHP 标签到可训练数据集"和"从权重到大图掩膜"这两段补齐了。

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
├── scripts/download_models.py
└── requirements.txt
```

运行后自动生成 `dataset/`、`work_dirs/`、`test/`（均已在 `.gitignore` 中排除）。

## 已知限制

- **标签质量决定上限**。矢量和影像套合不准（坐标系或时相不一致）时，模型学到的是系统性边界偏移，加数据也没用。训之前先抽几处叠加检查。
- **只有建筑/背景两类**。要区分屋面类型、层数得改 `METAINFO` 的 `classes` 并重做标签。
- **`prepare/check.py` 是 YOLO 标注格式的校验脚本**，与主链路（语义分割）不是同一条线，按需使用。
- **显存与切片尺寸强相关**。1024 切片 + W48 约需 8 GB；8 GB 卡上想再大就换 W32 或降 `CROP_SIZE`。

## 引用

```bibtex
@software{hrnetfootprinttrainer2026,
  title  = {HRNet-Footprint-Trainer: Building footprint segmentation training pipeline},
  author = {UVE},
  year   = {2026},
  url    = {https://github.com/UVE1048/HRNet-Footprint-Trainer}
}
```

若在论文中使用，也建议一并引用骨干网络原文：

```bibtex
@inproceedings{wang2020hrnet,
  title     = {Deep High-Resolution Representation Learning for Visual Recognition},
  author    = {Wang, Jingdong and Sun, Ke and Cheng, Tianheng and others},
  booktitle = {IEEE TPAMI},
  year      = {2020}
}
```

## 相关项目

同一套实景三维链路上的其他工具：

- [ObjOrthoRender](https://github.com/UVE1048/ObjOrthoRender)：OBJ 模型转带地理参考的 GeoTIFF 正射影像
- [Footprint-Creater](https://github.com/UVE1048/Footprint-Creater)：正射影像建筑物轮廓提取，输出 Shapefile
- [HRNet-Footprint-Trainer](https://github.com/UVE1048/HRNet-Footprint-Trainer)：建筑物分割模型训练：切片、训练、大图推理 ←
- [VideoFusion-Web](https://github.com/UVE1048/VideoFusion-Web)：无人机视频融合投射到三维实景模型

在线文档：<https://uve1048.github.io/hrnet-footprint-trainer/>

## 许可

代码以 [MIT](LICENSE) 协议开源。

基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)（Apache-2.0）构建，使用时请遵循其原始协议。其余依赖：PyTorch(BSD-3)、rasterio(BSD-3)、GeoPandas(BSD-3)、OpenCV(Apache-2.0)、tqdm(MIT)。

**训练数据不随本仓库分发**。使用者需自行确保影像与矢量的获取、使用符合当地测绘与数据合规要求。
