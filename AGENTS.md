# AGENTS.md

面向 AI 编码代理的仓库说明。人类读者请先看 [README](README.md)。

## 这是什么

在自有正射影像上训练建筑物轮廓分割模型，并对大图做滑窗推理。基于 MMSegmentation。

## 关键文件

- `prepare/prepare_dataset_multiprocess.py` — 滑窗切片、前景占比过滤、标签栅格化、增量式跳过已存在切片。
- `train.py` — 完整的 MMSegmentation 配置（Python dict，无外部 config 文件），定义 `FootprintDataset` 与 `MaskNoData` 变换。
- `predict.py` — 重叠滑窗推理 + 拼接，输出掩膜与可视化版。
- `prepare/DATA.example.TXT` — 数据集清单模板，格式为 `datasets = [(tif, shp), ...]`，用 `ast.literal_eval` 解析。

## 命令

```bash
conda create -n footprint python=3.10 -y && conda activate footprint
conda install -c conda-forge geopandas rasterio -y
pip install -r requirements.txt

cp prepare/DATA.example.TXT prepare/DATA.TXT
python prepare/prepare_dataset_multiprocess.py
python train.py
python predict.py --input test/mosaic.tif --checkpoint work_dirs/hrnet_w48_footprint/iter_40000.pth
```

## 约定

- `MaskNoData` 把纯黑 `(0,0,0)` 与纯白 `(255,255,255)` 在标签里置为 `255`（`ignore_index`）。这是核心设计，不要移除。
- `train.py` 里 `mmcv.__version__ = '2.1.0'` 是绕过版本强校验的 hack，让 mmcv-lite 能跑，不要当成 bug 修掉。
- `predict.py` 顶部有 `mmcv.ops` 的 polyfill，同理保留。
- 切片尺寸训练与推理必须一致（默认 1024 / 512），否则边界效应会显现。
- 数据集不入库（`.gitignore` 排除 `dataset/`、`work_dirs/`、`test/`）。

## 已知坑

- 标签与影像套合不准时模型学的是系统性偏移，加数据也救不回来。改完切片逻辑务必抽样叠加核对。
- `ast.literal_eval` 解析 LIST，文件里 `datasets =` 之后的方括号结构不能破坏。
