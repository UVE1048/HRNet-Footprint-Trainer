import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

# ================= Patch for mmcv-lite (Mocking mmcv.ops) =================
try:
    import mmcv
    try:
        import mmcv.ops
    except ImportError:
        print("[Warning] mmcv.ops not found. Applying patch for mmcv-lite...")
        
        class FlexibleMock(MagicMock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def __reduce__(self):
                return (MagicMock, ())

        class MockOpsModule(ModuleType):
            def __getattr__(self, name):
                if name in ('__file__', '__path__', '__loader__', '__spec__'):
                    return super().__getattr__(name)
                return FlexibleMock()
        
        fake_ops = MockOpsModule('mmcv.ops')
        fake_ops.__file__ = "mock_mmcv_ops.py"
        fake_ops.__path__ = []
        sys.modules['mmcv.ops'] = fake_ops
        if 'mmcv' in sys.modules:
            sys.modules['mmcv'].ops = fake_ops
        print("[Success] mmcv.ops patched successfully.")
except Exception as e:
    print(f"[Error] Failed to patch mmcv.ops: {e}")
# ==========================================================================

import torch
# Monkeypatch torch.load to default weights_only=False for PyTorch 2.6+ compatibility
# This fixes: _pickle.UnpicklingError: Weights only load failed.
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
from train import get_cfg, FootprintDataset  # 复用训练配置
import cv2

# ================= 配置区域 =================
# 以下默认值均可通过命令行参数覆盖，见文件末尾的 argparse 定义。

# 1. 待预测的影像路径
INPUT_TIF = os.path.join("test", "sample.tif")

# 2. 模型权重文件
CHECKPOINT_PATH = os.path.join("work_dirs", "hrnet_w48_footprint", "iter_40000.pth")

# 3. 输出路径
OUTPUT_TIF = os.path.join("test", "prediction_result.tif")

# 4. 预测参数
CROP_SIZE = 1024       # 预测时的窗口大小 (与训练保持一致)
STRIDE = 512           # 滑动步长 (小于 CROP_SIZE 以产生重叠，消除边缘效应)
DEVICE = 'cuda:0'      # 无 GPU 时改为 'cpu'
# ===========================================

def predict_large_image():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("请等待训练完成，或修改 CHECKPOINT_PATH 指向存在的 .pth 文件")
        return

    # 1. 初始化模型
    print("Initializing model...")
    register_all_modules()
    cfg = get_cfg()
    
    # 强制修改 cfg.model 以匹配最新的 train.py 配置
    # 关键：预测时不需要辅助头计算 loss，但结构必须一致
    if isinstance(cfg.model.decode_head, list):
        for head in cfg.model.decode_head:
            head.norm_cfg = dict(type='BN', requires_grad=False)
    else:
        cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=False)
        
    cfg.model.backbone.norm_cfg = dict(type='BN', requires_grad=False)

    # Ensure test_pipeline is accessible at top level
    if not hasattr(cfg, 'test_pipeline'):
        try:
            cfg.test_pipeline = cfg.test_dataloader.dataset.pipeline
        except Exception:
            # Fallback if structure is different
            cfg.test_pipeline = [
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs')
            ]

    # 强制在测试模式下使用全图尺寸的 pipeline (虽然我们是滑窗，但保持一致性)
    # cfg.test_pipeline[1].type = 'LoadAnnotations' # 临时修复 pipeline 里的 LoadAnnotations 依赖
    
    # Remove LoadAnnotations for inference if present, as we don't have labels for prediction
    cfg.test_pipeline = [p for p in cfg.test_pipeline if p['type'] != 'LoadAnnotations']

    # 关键：手动加载 Checkpoint，跳过不匹配的键
    # 因为我们训练时用了 CascadeEncoderDecoder，有些 key 可能有变化
    # 但 init_model 会自动处理大部分情况
    
    model = init_model(cfg, CHECKPOINT_PATH, device=DEVICE)

    # 2. 打开原始影像
    print(f"Opening input image: {INPUT_TIF}")
    with rasterio.open(INPUT_TIF) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()
        
        # 更新 profile 为单波段 (预测结果只有 0/1)
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

        # 3. 初始化结果数组 (内存中)
        # 注意：如果图太大 (如 > 20000x20000)，可能需要改成写文件模式
        print(f"Image size: {width}x{height}. Allocating memory for result...")
        result_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 辅助数组，用于记录每个像素被预测了多少次 (处理重叠区域)
        # 为了节省内存，我们可以简单地直接覆盖，或者取最大值。
        # 这里采用简单的策略：中心区域优先。但为了代码简单且有效，
        # 我们使用 MMSegmentation 的 slide_inference 风格：
        # 由于我们手动滑窗，我们简单地只取置信度高的，或者直接覆盖。
        # 改进策略：直接使用带有重叠的滑窗，取中间部分，或者直接覆盖。
        # 鉴于 STRIDE=512, CROP=1024，重叠很大。
        
        # 为了效果最好，我们逐块预测并填入
        
        ys = list(range(0, height - CROP_SIZE + 1, STRIDE))
        xs = list(range(0, width - CROP_SIZE + 1, STRIDE))
        
        # 补齐边缘
        if ys[-1] != height - CROP_SIZE: ys.append(height - CROP_SIZE)
        if xs[-1] != width - CROP_SIZE: xs.append(width - CROP_SIZE)

        total_patches = len(ys) * len(xs)
        print(f"Start prediction on {total_patches} patches...")

        with tqdm(total=total_patches) as pbar:
            for y in ys:
                for x in xs:
                    # 读取窗口数据
                    window = Window(x, y, CROP_SIZE, CROP_SIZE)
                    img_data = src.read(window=window) # (C, H, W)
                    
                    # 转换格式: (C, H, W) -> (H, W, C) -> BGR (opencv)
                    img_data = np.transpose(img_data, (1, 2, 0))
                    
                    # 处理通道数
                    if img_data.shape[2] > 3:
                        img_data = img_data[:, :, :3]
                    elif img_data.shape[2] == 1:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                    
                    # 确保是 BGR (rasterio 读出来通常是 RGB，OpenCV 需要 BGR)
                    # MMSegmentation 的 LoadImageFromFile 默认是用 mmcv (cv2) 读的 BGR
                    # 但 rasterio 读的是 RGB。所以转一下
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

                    # 预测
                    result = inference_model(model, img_data)
                    
                    # 获取预测的 mask (pred_sem_seg)
                    pred_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8) # (H, W)

                    # 填充回大图
                    # 简单的策略：直接覆盖。
                    # 高级策略：为了避免边缘效应，我们只保留中心区域 (Gaussian weight)，但这里简化处理
                    # 由于我们有重叠，后预测的会覆盖先预测的。
                    # 为了平滑，我们可以只更新 result_mask 中为 0 的地方，或者取并集？
                    # 建筑提取通常取并集比较好 (防止漏检)，或者覆盖。
                    # 这里采用直接覆盖，但在边缘处可能会有痕迹。
                    # 更好的方法：只取中心 STRIDE 大小的区域写入，除非是图像边缘。
                    
                    # 简单的中心裁剪写入策略：
                    # 定义有效写入区域
                    y_start_write = y
                    y_end_write = y + CROP_SIZE
                    x_start_write = x
                    x_end_write = x + CROP_SIZE
                    
                    # 如果不是第一行，避开顶部重叠区
                    if y > 0 and y < height - CROP_SIZE:
                        y_start_write += (CROP_SIZE - STRIDE) // 2
                        
                    # 如果不是最后一行，避开底部重叠区
                    if y < height - CROP_SIZE and y > 0:
                        y_end_write -= (CROP_SIZE - STRIDE) // 2

                    # 同理处理 X 轴
                    if x > 0 and x < width - CROP_SIZE:
                        x_start_write += (CROP_SIZE - STRIDE) // 2
                    if x < width - CROP_SIZE and x > 0:
                        x_end_write -= (CROP_SIZE - STRIDE) // 2
                        
                    # 计算在 crop 中的相对坐标
                    crop_y1 = y_start_write - y
                    crop_y2 = crop_y1 + (y_end_write - y_start_write)
                    crop_x1 = x_start_write - x
                    crop_x2 = crop_x1 + (x_end_write - x_start_write)
                    
                    result_mask[y_start_write:y_end_write, x_start_write:x_end_write] = \
                        pred_mask[crop_y1:crop_y2, crop_x1:crop_x2]

                    pbar.update(1)

        # 4. 保存结果
        profile = src.profile.copy()
        profile.update(count=1, dtype=rasterio.uint8)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(OUTPUT_TIF), exist_ok=True)
        
        try:
            print(f"Saving result to {OUTPUT_TIF}...")
            with rasterio.open(OUTPUT_TIF, 'w', **profile) as dst:
                dst.write(result_mask, 1)
        except Exception as e:
            print(f"Error saving result to {OUTPUT_TIF}: {e}")
            traceback.print_exc()
            
        # 5. 保存可视化结果 (白色掩膜)
        VIS_OUTPUT_TIF = OUTPUT_TIF.replace('.tif', '_vis.tif')
        print(f"Saving visualization to {VIS_OUTPUT_TIF}...")
        
        # 创建 RGB 图像: 0 -> 黑色, 1 -> 白色
        # result_mask 是 (H, W), rasterio 需要 (3, H, W)
        vis_mask = np.zeros((3, height, width), dtype=np.uint8)
        
        # 将掩膜为 1 的区域设为 255 (白色)
        mask_indices = (result_mask == 1)
        vis_mask[0][mask_indices] = 255 # R
        vis_mask[1][mask_indices] = 255 # G
        vis_mask[2][mask_indices] = 255 # B
        
        vis_profile = profile.copy()
        vis_profile.update(count=3, photometric='RGB')
        
        with rasterio.open(VIS_OUTPUT_TIF, 'w', **vis_profile) as dst:
            dst.write(vis_mask)
        
        print("Done!")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Sliding-window building-footprint inference over a large GeoTIFF."
    )
    parser.add_argument("--input", default=INPUT_TIF, help="input GeoTIFF path")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="model weights (.pth)")
    parser.add_argument("--output", default=OUTPUT_TIF, help="output GeoTIFF path")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE, help="sliding window size")
    parser.add_argument("--stride", type=int, default=STRIDE, help="sliding window stride")
    parser.add_argument("--device", default=DEVICE, help="torch device, e.g. cuda:0 or cpu")
    args = parser.parse_args()

    # 覆盖模块级配置，predict_large_image() 在调用时才读取这些全局量
    INPUT_TIF = args.input
    CHECKPOINT_PATH = args.checkpoint
    OUTPUT_TIF = args.output
    CROP_SIZE = args.crop_size
    STRIDE = args.stride
    DEVICE = args.device

    predict_large_image()
