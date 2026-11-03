from mmengine.config import Config
from mmengine.runner import Runner
import mmcv
# Hack: Bypass mmseg version check for mmcv 2.2.0
mmcv.__version__ = '2.1.0'
from mmseg.utils import register_all_modules
import mmseg.models
from mmseg.datasets import BaseSegDataset
from mmengine.registry import DATASETS, TRANSFORMS
from mmcv.transforms import BaseTransform
import numpy as np
import os.path as osp

@TRANSFORMS.register_module()
class MaskNoData(BaseTransform):
    """将图像中纯黑(0,0,0)和纯白(255,255,255)区域在Mask中设为ignore_index(255)"""
    def transform(self, results):
        img = results['img'] # (H, W, 3) BGR
        gt_seg_map = results['gt_seg_map'] # (H, W)
        
        # 找到所有通道都为0的像素 (纯黑)
        is_black = np.all(img == 0, axis=2)
        # 找到所有通道都为255的像素 (纯白)
        is_white = np.all(img == 255, axis=2)
        
        # 将这些位置的 Label 设为 255 (MMSeg 默认的 ignore_index)
        gt_seg_map[is_black] = 255
        gt_seg_map[is_white] = 255
        
        results['gt_seg_map'] = gt_seg_map
        return results

@DATASETS.register_module()
class FootprintDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 0, 0]]
    )
    def load_data_list(self):
        data_list = []
        with open(self.ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, seg_map_path = line.split()
                data_list.append(dict(
                    img_path=osp.join(self.data_root, img_path),
                    seg_map_path=osp.join(self.data_root, seg_map_path),
                    reduce_zero_label=False,
                    seg_fields=[]
                ))
        return data_list

def get_cfg():
    dataset_name = "footprint"
    data_root = "dataset"
    crop_size = (1024, 1024)
    num_classes = 2

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='MaskNoData'),  # 在这里处理 NoData，必须在任何 Resize/Crop 之前！
        dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 1.2), keep_ratio=True),
        dict(type='Pad', size=(1024, 1024), pad_val=dict(img=0, seg=255)), # 确保缩小后图像尺寸不小于 Crop 尺寸
        dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', prob=0.5, degree=180),
        dict(type='PhotoMetricDistortion'),
        dict(type='PackSegInputs')
    ]

    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    ]

    cfg = Config(dict(
        default_scope='mmseg',
        launcher='none',
        work_dir='./work_dirs/hrnet_w48_footprint',
        randomness=dict(seed=42),

        model=dict(
            type='CascadeEncoderDecoder',
            num_stages=2,
            data_preprocessor=dict(
                type='SegDataPreProcessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                pad_val=0,
                seg_pad_val=255,
                size_divisor=32
            ),
            backbone=dict(
                type='HRNet',
                init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w48'),
                norm_cfg=dict(type='BN', requires_grad=False),
                norm_eval=True,
                extra=dict(
                    stage1=dict(
                        num_modules=1,
                        num_branches=1,
                        block='BOTTLENECK',
                        num_blocks=(4,),
                        num_channels=(64,)
                    ),
                    stage2=dict(
                        num_modules=1,
                        num_branches=2,
                        block='BASIC',
                        num_blocks=(4, 4),
                        num_channels=(48, 96)
                    ),
                    stage3=dict(
                        num_modules=4,
                        num_branches=3,
                        block='BASIC',
                        num_blocks=(4, 4, 4),
                        num_channels=(48, 96, 192)
                    ),
                    stage4=dict(
                        num_modules=3,
                        num_branches=4,
                        block='BASIC',
                        num_blocks=(4, 4, 4, 4),
                        num_channels=(48, 96, 192, 384)
                    )
                )
            ),
            decode_head=[
                dict(
                    type='FCNHead',
                    in_channels=[48, 96, 192, 384],
                    in_index=(0, 1, 2, 3),
                    input_transform='resize_concat',
                    channels=256,
                    kernel_size=1,
                    num_convs=1,
                    concat_input=False,
                    dropout_ratio=0.1,
                    num_classes=num_classes,
                    norm_cfg=dict(type='BN', requires_grad=False),
                    align_corners=False,
                    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
                ),
                dict(
                    type='OCRHead',
                    in_channels=[48, 96, 192, 384],
                    in_index=(0, 1, 2, 3),
                    input_transform='resize_concat',
                    channels=512,
                    ocr_channels=256,
                    dropout_ratio=0.1,
                    num_classes=num_classes,
                    norm_cfg=dict(type='BN', requires_grad=False),
                    align_corners=False,
                    sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
                    loss_decode=[
                        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                        # dict(type='DiceLoss', loss_weight=1.0) # DiceLoss in OCRHead with Cascade might be problematic
                    ]
                )
            ],
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        ),

        train_dataloader=dict(
            batch_size=2,
            num_workers=2,
            persistent_workers=True,
            sampler=dict(type='InfiniteSampler', shuffle=True),
            dataset=dict(
                type='FootprintDataset',
                data_root=data_root,
                data_prefix=dict(img_path='', seg_map_path=''),
                ann_file='train_list.txt',
                pipeline=train_pipeline
            )
        ),

        val_dataloader=dict(
            batch_size=1,
            num_workers=2,
            persistent_workers=True,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type='FootprintDataset',
                data_root=data_root,
                data_prefix=dict(img_path='', seg_map_path=''),
                ann_file='val_list.txt',
                pipeline=test_pipeline
            )
        ),

        test_dataloader=dict(
            batch_size=1,
            num_workers=2,
            persistent_workers=True,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type='FootprintDataset',
                data_root=data_root,
                data_prefix=dict(img_path='', seg_map_path=''),
                ann_file='val_list.txt',
                pipeline=test_pipeline
            )
        ),

        val_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU']),
        test_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU']),

        optim_wrapper=dict(
            type='OptimWrapper',
            optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4),
            accumulative_counts=2,
            clip_grad=dict(max_norm=0.1, norm_type=2)
        ),

        param_scheduler=[
            dict(
                type='PolyLR',
                eta_min=1e-4,
                power=0.9,
                begin=0,
                end=40000,
                by_epoch=False
            )
        ],

        train_cfg=dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000),
        val_cfg=dict(type='ValLoop'),
        test_cfg=dict(type='TestLoop'),

        default_hooks=dict(
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            visualization=dict(type='SegVisualizationHook')
        ),

        env_cfg=dict(
            cudnn_benchmark=True,
            mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
            dist_cfg=dict(backend='nccl')
        ),
    ))
    return cfg

def main():
    cfg = get_cfg()
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()