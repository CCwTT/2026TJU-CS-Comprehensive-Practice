"""
Test 集评估脚本.

用法:
    python test_model.py

功能:
  1. val(split='test') — 最终 mAP/Precision/Recall
  2. predict 全部 test 图片 — 保存可视化
"""
import os
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================

DATA_YAML = '../data/training/dataset.yaml'
WEIGHTS = 'runs/detect/runs/train/weights/best.pt'
TEST_IMAGES = '../data/training/test/images'
OUTPUT_DIR = 'runs/detect/test_viz'
DEVICE = 0

# ============================================================
# 主流程
# ============================================================

def main():
    model = YOLO(WEIGHTS)
    print(f'模型: {WEIGHTS}')
    print(f'设备: {DEVICE}')

    # ---- 1. 指标评估 ----
    print('\n' + '=' * 60)
    print('  Test 集指标评估')
    print('=' * 60)
    metrics = model.val(data=DATA_YAML, split='test', device=DEVICE)

    # ---- 2. 全部图片预测 ----
    print('\n' + '=' * 60)
    print('  Test 集图片预测 (保存可视化)')
    print('=' * 60)

    test_imgs = sorted(Path(TEST_IMAGES).glob('*.jpg'))
    print(f'找到 {len(test_imgs)} 张 test 图片')
    print(f'输出到: {OUTPUT_DIR}')

    model.predict(
        source=TEST_IMAGES,
        save=True,
        project='runs/detect',
        name='test_viz',
        exist_ok=True,
        device=DEVICE,
        conf=0.25,
        iou=0.45,
    )

    print('\n' + '=' * 60)
    print('  评估完成!')
    print(f'  指标报告: 见上方输出')
    print(f'  可视化:   {OUTPUT_DIR}/')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()