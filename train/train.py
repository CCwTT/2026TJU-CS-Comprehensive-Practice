"""
YOLOv8 危险标志检测训练脚本.

基于预训练 yolov8n.pt 微调, 5 类警告标志检测.
200 epochs, freeze=5, batch=16.

用法:
    python train.py
"""
from ultralytics import YOLO

# ============================================================
# 训练配置
# ============================================================

# 数据集配置 (相对于 train/ 目录)
DATA_YAML = '../data/training/dataset.yaml'

# 模型 (预训练权重, 首次运行自动下载)
MODEL = 'yolov8n.pt'

# 训练参数
EPOCHS = 200
BATCH = 16
IMGSZ = 640
FREEZE = 5           # 冻结 backbone 前 5 层
PATIENCE = 50        # 50 epoch 不提升自动早停

# 优化器
LR0 = 0.01
LRF = 0.01

# 硬件
DEVICE = 0           # GPU 0; CPU 用 'cpu'

# 输出 (最终路径: runs/detect/train/)
PROJECT = 'runs'
NAME = 'train'

# ============================================================
# 训练
# ============================================================

if __name__ == '__main__':
    print(f'[INFO] 加载模型: {MODEL}')
    model = YOLO(MODEL)

    print(f'[INFO] 数据集配置: {DATA_YAML}')
    print(f'[INFO] 训练参数: epochs={EPOCHS}, batch={BATCH}, '
          f'imgsz={IMGSZ}, freeze={FREEZE}, device={DEVICE}')
    print(f'[INFO] 输出目录: {PROJECT}/detect/{NAME}')

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        freeze=FREEZE,
        patience=PATIENCE,
        lr0=LR0,
        lrf=LRF,
        device=DEVICE,
        workers=0,             # Windows spawn
        cache=True,            # 全量加载到 RAM
        optimizer='auto',      # AdamW
        warmup_epochs=3,
        cos_lr=True,
        project=PROJECT,
        name=NAME,
        save=True,
        save_period=10,        # 每 10 epoch 存一次
        plots=True,
        exist_ok=True,
        verbose=True,
    )

    print(f'\n[INFO] 训练完成!')
    save_dir = getattr(results, 'save_dir', f'{PROJECT}/detect/{NAME}')
    print(f'  最佳权重: {save_dir}/weights/best.pt')
    print(f'  结果目录: {save_dir}')