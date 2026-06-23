"""
危险标志检测器.

封装 YOLOv8 模型, 提供统一检测接口.

用法:
    from src.models.detector import SignDetector
    detector = SignDetector()
    results = detector.detect(image)  # image: ndarray 或 文件路径
"""
import os
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

from .config import (
    CLASS_NAMES,
    CLASS_NAMES_ZH,
    DEFAULT_CONF,
    DEFAULT_IOU,
)

# 权重路径 (与 detector.py 同目录)
_WEIGHTS_PATH = str(Path(__file__).resolve().parent / 'best.pt')


class SignDetector:
    """
    危险标志检测器.

    Attributes:
        conf: 置信度阈值 (默认 0.25)
        iou: IoU 阈值 (默认 0.45)
    """

    def __init__(self, device: int | str = 0, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU):
        """
        Args:
            device: 推理设备 (0=CUDA:0, 'cpu'=CPU)
            conf: 置信度阈值
            iou: NMS IoU 阈值
        """
        if not os.path.exists(_WEIGHTS_PATH):
            raise FileNotFoundError(
                f'模型权重文件不存在: {_WEIGHTS_PATH}\n'
                f'请先运行 train/train.py 训练模型'
            )

        self.conf = conf
        self.iou = iou
        self._model = YOLO(_WEIGHTS_PATH)
        if device is not None:
            self._model.to(device)

    @property
    def class_names(self) -> dict:
        """类别 id → 英文名."""
        return CLASS_NAMES.copy()

    @property
    def class_names_zh(self) -> dict:
        """类别英文名 → 中文名."""
        return CLASS_NAMES_ZH.copy()

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray | str) -> list[dict]:
        """
        对单张图片执行检测.

        Args:
            image: BGR numpy 数组 (H, W, 3) 或图片文件路径

        Returns:
            [
                {
                    "bbox": [x1, y1, x2, y2],     # 整数像素坐标
                    "confidence": 0.95,
                    "class_id": 0,
                    "class_name": "danger",
                    "class_name_zh": "注意安全",
                },
                ...
            ]
        """
        results = self._model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_with_raw(self, image: np.ndarray | str) -> tuple[list[dict], object]:
        """
        返回解析后的结果 + 原始 ultralytics Results 对象.

        用于需要额外信息 (如可视化) 的场景.
        """
        results = self._model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        return self._parse_results(results), results[0]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _parse_results(self, results: list) -> list[dict]:
        """将 ultralytics Results 转换为统一的 dict 列表."""
        parsed = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()      # (N, 4)
            confs = boxes.conf.cpu().numpy()     # (N,)
            clss = boxes.cls.cpu().numpy().astype(int)  # (N,)

            for i in range(len(xyxy)):
                cls_id = int(clss[i])
                cls_name = CLASS_NAMES.get(cls_id, 'unknown')
                parsed.append({
                    'bbox': [int(xyxy[i][0]), int(xyxy[i][1]),
                             int(xyxy[i][2]), int(xyxy[i][3])],
                    'confidence': float(confs[i]),
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'class_name_zh': CLASS_NAMES_ZH.get(cls_name, '未知'),
                })

        return parsed


# ------------------------------------------------------------------
# 便捷函数
# ------------------------------------------------------------------

def create_detector(device: int | str = 0, **kwargs) -> SignDetector:
    """
    快速创建检测器实例.

    用法:
        detector = create_detector()
        results = detector.detect(cv2.imread('test.jpg'))
    """
    return SignDetector(device=device, **kwargs)