"""
时空一致性校验.

通过对连续帧中检测结果的 IoU 匹配, 过滤低置信度的孤立误检.
"""
from typing import List, Dict, Tuple
import numpy as np


def _iou(box1: List[int], box2: List[int]) -> float:
    """计算两个 bbox 的 IoU."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


class TemporalFilter:
    """
    基于滑动窗口的时空间过滤.

    - 高置信度 (≥ high_conf) 直接确认
    - 中置信度 (low_conf ≤ conf < high_conf) 需 N 帧内出现 ≥ min_hits 次
    - 低置信度 (< low_conf) 丢弃
    """

    def __init__(self, window_size: int = 10, min_hits: int = 6,
                 high_conf: float = 0.7, low_conf: float = 0.4,
                 iou_thresh: float = 0.5, ttl: int = 5):
        """
        Args:
            window_size: 滑动窗口帧数
            min_hits: 窗口内最少命中次数
            high_conf: 高置信度阈值
            low_conf: 低置信度阈值
            iou_thresh: 帧间 IoU 匹配阈值
            ttl: 跟踪失活帧数 (连续无匹配则清除)
        """
        self.window_size = window_size
        self.min_hits = min_hits
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.iou_thresh = iou_thresh
        self.ttl = ttl

        self._tracks: Dict[int, dict] = {}   # track_id → track info
        self._next_id = 0
        self._frame_count = 0

    def update(self, detections: List[dict]) -> List[dict]:
        """
        输入当前帧检测, 返回确认的检测.

        Args:
            detections: SignDetector.detect() 的输出

        Returns:
            经过时空确认的检测列表 (格式与输入相同)
        """
        self._frame_count += 1
        confirmed = []

        # 分离高/中/低置信度
        high_dets = [d for d in detections if d['confidence'] >= self.high_conf]
        mid_dets = [d for d in detections if self.low_conf <= d['confidence'] < self.high_conf]

        # 高置信度: 直接确认
        confirmed.extend(high_dets)

        # 中置信度: 匹配已有轨道
        matched_track_ids = set()
        unmatched_dets = list(mid_dets)

        for tid, track in self._tracks.items():
            best_iou = 0
            best_det = None
            best_idx = -1

            for i, det in enumerate(unmatched_dets):
                iou = _iou(track['last_bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
                    best_idx = i

            if best_iou >= self.iou_thresh and best_det is not None:
                # 匹配成功
                track['hits'].append(self._frame_count)
                track['last_bbox'] = best_det['bbox']
                track['last_det'] = best_det
                track['missed'] = 0
                matched_track_ids.add(tid)
                unmatched_dets.pop(best_idx)

                # 窗口检查
                if self._is_confirmed(track):
                    confirmed.append(best_det)
            else:
                # 失配
                track['missed'] += 1

        # 新建轨道 (未匹配的中置信度检测)
        for det in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                'hits': [self._frame_count],
                'last_bbox': det['bbox'],
                'last_det': det,
                'missed': 0,
            }

        # 清理过期轨道
        expired = []
        for tid, track in self._tracks.items():
            # 失活超过 ttl
            if track['missed'] > self.ttl:
                expired.append(tid)
                continue
            # 移除窗口外旧命中
            track['hits'] = [h for h in track['hits']
                             if self._frame_count - h < self.window_size]
        for tid in expired:
            del self._tracks[tid]

        return confirmed

    # ---------- 内部 ----------

    def _is_confirmed(self, track: dict) -> bool:
        """检查轨道是否满足确认条件."""
        return len(track['hits']) >= self.min_hits

    @property
    def active_tracks(self) -> int:
        """当前活跃轨道数."""
        return len(self._tracks)