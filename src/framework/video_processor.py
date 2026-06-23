"""
视频/摄像头帧读取器.

用法:
    vp = VideoProcessor("video.mp4")       # 视频文件
    vp = VideoProcessor(0)                  # 摄像头
    for frame_id, frame in vp:
        ...  # 逐帧处理
    vp.close()
"""
import cv2
from typing import Iterator, Tuple
import numpy as np


class VideoProcessor:
    """视频源读取器, 支持文件和摄像头."""

    def __init__(self, source, skip_frames: int = 0):
        """
        Args:
            source: 视频文件路径 或 摄像头编号 (0, 1, ...)
            skip_frames: 跳帧间隔 (0 = 不跳帧, 2 = 每3帧处理1帧)
        """
        self.source = source
        self.skip_frames = skip_frames
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f'无法打开视频源: {source}')
        self._frame_idx = -1

    # ---------- 属性 ----------

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------- 迭代 ----------

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """逐帧迭代: yield (frame_id, BGR image)."""
        while True:
            self._frame_idx += 1
            ret, frame = self._cap.read()
            if not ret:
                break

            # 跳帧
            if self.skip_frames > 0 and (self._frame_idx % (self.skip_frames + 1) != 0):
                continue

            yield self._frame_idx, frame

    def read(self) -> Tuple[int, np.ndarray] | None:
        """读取下一帧, 不迭代. 返回 (frame_id, frame) 或 None."""
        self._frame_idx += 1
        ret, frame = self._cap.read()
        if not ret:
            return None
        return self._frame_idx, frame

    # ---------- 资源 ----------

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()