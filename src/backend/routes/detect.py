"""检测路由: 图片检测 / 视频检测 / WebSocket 实时流"""
import io
import cv2
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from ..auth import is_authenticated
from ...models.detector import create_detector
from ...framework.temporal_filter import TemporalFilter

router = APIRouter(prefix="/api/detect", tags=["detect"])

# 全局检测器 (单例)
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = create_detector()
    return _detector


# ============================================================
# 图片检测
# ============================================================

@router.post("/image")
async def detect_image(file: UploadFile = File(...)):
    """上传图片, 返回标注后的 JPEG."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detector = _get_detector()
    dets = detector.detect(img)

    # 画框
    for d in dets:
        bbox = d['bbox']
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(img, label, (bbox[0], bbox[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 编码为 JPEG
    _, buf = cv2.imencode('.jpg', img)
    return {
        "detections": dets,
        "count": len(dets),
        "image_base64": buf.tobytes().hex()  # 前端用 hex 解码显示
    }


# ============================================================
# 视频检测
# ============================================================

@router.post("/video")
async def detect_video(file: UploadFile = File(...)):
    """
    上传视频, 逐帧检测.
    返回 JSON: frames (base64 JPEG 列表) + fps + 统计, 前端 canvas 渲染.
    """
    import tempfile, os
    suffix = Path(file.filename).suffix or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    from ...framework.video_processor import VideoProcessor
    detector = _get_detector()
    tf = TemporalFilter()

    vp = VideoProcessor(input_path)
    fps = vp.fps

    frames_base64 = []
    total_raw = 0
    total_conf = 0
    all_dets = []

    for frame_idx, frame in vp:
        dets = detector.detect(frame)
        total_raw += len(dets)
        confirmed = tf.update(dets)
        total_conf += len(confirmed)

        for d in confirmed:
            b = d['bbox']
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['class_name']} {d['confidence']:.2f}",
                        (b[0], b[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        all_dets.extend(confirmed)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frames_base64.append(buf.tobytes().hex())

    vp.close()
    os.unlink(input_path)

    from collections import Counter
    class_counts = Counter(d['class_name'] for d in all_dets)
    class_summary = [
        {"class_name": cls, "class_name_zh": detector.class_names_zh.get(cls, cls), "count": cnt}
        for cls, cnt in class_counts.most_common()
    ]

    return {
        "ok": True,
        "fps": fps,
        "total_raw": total_raw,
        "total_confirmed": total_conf,
        "filter_rate": round((1 - total_conf / max(total_raw, 1)) * 100, 1),
        "class_summary": class_summary,
        "frames": frames_base64,
    }


# ============================================================
# WebSocket 实时检测
# ============================================================

@router.websocket("/live")
async def detect_live(ws: WebSocket):
    """
    WebSocket 实时检测.
    前端发送 JPEG 帧 (bytes), 后端返回标注后的 JPEG 帧.
    首条消息为 token: "token:<token>"
    """
    await ws.accept()
    detector = _get_detector()
    tf = TemporalFilter()

    # 等待认证 token (前端发送 text)
    token = await ws.receive_text()
    if not is_authenticated(token):
        await ws.close(code=1008, reason="Unauthorized")
        return

    await ws.send_text("ready")

    try:
        while True:
            data = await ws.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            dets = detector.detect(frame)
            confirmed = tf.update(dets)

            # 画框
            for d in confirmed:
                bbox = d['bbox']
                label = f"{d['class_name']} {d['confidence']:.2f}"
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            await ws.send_bytes(buf.tobytes())
    except WebSocketDisconnect:
        pass