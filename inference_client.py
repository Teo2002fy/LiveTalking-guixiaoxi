###############################################################################
#  远程推理客户端 — 本地 app.py 通过此模块调用远程 inference_server.py
#
#  当 conf.ini [inference] remote=true 时，avatar 的音频特征提取和推理
#  都通过 HTTP POST 调远程 GPU 服务器完成，本地不需要 GPU。
###############################################################################

import io
import base64
import json
import threading
import requests
import numpy as np
import cv2
import pickle
from utils.logger import logger

try:
    import websocket  # websocket-client (同步)
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


class InferenceClient:
    def __init__(self, server_url: str, api_key: str = "", use_ws: bool = True):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        # WebSocket 流式推理（参照 OmniRT），失败自动回退 HTTP
        self.use_ws = use_ws and _WS_AVAILABLE
        self._ws = None
        self._ws_lock = threading.Lock()
        logger.info(f"InferenceClient: connecting to {self.server_url} "
                    f"(auth={'yes' if api_key else 'no'}, ws={self.use_ws})")
        try:
            r = requests.get(f"{self.server_url}/health", timeout=10)
            logger.info(f"InferenceClient: remote server OK - {r.json()}")
        except Exception as e:
            logger.error(f"InferenceClient: cannot reach {self.server_url}: {e}")

    def _ws_url(self) -> str:
        base = self.server_url.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}/ws/infer"

    def _ensure_ws(self):
        """建立/复用 WebSocket 长连接，首帧鉴权"""
        if self._ws is not None:
            return self._ws
        ws = websocket.create_connection(self._ws_url(), timeout=30)
        if self.api_key:
            ws.send(json.dumps({"type": "auth", "api_key": self.api_key}))
            resp = json.loads(ws.recv())
            if not resp.get("ok"):
                ws.close()
                raise RuntimeError("WebSocket auth failed")
        self._ws = ws
        logger.info("InferenceClient: WebSocket connected")
        return ws

    def health(self) -> dict:
        r = requests.get(f"{self.server_url}/health", timeout=10)
        return r.json()

    def get_avatar_info(self) -> dict:
        r = requests.get(f"{self.server_url}/avatar_info",
                         headers=self._headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def switch_avatar(self, avatar_id: str) -> dict:
        r = requests.post(f"{self.server_url}/switch_avatar",
                          json={"avatar_id": avatar_id},
                          headers=self._headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def audio_feature(self, audio_frames: np.ndarray,
                      stride_left_size: int = 10,
                      stride_right_size: int = 10,
                      batch_size: int = 8,
                      fps: int = 25) -> np.ndarray:
        """发送音频帧，返回特征 batch (numpy array)"""
        payload = {
            "audio_frames_b64": _np_to_b64(audio_frames),
            "stride_left_size": stride_left_size,
            "stride_right_size": stride_right_size,
            "batch_size": batch_size,
            "fps": fps,
        }
        r = requests.post(f"{self.server_url}/audio_feature",
                          json=payload, headers=self._headers, timeout=30)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            logger.error("audio_feature HTTP %s: %s", r.status_code, r.text[:1000])
            raise
        return _b64_to_np(r.json()["features_b64"])

    def inference(self, audiofeat_batch: np.ndarray,
                  index: int, batch_size: int = 8) -> np.ndarray:
        """发送特征 batch + index，返回推理出的脸帧 batch。优先走 WS 流式。"""
        if self.use_ws:
            try:
                return self._inference_ws(audiofeat_batch, index, batch_size)
            except Exception as e:
                logger.warning(f"WS inference failed ({e}), fallback to HTTP for this batch")
                self._close_ws()
        return self._inference_http(audiofeat_batch, index, batch_size)

    def _inference_ws(self, audiofeat_batch, index, batch_size) -> np.ndarray:
        """WebSocket 流式推理：长连接 + 二进制 JPEG 帧"""
        with self._ws_lock:
            ws = self._ensure_ws()
            ws.send(json.dumps({
                "type": "infer",
                "feat_b64": _np_to_b64(audiofeat_batch),
                "index": index,
                "batch": batch_size,
            }))
            # 先收 JSON 头
            head = json.loads(ws.recv())
            if head.get("type") != "frames":
                raise RuntimeError(f"unexpected ws head: {head}")
            count = head["count"]
            # 再收 count 个二进制 JPEG 帧
            frames = []
            for _ in range(count):
                jb = ws.recv()  # bytes
                buf = np.frombuffer(jb, dtype=np.uint8)
                frames.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
            return np.stack(frames)

    def _inference_http(self, audiofeat_batch, index, batch_size) -> np.ndarray:
        payload = {
            "audiofeat_batch_b64": _np_to_b64(audiofeat_batch),
            "index": index,
            "batch_size": batch_size,
        }
        r = requests.post(f"{self.server_url}/inference",
                          json=payload, headers=self._headers, timeout=30)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            logger.error("inference HTTP %s: %s", r.status_code, r.text[:1000])
            raise
        data = r.json()
        if data.get("format") == "jpeg":
            return np.stack([_jpeg_b64_to_np(s) for s in data["frames_jpeg_b64"]])
        return _b64_to_np(data["pred_frames_b64"])

    def _close_ws(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()

def _b64_to_np(s: str) -> np.ndarray:
    buf = io.BytesIO(base64.b64decode(s))
    return np.load(buf, allow_pickle=True)

def _jpeg_b64_to_np(s: str) -> np.ndarray:
    """JPEG base64 → numpy 图像 (H,W,3) uint8"""
    buf = np.frombuffer(base64.b64decode(s), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
