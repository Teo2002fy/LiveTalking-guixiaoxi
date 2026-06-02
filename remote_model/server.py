"""
 * Author: JimZhang
 * Date: 2026-06-01 21:14:32
 * LastEditors: 很拉风的James
 * LastEditTime: 2026-06-01 23:47:05
 * FilePath: /remote_model/server.py
 * Description: 口型推理服务；wav2lip/musetalk 推理 HTTP API，支持 API Key 鉴权。
"""

import io
import os
import sys
import json
import pickle
import glob
import base64
import struct
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# 自包含：把当前目录加入 path 以 import avatars/ utils/
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)

from utils.logger import logger
from utils.image import read_imgs, mirror_index
from config import parse_args

app = FastAPI(title="LiveTalking Inference Server")

# 鉴权
_security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(_security)):
    api_key = _state.get("api_key", "")
    if not api_key:
        return
    if credentials is None or credentials.credentials != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# 全局状态
_state = {
    "model_type": None,
    "model": None,
    "avatar_id": None,
    "avatar": None,
    "device": None,
    "batch_size": 8,
    "api_key": "",
    "models_dir": "./models",
    "avatars_dir": "./data/avatars",
    "jpeg_quality": 60,
}

# JPEG 并行编码线程池（参照 OmniRT FLASHTALK_JPEG_WORKERS）
_jpeg_pool = ThreadPoolExecutor(max_workers=4)


# 请求模型

class AudioFeatureRequest(BaseModel):
    audio_frames_b64: str
    stride_left_size: int = 10
    stride_right_size: int = 10
    batch_size: int = 8
    fps: int = 25

class InferenceRequest(BaseModel):
    audiofeat_batch_b64: str
    index: int
    batch_size: int = 8

class SwitchAvatarRequest(BaseModel):
    avatar_id: str


# numpy <-> base64

def np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_np(s: str) -> np.ndarray:
    buf = io.BytesIO(base64.b64decode(s))
    return np.load(buf, allow_pickle=True)

def frames_to_jpeg_b64(frames: np.ndarray, quality: int = 90) -> list:
    """脸帧 batch (N,H,W,3) uint8 → JPEG base64 列表，大幅压缩传输体积"""
    out = []
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for f in frames:
        ok, buf = cv2.imencode('.jpg', f.astype(np.uint8), enc)
        out.append(base64.b64encode(buf.tobytes()).decode())
    return out

def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode('.jpg', frame.astype(np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes()

def frames_to_jpeg_bytes(frames: np.ndarray, quality: int) -> list:
    """并行把脸帧 batch 编码为 JPEG 二进制列表（用于 WebSocket 二进制传输）"""
    return list(_jpeg_pool.map(lambda f: _encode_jpeg(f, quality), list(frames)))


def _run_inference(audiofeat_batch: np.ndarray, index: int, batch_size: int) -> np.ndarray:
    """核心推理：特征 batch + index → 脸帧 batch (N,H,W,3) uint8。HTTP/WS 共用。"""
    device = _state["device"]
    if _state["model_type"] == "wav2lip":
        model = _state["model"]
        frame_list, face_list, coord_list = _state["avatar"]
        length = len(face_list)
        img_batch = np.asarray([face_list[mirror_index(length, index + i)] for i in range(batch_size)])
        audiofeat_batch = np.asarray(audiofeat_batch)
        img_masked = img_batch.copy()
        img_masked[:, img_batch.shape[1] // 2:] = 0
        img_batch_input = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        audiofeat_input = np.reshape(audiofeat_batch,
                                     [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])
        img_tensor = torch.FloatTensor(np.transpose(img_batch_input, (0, 3, 1, 2))).to(device)
        audio_tensor = torch.FloatTensor(np.transpose(audiofeat_input, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(audio_tensor, img_tensor)
        return (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
    else:
        vae, unet, pe, timesteps, audio_processor = _state["model"]
        frame_list, mask_list, coord_list, mask_coords_list, input_latent_list = _state["avatar"]
        length = len(input_latent_list)
        whisper_batch = np.stack(audiofeat_batch)
        latent_batch = torch.cat(
            [input_latent_list[mirror_index(length, index + i)] for i in range(batch_size)], dim=0)
        audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=unet.model.dtype)
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)
        pred_latents = unet.model(latent_batch, timesteps,
                                  encoder_hidden_states=audio_feature_batch).sample
        return vae.decode_latents(pred_latents).astype(np.uint8)


# 模型加载

def load_wav2lip(model_path):
    from avatars.wav2lip.models.wav2lip import Wav2Lip
    device = _state["device"]
    model = Wav2Lip()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    model = model.to(device).eval()
    logger.info(f"wav2lip model loaded from {model_path}")
    return model

def load_musetalk():
    from avatars.musetalk.utils.utils import load_all_model
    from avatars.musetalk.whisper.audio2feature import Audio2Feature
    device = _state["device"]
    models_dir = _state["models_dir"]
    vae, unet, pe = load_all_model(
        unet_model_path=os.path.join(models_dir, "musetalkV15", "unet.pth"),
        vae_type="sd-vae",
        unet_config=os.path.join(models_dir, "musetalkV15", "musetalk.json"),
        device=device)
    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    audio_processor = Audio2Feature(model_path=os.path.join(models_dir, "whisper"))
    logger.info("musetalk model loaded")
    return vae, unet, pe, timesteps, audio_processor

def load_avatar_wav2lip(avatar_id):
    avatar_path = os.path.join(_state["avatars_dir"], avatar_id)
    coords_path = os.path.join(avatar_path, "coords.pkl")
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    full = sorted(glob.glob(os.path.join(avatar_path, "full_imgs", '*.[jpJP][pnPN]*[gG]')),
                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(full)
    faces = sorted(glob.glob(os.path.join(avatar_path, "face_imgs", '*.[jpJP][pnPN]*[gG]')),
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(faces)
    logger.info(f"wav2lip avatar loaded: {avatar_id}, {len(frame_list_cycle)} frames")
    return frame_list_cycle, face_list_cycle, coord_list_cycle

def load_avatar_musetalk(avatar_id):
    avatar_path = os.path.join(_state["avatars_dir"], avatar_id)
    input_latent_list_cycle = torch.load(os.path.join(avatar_path, "latents.pt"),
                                         map_location=_state["device"], weights_only=False)
    with open(os.path.join(avatar_path, "coords.pkl"), 'rb') as f:
        coord_list_cycle = pickle.load(f)
    full = sorted(glob.glob(os.path.join(avatar_path, "full_imgs", '*.[jpJP][pnPN]*[gG]')),
                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(full)
    with open(os.path.join(avatar_path, "mask_coords.pkl"), 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    masks = sorted(glob.glob(os.path.join(avatar_path, "mask", '*.[jpJP][pnPN]*[gG]')),
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(masks)
    logger.info(f"musetalk avatar loaded: {avatar_id}, {len(frame_list_cycle)} frames")
    return frame_list_cycle, mask_list_cycle, coord_list_cycle, mask_coords_list_cycle, input_latent_list_cycle


# API 路由

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": _state["model_type"],
        "avatar_id": _state["avatar_id"],
        "device": str(_state["device"]),
        "auth_required": bool(_state["api_key"]),
    }

@app.get("/avatar_info", dependencies=[Depends(verify_api_key)])
def avatar_info():
    if _state["avatar"] is None:
        raise HTTPException(400, "No avatar loaded")
    if _state["model_type"] == "wav2lip":
        frame_list, face_list, coord_list = _state["avatar"]
        return {
            "model_type": "wav2lip", "avatar_id": _state["avatar_id"],
            "num_frames": len(frame_list), "frame_shape": list(frame_list[0].shape),
            "face_shape": list(face_list[0].shape),
            "coords_b64": base64.b64encode(pickle.dumps(coord_list)).decode(),
        }
    else:
        frame_list, mask_list, coord_list, mask_coords_list, latent_list = _state["avatar"]
        return {
            "model_type": "musetalk", "avatar_id": _state["avatar_id"],
            "num_frames": len(frame_list), "frame_shape": list(frame_list[0].shape),
            "coords_b64": base64.b64encode(pickle.dumps(coord_list)).decode(),
            "mask_coords_b64": base64.b64encode(pickle.dumps(mask_coords_list)).decode(),
        }

@app.post("/switch_avatar", dependencies=[Depends(verify_api_key)])
def switch_avatar(req: SwitchAvatarRequest):
    try:
        if _state["model_type"] == "wav2lip":
            _state["avatar"] = load_avatar_wav2lip(req.avatar_id)
        else:
            _state["avatar"] = load_avatar_musetalk(req.avatar_id)
        _state["avatar_id"] = req.avatar_id
        return {"status": "ok", "avatar_id": req.avatar_id}
    except Exception as e:
        raise HTTPException(500, str(e))

def _extract_feature(audio_frames, stride_left_size, stride_right_size, batch_size, fps):
    """核心特征提取：音频 → 特征 batch。HTTP/WS 共用。"""
    if _state["model_type"] == "wav2lip":
        from avatars.wav2lip.audio import melspectrogram
        mel = melspectrogram(audio_frames)
        mel_idx_multiplier = 80.0 / fps
        mel_step_size = 16
        left = max(0, stride_left_size * 80 / 50)
        mel_chunks = []
        i = 0
        total_frames = len(audio_frames) // 320
        num_steps = (total_frames - stride_left_size - stride_right_size) // 2
        while i < num_steps:
            start_idx = int(left + i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            else:
                mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1
        return np.array(mel_chunks)
    else:
        vae, unet, pe, timesteps, audio_processor = _state["model"]
        whisper_feature = audio_processor.audio2feat(audio_frames)
        feature_chunks = []
        start = stride_left_size / 2
        for i in range(batch_size):
            center_idx = int((i + start) * 2)
            selected = []
            for idx in range(center_idx, center_idx + 5):
                idx = max(0, min(len(whisper_feature) - 1, idx))
                selected.append(whisper_feature[idx])
            feature_chunks.append(np.asarray(selected).reshape(-1, 384))
        return np.array(feature_chunks)


@app.post("/audio_feature", dependencies=[Depends(verify_api_key)])
def audio_feature(req: AudioFeatureRequest):
    try:
        audio_frames = b64_to_np(req.audio_frames_b64)
        feat = _extract_feature(audio_frames, req.stride_left_size,
                                req.stride_right_size, req.batch_size, req.fps)
        return {"features_b64": np_to_b64(feat)}
    except Exception as e:
        logger.exception("audio_feature error")
        raise HTTPException(500, str(e))

@app.post("/inference", dependencies=[Depends(verify_api_key)])
def inference(req: InferenceRequest):
    try:
        audiofeat_batch = b64_to_np(req.audiofeat_batch_b64)
        pred = _run_inference(audiofeat_batch, req.index, req.batch_size)
        return {"format": "jpeg",
                "frames_jpeg_b64": frames_to_jpeg_b64(pred, _state["jpeg_quality"])}
    except Exception as e:
        logger.exception("inference error")
        raise HTTPException(500, str(e))


# WebSocket 流式推理（参照 OmniRT AUDI/VIDX：长连接 + 二进制 JPEG 帧）
#
# 协议（客户端 → 服务端，JSON 文本帧）:
#   {"type":"auth","api_key":"..."}                              首帧鉴权
#   {"type":"infer","feat_b64":"<npy特征>","index":N,"batch":8}  请求推理
# 服务端 → 客户端:
#   对每个 infer 请求，先发一个 JSON 头 {"type":"frames","count":N,"index":I}
#   紧接着发 N 个二进制 JPEG 帧（bytes），客户端按 count 接收

@app.websocket("/ws/infer")
async def ws_infer(ws: WebSocket):
    await ws.accept()
    authed = not _state["api_key"]  # 未设 key 则默认通过
    try:
        while True:
            msg = await ws.receive_text()
            req = json.loads(msg)
            mtype = req.get("type")

            if mtype == "auth":
                authed = (req.get("api_key") == _state["api_key"])
                await ws.send_text(json.dumps({"type": "auth", "ok": authed}))
                if not authed:
                    await ws.close(code=4001)
                    return
                continue

            if not authed:
                await ws.send_text(json.dumps({"type": "error", "msg": "unauthorized"}))
                await ws.close(code=4001)
                return

            if mtype == "infer":
                feat = b64_to_np(req["feat_b64"])
                index = int(req.get("index", 0))
                batch = int(req.get("batch", _state["batch_size"]))
                pred = _run_inference(feat, index, batch)
                jpegs = frames_to_jpeg_bytes(pred, _state["jpeg_quality"])
                # 先发头，再逐帧发二进制
                await ws.send_text(json.dumps(
                    {"type": "frames", "count": len(jpegs), "index": index}))
                for jb in jpegs:
                    await ws.send_bytes(jb)
            elif mtype == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
            else:
                await ws.send_text(json.dumps({"type": "error", "msg": f"unknown type {mtype}"}))
    except WebSocketDisconnect:
        logger.info("ws_infer client disconnected")
    except Exception as e:
        logger.exception("ws_infer error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass


# 启动

def main():
    args = parse_args()

    _state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state["model_type"] = args.model
    _state["batch_size"] = args.batch_size
    _state["api_key"] = args.api_key
    _state["models_dir"] = args.models_dir
    _state["avatars_dir"] = args.avatars_dir
    _state["jpeg_quality"] = args.jpeg_quality

    logger.info(f"Loading model={args.model} avatar={args.avatar_id} on {_state['device']}")
    logger.info(f"Auth: {'enabled' if args.api_key else 'disabled'}")

    if args.model == "wav2lip":
        _state["model"] = load_wav2lip(args.wav2lip_model_path)
        _state["avatar"] = load_avatar_wav2lip(args.avatar_id)
    else:
        _state["model"] = load_musetalk()
        _state["avatar"] = load_avatar_musetalk(args.avatar_id)
    _state["avatar_id"] = args.avatar_id

    logger.info(f"Inference server ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
