###############################################################################
#  Xinference TTS 插件 — 对接 Xinference 的 OpenAI 兼容语音合成接口
#  (POST /v1/audio/speech)，已在 CosyVoice2-0.5B 上验证。
#
#  启动示例:
#    python app.py --tts xinference \
#      --TTS_SERVER http://<your-tts-host>:8005 \
#      --REF_FILE "中文女"          # REF_FILE 复用为 voice 音色名
#
#  额外可用环境变量:
#    XINFERENCE_TTS_KEY    Xinference API Key (Authorization: Bearer)
#    XINFERENCE_TTS_MODEL  TTS 模型名，默认 CosyVoice2-0.5B
###############################################################################

import os
import time
import numpy as np
import resampy
import requests
from typing import Iterator

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

# CosyVoice2 输出固定 24kHz / mono / 16-bit PCM
_SRC_SR = 24000


@register("tts", "xinference")
class XinferenceTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.server_url = opt.TTS_SERVER.rstrip('/')
        # 优先用 conf.ini [tts] (经 config.py 挂到 opt)，否则回退环境变量
        self.api_key = getattr(opt, "xinference_tts_key", "") or os.getenv("XINFERENCE_TTS_KEY", "")
        self.model = getattr(opt, "xinference_tts_model", "") or os.getenv("XINFERENCE_TTS_MODEL", "CosyVoice2-0.5B")
        logger.info(f"XinferenceTTS init: url={self.server_url} model={self.model}")

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        # 音色：优先取请求透传的 tts.ref_file，否则用启动参数 REF_FILE
        voice = textevent.get('tts', {}).get('ref_file', self.opt.REF_FILE) or "中文女"
        self.stream_tts(
            self.xinference_speech(text, voice),
            msg
        )

    def xinference_speech(self, text, voice) -> Iterator[bytes]:
        start = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "input": text,
            "voice": voice,
            "response_format": "pcm",   # 直接拿原始 PCM，省去解码
            "stream": True,
        }
        try:
            res = requests.post(
                f"{self.server_url}/v1/audio/speech",
                json=payload,
                headers=headers,
                stream=True,
            )
            logger.info(f"xinference tts POST: {time.perf_counter()-start:.3f}s")
            if res.status_code != 200:
                logger.error("Xinference TTS error %s: %s", res.status_code, res.text[:300])
                return

            first = True
            # 16K 24kHz*20ms*2 ≈ 960 bytes/帧, 取较大块降低请求开销
            for chunk in res.iter_content(chunk_size=9600):
                if first:
                    logger.info(f"xinference tts first chunk: {time.perf_counter()-start:.3f}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception:
            logger.exception('xinference tts')

    def stream_tts(self, audio_stream, msg: tuple[str, dict]):
        text, textevent = msg
        first = True
        last_stream = np.array([], dtype=np.float32)
        # PCM 是 16-bit，按 2 字节对齐累积，避免跨 chunk 截断样本
        carry = b''
        for chunk in audio_stream:
            if not chunk:
                continue
            buf = carry + chunk
            valid = len(buf) - (len(buf) % 2)
            carry = buf[valid:]
            if valid <= 0:
                continue
            stream = np.frombuffer(buf[:valid], dtype=np.int16).astype(np.float32) / 32767
            # 24kHz -> 16kHz
            stream = resampy.resample(x=stream, sr_orig=_SRC_SR, sr_new=self.sample_rate)
            stream = np.concatenate((last_stream, stream))
            streamlen = stream.shape[0]
            idx = 0
            while streamlen >= self.chunk:
                eventpoint = {}
                if first:
                    eventpoint = {'status': 'start', 'text': text}
                    first = False
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                streamlen -= self.chunk
                idx += self.chunk
            last_stream = stream[idx:]  # 余量留到下一块

        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
