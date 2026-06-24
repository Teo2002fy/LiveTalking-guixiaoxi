import time
import numpy as np
import resampy
import soundfile as sf
import requests
import json
import shutil
import subprocess
from io import BytesIO
from typing import Iterator

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "gpt-sovits")
class SovitsTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]): 
        text,textevent = msg
        ref_file = textevent.get('tts', {}).get('ref_file',self.opt.REF_FILE)
        ref_text = textevent.get('tts', {}).get('ref_text',self.opt.REF_TEXT)
        self.stream_tts(
            self.gpt_sovits(
                text=text,
                reffile=ref_file,
                reftext=ref_text,
                language="zh", #en args.language,
                server_url=self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
                proxy=getattr(self.opt, 'tts_proxy', '') or getattr(self.opt, 'TTS_PROXY', ''),
            ),
            msg
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url, proxy='') -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'text_split_method': 'cut5',
            'batch_size': 1,
            'media_type':'raw',
            'streaming_mode':True
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            proxy = (proxy or '').strip()
            if proxy and proxy.startswith(('socks5://', 'socks5h://')):
                curl_bin = shutil.which('curl.exe') or shutil.which('curl')
                if not curl_bin:
                    logger.error('gpt_sovits socks proxy requires curl, but curl was not found')
                    return
                cmd = [
                    curl_bin,
                    '--socks5-hostname', proxy.split('://', 1)[1],
                    '--max-time', '120',
                    '-sS',
                    '--fail',
                    '-X', 'POST',
                    f'{server_url.rstrip("/")}/tts',
                    '-H', 'Content-Type: application/json',
                    '--data-binary', '@-',
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                )
                assert proc.stdin is not None
                proc.stdin.write(json.dumps(req, ensure_ascii=False).encode('utf-8'))
                proc.stdin.close()
                end = time.perf_counter()
                logger.info(f"gpt_sovits curl POST started: {end-start}s via {proxy}")

                first = True
                while self.state == State.RUNNING:
                    chunk = proc.stdout.read(4096) if proc.stdout else b''
                    if not chunk:
                        break
                    if first:
                        end = time.perf_counter()
                        logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                        first = False
                    yield chunk

                stderr = proc.stderr.read().decode('utf-8', errors='ignore') if proc.stderr else ''
                return_code = proc.wait(timeout=5)
                if return_code != 0:
                    logger.error('gpt_sovits curl failed code=%s stderr=%s', return_code, stderr.strip())
                return

            session = requests.Session()
            session.trust_env = False
            proxies = {'http': proxy, 'https': proxy} if proxy else None
            res = session.post(
                f"{server_url.rstrip('/')}/tts",
                json=req,
                stream=True,
                timeout=(10, 120),
                proxies=proxies,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return

            first = True
            for chunk in res.iter_content(chunk_size=4096): # raw PCM chunks; smaller chunks reduce playback latency
                if self.state != State.RUNNING:
                    break
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def stream_tts(self,audio_stream,msg:tuple[str, dict]):
        text,textevent = msg
        first = True
        byte_buffer = b""
        sample_buffer = np.empty(0, dtype=np.float32)

        for chunk in audio_stream:
            if self.state != State.RUNNING:
                break
            if chunk is None or len(chunk) <= 0:
                continue

            byte_buffer += chunk
            usable_bytes = len(byte_buffer) - (len(byte_buffer) % 2)
            if usable_bytes <= 0:
                continue

            pcm32k = np.frombuffer(byte_buffer[:usable_bytes], dtype=np.int16).astype(np.float32) / 32768.0
            byte_buffer = byte_buffer[usable_bytes:]
            if pcm32k.size == 0:
                continue

            # GPT-SoVITS raw stream is mono s16le at 32 kHz; LiveTalking expects 16 kHz.
            stream = pcm32k[::2]
            if sample_buffer.size:
                stream = np.concatenate((sample_buffer, stream))

            idx=0
            while stream.shape[0] - idx >= self.chunk:
                eventpoint={}
                if first:
                    eventpoint={'status':'start','text':text}
                    first = False
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                idx += self.chunk
            sample_buffer = stream[idx:]

        if sample_buffer.size:
            padded = np.zeros(self.chunk, dtype=np.float32)
            n = min(sample_buffer.size, self.chunk)
            padded[:n] = sample_buffer[:n]
            eventpoint={}
            if first:
                eventpoint={'status':'start','text':text}
                first = False
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(padded,eventpoint)

        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)
