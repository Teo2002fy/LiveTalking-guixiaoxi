###############################################################################
#  远程推理 Avatar — 口型推理通过 HTTP 调用远程 GPU 服务器
#
#  当 conf.ini [inference] remote=true 时，app.py 使用此类替代
#  musetalk_avatar / wav2lip_avatar 的本地 GPU 推理。
#
#  本地只做: 音频帧收集 → 远程特征提取 → 远程推理 → 本地贴回 → 本地推流
#  不需要本地有 GPU。
###############################################################################

import time
import copy
import numpy as np
import cv2
import pickle
import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.base_avatar import BaseAvatar, AudioFrameData
from inference_client import InferenceClient
from utils.logger import logger
from utils.image import mirror_index
from registry import register


class RemoteASR:
    """
    远程音频特征提取 — 替代本地的 MelASR / WhisperASR。
    收集音频帧，攒够一个 batch 后发送到远程服务器提取特征。
    """
    def __init__(self, opt, parent: BaseAvatar, client: InferenceClient):
        self.opt = opt
        self.parent = parent
        self.client = client

        self.fps = opt.fps
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps * 2)  # 320
        self.batch_size = opt.batch_size

        self.queue: Queue[AudioFrameData] = Queue()
        self.output_queue: Queue[AudioFrameData] = Queue()
        self.feat_queue: Queue = Queue(maxsize=2)

        self.frames: list[np.ndarray] = []
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r

    def flush_talk(self):
        self.queue.queue.clear()

    def put_audio_frame(self, audio_chunk: np.ndarray, datainfo: dict):
        self.queue.put(AudioFrameData(data=audio_chunk, type=0, userdata=datainfo))

    def get_audio_frame(self) -> AudioFrameData:
        try:
            if self.parent and self.parent.custom_audiotype > 1:
                frame = self.parent.get_custom_audio_stream(self.parent.custom_audiotype)
                return AudioFrameData(data=frame, type=self.parent.custom_audiotype, userdata={})
            else:
                return self.queue.get(block=True, timeout=0.01)
        except queue.Empty:
            return AudioFrameData(data=np.zeros(self.chunk, dtype=np.float32), type=1, userdata={})

    def warm_up(self):
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame = self.get_audio_frame()
            self.frames.append(audio_frame.data)
            self.output_queue.put(audio_frame)
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        """收集音频帧，发送到远程提取特征"""
        for _ in range(self.batch_size * 2):
            audio_frame = self.get_audio_frame()
            self.frames.append(audio_frame.data)
            self.output_queue.put(audio_frame)

        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return

        inputs = np.concatenate(self.frames)

        try:
            features = self.client.audio_feature(
                audio_frames=inputs,
                stride_left_size=self.stride_left_size,
                stride_right_size=self.stride_right_size,
                batch_size=self.batch_size,
                fps=self.fps,
            )
            self.feat_queue.put(features)
        except Exception as e:
            logger.error(f"RemoteASR audio_feature error: {e}")

        # 丢弃旧帧
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]


@register("avatar", "remote_musetalk")
@register("avatar", "remote_wav2lip")
class RemoteAvatar(BaseAvatar):
    """
    远程推理 Avatar。
    初始化时从远程服务器拉取 avatar 素材（full_imgs, coords, masks），
    推理时通过 HTTP 调远程 inference_batch。
    """

    def __init__(self, opt, model=None, avatar=None):
        # 先初始化 TTS/Output（BaseAvatar.__init__ 会做）
        # 但不初始化本地 ASR 和模型——我们用远程的
        # 调 BaseAvatar.__init__ 前先设置一个 flag 防止子类初始化 GPU 模型
        super().__init__(opt)

        self.client = InferenceClient(opt.inference_server_url,
                                      api_key=getattr(opt, 'inference_api_key', ''))

        # 从远程拉取 avatar 信息
        info = self.client.get_avatar_info()
        self.num_frames = info["num_frames"]
        self.frame_shape = info["frame_shape"]
        self._model_type = info["model_type"]

        # 解码坐标
        self.coord_list_cycle = pickle.loads(
            __import__('base64').b64decode(info["coords_b64"]))

        if self._model_type == "musetalk":
            self.mask_coords_list_cycle = pickle.loads(
                __import__('base64').b64decode(info["mask_coords_b64"]))

        # 本地需要 full_imgs 和 mask 用于 paste_back_frame
        # 这些数据量大，从远程拉不现实，需要本地也有一份 avatar 数据
        # 从本地 data/avatars/<id> 读取
        self._load_local_frames(opt.avatar_id)

        # 用远程 ASR 替代本地 GPU ASR
        self.asr = RemoteASR(opt, self, self.client)
        self.asr.warm_up()

    def _load_local_frames(self, avatar_id):
        """加载本地的 full_imgs 和 mask（贴回用，纯 CPU 操作）"""
        import glob
        from utils.image import read_imgs

        avatar_path = f"./data/avatars/{avatar_id}"
        full_imgs_path = f"{avatar_path}/full_imgs"

        input_img_list = sorted(
            glob.glob(f"{full_imgs_path}/*.[jpJP][pnPN]*[gG]"),
            key=lambda x: int(__import__('os').path.splitext(
                __import__('os').path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)

        if self._model_type == "musetalk":
            mask_path = f"{avatar_path}/mask"
            input_mask_list = sorted(
                glob.glob(f"{mask_path}/*.[jpJP][pnPN]*[gG]"),
                key=lambda x: int(__import__('os').path.splitext(
                    __import__('os').path.basename(x))[0]))
            self.mask_list_cycle = read_imgs(input_mask_list)

    def get_avatar_length(self):
        return len(self.frame_list_cycle)

    def inference_batch(self, index, audiofeat_batch):
        """调远程推理服务"""
        pred = self.client.inference(
            audiofeat_batch=np.array(audiofeat_batch),
            index=index,
            batch_size=self.batch_size,
        )
        return pred

    def paste_back_frame(self, pred_frame, idx: int):
        """本地贴回（CPU）"""
        if self._model_type == "wav2lip":
            bbox = self.coord_list_cycle[idx]
            combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
            y1, y2, x1, x2 = bbox
            res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            combine_frame[y1:y2, x1:x2] = res_frame
            return combine_frame
        else:  # musetalk
            from avatars.musetalk.myutil import get_image_blending
            bbox = self.coord_list_cycle[idx]
            ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
            x1, y1, x2, y2 = bbox
            res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            mask = self.mask_list_cycle[idx]
            mask_crop_box = self.mask_coords_list_cycle[idx]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            return combine_frame
