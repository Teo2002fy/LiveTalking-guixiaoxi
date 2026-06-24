###############################################################################
# Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
# Licensed under the Apache License, Version 2.0
###############################################################################

import asyncio
import fractions
import logging
import queue
import threading
import time
from typing import Optional, Set, Tuple, Union

import cv2
import numpy as np
from av import AudioFrame, VideoFrame
from av.frame import Frame
from av.packet import Packet
from aiortc import MediaStreamTrack

VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

logging.basicConfig()
logger = logging.getLogger(__name__)
from utils.logger import logger as mylogger


class PlayerStreamTrack(MediaStreamTrack):
    def __init__(self, player, kind):
        super().__init__()
        self.kind = kind
        self._player = player
        if kind == "video":
            queue_size = max(1, int(getattr(player, "video_queue_size", 2)))
        else:
            queue_size = max(1, int(getattr(player, "audio_queue_size", 60)))
        self._queue = queue.Queue(maxsize=queue_size)
        fps = max(1, int(getattr(player, "fps", 15)))
        self.video_ptime = 1.0 / fps
        self.timelist = []
        self.current_frame_count = 0
        self._last_frame = None
        self._last_audio_samples = 320
        if self.kind == "video":
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        now = time.time()
        if self.kind == "video":
            if hasattr(self, "_timestamp"):
                self._timestamp += int(self.video_ptime * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * self.video_ptime - now
                if wait < -self.video_ptime * 3:
                    self._start = now - self.current_frame_count * self.video_ptime
                    wait = 0
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = now
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info("video start:%f", self._start)
            return self._timestamp, VIDEO_TIME_BASE

        if hasattr(self, "_timestamp"):
            samples = int(getattr(self, "_last_audio_samples", 320))
            self._timestamp += samples
            self.current_frame_count += 1
            wait = self._start + self._timestamp / SAMPLE_RATE - now
            if wait < -0.12:
                self._start = now - self._timestamp / SAMPLE_RATE
                wait = 0
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = now
            self._timestamp = 0
            self.timelist.append(self._start)
            mylogger.info("audio start:%f", self._start)
        return self._timestamp, AUDIO_TIME_BASE

    def _silence_frame(self) -> AudioFrame:
        audio = np.zeros((1, 320), dtype=np.int16)
        frame = AudioFrame.from_ndarray(audio, layout="mono", format="s16")
        frame.sample_rate = SAMPLE_RATE
        self._last_audio_samples = 320
        return frame

    async def recv(self) -> Union[Frame, Packet]:
        self._player._start(self)
        eventpoint = None

        while True:
            try:
                frame, eventpoint = self._queue.get_nowait()
                if self.kind == "video":
                    dropped = 0
                    while True:
                        try:
                            frame, eventpoint = self._queue.get_nowait()
                            dropped += 1
                        except queue.Empty:
                            break
                    if dropped:
                        self._player._note_video_drop(dropped)
                    self._last_frame = frame
                else:
                    self._last_audio_samples = getattr(frame, "samples", 320)
                break
            except queue.Empty:
                if self.kind == "video" and self._last_frame is not None:
                    frame = self._last_frame
                    eventpoint = None
                    break
                if self.kind == "audio":
                    frame = self._silence_frame()
                    eventpoint = None
                    break
                await asyncio.sleep(0.005)

        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base

        if eventpoint and self._player is not None:
            self._player.notify(eventpoint)
        if frame is None:
            self.stop()
            raise Exception

        if self.kind == "video":
            self.totaltime += time.perf_counter() - self.lasttime
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                mylogger.info("------actual avg final fps:%.4f", self.framecount / self.totaltime)
                self.framecount = 0
                self.totaltime = 0
        return frame

    def stop(self):
        super().stop()
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                del item
            except queue.Empty:
                break
        if self._player is not None:
            self._player._stop(self)
            self._player = None


def player_worker_thread(quit_event, container):
    container.render(quit_event)


class HumanPlayer:
    def __init__(self, avatar_session, format=None, options=None, timeout=None, loop=False, decode=True):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None
        self.__container = avatar_session

        opt = getattr(avatar_session, "opt", None)
        self.fps = max(1, int(getattr(opt, "webrtc_fps", getattr(opt, "fps", 15))))
        self.video_queue_size = max(1, int(getattr(opt, "webrtc_video_queue", 2)))
        self.audio_queue_size = max(1, int(getattr(opt, "webrtc_audio_queue", 60)))
        self.max_video_width = max(0, int(getattr(opt, "webrtc_max_width", 576)))
        self._video_drop_count = 0
        self._last_drop_log = time.perf_counter()

        mylogger.info(
            "WebRTC media config fps=%s max_width=%s video_queue=%s audio_queue=%s",
            self.fps,
            self.max_video_width,
            self.video_queue_size,
            self.audio_queue_size,
        )

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")

        if hasattr(self.__container, "output"):
            self.__container.output._player = self

    def _note_video_drop(self, dropped: int):
        self._video_drop_count += dropped
        now = time.perf_counter()
        if now - self._last_drop_log > 5:
            mylogger.info("WebRTC dropped stale video frames: %d", self._video_drop_count)
            self._video_drop_count = 0
            self._last_drop_log = now

    def _resize_video(self, frame):
        if not self.max_video_width:
            return frame
        h, w = frame.shape[:2]
        if w <= self.max_video_width:
            return frame
        scale = self.max_video_width / float(w)
        new_size = (self.max_video_width, max(1, int(h * scale)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def push_video(self, frame):
        frame = self._resize_video(frame)
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        q = self.__video._queue
        dropped = 0
        while True:
            try:
                q.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            self._note_video_drop(dropped)
        try:
            q.put_nowait((new_frame, None))
        except queue.Full:
            self._note_video_drop(1)

    def push_audio(self, frame, eventpoint=None):
        pcm = np.asarray(frame)
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        new_frame = AudioFrame(format="s16", layout="mono", samples=pcm.shape[0])
        new_frame.planes[0].update(pcm.tobytes())
        new_frame.sample_rate = SAMPLE_RATE
        q = self.__audio._queue
        try:
            q.put_nowait((new_frame, eventpoint))
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait((new_frame, eventpoint))
            except queue.Full:
                pass

    def get_buffer_size(self) -> int:
        return self.__video._queue.qsize()

    def notify(self, eventpoint):
        if self.__container is not None:
            self.__container.notify(eventpoint)

    @property
    def audio(self) -> MediaStreamTrack:
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(self.__thread_quit, self.__container),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)
        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None
        if not self.__started and self.__container is not None:
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)