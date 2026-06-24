###############################################################################
# WebRTC connection manager.
###############################################################################

import asyncio
import json

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger
from server.session_manager import session_manager


class RTCManager:
    def __init__(self, opt):
        self.opt = opt
        self.pcs: set = set()

    def _video_codec_preferences(self):
        capabilities = RTCRtpSender.getCapabilities("video")
        preferred = (getattr(self.opt, "webrtc_codec", "H264") or "H264").upper()
        order = [preferred]
        for name in ("H264", "VP8", "rtx"):
            if name not in order:
                order.append(name)
        codecs = []
        for name in order:
            codecs.extend([c for c in capabilities.codecs if c.name.upper() == name.upper()])
        logger.info("WebRTC codec preference: %s -> %s", preferred, [c.name for c in codecs])
        return codecs

    async def handle_offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        sessionid = await session_manager.create_session(params)
        logger.info("offer sessionid=%s", sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                self.pcs.discard(pc)
                session_manager.remove_session(sessionid)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        video_transceiver = None
        for transceiver in pc.getTransceivers():
            sender = getattr(transceiver, "sender", None)
            track = getattr(sender, "track", None)
            if getattr(track, "kind", None) == "video":
                video_transceiver = transceiver
                break
        if video_transceiver is not None:
            preferences = self._video_codec_preferences()
            if preferences:
                video_transceiver.setCodecPreferences(preferences)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())
        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

    async def shutdown(self):
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()