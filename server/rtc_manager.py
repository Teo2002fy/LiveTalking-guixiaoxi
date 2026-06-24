###############################################################################
# WebRTC connection manager.
###############################################################################

import asyncio
import json

from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger
from server.session_manager import session_manager


class RTCManager:
    def __init__(self, opt):
        self.opt = opt
        self.pcs: set = set()
        self._apply_encoder_defaults()

    def _apply_encoder_defaults(self):
        bitrate = int(getattr(self.opt, "webrtc_video_bitrate", 800000) or 800000)
        fps = max(1, int(getattr(self.opt, "webrtc_fps", 15) or 15))
        try:
            from aiortc.codecs import h264, vpx

            h264.DEFAULT_BITRATE = max(getattr(h264, "MIN_BITRATE", 500000), min(bitrate, getattr(h264, "MAX_BITRATE", 3000000)))
            vpx.DEFAULT_BITRATE = max(getattr(vpx, "MIN_BITRATE", 250000), min(bitrate, getattr(vpx, "MAX_BITRATE", 1500000)))
            h264.MAX_FRAME_RATE = min(30, fps)
            vpx.MAX_FRAME_RATE = min(30, fps)
            logger.info(
                "WebRTC encoder defaults bitrate=%s h264=%s vp8=%s fps=%s",
                bitrate,
                h264.DEFAULT_BITRATE,
                vpx.DEFAULT_BITRATE,
                fps,
            )
        except Exception as exc:
            logger.warning("Failed to apply WebRTC encoder defaults: %s", exc)

    def _ice_servers(self):
        servers = []
        for item in getattr(self.opt, "webrtc_ice_servers", []) or []:
            if not isinstance(item, dict) or not item.get("urls"):
                continue
            servers.append(
                RTCIceServer(
                    urls=item["urls"],
                    username=item.get("username") or None,
                    credential=item.get("credential") or None,
                    credentialType=item.get("credentialType") or "password",
                )
            )
        return servers

    def _rtc_configuration(self):
        servers = self._ice_servers()
        if servers:
            logger.info("WebRTC ICE servers enabled: %s", [s.urls for s in servers])
        try:
            from aiortc import RTCBundlePolicy

            return RTCConfiguration(iceServers=servers, bundlePolicy=RTCBundlePolicy.MAX_BUNDLE)
        except Exception:
            return RTCConfiguration(iceServers=servers)

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

        pc = RTCPeerConnection(self._rtc_configuration())
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

        sdp = _limit_video_bandwidth(
            pc.localDescription.sdp,
            int(getattr(self.opt, "webrtc_video_bitrate", 800000) or 800000),
        )
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection(self._rtc_configuration())
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


def _limit_video_bandwidth(sdp: str, bitrate: int) -> str:
    """Add conservative video bandwidth hints to SDP answer.

    aiortc 1.x does not expose RTCRtpSender.setParameters(maxBitrate), so this
    keeps the browser-side estimator aligned with the server's low-latency
    target while encoder defaults are patched separately above.
    """
    if not sdp or bitrate <= 0:
        return sdp

    as_kbps = max(1, bitrate // 1000)
    tias = max(1, bitrate)
    lines = sdp.splitlines()
    out = []
    in_video = False
    inserted = False

    for line in lines:
        if line.startswith("m="):
            if in_video and not inserted:
                out.append(f"b=AS:{as_kbps}")
                out.append(f"b=TIAS:{tias}")
            in_video = line.startswith("m=video")
            inserted = False
            out.append(line)
            continue

        if in_video and line.startswith("b="):
            continue

        out.append(line)
        if in_video and line.startswith("c=") and not inserted:
            out.append(f"b=AS:{as_kbps}")
            out.append(f"b=TIAS:{tias}")
            inserted = True

    if in_video and not inserted:
        out.append(f"b=AS:{as_kbps}")
        out.append(f"b=TIAS:{tias}")

    return "\r\n".join(out) + "\r\n"
