###############################################################################
#  配置解析 — conf.ini + CLI 参数
#
#  优先级: 命令行参数 > conf.ini > 代码内置默认值
#
#  conf.ini 分段:
#    [app]   启动参数 (model/avatar_id/batch_size/transport/...)
#    [audio] 音频特征窗口 (l/m/r)
#    [tts]   TTS 配置 (tts/REF_FILE/TTS_SERVER/xinference_*)
#    [llm]   LLM 配置 (base_url/api_key/model/...)
#
#  [tts] 和 [llm] 段由 llm.py / tts 插件通过 get_config() 读取，
#  这里只负责把它们也暴露到 opt 上，便于统一访问。
###############################################################################

import argparse
import configparser
import json
import os

# conf.ini 默认路径 (项目根目录)
_DEFAULT_CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf.ini")

_config_cache = None


def get_config(config_path: str = None) -> configparser.ConfigParser:
    """加载并缓存 conf.ini。供 config.py / llm.py / tts 插件共享同一份配置。"""
    global _config_cache
    if _config_cache is not None and config_path is None:
        return _config_cache
    path = config_path or os.getenv("LIVETALKING_CONF", _DEFAULT_CONF)
    cfg = configparser.ConfigParser()
    # 保留键的大小写 (REF_FILE / TTS_SERVER 等)
    cfg.optionxform = str
    if os.path.isfile(path):
        cfg.read(path, encoding="utf-8")
    if config_path is None:
        _config_cache = cfg
    return cfg


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def _ini_get(cfg, section, key, fallback=None):
    """从 ini 取字符串值，缺失返回 fallback"""
    if cfg.has_option(section, key):
        return cfg.get(section, key)
    return fallback


def parse_args():
    """解析配置: conf.ini 作为默认值，命令行可覆盖"""
    # 先解析 --config，确定使用哪个 ini
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=_DEFAULT_CONF,
                     help="配置文件路径，默认项目根目录 conf.ini")
    pre_args, _ = pre.parse_known_args()
    cfg = get_config(pre_args.config)

    A = lambda k, d: _ini_get(cfg, 'app', k, d)        # [app]
    AU = lambda k, d: _ini_get(cfg, 'audio', k, d)     # [audio]
    T = lambda k, d: _ini_get(cfg, 'tts', k, d)        # [tts]

    parser = argparse.ArgumentParser(
        description="LiveTalking Digital Human Server", parents=[pre])

    # ─── 音频特征窗口 ──────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=int(A('fps', 25)),
                        help="video fps, must be 25")
    parser.add_argument('-l', type=int, default=int(AU('l', 10)))
    parser.add_argument('-m', type=int, default=int(AU('m', 8)))
    parser.add_argument('-r', type=int, default=int(AU('r', 10)))

    # ─── 数字人模型 ────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default=A('model', 'wav2lip'),
                        help="avatar model: musetalk/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str,
                        default=A('avatar_id', 'wav2lip256_avatar1'),
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int,
                        default=int(A('batch_size', 16)), help="infer batch")
    parser.add_argument('--modelres', type=int, default=int(A('modelres', 192)))
    parser.add_argument('--modelfile', type=str, default=A('modelfile', '') or '')

    # ─── 自定义动作和多形象 ────────────────────────────────────────────
    parser.add_argument('--customvideo_config', type=str,
                        default=A('customvideo_config', '') or '',
                        help="custom action json")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str, default=T('tts', 'edgetts'),
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/xinference/"
                             "fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str,
                        default=T('REF_FILE', 'zh-CN-YunxiaNeural'),
                        help="参考文件名或语音模型ID")
    parser.add_argument('--REF_TEXT', type=str, default=T('REF_TEXT', None) or None)
    parser.add_argument('--TTS_SERVER', type=str,
                        default=T('TTS_SERVER', 'http://127.0.0.1:9880'))

    # ─── 传输 ─────────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str, default=A('transport', 'webrtc'),
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default=A('push_url',
                                  'http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream'))
    parser.add_argument('--max_session', type=int, default=int(A('max_session', 1)))
    parser.add_argument('--listenport', type=int, default=int(A('listenport', 8010)),
                        help="web listen port")

    opt = parser.parse_args()

    # ─── 后处理 ────────────────────────────────────────────────────────
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    # 把 [tts] / [llm] / [inference] 段也挂到 opt 上，便于插件直接访问
    opt.config_path = pre_args.config
    opt.xinference_tts_key = T('xinference_tts_key', '') or ''
    opt.xinference_tts_model = T('xinference_tts_model', 'CosyVoice2-0.5B')

    # [inference] 远程推理
    I = lambda k, d: _ini_get(cfg, 'inference', k, d)
    opt.inference_remote = (I('remote', 'false') or 'false').lower() == 'true'
    opt.inference_server_url = I('base_url', 'http://127.0.0.1:8020') or 'http://127.0.0.1:8020'
    opt.inference_api_key = I('api_key', '') or ''

    return opt
