"""
 * Author: JimZhang
 * Date: 2026-06-01 20:42:18
 * LastEditors: 很拉风的James
 * LastEditTime: 2026-06-01 22:55:31
 * FilePath: /remote_model/config.py
 * Description: 配置解析；读取 conf.ini，命令行参数可覆盖。
"""

import argparse
import configparser
import os

_DEFAULT_CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf.ini")


def _load_ini(path):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # 保留键大小写
    if os.path.isfile(path):
        cfg.read(path, encoding="utf-8")
    return cfg


def _ini_get(cfg, section, key, fallback=None):
    if cfg.has_option(section, key):
        return cfg.get(section, key)
    return fallback


def parse_args():
    """解析配置: conf.ini 作为默认值，命令行可覆盖。返回 argparse.Namespace。"""
    # 先取 --config 确定配置文件
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=_DEFAULT_CONF,
                     help="配置文件路径，默认同目录 conf.ini")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_ini(pre_args.config)

    S = lambda k, d: _ini_get(cfg, 'server', k, d)
    M = lambda k, d: _ini_get(cfg, 'model', k, d)
    P = lambda k, d: _ini_get(cfg, 'path', k, d)

    parser = argparse.ArgumentParser(
        description="LiveTalking Inference Server", parents=[pre])

    parser.add_argument("--host", type=str, default=S('host', '0.0.0.0'))
    parser.add_argument("--port", type=int, default=int(S('port', 8020)))
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("INFERENCE_API_KEY", S('api_key', '') or ''))

    parser.add_argument("--model", type=str, default=M('model', 'musetalk'),
                        choices=["musetalk", "wav2lip"])
    parser.add_argument("--avatar_id", type=str, default=M('avatar_id', 'musetalk_avatar1'))
    parser.add_argument("--batch_size", type=int, default=int(M('batch_size', 8)))
    parser.add_argument("--jpeg_quality", type=int, default=int(M('jpeg_quality', 60)))
    parser.add_argument("--serialize_model_tasks", type=str,
                        default=M('serialize_model_tasks', 'true'),
                        help="true: serialize audio_feature/inference to avoid GPU contention")
    parser.add_argument("--wav2lip_model_path", type=str,
                        default=M('wav2lip_model_path', './models/wav2lip.pth'))

    parser.add_argument("--models_dir", type=str, default=P('models_dir', './models'))
    parser.add_argument("--avatars_dir", type=str, default=P('avatars_dir', './data/avatars'))

    return parser.parse_args()
