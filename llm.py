import time
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

# ─────────────────────────────────────────────────────────────────────────────
#  LLM 配置
#
#  优先级: 环境变量 > conf.ini [llm] 段 > 内置默认
#  便于在 vLLM(qwen3) / Dify / DashScope 之间切换。
#
#  conf.ini 配置见 [llm] 段; 环境变量(可临时覆盖):
#    LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_MAX_TOKENS /
#    LLM_ENABLE_THINKING / LLM_SYSTEM_PROMPT
#
#  切换到 Dify 示例(改 conf.ini 或设环境变量):
#    base_url = http://<dify-host>/v1
#    api_key  = app-xxxxxx
#    model    = dify
# ─────────────────────────────────────────────────────────────────────────────

def _ini(key, default=None):
    try:
        from config import get_config
        cfg = get_config()
        if cfg.has_option('llm', key):
            return cfg.get('llm', key)
    except Exception:
        pass
    return default


def _conf(env_key, ini_key, default):
    """环境变量 > conf.ini > 默认"""
    v = os.getenv(env_key)
    if v is not None:
        return v
    return _ini(ini_key, default)


LLM_BASE_URL = _conf("LLM_BASE_URL", "base_url", "http://127.0.0.1:9380/v1")
LLM_API_KEY = _conf("LLM_API_KEY", "api_key", "EMPTY")
LLM_MODEL = _conf("LLM_MODEL", "model", "qwen3-14b")
LLM_MAX_TOKENS = int(_conf("LLM_MAX_TOKENS", "max_tokens", "10240"))
LLM_ENABLE_THINKING = str(_conf("LLM_ENABLE_THINKING", "enable_thinking", "false")).lower() == "true"
LLM_SYSTEM_PROMPT = _conf("LLM_SYSTEM_PROMPT", "system_prompt",
                          "你是一个知识助手，尽量以简短、口语化的方式输出")

# 触发数字人 put_msg_txt 的最小累积字符数（攒够一个短句再合成，降低首句延迟）
_FLUSH_MIN_CHARS = 10
_PUNCTUATION = ",.!;:，。！？：；"


def llm_response(message, avatar_session: 'BaseAvatar', datainfo: dict = {}):
    try:
        start = time.perf_counter()
        from openai import OpenAI
        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )
        logger.info(f"llm init: base_url={LLM_BASE_URL} model={LLM_MODEL} msg={message}")

        # 构造请求参数。qwen3 (vLLM) 支持通过 chat_template_kwargs 关闭思考模式，
        # 关闭后直接流式输出 content，显著降低首字延迟，避免把思考过程念出来。
        create_kwargs = dict(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
                {'role': 'user', 'content': message},
            ],
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )
        if not LLM_ENABLE_THINKING:
            # 仅 qwen3 等支持该参数的服务有效；Dify/DashScope 会忽略 extra_body
            create_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        try:
            completion = client.chat.completions.create(**create_kwargs)
        except Exception:
            # 某些端点(如 Dify)不接受 extra_body，去掉后重试一次
            create_kwargs.pop("extra_body", None)
            completion = client.chat.completions.create(**create_kwargs)

        result = ""
        first = True
        for chunk in completion:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            # 只取正式回答 content，丢弃 qwen3 的思考过程(reasoning/reasoning_content)
            msg = getattr(delta, "content", None)
            if msg is None:
                continue
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start}s")
                first = False

            lastpos = 0
            for i, char in enumerate(msg):
                if char in _PUNCTUATION:
                    result = result + msg[lastpos:i + 1]
                    lastpos = i + 1
                    if len(result) > _FLUSH_MIN_CHARS:
                        logger.info(result)
                        avatar_session.put_msg_txt(result, datainfo)
                        result = ""
            result = result + msg[lastpos:]

        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result, datainfo)

    except Exception as e:
        logger.exception('llm exception:')
        return
