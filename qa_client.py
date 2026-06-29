import os
from typing import Any, Dict, Tuple

import aiohttp

from utils.logger import logger


def _ini(section: str, key: str, default: str = "") -> str:
    try:
        from config import get_config
        cfg = get_config()
        if cfg.has_option(section, key):
            return cfg.get(section, key)
    except Exception:
        pass
    return default


def _conf(env_key: str, section: str, key: str, default: str = "") -> str:
    value = os.getenv(env_key)
    if value is not None:
        return value
    return _ini(section, key, default)


def _is_placeholder(value: str) -> bool:
    text = (value or "").strip()
    return not text or "<" in text or "your-" in text.lower()


def _chat_config() -> Dict[str, Any]:
    base_url = _conf("LLM_BASE_URL", "llm", "base_url", "http://127.0.0.1:9380/v1").strip()
    api_key = _conf("LLM_API_KEY", "llm", "api_key", "").strip()
    model = _conf("LLM_MODEL", "llm", "model", "qwen3-14b").strip()
    provider = _conf("LLM_PROVIDER", "llm", "provider", "").strip().lower()
    max_tokens = int(_conf("LLM_MAX_TOKENS", "llm", "max_tokens", "1024") or "1024")
    system_prompt = _conf(
        "LLM_SYSTEM_PROMPT",
        "llm",
        "system_prompt",
        "你是 GuiXiaoxi 数字人助手，请用简洁、自然的中文回答。",
    )
    fallback_answer = _conf(
        "LLM_FALLBACK_ANSWER",
        "llm",
        "fallback_answer",
        "我已经收到你的问题，但当前问答服务还没有配置完成。",
    )
    if not provider:
        provider = "dify" if model.lower() == "dify" or "dify" in base_url.lower() else "openai"
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "provider": provider,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
        "fallback_answer": fallback_answer,
    }


def _dify_chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat-messages"):
        return url
    return url + "/chat-messages"


def _openai_chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


async def ask_question(
    question: str,
    *,
    conversation_id: str = "",
    user: str = "guixiaoxi",
    inputs: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    cfg = _chat_config()
    if _is_placeholder(cfg["base_url"]):
        return cfg["fallback_answer"], {"provider": "fallback", "configured": False}

    timeout = aiohttp.ClientTimeout(total=90)
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    if cfg["provider"] == "dify":
        payload = {
            "inputs": inputs or {},
            "query": question,
            "response_mode": "blocking",
            "conversation_id": conversation_id or "",
            "user": user or "guixiaoxi",
        }
        url = _dify_chat_url(cfg["base_url"])
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise RuntimeError(f"Dify chat failed {resp.status}: {data}")
        answer = data.get("answer") or data.get("message") or ""
        return answer, {
            "provider": "dify",
            "conversation_id": data.get("conversation_id") or conversation_id or "",
            "message_id": data.get("message_id") or data.get("id") or "",
        }

    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": question},
        ],
        "max_tokens": cfg["max_tokens"],
        "stream": False,
    }
    url = _openai_chat_url(cfg["base_url"])
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                raise RuntimeError(f"OpenAI-compatible chat failed {resp.status}: {data}")

    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        answer = message.get("content") or choices[0].get("text") or ""
    else:
        answer = data.get("answer") or data.get("content") or ""
    logger.info("qa answer provider=%s chars=%d", cfg["provider"], len(answer))
    return answer, {"provider": cfg["provider"], "id": data.get("id", "")}
