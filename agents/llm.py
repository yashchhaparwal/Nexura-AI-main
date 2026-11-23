"""LLM access shared by the coder and debugger: lazy construction (no import-time side effects) and rate-limit backoff."""
import random
import re
import threading
import time
from typing import Any

from langchain_core.messages import AIMessage

import config
from core import execution_manager as em

_override: Any = None  # tests inject a fake chat model here
_cache: dict[tuple, Any] = {}


def set_llm(llm: Any) -> None:
    global _override
    _override = llm


def get_llm(model_name: str):
    """Factory function to initialize the correct LLM provider."""
    if any(m in model_name.lower() for m in ["llama", "mixtral", "gemma"]):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, temperature=config.TEMPERATURE, request_timeout=config.LLM_TIMEOUT_S, max_retries=1)
    from langchain_google_genai import ChatGoogleGenerativeAI
    # the library defaults are NO timeout and 6 internal retries; we apply our own bounded backoff instead
    return ChatGoogleGenerativeAI(model=model_name, temperature=config.TEMPERATURE, timeout=config.LLM_TIMEOUT_S, max_retries=1)


def model_for(role: str) -> str:
    if _override is not None:
        return "fake"
    return config.model_name("coder")  # the debugger uses the coder's model


FAILOVER_AFTER = 2     # refusals in a row from one model before the next model is tried
STICKY_S = 600         # after a switch, keep using the model that answered this long, then try the first one again
_sticky: dict = {"model": "", "until": 0.0}


def models_for(role: str) -> list[str]:
    """The main model first, then the configured fallbacks (config.FALLBACK_MODELS). Just the main one when none are set."""
    primary = model_for(role)
    if _override is not None or not config.FALLBACK_MODELS:
        return [primary]
    return [primary] + [m for m in config.FALLBACK_MODELS if m != primary]


def llm_with_tools(role: str, model: str | None = None):
    if _override is not None:
        return _override
    name = model or model_for(role)
    key = (name, config.PATCH_MODE)
    if key not in _cache:
        _cache[key] = get_llm(name).bind_tools(em.tool_schemas())
    return _cache[key]


def _short(text: Any, n: int = 200) -> str:
    text = text if isinstance(text, str) else str(text)
    return text if len(text) <= n else text[:n] + "..."


# Errors worth retrying with backoff: rate limits, server overload (5xx / "high demand"), timeouts
TRANSIENT = re.compile(r"\b(429|500|502|503|504)\b|UNAVAILABLE|RESOURCE_EXHAUSTED|DEADLINE_EXCEEDED|timed? ?out|overloaded|"
                       r"ConnectionError|ReadError|RemoteProtocolError|"
                       # a dropped connection is as temporary as a timeout (a live run ended with WinError 10054 on its first call)
                       r"WinError 1005[34]|forcibly closed|Connection (?:reset|aborted)|ConnectionReset|ConnectionAborted|RemoteDisconnected|Errno 104",
                       re.IGNORECASE)


QUOTA = re.compile(r"PerDay|per day", re.IGNORECASE)    # per-MINUTE limits ("tokens per minute ... exceeded on Free Tier") are transient
RETRY_DELAY = re.compile(r"(?:retryDelay['\": ]+|retry in )(\d+(?:\.\d+)?)s", re.IGNORECASE)


def quota_message(error_text: str) -> str | None:
    """A DAILY/long quota exhaustion is not transient: waiting a few seconds will not help, so fail fast with a clear message."""
    m = RETRY_DELAY.search(error_text)
    delay = float(m.group(1)) if m else 0.0
    if QUOTA.search(error_text) or delay > 300:
        when = f" (resets in about {delay / 3600:.0f}h)" if delay > 3600 else ""
        return f"LLM_QUOTA: the API quota is exhausted{when}. Check your plan/billing or try again later."
    return None


def is_rate_limit(error_text: str) -> bool:
    return bool(TRANSIENT.search(error_text))


def backoff_delay(attempt: int, error_text: str) -> float:
    """Seconds to wait after a rate limit: server's retryDelay if present, else exponential; plus jitter."""
    m = RETRY_DELAY.search(error_text)
    base = float(m.group(1)) + 2 if m else min(5 * 2 ** attempt, 60)
    return min(base, 120) + random.uniform(0, base * 0.25)


def call_with_deadline(fn, seconds: float | None = None):
    """Run fn() and return its result, or raise TimeoutError after `seconds` of wall-clock time. The client libraries'
    own timeouts do not always stop a stalled connection, so a call could block a run forever. The worker is a daemon
    thread: if it never returns it is abandoned (it cannot be killed) and does not block process exit."""
    seconds = config.LLM_HARD_TIMEOUT_S if seconds is None else seconds
    box: dict[str, Any] = {}

    def work():
        try:
            box["value"] = fn()
        except BaseException as e:   # re-raised in the caller's thread
            box["error"] = e

    t = threading.Thread(target=work, name="llm-call", daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"Request timed out: no answer from the model after {seconds:.0f}s (call abandoned)")
    if "error" in box:
        raise box["error"]
    return box["value"]


def invoke(role: str, messages: list, prompt_version: str, on_log=None) -> tuple[AIMessage | None, str | None]:
    """Call the model. Returns (response, None) or (None, failure_reason). Rate limits retry with backoff. With fallback models
    configured, a model that keeps refusing (or whose daily quota is used up) is replaced by the next one."""
    models = models_for(role)
    idx = models.index(_sticky["model"]) if _sticky["model"] in models and time.time() < _sticky["until"] else 0
    refused_here, quota_hit, detail = 0, set(), ""
    total = config.LLM_MAX_ATTEMPTS + 2 * (len(models) - 1)     # room to try every model, not only the first
    for attempt in range(total):
        model = models[idx]
        try:
            response = call_with_deadline(lambda m=model: llm_with_tools(role, m).invoke(messages))
            if len(models) > 1:
                if idx and not (_sticky["model"] == model and time.time() < _sticky["until"]):
                    print(f"  [!] using {model} (the main model {models[0]} was not answering); the main model is tried again in {STICKY_S // 60} min")
                _sticky.update(model=models[idx] if idx else "", until=time.time() + STICKY_S if idx else 0.0)
            if on_log:
                calls = [c["name"] for c in (getattr(response, "tool_calls", None) or [])]
                usage = getattr(response, "usage_metadata", None) or {}
                on_log(role=role, prompt_version=prompt_version, model=model, result="ok",
                       input_summary=_short(getattr(messages[-1], "content", ""), 300),
                       output_summary=("tool_calls: " + ", ".join(calls)) if calls else _short(getattr(response, "content", ""), 300),
                       tokens=usage.get("total_tokens"))
            return response, None
        except config.ConfigError as e:
            return None, f"LLM_CONFIG: {e}"
        except Exception as e:
            err = str(e)
            quota = quota_message(err)
            if quota:
                quota_hit.add(idx)
                if len(quota_hit) < len(models):   # the daily quota is per model: another model may still have some left
                    idx = next(i for i in range(len(models)) if i not in quota_hit)
                    print(f"  [!] daily quota of {model} is used up; trying {models[idx]}")
                    continue
                if on_log:
                    on_log(role=role, prompt_version=prompt_version, model=model, result="quota_exhausted")
                return None, quota
            if not is_rate_limit(err):
                if on_log:
                    on_log(role=role, prompt_version=prompt_version, model=model, result="error")
                return None, f"LLM_ERROR: {_short(err, 150)}"
            detail = _short(err, 150)
            refused_here += 1
            if attempt + 1 < total:
                if on_log:   # a failed attempt still counts against the provider's request quota: record it for the usage budget
                    on_log(role=role, prompt_version=prompt_version, model=model, result="retry_after_rate_limit")
                if len(models) > 1 and refused_here >= FAILOVER_AFTER:
                    idx, refused_here = (idx + 1) % len(models), 0
                    print(f"  [!] {model} keeps refusing ({detail[:50]}...); trying {models[idx]}")
                    continue   # a different model: no reason to wait first
                wait = backoff_delay(attempt, err)
                print(f"  [!] Rate limited (attempt {attempt + 1}/{total}). Waiting {wait:.1f}s...")
                time.sleep(wait)
    if on_log:
        on_log(role=role, prompt_version=prompt_version, model=models[idx], result="rate_limited")
    return None, f"LLM_UNAVAILABLE: {detail}"
