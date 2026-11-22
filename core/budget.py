"""Usage budget for free-tier model quotas.

Counts the `llm_call` events (planner, coder, debugger) over a rolling window across every run, compares them with the caps in
config (NEXURA_BUDGET_MAX_REQUESTS / NEXURA_BUDGET_MAX_TOKENS), and estimates what a plan will cost. Read-only: it never
blocks by itself; the orchestrator asks `check()` before each model call and ends the task cleanly, so `--resume` can continue.
"""
import time

import config

# Measured on the 14 V1 eval projects (docs/V1_EVAL.md, evals/results/2026-11-17_011854.json): about 3.7 model requests and
# 9.8k tokens per plan step. The margin covers the planner call, retries and debugger loops, which the eval did not count.
REQUESTS_PER_STEP = 4
TOKENS_PER_STEP = 10_000
MARGIN = 1.3


def enabled() -> bool:
    return bool(config.BUDGET_MAX_REQUESTS or config.BUDGET_MAX_TOKENS)


def window_start(now: float | None = None) -> float:
    return (time.time() if now is None else now) - config.BUDGET_WINDOW_HOURS * 3600


def used(sm, now: float | None = None) -> dict:
    """Requests and tokens spent in the current window (all runs)."""
    requests, tokens = sm.usage_since(window_start(now))
    return {"requests": requests, "tokens": tokens}


def remaining(sm, now: float | None = None) -> dict:
    """Left in the window; None for a resource without a cap."""
    u = used(sm, now)
    return {
        "requests": max(config.BUDGET_MAX_REQUESTS - u["requests"], 0) if config.BUDGET_MAX_REQUESTS else None,
        "tokens": max(config.BUDGET_MAX_TOKENS - u["tokens"], 0) if config.BUDGET_MAX_TOKENS else None,
    }


def check(sm, now: float | None = None) -> str | None:
    """None if another model call is allowed, else a readable reason."""
    if not enabled():
        return None
    u = used(sm, now)
    window = f"{config.BUDGET_WINDOW_HOURS}h"
    if config.BUDGET_MAX_REQUESTS and u["requests"] >= config.BUDGET_MAX_REQUESTS:
        return (f"request budget reached ({u['requests']}/{config.BUDGET_MAX_REQUESTS} in the last {window}). "
                f"Resume later, or raise NEXURA_BUDGET_MAX_REQUESTS.")
    if config.BUDGET_MAX_TOKENS and u["tokens"] >= config.BUDGET_MAX_TOKENS:
        return (f"token budget reached ({u['tokens']}/{config.BUDGET_MAX_TOKENS} in the last {window}). "
                f"Resume later, or raise NEXURA_BUDGET_MAX_TOKENS.")
    return None


def estimate(steps: int) -> dict:
    """Rough cost of a plan with `steps` steps (an average, not a promise: a failing step costs several times more)."""
    return {"steps": steps, "requests": round(steps * REQUESTS_PER_STEP * MARGIN),
            "tokens": round(steps * TOKENS_PER_STEP * MARGIN, -3)}


def estimate_text(steps: int, sm=None, now: float | None = None) -> str:
    e = estimate(steps)
    text = f"Estimated cost of this plan: about {e['requests']} model requests and {int(e['tokens']):,} tokens ({steps} steps)."
    if sm is not None and enabled():
        left, parts = remaining(sm, now), []
        if left["requests"] is not None:
            parts.append(f"{left['requests']} requests")
            if e["requests"] > left["requests"]:
                parts.append("NOT ENOUGH for the whole plan: the run will pause and can be resumed later")
        if left["tokens"] is not None:
            parts.append(f"{left['tokens']:,} tokens")
        text += f" Budget left in the last {config.BUDGET_WINDOW_HOURS}h window: " + ", ".join(parts) + "."
    return text
