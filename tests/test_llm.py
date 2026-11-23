"""LLM access: every model client has a timeout, and transient errors (rate limit, overload, timeouts) are retried with backoff."""
import time

import pytest

import agents.llm as llm
import config
from tests.conftest import FakeChatModel, tool_call_msg


def test_google_and_groq_clients_get_an_explicit_timeout(monkeypatch):
    """The library default is no timeout plus 6 internal retries: a stalled connection hung a whole eval run."""
    seen = {}

    class Rec:
        def __init__(self, **kw):
            seen.update(kw)
    import langchain_google_genai
    import langchain_groq
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", Rec)
    monkeypatch.setattr(langchain_groq, "ChatGroq", Rec)
    llm.get_llm("gemini-test")
    assert seen["timeout"] == config.LLM_TIMEOUT_S and seen["max_retries"] <= 2
    seen.clear()
    llm.get_llm("llama-3-test")
    assert seen["request_timeout"] == config.LLM_TIMEOUT_S


@pytest.mark.parametrize("text,transient", [
    ("503 UNAVAILABLE. This model is currently experiencing high demand.", True), ("Request timed out.", True),
    ("429 RESOURCE_EXHAUSTED", True), ("httpx.ReadError: connection closed", True), ("ConnectionError: reset", True),
    ("[WinError 10054] An existing connection was forcibly closed by the remote host", True),
    ("[WinError 10053] An established connection was aborted by the software in your host machine", True),
    ("ConnectionResetError: [Errno 104] Connection reset by peer", True), ("http.client.RemoteDisconnected: Remote end closed connection", True),
    ("401 API key not valid", False), ("400 invalid argument", False), ("403 permission denied for this key", False)])
def test_transient_classification(text, transient):
    assert llm.is_rate_limit(text) is transient


def test_invoke_retries_a_timeout_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    fake = FakeChatModel(script=[TimeoutError("Request timed out."), tool_call_msg("task_complete", {"summary": "x"})], calls=[])
    monkeypatch.setattr(llm, "_override", fake)
    resp, failure = llm.invoke("coder", ["hi"], "1")
    assert failure is None and resp.tool_calls[0]["name"] == "task_complete" and len(sleeps) == 1


def test_invoke_reports_persistent_overload_and_hard_errors_differently(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "_override", FakeChatModel(script=[Exception("503 UNAVAILABLE")] * config.LLM_MAX_ATTEMPTS, calls=[]))
    assert llm.invoke("coder", ["hi"], "1")[1].startswith("LLM_UNAVAILABLE")
    monkeypatch.setattr(llm, "_override", FakeChatModel(script=[Exception("401 API key not valid")], calls=[]))
    assert llm.invoke("coder", ["hi"], "1")[1].startswith("LLM_ERROR")


GEMINI_DAILY_QUOTA = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota ... Generate content "
                      "requests per day limit exceeded on Free Tier.', 'details': [{'violations': [{'quotaId': "
                      "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'retryDelay': '36891s'}]}}")


def test_daily_quota_exhaustion_fails_fast_with_a_clear_reason(monkeypatch):
    """A free-tier daily limit used to be 'retried' for minutes; it must stop at once and say what happened."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    fake = FakeChatModel(script=[Exception(GEMINI_DAILY_QUOTA)] * 3, calls=[])
    monkeypatch.setattr(llm, "_override", fake)
    resp, failure = llm.invoke("coder", ["hi"], "1")
    assert resp is None and failure.startswith("LLM_QUOTA") and "10h" in failure
    assert sleeps == [] and len(fake.calls) == 1


GEMINI_PER_MINUTE = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. * Quota exceeded for metric: "
                     "generate_content_free_tier_input_token_count, limit: 250000, description: Generate content input tokens per minute "
                     "limit exceeded on Free Tier.\nPlease retry in 5.5s.'}}")


def test_per_minute_token_limits_are_retried_not_treated_as_a_daily_quota(monkeypatch):
    """A tokens-per-minute limit on the free tier says 'exceeded on Free Tier' too, but waiting a few seconds fixes it."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(llm, "_override", FakeChatModel(script=[Exception(GEMINI_PER_MINUTE), tool_call_msg("task_complete", {"summary": "x"})], calls=[]))
    assert llm.quota_message(GEMINI_PER_MINUTE) is None
    resp, failure = llm.invoke("coder", ["hi"], "1")
    assert failure is None and len(sleeps) == 1 and 7 <= sleeps[0] < 12     # the server's "retry in 5.5s" (+2s) is honoured


def test_short_rate_limits_are_still_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    err = "429 RESOURCE_EXHAUSTED. {'retryDelay': '12s'}"
    monkeypatch.setattr(llm, "_override", FakeChatModel(script=[Exception(err), tool_call_msg("task_complete", {"summary": "x"})], calls=[]))
    resp, failure = llm.invoke("coder", ["hi"], "1")
    assert failure is None and len(sleeps) == 1 and sleeps[0] >= 12


def test_retried_attempts_are_logged_so_the_usage_budget_counts_them(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    logged = []
    monkeypatch.setattr(llm, "_override", FakeChatModel(
        script=[Exception("503 UNAVAILABLE"), Exception("429 RESOURCE_EXHAUSTED"), tool_call_msg("task_complete", {"summary": "x"})], calls=[]))
    resp, failure = llm.invoke("coder", ["hi"], "1", on_log=lambda **kw: logged.append(kw["result"]))
    assert failure is None and logged == ["retry_after_rate_limit", "retry_after_rate_limit", "ok"]   # 3 requests were made


def test_call_with_deadline_returns_raises_and_abandons():
    import threading
    assert llm.call_with_deadline(lambda: 42, 2) == 42
    with pytest.raises(ValueError, match="boom"):
        llm.call_with_deadline(lambda: (_ for _ in ()).throw(ValueError("boom")), 2)
    stall = threading.Event()
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        llm.call_with_deadline(lambda: stall.wait(30), 0.2)
    assert time.monotonic() - start < 5          # did not wait for the stalled call
    stall.set()                                  # let the abandoned thread finish


def test_a_stalled_model_call_is_cut_off_and_retried(monkeypatch):
    """A live eval hung ~53 min on one stalled request: the call now has a hard deadline and the stall is retried like any timeout."""
    import threading
    stall = threading.Event()
    monkeypatch.setattr(config, "LLM_HARD_TIMEOUT_S", 0.3)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    fake = FakeChatModel(script=[lambda: stall.wait(30), tool_call_msg("task_complete", {"summary": "x"})], calls=[])
    monkeypatch.setattr(llm, "_override", fake)
    logged = []
    start = time.monotonic()
    resp, failure = llm.invoke("coder", ["hi"], "1", on_log=lambda **kw: logged.append(kw["result"]))
    stall.set()
    assert failure is None and resp.tool_calls[0]["name"] == "task_complete"
    assert logged == ["retry_after_rate_limit", "ok"] and time.monotonic() - start < 10


def test_a_model_that_always_stalls_fails_cleanly_instead_of_hanging(monkeypatch):
    import threading
    stall = threading.Event()
    monkeypatch.setattr(config, "LLM_HARD_TIMEOUT_S", 0.2)
    monkeypatch.setattr(config, "LLM_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "_override", FakeChatModel(script=[lambda: stall.wait(30), lambda: stall.wait(30)], calls=[]))
    resp, failure = llm.invoke("coder", ["hi"], "1")
    stall.set()
    assert resp is None and failure.startswith("LLM_UNAVAILABLE") and "timed out" in failure


def test_the_planner_call_has_the_same_deadline(tmp_path, monkeypatch):
    import threading
    monkeypatch.setenv("PLANNER_MODEL_NAME", "m")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(config, "LLM_HARD_TIMEOUT_S", 0.2)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    from agents.planner import PlannerAgent
    stall, p = threading.Event(), PlannerAgent()
    calls = []

    def call_llm(prompt):
        calls.append(1)
        if len(calls) == 1:
            stall.wait(30)
        return '{"ok": true}'
    p.call_llm = call_llm
    assert p._call_with_retry("x") == '{"ok": true}' and len(calls) == 2 and p.requests == 2
    stall.set()


def test_planner_reports_quota_exhaustion_immediately(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANNER_MODEL_NAME", "m")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(time, "sleep", lambda s: pytest.fail("must not sleep"))
    from agents.planner import PlanError, PlannerAgent
    p = PlannerAgent()
    calls = []

    def boom(prompt):
        calls.append(1)
        raise Exception(GEMINI_DAILY_QUOTA)
    p.call_llm = boom
    with pytest.raises(PlanError, match="LLM_QUOTA"):
        p.plan("x", str(tmp_path))
    assert len(calls) == 1


def test_planner_client_uses_v1beta_so_preview_models_work(monkeypatch):
    monkeypatch.setenv("PLANNER_MODEL_NAME", "m")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    from agents.planner import PlannerAgent
    opts = PlannerAgent().client._api_client._http_options
    assert opts.api_version == "v1beta"


def test_a_dropped_connection_is_retried_like_a_timeout_instead_of_ending_the_run(monkeypatch):
    """A live run died on its very first model call with WinError 10054 (3 steps lost); a reset connection is temporary."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    logged = []
    reset = "[WinError 10054] An existing connection was forcibly closed by the remote host"
    monkeypatch.setattr(llm, "_override", FakeChatModel(script=[ConnectionResetError(reset), Exception(reset),
                                                                tool_call_msg("task_complete", {"summary": "x"})], calls=[]))
    resp, failure = llm.invoke("coder", ["hi"], "1", on_log=lambda **kw: logged.append(kw["result"]))
    assert failure is None and resp.tool_calls[0]["name"] == "task_complete"
    assert logged == ["retry_after_rate_limit", "retry_after_rate_limit", "ok"]


def test_a_connection_that_keeps_dropping_fails_cleanly_after_the_allowed_attempts(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    reset = "[WinError 10054] An existing connection was forcibly closed by the remote host"
    fake = FakeChatModel(script=[Exception(reset)] * 3, calls=[])
    monkeypatch.setattr(llm, "_override", fake)
    resp, failure = llm.invoke("coder", ["hi"], "1")
    assert resp is None and failure.startswith("LLM_UNAVAILABLE") and "10054" in failure and len(fake.calls) == 3


def test_a_real_error_is_still_not_retried(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: pytest.fail("must not retry or sleep"))
    fake = FakeChatModel(script=[Exception("401 API key not valid")], calls=[])
    monkeypatch.setattr(llm, "_override", fake)
    resp, failure = llm.invoke("coder", ["hi"], "1")
    assert resp is None and failure.startswith("LLM_ERROR") and len(fake.calls) == 1


def test_the_planner_retries_a_dropped_connection_too(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANNER_MODEL_NAME", "m")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    from agents.planner import PlannerAgent
    p = PlannerAgent()
    calls = []

    def call_llm(prompt):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionResetError("[WinError 10054] An existing connection was forcibly closed by the remote host")
        return '{"ok": true}'
    p.call_llm = call_llm
    assert p._call_with_retry("x") == '{"ok": true}' and len(calls) == 2 and p.requests == 2
