"""Usage budget: counting across runs, caps, estimate, planner usage, and a clean pause that can be resumed."""
import time

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

import config
from core import budget, metrics
from core.runner import prepare_resume
from core.state_manager import StateManager
from tests.test_persistence import PLAN, env, run_new  # noqa: F401  (env is a fixture)


@pytest.fixture
def sm(tmp_path):
    s = StateManager(tmp_path / "b.db")
    s.upsert_project("p", "/w")
    s.create_run("r1", "p", "prompt", "ask_risky")
    s.create_run("r2", "p", "prompt", "ask_risky")
    return s


def caps(monkeypatch, requests=0, tokens=0, hours=24):
    monkeypatch.setattr(config, "BUDGET_MAX_REQUESTS", requests)
    monkeypatch.setattr(config, "BUDGET_MAX_TOKENS", tokens)
    monkeypatch.setattr(config, "BUDGET_WINDOW_HOURS", hours)


def test_usage_counts_all_runs_and_the_planner_requests_field(sm):
    sm.add_event("r1", "llm_call", 1, role="coder", tokens=100)
    sm.add_event("r2", "llm_call", 1, role="coder", tokens=None)
    sm.add_event("r2", "llm_call", None, role="planner", tokens=500, requests=3)   # one event, three requests
    sm.add_event("r1", "task_started", 1)                                          # not a model call
    assert sm.usage_since(0) == (5, 600)


def test_usage_respects_the_window(sm, monkeypatch):
    caps(monkeypatch, requests=10, hours=1)
    sm.add_event("r1", "llm_call", 1, role="coder", tokens=10)
    assert budget.used(sm) == {"requests": 1, "tokens": 10}
    assert budget.used(sm, now=time.time() + 2 * 3600) == {"requests": 0, "tokens": 0}   # the window moved past the call


def test_no_caps_means_no_limit(sm, monkeypatch):
    caps(monkeypatch)
    for _ in range(50):
        sm.add_event("r1", "llm_call", 1, role="coder", tokens=1_000_000)
    assert not budget.enabled() and budget.check(sm) is None


def test_request_cap_and_token_cap(sm, monkeypatch):
    caps(monkeypatch, requests=3)
    for _ in range(2):
        sm.add_event("r1", "llm_call", 1, role="coder", tokens=5)
    assert budget.check(sm) is None
    sm.add_event("r2", "llm_call", 1, role="coder", tokens=5)      # a different run still counts: the quota is per key
    assert "request budget reached (3/3" in budget.check(sm)
    caps(monkeypatch, tokens=15)
    assert "token budget reached (15/15" in budget.check(sm)


def test_remaining(sm, monkeypatch):
    caps(monkeypatch, requests=10)
    sm.add_event("r1", "llm_call", 1, role="coder", tokens=7)
    assert budget.remaining(sm) == {"requests": 9, "tokens": None}


def test_estimate_scales_with_steps_and_warns_when_the_budget_is_too_small(sm, monkeypatch):
    assert budget.estimate(10) == {"steps": 10, "requests": 52, "tokens": 130_000}
    caps(monkeypatch, requests=20)
    text = budget.estimate_text(10, sm)
    assert "52 model requests" in text and "20 requests" in text and "NOT ENOUGH" in text
    caps(monkeypatch)
    assert "Budget left" not in budget.estimate_text(10, sm)       # without caps it is only the estimate


def test_config_rejects_a_bad_cap():
    assert config.env_limit("X", {}) == 0 and config.env_limit("X", {"X": "450"}) == 450 and config.env_limit("X", {"X": "0"}) == 0
    for bad in ("-1", "abc"):
        with pytest.raises(config.ConfigError):
            config.env_limit("X", {"X": bad})


def test_metrics_report_model_requests(sm):
    sm.add_event("r1", "llm_call", 1, role="coder", tokens=10)
    sm.add_event("r1", "llm_call", None, role="planner", tokens=20, requests=2)
    m = metrics.compute(sm, "r1")
    assert m["model_requests"] == 3 and m["tokens"] == 30


def test_planner_logs_requests_and_tokens_once(sm):
    from core.orchestrator import _log_planner_usage
    from core.schemas import ProjectContext

    class P:
        requests, tokens = 4, 1234
    ctx = ProjectContext(run_id="r1", root=".")
    ctx.state_manager = sm
    _log_planner_usage(ctx, P())
    _log_planner_usage(ctx, None)                                   # planner never built: nothing to log
    assert sm.usage_since(0) == (4, 1234)


def test_run_pauses_cleanly_at_the_cap_and_can_be_resumed(env, monkeypatch):    # noqa: F811
    caps(monkeypatch, requests=1)
    monkeypatch.setattr(config, "FAILURE_POLICY", "continue")      # the pause must stop the run even under "continue"
    # task 1: the first model call is allowed (0 < 1) but does not finish the task; the second is blocked
    first = [AIMessage(content="", tool_calls=[{"name": "create_file", "args": {"target_file": "a.txt", "content": "hi"}, "id": "c1"}])]
    ctx, graph, cfg = run_new(env, first)
    graph.invoke(Command(resume="yes"), cfg)                        # approve the plan
    run = env.sm.get_run(ctx.run_id)
    rows = {r["task_id"]: r for r in env.sm.list_tasks(ctx.run_id)}
    assert run.status == "failed" and "--resume" in run.error and "BUDGET_REACHED" in run.error
    assert rows[1]["status"] == "failed" and rows[2]["status"] == "skipped"          # task 2 was not started
    assert len(env.llm.calls) == 1                                                    # the blocked call never reached the model
    assert any(e["type"] == "budget_reached" for e in env.sm.list_events(ctx.run_id))
    caps(monkeypatch)                                                                 # budget lifted: the run can continue
    _, state = prepare_resume(env.sm, ctx.run_id, require_git=False)
    assert state["resuming"] and state["current_task_index"] == 0


def test_budget_and_git_events_reach_the_api_stream():
    """The extension shows these; the server only forwards the event types it knows."""
    from core.run_manager import _map_db_event
    for t, data in (("budget_estimate", {"steps": 2, "requests": 10, "tokens": 26000.0}), ("budget_reached", {"reason": "request budget reached"}),
                    ("git_committed", {"branch": "nexura/r", "commit": "abc", "previous": "main", "files": 1}),
                    ("git_commit_skipped", {"reason": "x"})):
        out = _map_db_event({"type": t, "task_id": None, "data": data})
        assert [name for name, _ in out] == [t] and out[0][1]["task_id"] is None and all(out[0][1][k] == v for k, v in data.items())
