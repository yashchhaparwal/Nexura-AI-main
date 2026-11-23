"""Opt-in model failover (NEXURA_FALLBACK_MODELS) for the coder/debugger, and an honest ending when the model service is down."""
import time

import pytest
from langgraph.types import Command

import agents.llm as llm
import config
from tests.conftest import FakeChatModel, tool_call_msg
from tests.test_persistence import env, run_new  # noqa: F401  (env is a fixture)

BUSY = Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")
QUOTA = Exception("429 RESOURCE_EXHAUSTED. Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel-FreeTier")


def ok():
    return tool_call_msg("task_complete", {"summary": "x"})


@pytest.fixture
def world(monkeypatch):
    """Real model selection (no override) with scripted models: main, backup, third."""
    monkeypatch.setattr(llm, "_override", None)
    monkeypatch.setattr(llm, "_cache", {})
    monkeypatch.setattr(llm, "_sticky", {"model": "", "until": 0.0})
    monkeypatch.setattr(config, "model_name", lambda role: "main")
    monkeypatch.setattr(config, "FALLBACK_MODELS", ("backup", "third"))
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    chats = {}
    monkeypatch.setattr(llm, "get_llm", lambda name: chats[name])

    class W:
        pass
    w = W()
    w.chats, w.sleeps = chats, sleeps

    def script(**by_model):
        llm._cache.clear()           # bound models are cached per name: a new script needs fresh ones
        for name, items in by_model.items():
            chats[name] = FakeChatModel(script=list(items), calls=[])
    w.script = script
    return w


def call(on_log=None):
    return llm.invoke("coder", ["hi"], "1", on_log=on_log)


def test_two_refusals_in_a_row_move_to_the_next_model_without_waiting(world):
    world.script(main=[BUSY, BUSY], backup=[ok()])
    logged = []
    resp, failure = call(lambda **kw: logged.append((kw["model"], kw["result"])))
    assert failure is None and resp.tool_calls[0]["name"] == "task_complete"
    assert len(world.chats["main"].calls) == 2 and len(world.chats["backup"].calls) == 1
    assert len(world.sleeps) == 1                                     # one wait after the first refusal, none at the switch
    assert logged == [("main", "retry_after_rate_limit"), ("main", "retry_after_rate_limit"), ("backup", "ok")]


def test_a_single_refusal_does_not_switch(world):
    world.script(main=[BUSY, ok()], backup=[])
    resp, failure = call()
    assert failure is None and len(world.chats["main"].calls) == 2 and world.chats["backup"].calls == []


def test_after_a_switch_the_working_model_is_kept_for_a_while_then_the_main_one_is_tried_again(world):
    world.script(main=[BUSY, BUSY, ok()], backup=[ok(), ok()])
    call()                                                              # switches to backup
    call()                                                              # still backup: the main model is not asked again yet
    assert len(world.chats["main"].calls) == 2 and len(world.chats["backup"].calls) == 2
    llm._sticky["until"] = 0.0                                          # ten minutes later
    resp, failure = call()
    assert failure is None and len(world.chats["main"].calls) == 3      # the main model is tried first again and answers
    assert llm._sticky["model"] == ""


def test_when_every_model_refuses_it_gives_up_after_trying_each_of_them(world):
    world.script(main=[BUSY] * 10, backup=[BUSY] * 10, third=[BUSY] * 10)
    resp, failure = call()
    assert resp is None and failure.startswith("LLM_UNAVAILABLE")
    total = sum(len(c.calls) for c in world.chats.values())
    assert total == config.LLM_MAX_ATTEMPTS + 2 * 2                     # 10 attempts for three models
    assert all(len(c.calls) >= 2 for c in world.chats.values())


def test_a_used_up_daily_quota_moves_to_the_next_model_at_once(world):
    world.script(main=[QUOTA], backup=[ok()])
    resp, failure = call()
    assert failure is None and world.sleeps == [] and len(world.chats["backup"].calls) == 1
    world.script(main=[QUOTA], backup=[QUOTA], third=[QUOTA])
    llm._sticky.update(model="", until=0.0)
    resp, failure = call()
    assert resp is None and failure.startswith("LLM_QUOTA")             # every model is out of quota: say so


def test_a_real_error_is_never_a_reason_to_switch(world):
    world.script(main=[Exception("401 API key not valid")], backup=[ok()])
    resp, failure = call()
    assert resp is None and failure.startswith("LLM_ERROR") and world.chats["backup"].calls == []


def test_without_fallback_models_nothing_changes(world, monkeypatch):
    monkeypatch.setattr(config, "FALLBACK_MODELS", ())
    world.script(main=[BUSY] * 10)
    resp, failure = call()
    assert failure.startswith("LLM_UNAVAILABLE") and len(world.chats["main"].calls) == config.LLM_MAX_ATTEMPTS
    assert llm.models_for("coder") == ["main"]


def test_the_model_list_and_the_cache_are_per_model(world, monkeypatch):
    monkeypatch.setattr(config, "FALLBACK_MODELS", ("main", "backup", "backup"))
    world.script(main=[], backup=[])
    assert llm.models_for("coder")[0] == "main" and llm.models_for("coder").count("main") == 1
    assert llm.llm_with_tools("coder", "backup") is world.chats["backup"] and llm.llm_with_tools("coder") is world.chats["main"]


def test_the_environment_setting_is_read_as_a_list():
    import os
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", "import config; print(config.FALLBACK_MODELS)"], capture_output=True, text=True,
                         env={**os.environ, "NEXURA_FALLBACK_MODELS": " modelA , modelB,, "}).stdout.strip()
    assert out == "('modelA', 'modelB')"


# ─── an outage of the model service ends the run honestly ───

def test_an_outage_stops_the_run_with_a_resume_hint_and_blames_nobody(env, monkeypatch):    # noqa: F811
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "FAILURE_POLICY", "continue")        # an outage must stop the run even under "continue"
    ctx, graph, cfg = run_new(env, [BUSY] * config.LLM_MAX_ATTEMPTS)
    graph.invoke(Command(resume="yes"), cfg)
    run = env.sm.get_run(ctx.run_id)
    rows = {r["task_id"]: r["status"] for r in env.sm.list_tasks(ctx.run_id)}
    assert rows == {1: "failed", 2: "skipped"} and run.status != "completed"      # the second step was not even started
    assert "--resume" in run.error and "temporary outage" in run.error and "not a problem with your request" in run.error
    assert ctx.run_id in run.error
