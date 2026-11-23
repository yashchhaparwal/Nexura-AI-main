import json

import pytest

from core.schemas import ExecutionPlan
from core.task_manager import final_run_status, pick_next, run_summary, validate_plan


def step(i, deps=(), acc=True, doc=False):
    s = {"step_id": i, "title": f"t{i}", "description": "d", "dependencies": list(deps), "documentation_only": doc,
         "acceptance": [{"type": "file_exists", "path": "a.py"}] if acc else []}
    return s


def plan(*steps):
    return ExecutionPlan.model_validate({"project_id": "p", "goal": "g", "steps": list(steps)})


# ─── validation ─────────────────────────────────────────────

def test_valid_plan():
    assert validate_plan(plan(step(1), step(2, [1]))) == []


@pytest.mark.parametrize("steps,fragment", [
    ([], "no steps"),
    ([step(2)], "1.."),
    ([step(1), step(3)], "1.."),
    ([step(1, [2]), step(2)], "not earlier"),
    ([step(1), step(2, [9])], "unknown step 9"),
    ([step(1), step(2, acc=False)], "no acceptance"),
])
def test_invalid_plans(steps, fragment):
    assert fragment in "; ".join(validate_plan(plan(*steps)))


def test_doc_only_step_may_skip_acceptance():
    assert validate_plan(plan(step(1), step(2, acc=False, doc=True))) == []


def test_acceptance_required_fields():
    with pytest.raises(ValueError):
        ExecutionPlan.model_validate({"project_id": "p", "goal": "g", "steps": [
            {**step(1), "acceptance": [{"type": "http_status"}]}]})
    with pytest.raises(ValueError):
        ExecutionPlan.model_validate({"project_id": "p", "goal": "g", "steps": [
            {**step(1), "acceptance": [{"type": "output_contains", "command": "python x.py"}]}]})


# ─── pick_next / summary ────────────────────────────────────

TASKS = [{"step_id": 1, "dependencies": []}, {"step_id": 2, "dependencies": [1]}, {"step_id": 3, "dependencies": []}]


def test_pick_next_stop_policy_skips_everything_after_failure():
    idx, st = pick_next(TASKS, {"1": "failed", "2": "pending", "3": "pending"}, "stop")
    assert idx is None and st == {"1": "failed", "2": "skipped", "3": "skipped"}


def test_pick_next_continue_skips_only_dependents():
    idx, st = pick_next(TASKS, {"1": "failed", "2": "pending", "3": "pending"}, "continue")
    assert idx == 2 and st["2"] == "skipped"


def test_pick_next_in_order():
    assert pick_next(TASKS, {"1": "done", "2": "pending", "3": "pending"}, "stop")[0] == 1


def test_final_status_and_summary():
    assert final_run_status({"1": "done", "2": "done"}) == "completed"
    assert final_run_status({"1": "done", "2": "failed"}) == "completed_with_failures"
    assert final_run_status({"1": "failed", "2": "skipped"}) == "failed"
    s = run_summary({"tasks": [{"step_id": 1, "title": "a"}], "task_status": {"1": "failed"},
                     "task_errors": {"1": "TOOL_CAP"}, "run_status": "failed"})
    assert s["failed"] == [{"step_id": 1, "title": "a", "reason": "TOOL_CAP"}] and s["done"] == []


# ─── planner agent ──────────────────────────────────────────

@pytest.fixture
def planner(monkeypatch, tmp_path):
    monkeypatch.setenv("PLANNER_MODEL_NAME", "gemini-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    from agents.planner import PlannerAgent
    p = PlannerAgent()
    p.replies, p.prompts = [], []

    def fake(prompt):
        p.prompts.append(prompt)
        r = p.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    p.call_llm = fake
    p.call_llm_plain = fake   # the plain-text fallback reads the same scripted replies (no test ever reaches the network)
    p.path = str(tmp_path / "ws")
    return p


GOOD = json.dumps({"project_id": "p", "goal": "g", "steps": [step(1), step(2, [1])]})


def test_planner_valid_first_try_and_saves_files(planner):
    planner.replies = [GOOD]
    out = planner.plan("make x", planner.path)
    assert len(out.steps) == 2
    import os
    assert os.path.exists(os.path.join(planner.path, "plan.json")) and os.path.exists(os.path.join(planner.path, "plan.md"))


def test_planner_strips_code_fences(planner):
    planner.replies = ["```json\n" + GOOD + "\n```"]
    assert len(planner.plan("x", planner.path).steps) == 2


def test_planner_retries_with_error_feedback_then_succeeds(planner):
    planner.replies = ["not json at all", json.dumps({"project_id": "p", "goal": "g", "steps": [step(1, acc=False)]}), GOOD]
    assert len(planner.plan("x", planner.path).steps) == 2
    assert "INVALID" in planner.prompts[1] and "no acceptance" in planner.prompts[2]


def test_planner_gives_up_after_three_invalid_answers(planner):
    from agents.planner import PlanError
    planner.replies = ["bad", "bad", "bad"]
    with pytest.raises(PlanError, match="3 attempts"):
        planner.plan("x", planner.path)


def test_planner_permanent_api_failure_is_not_retried(planner):
    from agents.planner import PlanError
    planner.replies = [RuntimeError("401 API key not valid")]
    with pytest.raises(PlanError, match="LLM call failed"):
        planner.plan("x", planner.path)
    assert len(planner.prompts) == 1


def test_planner_retries_transient_api_errors_with_backoff(planner, monkeypatch):
    """A Gemini '503 high demand' used to kill the run instantly. (Checked with the plain-text fallback off: it has its own tests.)"""
    import time
    import config
    sleeps = []
    monkeypatch.setattr(config, "PLANNER_FALLBACK", False)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    planner.replies = [RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand."), RuntimeError("429 RESOURCE_EXHAUSTED"), GOOD]
    assert len(planner.plan("x", planner.path).steps) == 2
    assert len(sleeps) == 2 and sleeps[1] > 0


def test_planner_gives_up_on_persistent_transient_errors(planner, monkeypatch):
    import time
    import config
    from agents.planner import PlanError
    monkeypatch.setattr(time, "sleep", lambda s: None)
    planner.replies = [RuntimeError("503 UNAVAILABLE")] * config.LLM_MAX_ATTEMPTS
    with pytest.raises(PlanError, match="LLM call failed"):
        planner.plan("x", planner.path)


def test_acceptance_commands_that_can_never_run_are_sent_back_to_the_planner(planner):
    """`pip install` as an acceptance check can never pass (installs are a separate, approved tool): the planner must fix it."""
    def with_check(cmd):
        st = step(1)
        st["acceptance"] = [{"type": "command_ok", "command": cmd}]
        return json.dumps({"project_id": "p", "goal": "g", "steps": [st]})
    planner.replies = [with_check("pip install -r requirements.txt"), with_check("python a.py | more"), with_check("python -m pytest -q")]
    out = planner.plan("x", planner.path)
    assert out.steps[0].acceptance[0].command == "python -m pytest -q"
    assert "USE_INSTALL_DEPENDENCY" in planner.prompts[1] and "SHELL_OPERATOR" in planner.prompts[2]


def test_validate_plan_flags_unrunnable_acceptance_commands():
    p = plan({**step(1), "acceptance": [{"type": "http_status", "start_command": "curl http://x", "url": "http://127.0.0.1:1/"}]})
    assert "NOT_ALLOWED" in "; ".join(validate_plan(p))
    ok = plan({**step(1), "acceptance": [{"type": "output_contains", "command": "python -c \"print(1); print(2)\"", "text": "2"}]})
    assert validate_plan(ok) == []


def test_planner_prompt_version_and_granularity():
    import config
    from agents.planner import PROMPT_TEMPLATE
    assert config.PROMPT_VERSIONS["planner"] == "6" and "3-8 STEPS" in PROMPT_TEMPLATE and "acceptance" in PROMPT_TEMPLATE


def test_planner_node_failure_ends_run_not_crash(monkeypatch, planner):
    import agents.planner as ap
    monkeypatch.setattr(ap.PlannerAgent, "plan", lambda *a, **k: (_ for _ in ()).throw(ap.PlanError("boom")))
    from core.orchestrator import planner_node, route_after_planner
    from langgraph.graph import END
    from core.schemas import ProjectContext
    cfg = {"configurable": {"project": ProjectContext(run_id="r", root=planner.path)}}
    out = planner_node({"user_prompt": "x", "workspace_path": planner.path}, cfg)
    assert out["run_status"] == "failed" and "boom" in out["run_error"]
    assert route_after_planner(out) == END


def test_load_tasks_from_plan_dict():
    from core.task_manager import load_tasks
    tasks = load_tasks(plan(step(1), step(2, [1])).model_dump())
    assert [t.task_id for t in tasks] == [1, 2] and tasks[1].dependencies == [1] and tasks[0].status == "pending"
    assert tasks[0].acceptance[0].type == "file_exists"


@pytest.mark.parametrize("start,url", [
    ("python -m uvicorn app:app --port 8765", "http://127.0.0.1:8765/"), ("streamlit run app.py", "http://127.0.0.1:8501/"),
    ("python -m flask --app app run --port 8767", "http://127.0.0.1:8767/"), ("python manage.py runserver 0.0.0.0:9000", "http://127.0.0.1:9000/"),
    ("python -m http.server", "http://127.0.0.1:8000/")])
def test_http_status_url_is_inferred_from_the_start_command(start, url):
    """The live model often omits `url` and only gives the start command; that must not sink the whole plan."""
    from core.schemas import Acceptance
    assert Acceptance(type="http_status", start_command=start).url == url
    assert Acceptance(type="http_status", start_command=start, url="http://localhost:1/x").url == "http://localhost:1/x"


def test_http_status_without_any_way_to_find_a_url_is_still_invalid():
    from core.schemas import Acceptance
    with pytest.raises(ValueError, match="requires url"):
        Acceptance(type="http_status", start_command="python app.py")


def _plan_json(steps):
    return json.dumps({"project_id": "p", "goal": "g", "steps": steps})


def test_repair_drops_only_the_unrunnable_checks_and_warns(planner):
    """After 3 bad answers the plan is salvaged: unusable checks are dropped (visibly), good ones stay."""
    s1 = {**step(1), "acceptance": [{"type": "file_exists", "path": "app.py"},
                                    {"type": "command_ok", "command": "curl -s http://127.0.0.1:8000/health"}]}
    s2 = {**step(2, [1]), "acceptance": [{"type": "command_ok", "command": "pip install -r requirements.txt"},
                                         {"type": "http_status", "start_command": "python app.py"}]}   # no url, no port: invalid
    bad = _plan_json([s1, s2])
    planner.replies = [bad, bad, bad]
    out = planner.plan("x", planner.path)
    assert [a.type for a in out.steps[0].acceptance] == ["file_exists"]
    assert out.steps[1].acceptance == [] and out.steps[1].documentation_only        # default verification only
    joined = "\n".join(planner.last_warnings)
    assert "curl" in joined and "pip install" in joined and "default verification only" in joined
    assert len(planner.prompts) == 3


def test_repair_does_not_rescue_structurally_broken_plans(planner):
    from agents.planner import PlanError
    broken = _plan_json([step(1), step(3)])   # ids not sequential
    planner.replies = [broken, broken, broken]
    with pytest.raises(PlanError):
        planner.plan("x", planner.path)


def test_planner_node_records_warnings_in_the_plan(monkeypatch, planner):
    import agents.planner as ap
    from core.orchestrator import planner_node
    from core.schemas import ProjectContext
    bad = _plan_json([{**step(1), "acceptance": [{"type": "command_ok", "command": "cat a.txt"}, {"type": "file_exists", "path": "a.txt"}]}])
    planner.replies = [bad, bad, bad]
    monkeypatch.setattr(ap.PlannerAgent, "__new__", lambda cls: planner)
    monkeypatch.setattr(ap.PlannerAgent, "__init__", lambda self: None)
    cfg = {"configurable": {"project": ProjectContext(run_id="r", root=planner.path)}}
    out = planner_node({"user_prompt": "x", "workspace_path": planner.path}, cfg)
    assert any("cat a.txt" in w for w in out["plan"]["warnings"]) and out["tasks"][0]["acceptance"][0]["type"] == "file_exists"


# ─── plain-text fallback when structured plan requests keep failing ───

def _fallback_setup(planner, monkeypatch):
    import time
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    calls = {"structured": 0, "plain": 0}
    return sleeps, calls


def test_after_two_structured_failures_the_plan_is_requested_as_plain_text(planner, monkeypatch):
    from agents.planner import FALLBACK_WARNING
    sleeps, calls = _fallback_setup(planner, monkeypatch)

    def structured(prompt):
        calls["structured"] += 1
        raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")

    def plain(prompt):
        calls["plain"] += 1
        return GOOD
    planner.call_llm, planner.call_llm_plain = structured, plain
    out = planner.plan("x", planner.path)
    assert len(out.steps) == 2
    assert calls == {"structured": 2, "plain": 1}                  # two tries, then the other kind of request
    assert len(sleeps) == 1                                         # one wait after the first failure, none before switching
    assert planner.requests == 3 and planner.fallback_used and FALLBACK_WARNING in planner.last_warnings


def test_a_single_transient_failure_does_not_trigger_the_fallback(planner, monkeypatch):
    sleeps, calls = _fallback_setup(planner, monkeypatch)
    replies = [RuntimeError("503 UNAVAILABLE"), GOOD]

    def structured(prompt):
        calls["structured"] += 1
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    planner.call_llm = structured
    planner.call_llm_plain = lambda prompt: pytest.fail("the plain request must not be used")
    assert len(planner.plan("x", planner.path).steps) == 2
    assert calls["structured"] == 2 and not planner.fallback_used and planner.last_warnings == []


def test_the_fallback_can_be_switched_off(planner, monkeypatch):
    import config
    from agents.planner import PlanError
    sleeps, calls = _fallback_setup(planner, monkeypatch)
    monkeypatch.setattr(config, "PLANNER_FALLBACK", False)

    def structured(prompt):
        calls["structured"] += 1
        raise RuntimeError("503 UNAVAILABLE")
    planner.call_llm = structured
    planner.call_llm_plain = lambda prompt: pytest.fail("the plain request must not be used")
    with pytest.raises(PlanError, match="LLM call failed"):
        planner.plan("x", planner.path)
    assert calls["structured"] == config.LLM_MAX_ATTEMPTS and not planner.fallback_used


def test_real_errors_are_not_retried_in_plain_mode_either(planner, monkeypatch):
    from agents.planner import PlanError
    sleeps, calls = _fallback_setup(planner, monkeypatch)

    def structured(prompt):
        calls["structured"] += 1
        raise RuntimeError("401 API key not valid")
    planner.call_llm = structured
    planner.call_llm_plain = lambda prompt: pytest.fail("a bad key is not a reason to try the plain request")
    with pytest.raises(PlanError):
        planner.plan("x", planner.path)
    assert calls["structured"] == 1 and sleeps == []


def test_a_plain_answer_goes_through_the_same_validation_and_repair(planner, monkeypatch):
    """Fenced JSON is accepted; an invalid plan is sent back with the error, exactly like a structured one."""
    sleeps, calls = _fallback_setup(planner, monkeypatch)
    nl = chr(10)
    planner.call_llm = lambda prompt: (_ for _ in ()).throw(RuntimeError("503 UNAVAILABLE"))
    bad = json.dumps({"project_id": "p", "goal": "g", "steps": []})                      # no steps: invalid
    replies = [bad, "```json" + nl + GOOD + nl + "```"]

    def plain(prompt):
        calls["plain"] += 1
        return replies.pop(0)
    planner.call_llm_plain = plain
    out = planner.plan("x", planner.path)
    assert len(out.steps) == 2 and calls["plain"] == 2 and planner.fallback_used


def test_the_plain_request_really_has_no_structured_mode_and_counts_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("PLANNER_MODEL_NAME", "m")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    from agents.planner import PlannerAgent
    seen = []

    class Resp:
        text = " {} "

        class usage_metadata:
            total_token_count = 123

    class Models:
        def generate_content(self, **kw):
            seen.append(kw["config"])
            return Resp()

    class Client:
        models = Models()
    p = PlannerAgent()
    p.client = Client()
    assert p.call_llm("x") == "{}" and p.call_llm_plain("x") == "{}"
    structured, plain = seen
    assert structured.response_schema is not None and structured.response_mime_type == "application/json"
    assert plain.response_schema is None and plain.response_mime_type is None             # an ordinary request
    assert p.tokens == 246
