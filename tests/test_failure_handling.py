"""Failure handling found by the 2026-11-23 eval: a debugger that could not install a missing package, a 404 reported as
'port in use', and a check on '/' for an app described by other routes."""
import pytest

import config
from agents import debugger
from core import approvals, validator
from core.error_classifier import HINTS, classify
from core.retry_manager import decide
from core.schemas import Action, ProjectContext
from core.task_manager import root_url_warnings


@pytest.fixture
def ctx(tmp_path):
    return ProjectContext(run_id="r", root=tmp_path)


# ─── the debugger can install the missing package (and nothing else) ───

def test_debugger_can_install_but_every_install_is_still_validated(ctx):
    ok = validator.validate(Action(action="install_dependency", command="pip install streamlit"), ctx, "debug")
    assert ok.ok
    for cmd, code in (("pip install git+https://example.com/x.git", "BAD_INSTALL"), ("pip install -r ../outside.txt", None),
                      ("python -c \"print(1)\"", "NOT_AN_INSTALL")):
        r = validator.validate(Action(action="install_dependency", command=cmd), ctx, "debug")
        assert not r.ok and (code is None or r.code == code), cmd


def test_run_command_still_cannot_install_in_debug_mode(ctx):
    for cmd in ("pip install streamlit", "python -m pip install streamlit"):
        assert validator.validate(Action(action="run_command", command=cmd), ctx, "debug").code == "USE_INSTALL_DEPENDENCY"


def test_a_debugger_install_always_needs_the_users_approval(ctx):
    reasons = approvals.assess(Action(action="install_dependency", command="pip install streamlit"), [], ctx, set(), 0)
    assert any(kind == "hard" and "install" in text for kind, text in reasons)
    for level in ("ask_every_edit", "ask_risky", "auto_within_limits"):
        assert approvals.requires_approval(reasons, level)


def test_debugger_prompt_agrees_with_what_it_may_do(monkeypatch):
    text = debugger.build_prompt({"step_id": 3, "title": "t", "description": "d"}, "ModuleNotFoundError: No module named 'streamlit'",
                                 "missing_dependency", "hint", {})
    assert "install_dependency" in text and "do not install packages" not in text.lower() and "cannot change a check" in text
    assert "install_dependency" in HINTS["missing_dependency"]            # the hint and the prompt no longer contradict each other
    assert config.PROMPT_VERSIONS["debugger"] in ("3", "4")


# ─── a wrong HTTP status is not a busy port ───

BIND = ("ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8765): "
        "[winerror 10048] only one usage of each socket address is normally permitted")
MISMATCH = f"- http_status: http://127.0.0.1:8765/ expected 200, got status 404. Server output: INFO: Started server process [24124]\n{BIND}"


def test_a_404_with_a_bind_error_in_the_server_log_is_a_status_mismatch():
    assert classify(MISMATCH) == "http_status_mismatch"
    hint = HINTS["http_status_mismatch"]
    assert "route" in hint and "Do not change the port" in hint


def test_real_port_conflicts_and_other_categories_are_unchanged():
    assert classify("OSError: [Errno 98] Address already in use") == "port_in_use"
    assert classify(BIND) == "port_in_use"                                  # no status mismatch: still a busy port
    assert classify("ModuleNotFoundError: No module named 'x'\nexpected 200, got status 500") == "missing_dependency"
    assert classify("SyntaxError: invalid syntax") == "syntax_error"


def test_the_debugger_gets_the_mismatch_hint_on_the_second_attempt():
    d = decide({}, 2, MISMATCH)
    assert d.error_type == "http_status_mismatch" and "route" in d.hint and "port is busy" not in d.hint


# ─── a check on '/' for an app described by other routes ───

def plan_with(url):
    return {"steps": [{"step_id": 3, "acceptance": [{"type": "http_status", "url": url}]}]}


def test_root_check_for_an_app_described_by_other_routes_gets_a_warning():
    w = root_url_warnings(plan_with("http://127.0.0.1:8765/"),
                          'Create a FastAPI app with GET /health returning {"status": "ok"} and GET /users returning a list')
    assert len(w) == 1 and "/health" in w[0] and "/users" in w[0] and "step 3" in w[0] and "404" in w[0]


@pytest.mark.parametrize("prompt", [
    "App with GET / returning hello and GET /users", "A homepage and a /users page", "A flask hello world app",
    "uproot-free tool using and/or HTML/CSS in src/app.py"])
def test_no_warning_when_the_root_is_asked_for_or_no_routes_are_named(prompt):
    assert root_url_warnings(plan_with("http://127.0.0.1:5000/"), prompt) == []


def test_no_warning_when_the_check_already_targets_a_named_route():
    assert root_url_warnings(plan_with("http://127.0.0.1:8765/health"), "GET /health and GET /users") == []


def test_the_warning_reaches_the_plan_the_user_approves(monkeypatch, tmp_path):
    import agents.planner as ap
    from core.orchestrator import planner_node
    from core.schemas import ExecutionPlan

    plan = ExecutionPlan.model_validate({"project_id": "p", "goal": "g", "steps": [{
        "step_id": 1, "title": "api", "description": "d", "dependencies": [],
        "acceptance": [{"type": "http_status", "start_command": "python -m uvicorn app:app --port 8765"}]}]})   # no url: inferred as the root
    assert plan.steps[0].acceptance[0].url == "http://127.0.0.1:8765/"

    class FakePlanner:
        last_warnings: list = []
        requests = tokens = 0
        call_text = None

        def plan(self, **kw):
            return plan
    monkeypatch.setattr(ap, "PlannerAgent", FakePlanner)
    cfg = {"configurable": {"project": ProjectContext(run_id="r", root=tmp_path)}}
    out = planner_node({"user_prompt": "FastAPI app with GET /health and GET /users on port 8765", "workspace_path": str(tmp_path)}, cfg)
    assert any("root URL" in w for w in out["plan"]["warnings"])
