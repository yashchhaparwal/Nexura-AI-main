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
from tests.test_persistence import env  # noqa: F401  (a fixture)


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
    assert config.PROMPT_VERSIONS["debugger"] in ("5", "6")


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


# ─── a step that changed no files still lets the debugger see and edit the code ───

def _debug_state(file_changes, changed_files=()):
    from langchain_core.messages import HumanMessage
    return {"tasks": [{"step_id": 4, "title": "Verify the server", "description": "start it", "acceptance": [
                {"type": "http_status", "url": "http://127.0.0.1:8765/", "start_command": "python -m uvicorn app:app --port 8765"}]}],
            "current_task_index": 0, "messages": [HumanMessage(content="x")], "changed_files": list(changed_files),
            "file_changes": file_changes, "verify_failure": "- http_status: http://127.0.0.1:8765/ expected 200, got status 404",
            "error_type": "http_status_mismatch", "retry_hint": "hint", "attempts": 1, "tool_calls_count": 0}


def _run_debugger_node(monkeypatch, tmp_path, state):
    from langchain_core.messages import AIMessage
    from core import orchestrator
    seen = {}

    def propose(messages, on_log=None):
        seen["prompt"] = str(messages[-1].content)
        return AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"target_file": "app.py"}, "id": "c"}]), None
    monkeypatch.setattr(debugger, "propose", propose)
    ctx = ProjectContext(run_id="r", root=tmp_path)
    orchestrator.debugger_node(state, {"configurable": {"project": ctx}})
    return seen["prompt"], ctx


def test_debugger_sees_files_from_earlier_steps_when_its_step_changed_none(monkeypatch, tmp_path):
    (tmp_path / "app.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
    (tmp_path / "requirements.txt").write_text("fastapi\n")
    prompt, ctx = _run_debugger_node(monkeypatch, tmp_path, _debug_state([{"file": "app.py"}, {"file": "requirements.txt"}, {"file": "app.py"}]))
    assert "--- app.py ---" in prompt and "app = FastAPI()" in prompt and "changed earlier in this run" in prompt
    assert ctx.key(tmp_path / "app.py") in ctx.files_read            # shown = read, so an edit in this round is allowed


def test_the_earlier_files_note_is_not_shown_when_the_step_changed_files_itself(monkeypatch, tmp_path):
    (tmp_path / "app.py").write_text("OLD = 1\n")
    (tmp_path / "other.py").write_text("OTHER = 1\n")
    prompt, _ = _run_debugger_node(monkeypatch, tmp_path, _debug_state([{"file": "other.py"}], changed_files=["app.py"]))
    assert "OLD = 1" in prompt and "OTHER = 1" not in prompt and "changed earlier in this run" not in prompt


def test_no_files_anywhere_still_says_so(monkeypatch, tmp_path):
    prompt, _ = _run_debugger_node(monkeypatch, tmp_path, _debug_state([]))
    assert "(no files)" in prompt and "changed earlier in this run" not in prompt


def test_the_debugger_prompt_no_longer_forbids_what_the_check_needs():
    text = debugger.build_prompt({"step_id": 4, "title": "t", "description": "d"}, "f", "http_status_mismatch", "h", {"app.py": "x"})
    assert "Add NO features." not in text and "IS the fix" in text and "ONE round" in text


def test_pipeline_a_step_that_changes_nothing_is_fixed_by_the_debugger_in_one_round(env, monkeypatch):    # noqa: F811
    """The 2026-11-23 trace: step 2 only checks the result of step 1. The debugger now sees app.py and patches it in its single round."""
    from langchain_core.messages import AIMessage
    from langgraph.types import Command
    from core.runner import start_run
    from core.state_manager import make_checkpointer
    import tests.test_persistence as tp
    from tests.conftest import tool_call_msg

    plan = {"project_id": "p", "goal": "g", "steps": [
        {"step_id": 1, "title": "write app", "description": "d", "dependencies": [], "acceptance": [{"type": "file_exists", "path": "app.py"}]},
        {"step_id": 2, "title": "check app", "description": "d", "dependencies": [1],
         "acceptance": [{"type": "command_ok", "command": 'python -c "import app; assert app.ROUTE"'}]}]}
    monkeypatch.setattr(tp, "PLAN", plan)
    env.llm.script[:] = [
        AIMessage(content="", tool_calls=[{"name": "create_file", "args": {"target_file": "app.py", "content": "ROUTE = 0\n"}, "id": "a"},
                                          {"name": "task_complete", "args": {"summary": "ok"}, "id": "b"}]),
        tool_call_msg("task_complete", {"summary": "nothing to change"}, "c"),                       # step 2's coder changes no file
        tool_call_msg("patch_file", {"target_file": "app.py", "search_text": "ROUTE = 0", "replace_text": "ROUTE = 100"}, "d"),   # the debugger
    ]
    ctx, state = start_run(env.sm, "proj", env.ws, "make app", require_git=False)
    graph = env.build(checkpointer=make_checkpointer(env.ckpt_path))
    cfg = {"configurable": {"thread_id": ctx.run_id, "project": ctx}, "recursion_limit": 200}
    graph.invoke(state, cfg)
    graph.invoke(Command(resume="yes"), cfg)
    rows = {r["task_id"]: r["status"] for r in env.sm.list_tasks(ctx.run_id)}
    assert rows == {1: "done", 2: "done"} and env.sm.get_run(ctx.run_id).status == "completed"
    assert (env.ws / "app.py").read_text() == "ROUTE = 100\n"
    assert "ROUTE = 0" in str(env.llm.calls[2][-1].content)                                         # the debugger was shown the file


# ─── the check must not talk to an older copy of the app ───

import sys  # noqa: E402
import time  # noqa: E402

from core import verification  # noqa: E402
from core.processes import ProcessManager  # noqa: E402
from tests.test_control_plane import ctx as run_ctx, free_port, task as make_task  # noqa: E402,F401  (run_ctx is a fixture)

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_stop_same_server_matches_the_command_or_the_port_and_nothing_else(tmp_path):
    pm = ProcessManager()
    same = pm.start(SLEEP, str(tmp_path), "python -m uvicorn app:app --port 9100")
    same_port = pm.start(SLEEP, str(tmp_path), "uvicorn other:app --port 9100")
    other_port = pm.start(SLEEP, str(tmp_path), "python -m uvicorn app:app --port 9101")
    unrelated = pm.start(SLEEP, str(tmp_path), "python worker.py")
    try:
        assert pm.stop_same_server("python  -m uvicorn app:app --port  9100") == 2      # spacing is ignored; same port counts
        assert pm.pid(same) is None and pm.pid(same_port) is None
        assert pm.pid(other_port) is not None and pm.pid(unrelated) is not None        # untouched
        assert pm.stop_same_server("python -m uvicorn app:app --port 9100") == 0       # nothing left to stop
    finally:
        pm.stop_all()


def test_the_http_check_gets_a_fresh_server_not_the_agents_stale_one(run_ctx, workspace):    # noqa: F811
    """The 2026-11-23 live run: the route was added to app.py, but the server started earlier still ran the old code and held the
    port, so the check's own server could not bind and the check read the old answer (404) four times."""
    port = free_port()
    script = (
        "import http.server, sys\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(%d); self.end_headers(); self.wfile.write(b'x')\n"
        "    def log_message(self, *a): pass\n"
        "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n")
    (workspace / "server.py").write_text(script % 404)                       # the OLD code: no route, answers 404
    command = f"python server.py {port}"
    stale = run_ctx.backend.start_process([sys.executable, "server.py", str(port)], str(workspace), command)   # what the coder's start_process did
    try:
        for _ in range(100):                                              # wait until the old server answers
            try:
                import urllib.error
                import urllib.request
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
            except urllib.error.HTTPError:
                break
            except OSError:
                time.sleep(0.1)
        (workspace / "server.py").write_text(script % 200)                  # the debugger fixed the code; the old process is still running
        r = verification.verify(make_task({"type": "http_status", "start_command": command, "url": f"http://127.0.0.1:{port}/",
                                           "expect": 200, "wait_s": 15}), run_ctx, [], [])
        assert r.ok, r.failure_text()
        assert run_ctx.backend.pid(stale) is None                          # the stale server was stopped, not left running
    finally:
        run_ctx.backend.stop_all()


def test_a_check_without_a_start_command_still_uses_the_agents_running_server(run_ctx, workspace):    # noqa: F811
    port = free_port()
    (workspace / "server.py").write_text(
        "import http.server, sys\nhttp.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), http.server.SimpleHTTPRequestHandler).serve_forever()\n")
    handle = run_ctx.backend.start_process([sys.executable, "server.py", str(port)], str(workspace), f"python server.py {port}")
    try:
        r = verification.verify(make_task({"type": "http_status", "url": f"http://127.0.0.1:{port}/", "expect": 200, "wait_s": 15}), run_ctx, [], [])
        assert r.ok, r.failure_text()
        assert run_ctx.backend.pid(handle) is not None                      # nothing was stopped
    finally:
        run_ctx.backend.stop_all()
