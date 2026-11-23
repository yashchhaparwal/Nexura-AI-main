"""Failure handling found by the 2026-11-23 eval: a debugger that could not install a missing package, a 404 reported as
'port in use', and a check on '/' for an app described by other routes."""
import json

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


# ─── a check must see the code as it is now, and later checks must still find the server ───

import sys  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from core import verification  # noqa: E402
from core.processes import ProcessManager  # noqa: E402
from tests.test_control_plane import ctx as run_ctx, free_port, task as make_task  # noqa: E402,F401  (run_ctx is a fixture)

SERVER = (
    "import http.server, sys\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        self.send_response(%d); self.end_headers(); self.wfile.write(b'x')\n"
    "    def log_message(self, *a): pass\n"
    "http.server.HTTPServer(('127.0.0.1', int(sys.argv[-1])), H).serve_forever()\n")


def _status(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).status
    except urllib.error.HTTPError as e:
        return e.code
    except OSError:
        return None


def _wait_up(port):
    for _ in range(100):
        if _status(port) is not None:
            return
        time.sleep(0.1)


def test_restart_servers_restarts_web_servers_keeps_handles_and_leaves_other_processes_alone(tmp_path):
    port = free_port()
    (tmp_path / "server.py").write_text(SERVER % 200)
    pm = ProcessManager()
    server = pm.start([sys.executable, "server.py", "--port", str(port)], str(tmp_path), f"python server.py --port {port}")
    worker = pm.start([sys.executable, "-c", "import time; time.sleep(60)"], str(tmp_path), "python worker.py")
    try:
        _wait_up(port)
        old_server_pid, old_worker_pid = pm.pid(server), pm.pid(worker)
        assert pm.restart_servers() == 1
        assert pm.pid(server) not in (None, old_server_pid)                  # same handle, new process
        assert pm.pid(worker) == old_worker_pid                              # no recognisable port: untouched
        assert _status(port) == 200                                         # and it is already accepting connections when this returns
    finally:
        pm.stop_all()


def test_a_check_sees_the_new_code_not_the_agents_stale_server_and_later_checks_still_find_it(run_ctx, workspace):    # noqa: F811
    """The 2026-11-23 live runs: the route was added but the server started earlier ran the old code (404 four times). Stopping that
    server fixed this check and broke the checks after it, which call the same server; so it is restarted."""
    port = free_port()
    command = f"python server.py --port {port}"
    (workspace / "server.py").write_text(SERVER % 404)                       # the OLD code
    stale = run_ctx.backend.start_process([sys.executable, "server.py", "--port", str(port)], str(workspace), command)   # the coder's start_process
    try:
        _wait_up(port)
        assert _status(port) == 404
        (workspace / "server.py").write_text(SERVER % 200)                  # the debugger fixed the code; the old process keeps running
        probe = f'python -c "import urllib.request as u; print(u.urlopen(\'http://127.0.0.1:{port}/\').status)"'
        r = verification.verify(make_task(
            {"type": "http_status", "start_command": command, "url": f"http://127.0.0.1:{port}/", "expect": 200, "wait_s": 15},
            {"type": "output_contains", "command": probe, "text": "200"}), run_ctx, [], [])
        assert r.ok, r.failure_text()                                        # the second check needs the server the first one used
        assert run_ctx.backend.pid(stale) is not None                        # the agent's handle is still valid
    finally:
        run_ctx.backend.stop_all()


def test_a_check_without_a_start_command_also_gets_the_current_code(run_ctx, workspace):    # noqa: F811
    port = free_port()
    (workspace / "server.py").write_text(SERVER % 404)
    run_ctx.backend.start_process([sys.executable, "server.py", "--port", str(port)], str(workspace), f"python server.py --port {port}")
    try:
        _wait_up(port)
        (workspace / "server.py").write_text(SERVER % 200)
        r = verification.verify(make_task({"type": "http_status", "url": f"http://127.0.0.1:{port}/", "expect": 200, "wait_s": 15}), run_ctx, [], [])
        assert r.ok, r.failure_text()
    finally:
        run_ctx.backend.stop_all()


def test_nothing_is_restarted_when_the_agent_left_no_server_running(run_ctx, workspace):    # noqa: F811
    assert len(run_ctx.backend) == 0 and run_ctx.backend.restart_servers() == 0
    (workspace / "a.py").write_text("x = 1\n")
    assert verification.verify(make_task({"type": "file_exists", "path": "a.py"}), run_ctx, [], []).ok


# ─── a new project's check on "/" is rewritten to a route the request names ───

from core.task_manager import fix_root_url_checks, named_routes  # noqa: E402
from tests.test_planner_and_tasks import planner  # noqa: E402,F401  (a fixture)
from core.schemas import ExecutionPlan  # noqa: E402

API_PROMPT = 'Create a FastAPI app with GET /health returning ok and GET /users returning a list. Start it with uvicorn on --port 8765.'


def _api_plan(url="http://127.0.0.1:8765/", extra=None):
    acc = {"type": "http_status", "start_command": "python -m uvicorn app:app --port 8765", "url": url, **(extra or {})}
    return ExecutionPlan.model_validate({"project_id": "p", "goal": "g", "steps": [
        {"step_id": 1, "title": "write", "description": "d", "dependencies": [], "acceptance": [{"type": "file_exists", "path": "app.py"}]},
        {"step_id": 2, "title": "serve", "description": "d", "dependencies": [1], "acceptance": [acc]}]})


def test_named_routes_are_ordered_and_skip_path_parameters_and_prose():
    assert named_routes(API_PROMPT) == ["/health", "/users"]
    assert named_routes("GET /users/{id} and /items/<name> and and/or HTML/CSS in src/app.py") == []
    assert named_routes("GET /b then /a then /b again") == ["/b", "/a"]


def test_the_root_check_becomes_the_health_route_and_the_change_is_reported():
    plan = _api_plan()
    notes = fix_root_url_checks(plan, API_PROMPT)
    assert plan.steps[1].acceptance[0].url == "http://127.0.0.1:8765/health"
    assert len(notes) == 1 and "step 2" in notes[0] and "changed the check URL" in notes[0] and "/health" in notes[0]


def test_without_a_health_route_the_first_named_route_is_used_and_query_is_kept():
    plan = _api_plan(url="http://127.0.0.1:8765/?x=1")
    fix_root_url_checks(plan, "An API with GET /items and GET /users")
    assert plan.steps[1].acceptance[0].url == "http://127.0.0.1:8765/items?x=1"


@pytest.mark.parametrize("prompt", ["App with GET / returning hello and GET /health", "A homepage plus a /health page", "A flask hello world app"])
def test_nothing_is_rewritten_when_the_root_is_wanted_or_no_route_is_named(prompt):
    plan = _api_plan()
    assert fix_root_url_checks(plan, prompt) == [] and plan.steps[1].acceptance[0].url == "http://127.0.0.1:8765/"


def test_a_check_that_already_targets_a_route_is_left_alone():
    plan = _api_plan(url="http://127.0.0.1:8765/users")
    assert fix_root_url_checks(plan, API_PROMPT) == [] and plan.steps[1].acceptance[0].url == "http://127.0.0.1:8765/users"


def _plan_reply(url=None):
    acc = {"type": "http_status", "start_command": "python -m uvicorn app:app --port 8765"}
    if url:
        acc["url"] = url
    return json.dumps({"project_id": "p", "goal": "g", "steps": [
        {"step_id": 1, "title": "serve", "description": "d", "dependencies": [], "acceptance": [acc]}]})


def test_the_planner_rewrites_for_a_new_project_even_when_the_url_was_inferred(planner):    # noqa: F811
    planner.replies = [_plan_reply()]                                      # no url: Nexura infers the server root
    out = planner.plan(API_PROMPT, planner.path)
    assert out.steps[0].acceptance[0].url == "http://127.0.0.1:8765/health"
    assert any("changed the check URL" in w for w in planner.last_warnings)
    assert "/health" in open(planner.path + "/plan.json", encoding="utf-8").read()          # the saved plan agrees with what runs


def test_the_planner_only_warns_for_an_existing_project(planner):    # noqa: F811
    planner.replies = [_plan_reply(url="http://127.0.0.1:8765/")]
    out = planner.plan(API_PROMPT, planner.path, context={"exists": lambda p: True, "files": [], "files_total": 0, "hits": []})
    assert out.steps[0].acceptance[0].url == "http://127.0.0.1:8765/"      # unchanged: "/" may be that project's real home page
    assert not any("changed the check URL" in w for w in planner.last_warnings)


# ─── the debugger is told when its last round only looked ───

from core.orchestrator import _idle_debug_round  # noqa: E402


def _hist(*tools, step=4):
    return [{"step_id": step, "tool": t, "args": "", "result": "", "status": "ok"} for t in tools]


TASK4 = {"step_id": 4}


def test_a_debug_round_that_only_looked_is_reported():
    h = _hist("start_process", "task_complete", "verification", "list_directory", "verification")
    assert _idle_debug_round({"execution_history": h}, TASK4) == ["list_directory"]
    h2 = _hist("verification", "list_directory", "read_file", "verification")
    assert _idle_debug_round({"execution_history": h2}, TASK4) == ["list_directory", "read_file"]


def test_rounds_that_changed_something_or_belong_to_the_coder_or_other_steps_are_not_reported():
    assert _idle_debug_round({"execution_history": _hist("verification", "patch_file", "verification")}, TASK4) is None
    assert _idle_debug_round({"execution_history": _hist("verification", "list_directory", "patch_file", "verification")}, TASK4) is None
    assert _idle_debug_round({"execution_history": _hist("list_directory", "verification")}, TASK4) is None      # first debug round: the coder's work
    assert _idle_debug_round({"execution_history": _hist("verification", "verification")}, TASK4) is None        # nothing proposed
    other = _hist("verification", "list_directory", "verification", step=3)
    assert _idle_debug_round({"execution_history": other}, TASK4) is None


def test_the_note_reaches_the_debugger_prompt(monkeypatch, tmp_path):
    (tmp_path / "app.py").write_text("x = 1")
    state = _debug_state([{"file": "app.py"}])
    state["execution_history"] = _hist("verification", "list_directory", "verification")
    prompt, _ = _run_debugger_node(monkeypatch, tmp_path, state)
    assert "only used list_directory and changed nothing" in prompt and "Change the code now" in prompt
    quiet, _ = _run_debugger_node(monkeypatch, tmp_path, _debug_state([{"file": "app.py"}]))
    assert "changed nothing" not in quiet


# ─── a server a check starts stays up for the step's later checks ───

def _probe(port):
    q = chr(39)
    return 'python -c "import urllib.request as u; print(u.urlopen(' + q + f"http://127.0.0.1:{port}/" + q + ').status)"'


def test_a_server_started_by_a_check_is_still_up_for_the_checks_after_it_and_then_stopped(run_ctx, workspace):    # noqa: F811
    """The 2026-11-23 live run: the coder had stopped its own server, the plan's first check started one and shut it down again, and the
    two checks after it (requests.get on the same port) could never pass; the debugger has no tool to start a server."""
    port = free_port()
    (workspace / "server.py").write_text(SERVER % 200)
    assert len(run_ctx.backend) == 0                                            # the agent left nothing running
    r = verification.verify(make_task(
        {"type": "http_status", "start_command": f"python server.py --port {port}", "url": f"http://127.0.0.1:{port}/", "expect": 200, "wait_s": 15},
        {"type": "output_contains", "command": _probe(port), "text": "200"},
        {"type": "command_ok", "command": _probe(port)}), run_ctx, [], [])
    assert r.ok, r.failure_text()
    assert len(run_ctx.backend) == 0                                            # nothing is left running after the step's checks
    assert _status(port) is None


def test_the_check_server_is_stopped_even_when_a_later_check_fails(run_ctx, workspace):    # noqa: F811
    port = free_port()
    (workspace / "server.py").write_text(SERVER % 200)
    r = verification.verify(make_task(
        {"type": "http_status", "start_command": f"python server.py --port {port}", "url": f"http://127.0.0.1:{port}/", "expect": 200, "wait_s": 15},
        {"type": "output_contains", "command": _probe(port), "text": "999"}), run_ctx, [], [])
    assert not r.ok and "999" in r.failure_text()
    assert len(run_ctx.backend) == 0 and _status(port) is None


def test_the_check_server_is_stopped_when_its_own_check_fails(run_ctx, workspace):    # noqa: F811
    port = free_port()
    (workspace / "server.py").write_text(SERVER % 404)
    r = verification.verify(make_task(
        {"type": "http_status", "start_command": f"python server.py --port {port}", "url": f"http://127.0.0.1:{port}/", "expect": 200, "wait_s": 3}),
        run_ctx, [], [])
    assert not r.ok and "got status 404" in r.failure_text()
    assert len(run_ctx.backend) == 0 and _status(port) is None
