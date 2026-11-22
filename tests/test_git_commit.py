"""V3-2: opt-in commit of a finished run on a new branch. Never pushes, never mixes in the user's own work."""
import subprocess

import pytest
from langgraph.types import Command

from core import checkpoints
from core.runner import start_run
from core.state_manager import make_checkpointer
from tests.test_persistence import FIRST, SECOND, env  # noqa: F401  (env is a fixture)


def git(root, *a):
    return subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def branch(root):
    return git(root, "branch", "--show-current")


@pytest.fixture
def repo(git_repo):
    """A repo with three tracked files, like a project Nexura is about to change."""
    for name, text in (("mod.txt", "a\n"), ("del.txt", "b\n"), ("other.txt", "c\n")):
        (git_repo / name).write_text(text)
    git(git_repo, "add", "-A")
    git(git_repo, "commit", "-qm", "files")
    return git_repo


def simulate_run_changes(root):
    (root / "mod.txt").write_text("a2\n")
    (root / "del.txt").unlink()
    (root / "new.txt").write_text("n\n")
    return ["mod.txt", "del.txt", "new.txt"]


def test_commits_only_the_run_files_on_a_new_branch_and_leaves_the_users_work_alone(repo):
    start = branch(repo)
    files = simulate_run_changes(repo)
    (repo / "other.txt").write_text("c2\n")
    git(repo, "add", "other.txt")                       # the user's own STAGED change
    (repo / "scratch.txt").write_text("s\n")            # and an untracked file
    res = checkpoints.commit_run(repo, "run1", files, "nexura: test")
    assert res.committed and res.branch == "nexura/run1" and res.previous == start
    assert branch(repo) == "nexura/run1"
    assert set(git(repo, "show", "--name-only", "--format=", "HEAD").split()) == {"mod.txt", "del.txt", "new.txt"}
    status = git(repo, "status", "--short")
    assert "M  other.txt" in status and "?? scratch.txt" in status       # still staged / untracked, not committed
    assert git(repo, "log", "-1", "--format=%an") == "Nexura"
    git(repo, "switch", "-q", start)                    # going back is plain git
    assert branch(repo) == start and git(repo, "log", "-1", "--format=%s") == "files"


def test_refuses_when_the_users_own_uncommitted_edit_is_in_a_run_file(repo):
    (repo / "mod.txt").write_text("my own edit\n")      # dirty BEFORE the run
    checkpoints.snapshot(repo, "run2", "base")          # what the runner does for a dirty workspace
    (repo / "mod.txt").write_text("my own edit\nplus nexura\n")
    res = checkpoints.commit_run(repo, "run2", ["mod.txt"], "m")
    assert not res.committed and "mod.txt" in res.reason and "mix" in res.reason
    assert branch(repo) == git(repo, "rev-parse", "--abbrev-ref", "HEAD") and "nexura/run2" not in git(repo, "branch")


def test_nothing_to_commit_branch_exists_and_unsafe_paths(repo):
    assert "identical" in checkpoints.commit_run(repo, "r", ["mod.txt"], "m").reason
    assert "no files" in checkpoints.commit_run(repo, "r", [], "m").reason
    assert checkpoints.commit_run(repo, "r", ["../outside.txt"], "m").committed is False
    (repo / "mod.txt").write_text("x\n")
    git(repo, "branch", "nexura/r")
    assert "already exists" in checkpoints.commit_run(repo, "r", ["mod.txt"], "m").reason
    assert (repo / "mod.txt").read_text() == "x\n"


def test_a_rejecting_hook_leaves_everything_as_it_was(repo):
    start = branch(repo)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho no >&2\nexit 1\n")
    hook.chmod(0o755)
    (repo / "mod.txt").write_text("changed\n")
    res = checkpoints.commit_run(repo, "run3", ["mod.txt"], "m")
    assert not res.committed
    assert branch(repo) == start and "nexura/run3" not in git(repo, "branch")      # switched back, branch removed
    assert (repo / "mod.txt").read_text() == "changed\n"                          # the change is still there


def test_not_a_repo(tmp_path):
    assert checkpoints.commit_run(tmp_path, "r", ["a"], "m").reason == "not a git repository"


# ─── through the pipeline ───────────────────────────────────

def run_pipeline(env, git_repo, commit):  # noqa: F811
    env.llm.script[:] = FIRST + SECOND
    ctx, state = start_run(env.sm, "proj", git_repo, "make a.txt", commit_on_success=commit)
    graph = env.build(checkpointer=make_checkpointer(env.ckpt_path))
    cfg = {"configurable": {"thread_id": ctx.run_id, "project": ctx}, "recursion_limit": 200}
    graph.invoke(state, cfg)
    graph.invoke(Command(resume="yes"), cfg)
    return ctx


def test_a_successful_run_is_committed_when_asked(env, git_repo):    # noqa: F811
    start = branch(git_repo)
    ctx = run_pipeline(env, git_repo, commit=True)
    assert env.sm.get_run(ctx.run_id).status == "completed"
    assert branch(git_repo) == f"nexura/{ctx.run_id}" and "a.txt" in git(git_repo, "show", "--name-only", "--format=", "HEAD").split()
    ev = [e for e in env.sm.list_events(ctx.run_id) if e["type"] == "git_committed"]
    assert len(ev) == 1 and ev[0]["data"]["previous"] == start


def test_nothing_is_committed_unless_asked(env, git_repo):    # noqa: F811
    start = branch(git_repo)
    ctx = run_pipeline(env, git_repo, commit=False)
    assert env.sm.get_run(ctx.run_id).status == "completed"
    assert branch(git_repo) == start and not [e for e in env.sm.list_events(ctx.run_id) if e["type"].startswith("git_commit")]
    assert (git_repo / "a.txt").exists() and "a.txt" in git(git_repo, "status", "--short")      # left as plain changes


def test_a_run_that_did_not_fully_succeed_is_not_committed(repo, tmp_path):
    from core.orchestrator import _commit_if_requested
    from core.schemas import ProjectContext
    from core.state_manager import StateManager
    sm = StateManager(tmp_path / "s.db")
    sm.upsert_project("p", str(repo))
    sm.create_run("r9", "p", "prompt", "ask_risky")
    ctx = ProjectContext(run_id="r9", root=repo, commit_on_success=True, git_checkpoints=True, state_manager=sm)
    (repo / "mod.txt").write_text("x\n")
    _commit_if_requested(ctx, {"plan": {"goal": "g"}, "tasks": []}, "completed_with_failures")
    ev = [e for e in sm.list_events("r9") if e["type"].startswith("git_commit")]
    assert len(ev) == 1 and ev[0]["type"] == "git_commit_skipped" and "completed_with_failures" in ev[0]["data"]["reason"]
    assert "nexura/r9" not in git(repo, "branch")
