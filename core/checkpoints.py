"""Git-based checkpoints and undo. Uses the git CLI only (no GitPython).

Checkpoints are commit objects stored under refs/nexura/<run_id>/<task_id>, built from a TEMPORARY index, so HEAD, branches
and the user's index/staging area are never touched. Untracked files are included; .gitignore is respected.
The single exception is commit_run (V3-2), which the user must ask for: it creates one new branch for a finished run."""
import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.safety import SafetyError, resolve_in_root

IDENTITY = {"GIT_AUTHOR_NAME": "Nexura", "GIT_AUTHOR_EMAIL": "nexura@localhost",
            "GIT_COMMITTER_NAME": "Nexura", "GIT_COMMITTER_EMAIL": "nexura@localhost"}


class GitError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _git(root: Path | str, *args: str, env: dict | None = None, check: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **IDENTITY, **(env or {})}
    try:
        r = subprocess.run(["git", *args], cwd=str(root), env=full_env, capture_output=True, text=text,
                           **({"encoding": "utf-8", "errors": "replace"} if text else {}))
    except FileNotFoundError:
        raise GitError("GIT_NOT_INSTALLED", "git is not installed or not on PATH.")
    if check and r.returncode != 0:
        err = r.stderr if text else r.stderr.decode("utf-8", "replace")
        raise GitError("GIT_FAILED", f"git {' '.join(args)}: {err.strip()[:300]}")
    return r


# ─── repo state ─────────────────────────────────────────────

def is_repo(root: Path | str) -> bool:
    """True only if `root` IS the top level of a git work tree (not merely inside a parent repo)."""
    try:
        r = _git(root, "rev-parse", "--show-toplevel", check=False)
    except GitError:
        return False
    return r.returncode == 0 and os.path.normcase(os.path.realpath(r.stdout.strip())) == os.path.normcase(os.path.realpath(str(root)))


def require_repo(root: Path | str) -> None:
    if not is_repo(root):
        raise GitError("NOT_A_REPO", f"{root} is not a git repository. Initialize git first so every change can be undone.")


def init_repo(root: Path | str) -> None:
    """User-approved bootstrap: git init + an initial commit of the current files."""
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--allow-empty", "-m", "Initial commit (created by Nexura)")


def dirty_files(root: Path | str) -> list[str]:
    """Paths with uncommitted changes (modified, staged or untracked), relative with forward slashes."""
    r = _git(root, "status", "--porcelain", "-z", "--untracked-files=all", text=False)
    out = []
    for entry in r.stdout.split(b"\0"):
        if len(entry) > 3:
            out.append(entry[3:].decode("utf-8", "replace"))
    return out


# ─── checkpoints ────────────────────────────────────────────

def ref_name(run_id: str, task_id: int | str) -> str:
    return f"refs/nexura/{run_id}/{task_id}"


def snapshot(root: Path | str, run_id: str, task_id: int | str, message: str = "") -> str:
    """Snapshot the whole working tree (incl. untracked, minus ignored) as a commit under refs/nexura/... Returns the commit id."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": os.path.join(tmp, "index")}
        _git(root, "add", "-A", env=env)
        tree = _git(root, "write-tree", env=env).stdout.strip()
        head = _git(root, "rev-parse", "-q", "--verify", "HEAD", check=False)
        parent = ["-p", head.stdout.strip()] if head.returncode == 0 else []
        commit = _git(root, "commit-tree", tree, *parent, "-m", message or f"nexura checkpoint {run_id}/{task_id}").stdout.strip()
    _git(root, "update-ref", ref_name(run_id, task_id), commit)
    return commit


# ─── commit a finished run (opt-in) ─────────────────────────

@dataclass
class CommitResult:
    committed: bool
    branch: str | None = None
    commit: str | None = None
    previous: str | None = None            # branch (or short commit) the user was on: `git switch <previous>` goes back
    files: list[str] = field(default_factory=list)
    reason: str | None = None              # why nothing was committed


def _current_ref(root) -> str:
    r = _git(root, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else _git(root, "rev-parse", "--short", "HEAD").stdout.strip()


def dirty_before_run(root: Path | str, run_id: str) -> set[str]:
    """Files that had uncommitted changes of the USER when the run started (the 'base' checkpoint holds them)."""
    if not has_checkpoint(root, run_id, "base"):
        return set()
    base = ref_name(run_id, "base")
    r = _git(root, "diff", "--name-only", "-z", f"{base}^", base, check=False, text=False)
    return {p.decode("utf-8", "replace") for p in r.stdout.split(b"\0") if p} if r.returncode == 0 else set()


def commit_run(root: Path | str, run_id: str, files: list[str], message: str, branch: str | None = None) -> CommitResult:
    """Opt-in: put a finished run's changes on a NEW branch (nexura/<run_id>) as one commit. This is the one place Nexura moves
    HEAD, and only when the user asked for it. Only `files` (what the run changed) are committed (`git commit --only`): the
    user's other staged or unstaged work is not included and stays as it is. Never pushes. Refuses if one of the files already
    had uncommitted changes of the user's before the run (the commit would mix the two). Failure leaves everything as it was."""
    root = Path(root)
    branch = branch or f"nexura/{run_id}"
    if not is_repo(root):
        return CommitResult(False, reason="not a git repository")
    unique = list(dict.fromkeys(f.replace("\\", "/") for f in files))
    try:
        for f in unique:
            resolve_in_root(root, f)
    except SafetyError as e:
        return CommitResult(False, reason=str(e))
    if not unique:
        return CommitResult(False, reason="the run changed no files")
    mixed = sorted(set(unique) & dirty_before_run(root, run_id))
    if mixed:
        return CommitResult(False, files=unique, reason="these files already had uncommitted changes of yours before the run, so a commit would "
                                                         "mix your edits with Nexura's: " + ", ".join(mixed) + ". Commit them yourself.")
    if not _git(root, "status", "--porcelain", "--untracked-files=all", "--", *unique).stdout.strip():
        return CommitResult(False, files=unique, reason="the files are identical to the last commit (nothing to commit)")
    if _git(root, "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False).returncode == 0:
        return CommitResult(False, files=unique, reason=f"branch {branch} already exists")
    previous = _current_ref(root)
    switched = False
    try:
        _git(root, "switch", "-q", "-c", branch)
        switched = True
        present = [f for f in unique if (root / f).exists()]
        if present:
            _git(root, "add", "--", *present)
        gone = [f for f in unique if f not in present]
        if gone:
            _git(root, "rm", "--cached", "-q", "--ignore-unmatch", "--", *gone)
        _git(root, "commit", "-q", "--only", "-m", message, "--", *unique)
        sha = _git(root, "rev-parse", "--short", "HEAD").stdout.strip()
    except GitError as e:   # e.g. a pre-commit hook rejected it: put everything back
        if switched:
            _git(root, "switch", "-q", previous, check=False)
            _git(root, "branch", "-D", branch, check=False)
        return CommitResult(False, files=unique, reason=e.message)
    return CommitResult(True, branch=branch, commit=sha, previous=previous, files=unique)


def has_checkpoint(root: Path | str, run_id: str, task_id: int | str) -> bool:
    return _git(root, "rev-parse", "-q", "--verify", ref_name(run_id, task_id), check=False).returncode == 0


def _blob(root, commit: str, path: str, filters: bool = True) -> bytes | None:
    # --filters applies the working-tree conversion (autocrlf / .gitattributes) so the restore is byte-exact
    args = ["cat-file", "--filters", f"--path={path}"] if filters else ["cat-file", "blob"]
    r = _git(root, *args, f"{commit}:{path}", check=False, text=False)
    return r.stdout if r.returncode == 0 else None


# ─── undo ───────────────────────────────────────────────────

@dataclass
class UndoResult:
    applied: bool
    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)   # files edited since Nexura's last change
    error: str | None = None


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _plan_undo(changes: list[dict]) -> dict[str, dict]:
    """Per file: the EARLIEST task that touched it (its checkpoint is the state to restore) and the LAST recorded hash."""
    files: dict[str, dict] = {}
    for c in changes:  # ordered by id
        f = files.setdefault(c["file"], {"task_id": c["task_id"], "last_after": None, "first_before": c.get("sha256_before")})
        f["last_after"] = c["sha256_after"]
    return files


def undo(root: Path | str, run_id: str, changes: list[dict], confirm_conflicts: bool = False) -> UndoResult:
    """Restore exactly the files recorded in `changes` from their checkpoints; delete files the run created.
    If a file was edited after Nexura's last change, nothing is applied unless confirm_conflicts=True."""
    root = Path(root)
    if not changes:
        return UndoResult(applied=True)
    plan = _plan_undo(changes)
    try:
        for f, info in plan.items():
            resolve_in_root(root, f)
            if not has_checkpoint(root, run_id, info["task_id"]):
                return UndoResult(applied=False, error=f"No checkpoint for task {info['task_id']} of run {run_id}.")
    except SafetyError as e:
        return UndoResult(applied=False, error=str(e))

    conflicts = [f for f, info in plan.items() if _sha(root / f) != info["last_after"]]
    if conflicts and not confirm_conflicts:
        return UndoResult(applied=False, conflicts=conflicts)

    res = UndoResult(applied=True, conflicts=conflicts)
    for f, info in plan.items():
        commit = _git(root, "rev-parse", ref_name(run_id, info["task_id"])).stdout.strip()
        target = resolve_in_root(root, f)
        before = _blob(root, commit, f)
        want = info.get("first_before")
        if before is not None and want and hashlib.sha256(before).hexdigest() != want:
            # the working-tree conversion would change the bytes (e.g. LF files while core.autocrlf=true): use the stored bytes
            raw = _blob(root, commit, f, filters=False)
            if raw is not None and hashlib.sha256(raw).hexdigest() == want:
                before = raw
        if before is None:                      # did not exist before the task: remove it
            if target.exists():
                target.unlink()
            res.deleted.append(f)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(before)
            res.restored.append(f)
    return res
