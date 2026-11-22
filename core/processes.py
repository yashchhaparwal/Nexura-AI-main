"""Bounded command runner and managed long-running processes. Kills whole process trees."""
import os
import re
import subprocess
import threading
import uuid
from dataclasses import dataclass, field

import psutil

import config
from core.safety import truncate


def kill_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=5)


# Credential-looking variable names: any underscore-separated word api_key / token / secret / password / credentials / access_key /
# private_key (so GOOGLE_API_KEY, GROQ_API_KEY, NEXURA_TOKEN, AWS_SECRET_ACCESS_KEY, GITHUB_TOKEN match; TOKENIZERS_PARALLELISM and PATH do not).
_SECRET_NAME = re.compile(r"(^|_)(API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|ACCESS_?KEY|PRIVATE_?KEY)(_|$)", re.IGNORECASE)
_ALWAYS_HIDDEN = {"SSH_AUTH_SOCK"}   # would let a command use your SSH keys without ever seeing them


def scrub_env(env: dict | None = None) -> dict:
    """The environment for a command the agent runs: `env` (default: ours) without credentials. Never mutates its argument."""
    base = dict(os.environ if env is None else env)
    if not config.SCRUB_ENV:
        return base
    return {k: v for k, v in base.items()
            if k.upper() in config.ENV_PASSTHROUGH or not (_SECRET_NAME.search(k) or k.upper() in _ALWAYS_HIDDEN)}


def _popen(argv: list[str], cwd: str, env: dict | None = None) -> subprocess.Popen:
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    return subprocess.Popen(argv, cwd=cwd, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=scrub_env(env), **kw)


def _pump(proc: subprocess.Popen, lines: list[str], echo: bool) -> threading.Thread:
    def read():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                if echo:
                    print(f"    | {line}")
                lines.append(line)
    t = threading.Thread(target=read, daemon=True)
    t.start()
    return t


def run_bounded(argv: list[str], cwd: str, timeout: int, max_bytes: int, echo: bool = True,
                env: dict | None = None) -> tuple[int | None, str, bool]:
    """Run to completion. Returns (exit_code or None on timeout, output, timed_out)."""
    lines: list[str] = []
    proc = _popen(argv, cwd, env)
    t = _pump(proc, lines, echo)
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc.pid)
        proc.wait()
    t.join(timeout=2)
    out, _ = truncate("\n".join(lines), max_bytes)
    return (None if timed_out else proc.returncode), out, timed_out


@dataclass
class ManagedProcess:
    proc: subprocess.Popen
    command: str
    lines: list[str] = field(default_factory=list)
    read_upto: int = 0


class ProcessManager:
    """Registry of background processes for one project/run."""

    def __init__(self):
        self._procs: dict[str, ManagedProcess] = {}

    def start(self, argv: list[str], cwd: str, command: str) -> str:
        mp = ManagedProcess(_popen(argv, cwd), command)
        _pump(mp.proc, mp.lines, echo=False)
        handle = uuid.uuid4().hex[:8]
        self._procs[handle] = mp
        return handle

    def pid(self, handle: str) -> int | None:
        mp = self._procs.get(handle)
        return mp.proc.pid if mp else None

    def read(self, handle: str, max_bytes: int) -> str | None:
        mp = self._procs.get(handle)
        if mp is None:
            return None
        new = mp.lines[mp.read_upto:]
        mp.read_upto = len(mp.lines)
        status = "running" if mp.proc.poll() is None else f"exited with code {mp.proc.returncode}"
        out, _ = truncate("\n".join(new), max_bytes)
        return f"[process {handle}: {status}]\n{out}"

    def stop(self, handle: str) -> bool:
        mp = self._procs.pop(handle, None)
        if mp is None:
            return False
        kill_tree(mp.proc.pid)
        return True

    def stop_all(self) -> None:
        for h in list(self._procs):
            self.stop(h)

    def __len__(self):
        return len(self._procs)
