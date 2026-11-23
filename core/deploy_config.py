"""Deployment files for a finished project: a Dockerfile + .dockerignore, or a Procfile. Deterministic (no model, no network), derived
from the project profile. It only WRITES TEXT FILES into the workspace: it never builds an image, never runs docker and never deploys.

It says "not supported" instead of guessing when it cannot be sure (poetry/uv, yarn/pnpm, frontend frameworks, no requirements.txt):
a wrong Dockerfile is worse than none. It never overwrites an existing file unless asked (force), never writes through a symlink, and the
.dockerignore keeps .env*, keys and VCS data out of the image. Only the NAMES of environment variables the project needs are written."""
import re
from dataclasses import dataclass, field
from pathlib import Path

from core.project_profile import detect_profile

NL = chr(10)
TARGETS = ("docker", "procfile")
PYTHON_IMAGE, NODE_IMAGE, STATIC_IMAGE = "python:3.12-slim", "node:22-slim", "nginx:alpine"
FRONTENDS = {"React", "Vite", "Vue", "Next.js"}

DOCKERIGNORE = [
    "# Nothing secret or local goes into the image: pass secrets as environment variables when the container runs.",
    ".git", ".gitignore", ".env*", "*.pem", "*.key", "*.p12", "id_rsa*", "credentials*.json", "*.keystore",
    "__pycache__", "*.pyc", ".venv", "venv", "node_modules", ".pytest_cache", ".ruff_cache", ".vscode", ".idea",
    "Dockerfile", ".dockerignore", "*.md", "tests", "memory", ".nexura"]


class DeployConfigError(Exception):
    """The project cannot be turned into this target safely; the message says why and nothing was written."""


@dataclass
class App:
    kind: str                      # python-web | python-script | node | static
    framework: str = ""
    command: list[str] = field(default_factory=list)      # how the container starts it ($PORT is replaced for a Procfile)
    port: int | None = None
    needs_requirements: bool = False
    env_names: list[str] = field(default_factory=list)


@dataclass
class Result:
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)       # already existed (not overwritten) or a symlink
    notes: list[str] = field(default_factory=list)


def _first(pattern: str, commands: list[str]) -> str | None:
    for c in commands:
        m = re.search(pattern, c)
        if m:
            return m.group(1)
    return None


def describe(root: Path) -> App:
    p = detect_profile(root)
    managers = set(p.package_managers)
    if "python" in p.languages:
        if managers & {"poetry", "uv"}:
            raise DeployConfigError(f"this project uses {', '.join(sorted(managers & {'poetry', 'uv'}))}; a Dockerfile for it is not supported yet "
                                    "(the install steps differ). Nothing was written.")
        runs = p.run_commands
        env = list(p.env_needs)
        if "FastAPI" in p.frameworks and (target := _first(r"uvicorn (\S+:\S+)", runs)):
            return App("python-web", "FastAPI", ["python", "-m", "uvicorn", target, "--host", "0.0.0.0", "--port", "{port}"], 8000, True, env)
        if "Flask" in p.frameworks and (module := _first(r"--app (\S+)", runs)):
            return App("python-web", "Flask", ["python", "-m", "flask", "--app", module, "run", "--host", "0.0.0.0", "--port", "{port}"], 5000, True, env)
        if "Streamlit" in p.frameworks and (script := _first(r"streamlit run (\S+)", runs)):
            return App("python-web", "Streamlit", ["python", "-m", "streamlit", "run", script, "--server.address", "0.0.0.0",
                                                   "--server.port", "{port}"], 8501, True, env)
        if "Django" in p.frameworks and any("manage.py" in r for r in runs):
            return App("python-web", "Django", ["python", "manage.py", "runserver", "0.0.0.0:{port}"], 8000, True, env)
        script = _first(r"^python (\S+\.py)$", runs)
        if script:
            return App("python-script", "", ["python", script], None, (root / "requirements.txt").is_file(), env)
        raise DeployConfigError("Python project, but no entry point or web framework was recognised (looked for FastAPI, Flask, Streamlit, "
                                "Django or a main/app script). Nothing was written.")
    if p.languages and set(p.languages) <= {"javascript", "typescript"}:
        if managers - {"npm"}:
            raise DeployConfigError(f"this project uses {', '.join(sorted(managers - {'npm'}))}; only npm projects are supported. Nothing was written.")
        if set(p.frameworks) & FRONTENDS:
            raise DeployConfigError(f"{', '.join(sorted(set(p.frameworks) & FRONTENDS))} is a frontend build (build step plus static hosting): "
                                    "not supported yet. Nothing was written.")
        if "npm start" in p.run_commands:
            cmd = ["npm", "start"]
        else:
            entry = _first(r"^node (\S+)$", p.run_commands)
            if not entry:
                raise DeployConfigError("no `start` script and no entry file to run with node. Nothing was written.")
            cmd = ["node", entry]
        return App("node", p.frameworks[0] if p.frameworks else "", cmd, _node_port(root, p.entry_points), False, list(p.env_needs))
    if (root / "index.html").is_file() and not p.languages:
        return App("static", "", [], 80, False, [])
    raise DeployConfigError("could not tell what this project is (no recognisable Python or Node project and no index.html). Nothing was written.")


def _node_port(root: Path, entries: list[str]) -> int:
    for e in entries:
        try:
            m = re.search(r"listen\(\s*(?:process\.env\.PORT\s*(?:\|\||\?\?)\s*)?(\d{2,5})", (root / e).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if m:
            return int(m.group(1))
    return 3000


def _exec_form(parts: list[str], port: int | None) -> str:
    items = [p.replace("{port}", str(port)) for p in parts]
    return "[" + ", ".join('"' + i.replace('"', "") + '"' for i in items) + "]"


def dockerfile(root: Path, app: App) -> str:
    L: list[str] = []
    env = (f"# Set these environment variables when the container runs (they are never baked into the image): {', '.join(app.env_names)}"
           if app.env_names else "")
    if app.kind == "static":
        L += [f"FROM {STATIC_IMAGE}", "COPY . /usr/share/nginx/html", "EXPOSE 80"]
    elif app.kind.startswith("python"):
        L += [f"FROM {PYTHON_IMAGE}", "ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1", "WORKDIR /app"]
        if app.needs_requirements:
            L += ["COPY requirements.txt .", "RUN pip install --no-cache-dir -r requirements.txt"]
        L += ["COPY . .", "RUN useradd --create-home app && chown -R app /app", "USER app"]
        if env:
            L.append(env)
        if app.port:
            L.append(f"EXPOSE {app.port}")
        L.append(f"CMD {_exec_form(app.command, app.port)}")
        if app.framework in ("Flask", "Django"):
            L.append(f"# NOTE: this starts the {app.framework} development server. For real traffic put a production server (for example gunicorn) in front of it.")
    else:   # node
        lock = (root / "package-lock.json").is_file()
        L += [f"FROM {NODE_IMAGE}", "ENV NODE_ENV=production", "WORKDIR /app", "COPY package*.json ./",
              "RUN npm ci --omit=dev" if lock else "RUN npm install --omit=dev", "COPY . .", "USER node"]
        if env:
            L.append(env)
        L += [f"EXPOSE {app.port}", f"CMD {_exec_form(app.command, app.port)}"]
        if not lock:
            L.append("# NOTE: there is no package-lock.json, so versions are not pinned; commit one to make builds repeatable.")
    return NL.join(L) + NL


def procfile(app: App) -> str:
    if app.kind in ("python-script", "static"):
        raise DeployConfigError("a Procfile starts a web process, and this project is not a web app Nexura can recognise. Nothing was written.")
    cmd = " ".join(c.replace("{port}", "$PORT") for c in app.command)
    return f"web: {cmd}{NL}"


def files_for(root: Path, target: str) -> tuple[dict[str, str], list[str]]:
    """{file name: content} and notes. Raises DeployConfigError."""
    if target not in TARGETS:
        raise DeployConfigError(f"unknown target {target!r}; use one of: {', '.join(TARGETS)}")
    app = describe(root)
    notes: list[str] = []
    if target == "procfile":
        if app.kind == "node":
            notes.append("the app must read the port from the PORT environment variable")
        return {"Procfile": procfile(app)}, notes
    if app.needs_requirements and not (root / "requirements.txt").is_file():
        raise DeployConfigError("no requirements.txt: the image would not contain your dependencies. List them first. Nothing was written.")
    if app.kind == "node" and not (root / "package.json").is_file():
        raise DeployConfigError("no package.json. Nothing was written.")
    notes.append("the image was NOT built or tested by Nexura; build it yourself with `docker build .` before relying on it")
    return {"Dockerfile": dockerfile(root, app), ".dockerignore": NL.join(DOCKERIGNORE) + NL}, notes


def write(root: Path | str, target: str, force: bool = False) -> Result:
    """Write the files for `target` into `root`. Existing files are kept unless `force`; a symlink is never written through."""
    root = Path(root)
    if not root.is_dir():
        raise DeployConfigError(f"{root} is not a folder")
    files, notes = files_for(root, target)
    res = Result(notes=notes)
    for name, text in files.items():
        path = root / name
        if path.is_symlink() or (path.exists() and not force):
            res.skipped.append(name)
            continue
        path.write_text(text, encoding="utf-8", newline=NL)
        res.written.append(name)
    return res


def cli(root: Path | str, target: str, force: bool = False) -> int:
    """Used by `main.py --deploy-config`. Exit code 0 when something was written, 1 when nothing was."""
    try:
        res = write(root, target, force)
    except DeployConfigError as e:
        print(f"Deploy config not written: {e}")
        return 1
    for n in res.written:
        print(f"  wrote {n}")
    for n in res.skipped:
        print(f"  kept {n} (it already exists or is a symlink; use --force to replace a file)")
    for n in res.notes:
        print(f"  note: {n}")
    return 0 if res.written else 1
