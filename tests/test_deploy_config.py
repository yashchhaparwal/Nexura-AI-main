"""Deploy config: deterministic Dockerfile/.dockerignore/Procfile text from the project profile. Nothing is built, run or deployed."""
import json

import pytest

from core import deploy_config as dc

NL = chr(10)


def project(tmp_path, files: dict):
    for rel, lines in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(NL.join(lines) + NL, encoding="utf-8")
    return tmp_path


FASTAPI = {"app.py": ["from fastapi import FastAPI", "app = FastAPI()"], "requirements.txt": ["fastapi", "uvicorn"]}


def docker(root):
    return dc.files_for(root, "docker")[0]


# ─── what is generated for each kind of project ───

def test_fastapi_gets_a_non_root_image_with_the_right_command(tmp_path):
    files = docker(project(tmp_path, FASTAPI))
    d = files["Dockerfile"]
    assert d.startswith("FROM python:3.12-slim" + NL)
    assert "COPY requirements.txt ." in d and "pip install --no-cache-dir -r requirements.txt" in d
    assert "USER app" in d and "EXPOSE 8000" in d
    assert 'CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]' in d
    assert d.index("COPY requirements.txt") < d.index("COPY . .")                 # dependency layer is cached separately


def test_flask_and_streamlit_commands(tmp_path):
    flask = docker(project(tmp_path / "f", {"app.py": ["from flask import Flask", "app = Flask(__name__)"], "requirements.txt": ["flask"]}))["Dockerfile"]
    assert 'CMD ["python", "-m", "flask", "--app", "app", "run", "--host", "0.0.0.0", "--port", "5000"]' in flask and "development server" in flask
    st = docker(project(tmp_path / "s", {"app.py": ["import streamlit as st", "st.title('x')"], "requirements.txt": ["streamlit"]}))["Dockerfile"]
    assert '"streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"' in st and "EXPOSE 8501" in st


def test_a_plain_script_has_no_port(tmp_path):
    d = docker(project(tmp_path, {"main.py": ["print('hi')"], "requirements.txt": ["requests"]}))["Dockerfile"]
    assert 'CMD ["python", "main.py"]' in d and "EXPOSE" not in d


def test_node_uses_npm_ci_with_a_lockfile_and_the_start_script(tmp_path):
    pkg = json.dumps({"name": "x", "scripts": {"start": "node server.js"}, "dependencies": {"express": "4"}})
    root = project(tmp_path, {"package.json": [pkg], "server.js": ["app.listen(8080)"], "package-lock.json": ["{}"]})
    d = docker(root)["Dockerfile"]
    assert d.startswith("FROM node:22-slim") and "RUN npm ci --omit=dev" in d and 'CMD ["npm", "start"]' in d
    assert "EXPOSE 8080" in d and "USER node" in d
    (root / "package-lock.json").unlink()
    d2 = docker(root)["Dockerfile"]
    assert "RUN npm install --omit=dev" in d2 and "not pinned" in d2


def test_a_static_site_uses_nginx(tmp_path):
    d = docker(project(tmp_path, {"index.html": ["<h1>hi</h1>"]}))["Dockerfile"]
    assert d.startswith("FROM nginx:alpine") and "EXPOSE 80" in d


def test_procfile_uses_the_port_variable_and_refuses_non_web_apps(tmp_path):
    files = dc.files_for(project(tmp_path / "a", FASTAPI), "procfile")[0]
    assert files == {"Procfile": "web: python -m uvicorn app:app --host 0.0.0.0 --port $PORT" + NL}
    pkg = json.dumps({"name": "x", "scripts": {"start": "node s.js"}, "dependencies": {"express": "4"}})
    files, notes = dc.files_for(project(tmp_path / "n", {"package.json": [pkg], "s.js": ["x"]}), "procfile")
    assert files["Procfile"] == "web: npm start" + NL and any("PORT" in n for n in notes)
    with pytest.raises(dc.DeployConfigError, match="not a web app"):
        dc.files_for(project(tmp_path / "p", {"main.py": ["print(1)"]}), "procfile")


# ─── it refuses instead of guessing ───

@pytest.mark.parametrize("files,message", [
    ({"pyproject.toml": ["[tool.poetry]", 'name = "x"'], "app.py": ["from fastapi import FastAPI", "app = FastAPI()"]}, "poetry"),
    ({"app.py": ["from fastapi import FastAPI", "app = FastAPI()"]}, "no requirements.txt"),
    ({"package.json": [json.dumps({"dependencies": {"react": "18"}, "scripts": {"start": "x"}})]}, "frontend"),
    ({"package.json": [json.dumps({"scripts": {"start": "x"}})], "yarn.lock": [""]}, "yarn"),
    ({"README.md": ["nothing here"]}, "could not tell"),
])
def test_unsupported_projects_write_nothing(tmp_path, files, message):
    root = project(tmp_path, files)
    with pytest.raises(dc.DeployConfigError, match=message):
        dc.write(root, "docker")
    assert not (root / "Dockerfile").exists() and not (root / ".dockerignore").exists()


def test_unknown_target(tmp_path):
    with pytest.raises(dc.DeployConfigError, match="unknown target"):
        dc.files_for(project(tmp_path, FASTAPI), "kubernetes")


# ─── safety of what is written ───

def test_dockerignore_keeps_secrets_and_vcs_out_of_the_image(tmp_path):
    ignore = docker(project(tmp_path, FASTAPI))[".dockerignore"].splitlines()
    for must in (".git", ".env*", "*.pem", "*.key", "node_modules", ".venv"):
        assert must in ignore


def test_only_variable_names_are_written_never_values(tmp_path):
    root = project(tmp_path, {**FASTAPI, ".env.example": ["API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456", "DATABASE_URL=", "MODE=production"]})
    text = " ".join(docker(root).values())
    assert "DATABASE_URL" in text and "sk-abcdef" not in text and "production" not in text


def test_existing_files_are_kept_unless_forced(tmp_path):
    root = project(tmp_path, {**FASTAPI, "Dockerfile": ["# mine"]})
    res = dc.write(root, "docker")
    assert res.skipped == ["Dockerfile"] and res.written == [".dockerignore"]
    assert (root / "Dockerfile").read_text() == "# mine" + NL
    assert dc.write(root, "docker", force=True).written == ["Dockerfile", ".dockerignore"]
    assert (root / "Dockerfile").read_text().startswith("FROM python")


def test_a_symlink_is_never_written_through(tmp_path):
    root = project(tmp_path / "proj", FASTAPI)
    other = tmp_path / "target.txt"
    other.write_text("keep", encoding="utf-8")
    try:
        (root / "Procfile").symlink_to(other)
    except OSError:
        pytest.skip("symlinks need extra rights on this machine")
    assert dc.write(root, "procfile", force=True).skipped == ["Procfile"] and other.read_text() == "keep"


def test_files_use_unix_line_endings_and_nothing_else_is_created(tmp_path):
    root = project(tmp_path, FASTAPI)
    before = {p.name for p in root.iterdir()}
    dc.write(root, "docker")
    assert (root / "Dockerfile").read_bytes().count(b"\r") == 0
    assert {p.name for p in root.iterdir()} - before == {"Dockerfile", ".dockerignore"}


def test_the_module_has_no_way_to_build_or_deploy():
    import ast
    from pathlib import Path
    src = Path(dc.__file__).read_text(encoding="utf-8")
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import)}
    imported |= {n.module.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"subprocess", "socket", "requests", "urllib", "http", "docker", "os"}


# ─── the command line ───

def test_cli_exit_codes_and_messages(tmp_path, capsys):
    root = project(tmp_path, FASTAPI)
    assert dc.cli(root, "docker") == 0
    out = capsys.readouterr().out
    assert "wrote Dockerfile" in out and "wrote .dockerignore" in out and "NOT built or tested" in out
    assert dc.cli(root, "docker") == 1                                             # everything already exists: nothing written
    assert "kept Dockerfile" in capsys.readouterr().out
    assert dc.cli(project(tmp_path / "empty", {"x.txt": ["a"]}), "docker") == 1
    assert "not written" in capsys.readouterr().out
