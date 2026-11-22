"""Commands the agent runs must not inherit credentials from Nexura's environment."""
import sys

import pytest

import config
from core.processes import ProcessManager, run_bounded, scrub_env

SECRETS = {"GOOGLE_API_KEY": "g-secret", "GROQ_API_KEY": "q-secret", "LANGCHAIN_API_KEY": "l-secret", "NEXURA_TOKEN": "t-secret",
           "GITHUB_TOKEN": "gh-secret", "AWS_SECRET_ACCESS_KEY": "aws-secret", "DB_PASSWORD": "pw-secret", "SSH_AUTH_SOCK": "/tmp/agent.sock"}
HARMLESS = {"PATH": "p", "TOKENIZERS_PARALLELISM": "false", "PYTHONUTF8": "1", "HOME": "/h", "NEXURA_DATA_DIR": "/d", "MY_KEYBOARD": "us"}


def test_credentials_are_removed_and_ordinary_variables_kept():
    out = scrub_env({**SECRETS, **HARMLESS})
    assert not set(SECRETS) & set(out) and all(out[k] == v for k, v in HARMLESS.items())


def test_matching_is_case_insensitive_and_does_not_mutate_the_input():
    src = {"google_api_key": "x", "Path": "p"}
    assert scrub_env(src) == {"Path": "p"} and src == {"google_api_key": "x", "Path": "p"}


def test_passthrough_keeps_named_variables_and_the_switch_turns_it_all_off(monkeypatch):
    monkeypatch.setattr(config, "ENV_PASSTHROUGH", frozenset({"GITHUB_TOKEN"}))
    out = scrub_env(SECRETS)
    assert out.get("GITHUB_TOKEN") == "gh-secret" and "GOOGLE_API_KEY" not in out
    monkeypatch.setattr(config, "SCRUB_ENV", False)
    assert scrub_env(SECRETS) == SECRETS


PROBE = "import os; print('KEYS=' + ','.join(sorted(k for k in os.environ if k in %r)) + ';PATH=' + str(bool(os.environ.get('PATH'))))" % (tuple(SECRETS),)


def test_a_command_run_by_the_agent_cannot_see_the_keys(monkeypatch, tmp_path):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    code, out, _ = run_bounded([sys.executable, "-c", PROBE], str(tmp_path), 30, 4000, echo=False)
    assert code == 0 and "KEYS=;PATH=True" in out and "secret" not in out


def test_an_explicit_env_is_scrubbed_too(monkeypatch, tmp_path):
    code, out, _ = run_bounded([sys.executable, "-c", PROBE], str(tmp_path), 30, 4000, echo=False, env={**__import__("os").environ, "GROQ_API_KEY": "x"})
    assert code == 0 and "KEYS=;" in out


def test_background_processes_are_scrubbed_too(monkeypatch, tmp_path):
    import time
    monkeypatch.setenv("GOOGLE_API_KEY", "g-secret")
    pm = ProcessManager()
    handle = pm.start([sys.executable, "-c", PROBE], str(tmp_path), "probe")
    try:
        deadline, text = time.time() + 20, ""
        while time.time() < deadline and "KEYS=" not in text:
            time.sleep(0.2)
            text += pm.read(handle, 4000) or ""
        assert "KEYS=;" in text and "g-secret" not in text
    finally:
        pm.stop_all()


def test_the_default_is_on():
    assert config.SCRUB_ENV is True


@pytest.mark.parametrize("name", ["GOOGLE_API_KEY", "groq_api_key", "NEXURA_TOKEN", "OPENAI_API_KEY", "AZURE_CLIENT_SECRET", "NPM_TOKEN"])
def test_common_credential_names_match(name):
    assert name not in scrub_env({name: "v"})
