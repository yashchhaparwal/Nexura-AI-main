"""Debugger agent: fixes ONE verification failure. Proposes only; may not add features, create files or edit the plan."""
import config
from agents import llm


def prompt_version() -> str:
    return config.PROMPT_VERSIONS["debugger"]


def build_prompt(task: dict, failure_text: str, error_type: str, strategy_hint: str, files: dict[str, str]) -> str:
    file_block = "\n".join(f"--- {name} ---\n{body}" for name, body in files.items()) or "(no files)"
    v2 = config.PATCH_MODE == "v2"
    tools = ("read_file, apply_patch (preferred for fixes: each search_text must match exactly ONE place), patch_file, write_file (existing files only), "
             "run_command, install_dependency, list_directory, verify_code_syntax") if v2 else \
            "read_file, patch_file, write_file (existing files only), run_command, install_dependency, list_directory, verify_code_syntax"
    return f"""You are a Debugger Agent. (prompt version {prompt_version()})
Fix ONLY the failure below. Add NO features. Do not create new files or edit plans.
Packages: if the failure is a MISSING package, install exactly that package with install_dependency (the user is asked to approve it);
`run_command` can never install. Do not install anything else.
Checks are fixed by the plan: you cannot change a check's command, URL or port. Change the code so the check passes as written.
Allowed tools: {tools}.
Files shown below count as already read; read_file any other file before changing it. Do not call task_complete: the checks are re-run automatically after your fixes.

========================
TASK
========================
Task {task['step_id']}: {task['title']}
{task['description']}

========================
FAILING CHECKS
========================
{failure_text[-3000:]}

Error category: {error_type}
{strategy_hint}

========================
FILES INVOLVED
========================
{file_block}
"""


def propose(messages: list, on_log=None):
    return llm.invoke("debugger", messages, prompt_version(), on_log)
