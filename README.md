# Nexura AI

An orchestrator-driven coding agent. You describe what you want; Nexura **plans** it (you approve the plan), **builds** it step
by step, **validates** every action the model proposes, **executes** only through one controlled module, **verifies** each task
with objective checks, **debugs** failures with a bounded debugger, and **persists** its state so a run can resume. It edits
your real folder safely: every task is checkpointed in git, risky changes wait for your approval, and everything can be undone.

> Principle: **control over intelligence.** Agents only *propose*. The orchestrator decides, runs, verifies and records.

**V2 (optional, off by default)** adds code intelligence for existing projects: a local code index with hybrid retrieval,
project profile and summary, multi-hunk patches, per-file diff review, and a before/after test run that catches regressions.
Everything V2 sits behind flags whose defaults keep V1 behaviour. An opt-in code graph (where a function is used, name-based hints) and a local CI-aware profile exist; GitHub, live CI results and deploy features are not here; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How a run works

```
prompt -> Planner (structured plan + acceptance checks) -> you approve the plan
  for each task:   checkpoint (git ref) -> Coder proposes tool calls
                   -> validator (schema, paths, allowlist, read-before-write, protected/dirty/ignored files, trust)
                   -> [approval if risky] -> execution manager (the ONLY code that calls tools)
                   -> verification (the plan's acceptance checks)
                      pass -> task done
                      fail -> retry manager -> Debugger (fix only this failure) -> validate -> execute -> verify
                      exhausted -> task failed -> failure policy (stop, or skip dependents)
  end: run status = completed | completed_with_failures | failed | cancelled (the CLI exit code is non-zero unless completed)
```

Invariants (all covered by tests): agents never execute tools; every action is validated first; `task_complete` is only a
request (verification decides); only `state_manager` writes memory; agents cannot change the plan; everything is confined to
the workspace; every step is logged with prompt version and model; all loops are bounded; failures are never reported as success.

## Install

Requirements: **Python 3.11+**, **git**, Node 18+ (only for the VS Code extension and Node-based tasks), a Gemini or Groq API key.

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then edit .env (never commit it)
```

`.env`:

| Variable | Meaning |
|---|---|
| `GOOGLE_API_KEY` / `GROQ_API_KEY` | Model provider key (Gemini models use Google; llama/mixtral/gemma names use Groq) |
| `PLANNER_MODEL_NAME`, `CODER_MODEL_NAME` | Model names |
| `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | Optional LangSmith tracing |
| `NEXURA_DATA_DIR` | Where the state DB, checkpoints and run logs go (default `./memory`) |
| `NEXURA_APPROVAL_LEVEL` | `ask_every_edit`, `ask_risky` (default) or `auto_within_limits` |
| `NEXURA_FAILURE_POLICY` | `stop` (default) or `continue` |
| `NEXURA_SHELL_HINT` | Text telling the model which shell/OS it is on |
| `NEXURA_LLM_MAX_ATTEMPTS` | Tries per model call on rate limits / 503 "high demand" (default 6, backoff up to 60 s) |
| `NEXURA_SCRUB_ENV`, `NEXURA_ENV_PASSTHROUGH` | Commands the agent runs get a copy of the environment WITHOUT credentials (any variable whose name has the word API_KEY, TOKEN, SECRET, PASSWORD, CREDENTIALS, ACCESS_KEY or PRIVATE_KEY, plus `SSH_AUTH_SOCK`), so code the model wrote cannot read your keys. On by default; `NEXURA_SCRUB_ENV=false` turns it off; `NEXURA_ENV_PASSTHROUGH=NAME1,NAME2` keeps the named variables |
| `NEXURA_CI_AWARE` | `true` = for existing projects, also read the project's own `.github/workflows` files and add the test/lint commands they run (only exact known commands such as `python -m pytest -q`, `npm test`, `ruff check .` with plain arguments; nothing is fetched from GitHub and nothing is run by this; the baseline test run still needs your yes). Commands from the manifests stay first. Default `false` |
| `NEXURA_CODE_GRAPH` | `true` (with `NEXURA_RETRIEVAL_MODE=v2` and a ready index) = the coder also gets a USAGES block: for the functions/classes of the files it sees, where else they are used, and which files import those files (2 levels). Paths and line numbers only, never code. Name-based: searched in the defining file and the files that import it, so it can include look-alikes (a local variable with the same name) and misses indirect uses (aliases, dynamic calls). Default `false` |
| `NEXURA_LLM_HARD_TIMEOUT_S` | Hard wall-clock limit for one model call (default `2 x NEXURA_LLM_TIMEOUT_S + 30` = 270 s). A call that stalls longer is abandoned and retried like any timeout, so a dead connection cannot hang a run |
| `NEXURA_RETRIEVAL_MODE` | `v1` (default) or `v2`: retrieval context and a planner that knows the project (needs the index; falls back to v1). v2 also asks the planner for behaviour checks (not only syntax) and render-checks changed Streamlit pages |
| `NEXURA_INDEXING_ENABLED` | `false` (default) or `true`: build/use the local code index |
| `NEXURA_PATCH_MODE` | `v1` (default) or `v2`: `apply_patch` with unique-match, atomic multi-hunk edits |
| `NEXURA_ALLOW_MODEL_DOWNLOAD` | `1` = you consent to the one-time download of the local embedding model |
| `NEXURA_BUDGET_MAX_REQUESTS`, `NEXURA_BUDGET_MAX_TOKENS`, `NEXURA_BUDGET_WINDOW_HOURS` | Optional usage cap over a rolling window (default 24 h) across all runs, to protect a free-tier quota. The plan shows an estimated cost before you approve it; when a cap is reached the run pauses and `--resume <run_id>` continues it. Unset or 0 = no cap |

## Use it

**VS Code (recommended):** build and install the extension, see [extension/README.md](extension/README.md).

**Command line** (the workspace is `workspace/<project>`; it must be a git repo, or pass `--git-init`):

```
python main.py --project calc --prompt "Create a CLI calculator in calc.py" --git-init
python main.py --project calc --prompt "..." --approval ask_every_edit     # confirm every edit
python main.py --resume <run_id>                                            # continue an unfinished run
python main.py --undo <run_id> [--task N] [--confirm]                       # restore files from the checkpoints
python main.py --metrics <run_id>                                           # success rate, retries, error types, time
python main.py --project shop --deploy-config docker                        # write a Dockerfile + .dockerignore (or `procfile`) for the finished project
python main.py --project shop --prompt "..." --baseline                     # run the project's tests before/after; report new failures
python main.py --project shop --prompt "..." --commit                       # if every task is verified, commit the changes on a new branch nexura/<run_id> (never pushes)
```

Exit code is 0 only if every task was verified. Run folders (`plan.md` with live checkboxes, `plan.json`, `log.jsonl`) live in
`<data dir>/runs/<run_id>/`, not in your project.

**HTTP API:** `python server.py` with `NEXURA_TOKEN` set; see [docs/API.md](docs/API.md). Localhost only, bearer token required.

## Code indexing for existing projects (optional, V2)

Off by default; V1 behaviour needs none of this. To use it:

1. Install the extra packages: `pip install -r requirements-index.txt` (tree-sitter, chromadb, fastembed; a few hundred MB; the
   VS Code extension offers to do this for you after asking).
2. Consent to the one-time download of the small local embedding model (`BAAI/bge-small-en-v1.5`, about 130 MB):
   set `NEXURA_ALLOW_MODEL_DOWNLOAD=1` or accept the extension's prompt. Without consent Nexura never touches the network and
   tells you how to install the model offline (download it on another machine and copy the `models` folder to `<data dir>/models`).
3. Set `NEXURA_INDEXING_ENABLED=true` and `NEXURA_RETRIEVAL_MODE=v2` (VS Code: "Nexura: Enable Code Indexing").

What happens: your source files are chunked and embedded **locally**; the index lives in `<data dir>/index/` (never inside your
project). Ignored files, `.env*`, keys, lockfiles and files containing secrets are never indexed. Retrieved code is only a hint:
the agent still has to read a file before editing it, and if the index is missing or broken the run uses the V1 context instead.
Indexing a 200-file project takes about a minute on a CPU; later edits re-index only the changed files.

Other V2 options: `NEXURA_PATCH_MODE=v2` (patches must match exactly one place, all hunks or none, line endings preserved);
`baseline: true` on runs / `--baseline` / the extension's question runs your test command before and after the run and marks
the run `completed_with_failures` if a test that passed before now fails (tests that already failed are not blamed).
Commands such as `poetry`, `yarn`, `pnpm`, `make`, `uv` need your one-time approval per workspace.

## Deployment files (optional)

`python main.py --project NAME --deploy-config docker` writes a `Dockerfile` and a `.dockerignore` into the project; `--deploy-config procfile` writes a `Procfile` (Render, Railway, Heroku and similar). It reads the project profile, uses no model and no network, and **only writes text files: it never builds an image and never deploys anything.** Existing files are kept unless you pass `--force`. The image runs as a non-root user, and `.env*`, keys and `.git` are kept out of it (only the *names* of the environment variables the project needs are written, as a comment). Supported: FastAPI, Flask, Streamlit, Django, plain Python scripts (with a `requirements.txt`), npm projects with a `start` script or an entry file, and static sites. Not supported, and then nothing is written: poetry/uv, yarn/pnpm, React/Vite/Vue/Next.js builds. The files are **not tested by Nexura against a real build**: run `docker build .` before you rely on them. No `vercel.json` is generated on purpose (Vercel detects most projects itself, and a wrong one can break a working deploy).

## Safety model

Summary below; full threat model in [docs/SECURITY.md](docs/SECURITY.md).

- **Confinement:** all paths resolve (real paths, no `..`, no symlink escapes, Windows drives handled) inside the opened folder.
- **Secrets:** commands the agent runs never inherit your API keys or tokens (see `NEXURA_SCRUB_ENV`); a script cannot read `GOOGLE_API_KEY` from its environment. It is not a sandbox: a script can still read files your user can read.
- **Commands:** one allowlisted program (`python`, `pip`, `node`, `npm`, `npx`, `uvicorn`, `pytest`, `streamlit`, `flask`), no shell,
  no operators outside quotes; installs only through `install_dependency` and only with your approval; servers run as managed processes that are
  killed (whole tree) at task end.
- **Files:** read-before-write, size and binary limits, `.env*` and `.git/` never writable, lockfiles/keys/CI workflows need approval,
  files with unsaved editor changes are never touched, ignored and secret files never reach the model, credentials in output are redacted.
- **Git:** per-task checkpoints (temporary index; HEAD, branches and your staging area are never touched), exact undo, one active run per workspace.
  The only exception is `--commit` (API: `commit: true`), which you must ask for: after a fully verified run it creates one new branch `nexura/<run_id>` and commits only the files the run changed (your other staged or unstaged work is not included). It never pushes, refuses if one of those files already had your own uncommitted edits, and leaves everything as it was if the commit fails (for example a pre-commit hook). `git switch <previous branch>` goes back; the working folder then shows the files without Nexura's changes, because they live on the new branch.
- **Honesty:** a task is done only when its acceptance checks pass; failed work is reported as failed.

## Develop

```
pytest --runslow                            # ~460 tests, fake LLM and FakeEmbedder: no network, no API cost, no model download
ruff check .                                # lint
python evals/run.py --fake                  # CI smoke eval (scripted model)
python evals/run.py                         # real-model evals (costs API credits; uses a throwaway venv for agent commands)
cd extension && npm run setup && npm test   # build + end-to-end VS Code smoke test
```

Layout: `agents/` (planner, coder, debugger, llm access), `core/` (orchestrator, validator, execution manager, verification,
approvals, checkpoints, state manager, runner, run manager), `tools/` (file, directory, system, verification, search tools),
`server.py` (API), `main.py` (CLI), `extension/` + `ui/` (VS Code extension and sidebar), `evals/`, `tests/`, `docs/`.
`rag/` is the V2 code index and retrieval plus `graph_rag.py` (V3, opt-in usages and impact hints, standard library only); `tools/github_tool.py` is unused (V3).

## Known limits

- `python`/`node` are allowed, and they can run arbitrary code: there is an allowlist, not a sandbox (the `ExecutionBackend` seam for a future container backend exists; only the local backend is implemented).
- `pip install` goes to whichever Python is first on PATH; the approval prompt tells you which one.
- Without the optional index, large projects are summarised by path and first lines within a character budget. With it, retrieval quality was measured on small fixtures only (see [docs/V2_EVAL.md](docs/V2_EVAL.md)).
- The old Tauri prototype lives in `archive/App/`. It is superseded by the VS Code extension and is not used by anything.
