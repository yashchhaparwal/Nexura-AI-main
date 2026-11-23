"""Tiny regex classifier mapping command output to an error category (metrics + debugger hints)."""
import re

RULES = [
    ("missing_dependency", r"ModuleNotFoundError|No module named|Cannot find module|npm ERR! missing|command not found|is not recognized"),
    ("import_error", r"ImportError|cannot import name"),
    ("syntax_error", r"SyntaxError|IndentationError|TabError|Unexpected token|invalid syntax"),
    # an http_status check that got a response, just not the expected one: checked BEFORE port_in_use because its failure text
    # also carries the server log, which may contain a bind error from a second server on the same port
    ("http_status_mismatch", r"expected \d+, got status \d+"),
    ("port_in_use", r"Address already in use|EADDRINUSE|Errno 98|WinError 10048|port is already allocated"),
    # a check tried to connect and nothing was listening (before test_failure: the traceback of such a check also contains "assert")
    ("server_not_running", r"ConnectionRefusedError|Connection refused|WinError 10061|actively refused|Failed to establish a new connection"),
    ("test_failure", r"AssertionError|FAILED|\d+ failed|Tests? failed|assert "),
]
HINTS = {
    "missing_dependency": "A module/program is missing: add it with install_dependency (and to requirements.txt/package.json). "
                          "If it is a LOCAL package, run tests with `python -m pytest` (it puts the project root on sys.path; bare `pytest` does not).",
    "import_error": "An import failed: check the module/file names and the import path.",
    "syntax_error": "There is a syntax error: re-read the file and fix the exact line reported.",
    "http_status_mismatch": "A check requests a URL and the app answered with the wrong status (404 = that route does not exist). The check, its URL and its port are fixed by the plan: change the CODE so that exact URL answers as expected (for example add the missing route). Do not change the port. (A bind error in the server log means an older copy of the app may still be running on that port; Nexura stops its own before a check, so look at the code first.)",
    "port_in_use": "The port is busy: use a different port or stop the process using it.",
    "server_not_running": "A check could not connect: nothing was listening on that port while it ran. The step's servers are started by its http_status "
                          "check (and kept up for the later checks of the same step) or must be left running by the step itself; a check cannot start one. "
                          "Look for what stops the app from starting (import error, wrong module name, wrong port) and fix that.",
    "test_failure": "A test failed: read the assertion, compare expected vs actual, fix the code under test.",
    "unknown": "Read the error output carefully and fix the root cause.",
}


def classify(text: str) -> str:
    for name, pattern in RULES:
        if re.search(pattern, text or "", re.IGNORECASE):
            return name
    return "unknown"
