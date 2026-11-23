"""Code graph (usages and impact): name-based hints from the chunk table and the import map. No network, no model."""
import shutil
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
import config  # noqa: E402
from core import context_builder as cb  # noqa: E402
from core.schemas import ProjectContext  # noqa: E402
from rag.embedder import FakeEmbedder  # noqa: E402
from rag.graph_rag import CodeGraph  # noqa: E402
from rag.indexer import Indexer  # noqa: E402

FIX = Path(__file__).resolve().parent.parent / "evals" / "fixtures"
TASK = {"step_id": 1, "title": "Change reports.py", "description": "Change the value_report function in reports.py."}
NL = chr(10)


@pytest.fixture(autouse=True)
def index_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")


def build(root, name="ws"):
    ix = Indexer(root, name, FakeEmbedder(dim=32))
    ix.open()
    ix.build()
    return ix


def project(tmp_path, files: dict):
    root = tmp_path / "proj"
    for rel, lines in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(NL.join(lines) + NL, encoding="utf-8")
    return root


@pytest.fixture
def inventory(tmp_path):
    root = tmp_path / "inv"
    shutil.copytree(FIX / "inventory_pkg", root, ignore=shutil.ignore_patterns("__pycache__"))
    ix = build(root, "inv")
    yield root, ix, CodeGraph(ix.lexical, ix.imports)
    ix.close()


# ─── definitions and usages on a real fixture ───

def test_definitions_list_functions_and_skip_noise(tmp_path):
    root = project(tmp_path, {"m.py": [
        "class Cart:", "    def add_item(self, x):", "        return x", "    def __init__(self):", "        pass", "",
        "def go(a):", "    return a", "", "def test_cart():", "    pass", "", "def real_work():", "    pass"]})
    ix = build(root)
    names = {(d.qualified, d.kind) for d in CodeGraph(ix.lexical, ix.imports).definitions("m.py")}
    ix.close()
    assert ("Cart", "class") in names and ("Cart.add_item", "method") in names and ("real_work", "function") in names
    assert not any(n in ("go", "test_cart", "Cart.__init__") for n, _ in names)      # too short, a test, a dunder


def test_usages_match_what_grep_finds_in_the_inventory_fixture(inventory):
    _, ix, g = inventory
    got = g.usages("line_total", "inventory/pricing.py", (1, 3))
    assert got == [("inventory/pricing.py", 11), ("inventory/reports.py", 1), ("inventory/reports.py", 9), ("inventory/reports.py", 10),
                   ("tests/test_pricing.py", 2), ("tests/test_pricing.py", 6)]
    assert ("tests/test_pricing.py", 5) not in got                                 # def test_line_total: a longer name, not a use
    assert ("inventory/pricing.py", 1) not in got                                  # the definition itself


def test_impact_is_nearest_first_and_limited(inventory):
    _, ix, g = inventory
    imp = g.impact(["inventory/pricing.py"])
    assert imp[:2] == [("inventory/reports.py", 1), ("tests/test_pricing.py", 1)] and ("main.py", 2) in imp
    assert g.impact(["inventory/pricing.py"], depth=1) == [("inventory/reports.py", 1), ("tests/test_pricing.py", 1)]
    assert len(g.impact(["inventory/pricing.py"], limit=1)) == 1


def test_summary_is_paths_and_line_numbers_only(inventory):
    _, ix, g = inventory
    text = NL.join(g.summary(["inventory/pricing.py"]))
    assert "line_total (function, lines 1-3) is also used at: inventory/pricing.py:11; inventory/reports.py:1,9,10" in text
    assert "imported by: inventory/reports.py" in text
    assert "return" not in text and "def " not in text                              # never code


# ─── look-alikes, comments, cycles ───

def test_look_alikes_comments_and_longer_names_are_not_reported(tmp_path):
    root = project(tmp_path, {
        "a.py": ["def compute_total(x):", "    return x"],
        "b.py": ["from a import compute_total", "# compute_total is great", "compute_total_extra = 1", "y = compute_total(2)"],
        "c.py": ["def compute_total(x):", "    return compute_total(x - 1)"]})      # its own function, never imports a.py
    ix = build(root)
    got = CodeGraph(ix.lexical, ix.imports).usages("compute_total", "a.py", (1, 2))
    ix.close()
    assert got == [("b.py", 1), ("b.py", 4)]            # not the comment (line 2), not compute_total_extra (3), nothing from c.py


def test_import_cycles_do_not_loop(tmp_path):
    root = project(tmp_path, {"a.py": ["import b", "def fa():", "    return b.fb()"],
                              "b.py": ["import a", "def fb():", "    return a.fa()"]})
    ix = build(root)
    g = CodeGraph(ix.lexical, ix.imports)
    assert g.impact(["a.py"], depth=5) == [("b.py", 1)]
    assert "b.py" in NL.join(g.summary(["a.py"]))
    ix.close()


def test_javascript_imports_are_followed(tmp_path):
    root = project(tmp_path, {
        "util.js": ["export function formatName(first, last) {", "  return first + ' ' + last;", "}"],
        "app.js": ["import { formatName } from './util.js';", "console.log(formatName('a', 'b'));"],
        "other.js": ["function formatName() { return 1; }", "formatName();"]})
    ix = build(root)
    got = CodeGraph(ix.lexical, ix.imports).usages("formatName", "util.js", (1, 3))
    ix.close()
    assert got == [("app.js", 1), ("app.js", 2)]


def test_the_allow_filter_hides_paths(inventory):
    _, ix, g = inventory
    text = NL.join(g.summary(["inventory/pricing.py"], allow=lambda p: not p.startswith("tests/")))
    assert "tests/" not in text and "inventory/reports.py" in text
    assert g.summary(["inventory/pricing.py"], allow=lambda p: False) == []


# ─── in the coder's context ───

def make_ctx(root, ix):
    events = []

    class Sink:
        def add_event(self, run_id, type, task_id=None, **data):
            events.append((type, data))
    return ProjectContext(run_id="r", root=root, indexer=ix, state_manager=Sink()), events


def test_the_usages_block_is_opt_in_and_shows_no_code(inventory, monkeypatch):
    root, ix, _ = inventory
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "v2")
    ctx, _ = make_ctx(root, ix)
    off, b_off = cb.build_task_context(TASK, ctx, [])
    assert "USAGES" not in off and "usage_files" not in b_off and "usages" not in b_off["sections"]      # flag off: exactly the V2 shape
    monkeypatch.setattr(config, "CODE_GRAPH", True)
    on, b_on = cb.build_task_context(TASK, ctx, [])
    block = on[on.index("USAGES"):]
    assert "value_report (function" in block and "main.py:4,12" in block and "imported by: main.py" in block
    assert b_on["usage_files"] >= 1 and b_on["sections"]["usages"] > 0
    assert len(on) > len(off)                                                # the same context plus one more block


def test_a_graph_failure_never_breaks_the_context(inventory, monkeypatch):
    root, ix, _ = inventory
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "v2")
    monkeypatch.setattr(config, "CODE_GRAPH", True)
    import rag.graph_rag as gr

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(gr.CodeGraph, "summary", boom)
    ctx, events = make_ctx(root, ix)
    text, b = cb.build_task_context(TASK, ctx, [])
    assert "FILES" in text and "USAGES" not in text and b["mode"] == "v2" and b["usage_files"] == 0
    assert any(t == "code_graph_failed" for t, _ in events)


def test_ignored_files_are_not_named_in_the_usages(inventory, monkeypatch):
    root, ix, _ = inventory
    (root / ".nexuraignore").write_text("tests/" + NL, encoding="utf-8")
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "v2")
    monkeypatch.setattr(config, "CODE_GRAPH", True)
    ctx, _ = make_ctx(root, ix)
    text, _ = cb.build_task_context(TASK, ctx, [])
    block = text[text.index("USAGES"):]
    assert "tests/" not in block and "main.py" in block
