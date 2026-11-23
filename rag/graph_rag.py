"""Usages and impact, computed from data the index already holds (the chunk table and the import map). Standard library only.

It is deliberately NOT a type-aware call graph. Two facts are used:
  * which files import which (resolved paths from the import map, so only files of this workspace), and
  * where a defined name (function, class, method) appears as a whole word.
A name is searched only in the file that defines it and in the files that import that file, which keeps look-alike names in
unrelated files out. It can still over-report (a local variable with the same name) and it misses indirect uses (aliases,
`getattr`, dynamic dispatch); callers must present it as a hint, never as proof.
"""
import re
from collections import deque
from dataclasses import dataclass

DEF_KINDS = ("function", "class", "method")
MIN_NAME = 3                      # shorter names (i, db, id) match everything
COMMENT = ("#", "//", "*", "/*")


@dataclass(frozen=True)
class Definition:
    name: str                     # bare name; a method "Class.run" is searched as "run"
    qualified: str
    kind: str
    start: int
    end: int


class CodeGraph:
    def __init__(self, lexical, imports):
        self._lex, self._imp = lexical, imports

    # ── what a file defines ──
    def definitions(self, path: str) -> list[Definition]:
        out, seen = [], set()
        for c in self._lex.chunks_of(path):
            if c.kind not in DEF_KINDS or not c.symbol or c.symbol == "<module>":
                continue
            name = c.symbol.split(".")[-1]
            key = (c.symbol, c.kind)
            if key in seen or len(name) < MIN_NAME or name.startswith("__") or name.startswith("test_"):
                continue
            seen.add(key)
            out.append(Definition(name, c.symbol, c.kind, c.start_line, c.end_line))
        return out

    # ── where a name is used ──
    def usages(self, name: str, defined_in: str, own: tuple[int, int] | None = None, limit: int = 40) -> list[tuple[str, int]]:
        """(path, line) of whole-word occurrences of `name` in the defining file and in the files that import it. `own` is the line
        range of the definition itself, which is not a usage."""
        pattern = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])")
        files = [defined_in] + [f for f in self._imp.importers_of(defined_in) if f != defined_in]
        found: dict[tuple[str, int], None] = {}
        for path in files:
            for c in self._lex.chunks_of(path):
                for i, line in enumerate(c.text.splitlines()):
                    if not pattern.search(line) or line.lstrip().startswith(COMMENT):
                        continue
                    n = c.start_line + i
                    if path == defined_in and own and own[0] <= n <= own[1]:
                        continue
                    found[(path, n)] = None      # split chunks overlap: the same line can come twice
                    if len(found) >= limit:
                        return sorted(found)
        return sorted(found)

    # ── who is affected ──
    def impact(self, paths: list[str], depth: int = 2, limit: int = 20) -> list[tuple[str, int]]:
        """Files that import `paths` directly (1) or through another file (2, ...), breadth first, nearest first."""
        seen = set(paths)
        queue = deque((p, 0) for p in paths)
        out: list[tuple[str, int]] = []
        while queue and len(out) < limit:
            path, d = queue.popleft()
            if d >= depth:
                continue
            for importer in self._imp.importers_of(path):
                if importer not in seen:
                    seen.add(importer)
                    out.append((importer, d + 1))
                    queue.append((importer, d + 1))
                    if len(out) >= limit:
                        break
        return out

    # ── text for the coder ──
    def summary(self, paths: list[str], allow=lambda p: True, max_symbols: int = 6, max_sites: int = 4, max_files: int = 12) -> list[str]:
        """One block per file: its definitions with where they are used elsewhere, then the files that import it. `allow(path)` lets
        the caller drop paths it must not show (ignored or protected files). Only paths and line numbers are returned, never code."""
        lines: list[str] = []
        for path in paths:
            if not allow(path):
                continue
            rows = []
            for d in self.definitions(path)[:max_symbols]:
                sites = [(p, n) for p, n in self.usages(d.name, path, (d.start, d.end)) if allow(p)]
                if not sites:
                    continue
                by_file: dict[str, list[int]] = {}
                for p, n in sites:
                    by_file.setdefault(p, []).append(n)
                shown = []
                for p, ns in list(by_file.items())[:max_sites]:
                    shown.append(f"{p}:{','.join(str(n) for n in ns[:3])}")
                more = len(by_file) - max_sites
                rows.append(f"  {d.qualified} ({d.kind}, lines {d.start}-{d.end}) is also used at: {'; '.join(shown)}"
                            + (f"; and {more} more file(s)" if more > 0 else ""))
            ripple = [(p, d) for p, d in self.impact([path], limit=max_files) if allow(p)]
            if rows or ripple:
                lines.append(path)
                lines += rows
                if ripple:
                    lines.append("  imported by: " + ", ".join(p if d == 1 else f"{p} (via another file)" for p, d in ripple))
        return lines
