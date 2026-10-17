# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""The ```python blocks of README.md and docs/PLOTTING.md run, in order, as written.

Each file's blocks share one namespace, as a reader running them in one session would.
Illustrative fragments that are not meant to run use another fence (```text, ```bash).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def python_blocks(path: Path) -> list[str]:
    return BLOCK.findall(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("document", ["README.md", "docs/PLOTTING.md"])
def test_documented_code_runs(document, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)  # figures written by the snippets land here
    monkeypatch.setattr(sys, "path", [str(EXAMPLES), *sys.path])
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    blocks = python_blocks(ROOT / document)
    assert blocks, f"{document} has no python blocks"
    namespace: dict = {"__name__": "__docs__"}
    try:
        for index, code in enumerate(blocks):
            exec(compile(code, f"{document}#python-block-{index}", "exec"), namespace)
    finally:
        plt.close("all")


def test_readme_quick_start_shows_a_split_and_a_merge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [str(EXAMPLES), *sys.path])
    namespace: dict = {"__name__": "__docs__"}
    block = next(code for code in python_blocks(ROOT / "README.md")
                 if "xhw.build_lineage(" in code)
    exec(compile(block, "README.md#quick-start", "exec"), namespace)
    graph, mask = namespace["graph"], namespace["mask"]
    mask.check_invariants()
    namespace["catalog"].check_against(mask)
    namespace["objects"].check_against(mask)
    graph.check_against(namespace["objects"])
    assert int(graph.dataset.is_split.sum()) >= 1
    assert int(graph.dataset.is_merge.sum()) >= 1
