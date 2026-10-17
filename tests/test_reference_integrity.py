# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Every independent input/output file is present and matches its registered SHA-256."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent / "reference"
FOLDERS = ("marine", "climdex", "stl", "ssa", "objects", "lineage", "deeptrack")


@pytest.mark.parametrize("folder", FOLDERS)
def test_registered_reference_files_are_complete_and_intact(folder):
    root = ROOT / folder
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]
    for name, expected in manifest["files"].items():
        path = root / name
        assert path.resolve().is_relative_to(root.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
