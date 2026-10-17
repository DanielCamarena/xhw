# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Integrity-checked, independent numerical references; missing files are test failures."""

import hashlib
import json
from pathlib import Path

import numpy as np


def load_npz(folder: str, filename: str):
    """Load numeric arrays without pickle after checking the registered SHA-256."""
    root = Path(__file__).parent / "reference" / folder
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    path = root / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][filename]
    return np.load(path, allow_pickle=False)
