# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Portable fingerprints of the installed library and the example source being run."""

import hashlib
import subprocess
from pathlib import Path

import xhw


def code_fingerprint():
    """Identify actual imported bytes, rather than assuming a version identifies a build."""
    package = Path(xhw.__file__).resolve().parent
    modules = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))
    }
    examples = Path(__file__).resolve().parent
    helpers = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(examples.glob("*.py"))
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=examples.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=examples.parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "package_modules_sha256": modules,
        "example_sources_sha256": helpers,
        "example_base_revision": revision,
        "example_worktree_dirty": dirty,
    }
