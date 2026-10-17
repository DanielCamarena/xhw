# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Check a built wheel against actual source, plus the sdist's teaching assets."""

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel, sdist = next(args.dist_dir.glob("*.whl")), next(args.dist_dir.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as z:
        for p in (root/"src/xhw").glob("*.py"):
            assert z.read("xhw/"+p.name) == p.read_bytes(), f"stale wheel: {p.name}"
        assert not any(n.startswith(("docs/", "examples/", "tests/")) for n in z.namelist())
        assert any(n.endswith("/LICENSE") for n in z.namelist())
        assert any(n.endswith("/NOTICE") for n in z.namelist())
    with tarfile.open(sdist) as archive:
        names = {n.split("/", 1)[-1] for n in archive.getnames()}
        for folder in ("docs", "examples"):
            for p in (root/folder).rglob("*"):
                if p.is_file() and p.suffix in (
                    ".md",
                    ".json",
                    ".png",
                    ".svg",
                    ".gif",
                    ".ipynb",
                    ".py",
                    ".txt",
                ):
                    if "__pycache__" not in p.parts:
                        assert p.relative_to(root).as_posix() in names, f"sdist missing {p}"
    print(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (wheel, sdist)}, indent=2))


if __name__ == "__main__":
    main()
