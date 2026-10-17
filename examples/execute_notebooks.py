# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Execute each tutorial from a fresh kernel using this Python installation.

No data files and no downloads: both tutorials generate their input. Outputs are saved in
place, with PNGs and portable run reports in --output-dir.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path


def portable_warnings(notebook) -> None:
    """Keep warning messages and line numbers while removing machine paths."""
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream" and output.name == "stderr":
                output.text = re.sub(
                    r"(?m)^(?:[A-Za-z]:[\\/]|/)[^\n]*[\\/]([^\\/\n]+\.py:\d+:)",
                    r"\1",
                    output.text,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/notebooks"))
    parser.add_argument("--case", choices=("synthetic", "synthetic_peru"))
    args = parser.parse_args()
    try:
        import nbformat
        from jupyter_client import KernelManager
        from nbclient import NotebookClient
    except ImportError as exc:
        raise ImportError("Install examples/requirements.txt to execute notebooks.") from exc

    # Absolute paths are passed only through the process environment, never saved in notebooks.
    os.environ["XHW_OUTPUT_DIR"] = str(args.output_dir.resolve())
    directory = Path(__file__).resolve().parent
    notebooks = {
        "synthetic": "synthetic.ipynb",
        "synthetic_peru": "synthetic_peru.ipynb",
    }
    executions = []
    for case, name in notebooks.items():
        if args.case and args.case != case:
            continue
        path = directory / name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        start = time.perf_counter()
        client = NotebookClient(
            notebook, km=manager, timeout=1800, resources={"metadata": {"path": str(directory)}}
        )
        try:
            client.execute()
            notebook.metadata["xhw"]["output_status"] = "executed from a fresh kernel"
            notebook.metadata["xhw"]["executed_package_version"] = importlib.metadata.version("xhw")
        finally:
            # An externally supplied manager must be shut down explicitly, also after errors.
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        portable_warnings(notebook)
        path.write_text(nbformat.writes(notebook) + "\n", encoding="utf-8", newline="\n")
        seconds = time.perf_counter() - start
        images = sum(
            "image/png" in o.get("data", {}) for c in notebook.cells for o in c.get("outputs", [])
        )
        executions.append(
            {
                "case": case,
                "seconds": round(seconds, 2),
                "png_outputs": images,
                "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
            }
        )
        print(
            f"{case}: executed from a fresh kernel; {images} PNG outputs; {seconds:.1f} s",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "notebook_execution.json").write_text(
        json.dumps(executions, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
