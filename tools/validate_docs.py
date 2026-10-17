# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline local-link and saved-notebook structural checks. Run from any directory."""

import json
import re
from pathlib import Path
from urllib.parse import unquote


def anchors(path):
    text = path.read_text(encoding="utf8")
    result, seen = set(), {}
    for heading in re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M):
        heading = re.sub(r"<[^>]*>", "", heading).strip().lower()
        slug = re.sub(r"[^\w\s-]", "", heading).replace(" ", "-")
        number = seen.get(slug, 0)
        seen[slug] = number + 1
        result.add(slug + (f"-{number}" if number else ""))
    result.update(re.findall(r'''id=["']([^"']+)["']''', text))
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    failures, count = [], 0
    for p in [root/"README.md", *sorted((root/"docs").rglob("*.md")),
              *sorted((root/"examples").glob("*.ipynb"))]:
        text = p.read_text(encoding="utf8")
        if p.suffix == ".ipynb":
            text = "\n".join("".join(c["source"]) for c in json.loads(text)["cells"]
                             if c["cell_type"] == "markdown")
        for target in re.findall(r'!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)', text):
            if "://" in target or target.startswith(("mailto:", "data:")):
                continue
            target = unquote(target.strip("<>"))
            filename = target.split("#", 1)[0]
            dest = (p.parent/filename).resolve() if filename else p.resolve()
            count += 1
            if not dest.exists():
                failures.append(f"{p.relative_to(root)}: missing {target}")
            elif "#" in target and dest.suffix == ".md":
                anchor = target.split("#", 1)[1]
                if anchor and anchor not in anchors(dest):
                    failures.append(f"{p.relative_to(root)}: missing anchor {target}")
    for p in sorted((root/"examples").glob("*.ipynb")):
        import nbformat

        nb = nbformat.read(p, as_version=4)
        nbformat.validate(nb)
        for c in nb.cells:
            if c.cell_type == "code":
                if c.execution_count is None:
                    failures.append(f"{p.name}: unexecuted cell")
                if any(o.output_type == "error" for o in c.outputs):
                    failures.append(f"{p.name}: error output")
        widgets = nb.metadata.get("widgets", {}).get(
            "application/vnd.jupyter.widget-state+json", {}).get("state", {})
        for widget in widgets.values():
            outputs = widget.get("state", {}).get("outputs", [])
            if any(o.get("output_type") == "error" for o in outputs):
                failures.append(f"{p.name}: error in saved widget output")
    print(json.dumps({"local_links": count, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
