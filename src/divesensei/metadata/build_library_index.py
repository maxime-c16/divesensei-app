#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from divesensei.metadata.ui_contract import build_ui_library_index


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else (Path.cwd() / "outputs")
    output_path = Path(argv[1]).resolve() if len(argv) > 1 else (root / "ui_library_index.json")

    manifests = []
    for manifest_path in sorted(root.rglob("ui_session_manifest.json")):
        manifests.append(json.loads(manifest_path.read_text()))

    index = build_ui_library_index(manifests)
    output_path.write_text(json.dumps(index, indent=2))
    print(json.dumps({"sessions": len(manifests), "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
