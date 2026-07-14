#!/usr/bin/env python3
"""Clone third-party repositories at the audited commits in the manifest."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="third_party/manifest.yaml")
    parser.add_argument("--root", default="third_party")
    args = parser.parse_args()
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest["repositories"].items():
        target = root / name
        if not (target / ".git").exists():
            run("git", "clone", "--filter=blob:none", spec["url"], str(target))
        run("git", "fetch", "--depth", "1", "origin", spec["commit"], cwd=target)
        run("git", "checkout", "--detach", spec["commit"], cwd=target)


if __name__ == "__main__":
    main()
