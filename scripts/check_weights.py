#!/usr/bin/env python3
"""Verify separately downloaded checkpoints without loading them with torch."""

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "docs" / "weights_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check training and ablation models in addition to inference models.",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = [
        entry for entry in manifest["files"]
        if args.all or entry["required_for_inference"]
    ]
    failures = 0
    for entry in files:
        path = PROJECT_ROOT / entry["path"]
        if not path.is_file():
            print("MISSING: " + entry["path"])
            failures += 1
            continue
        digest = hashlib.sha256()
        with path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            print("CHECKSUM MISMATCH: " + entry["path"])
            failures += 1
        else:
            print("OK: " + entry["path"])
    if failures:
        print("See docs/PRETRAINED_MODELS.md for installation instructions.")
        return 1
    print("Verified {} checkpoint(s).".format(len(files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
