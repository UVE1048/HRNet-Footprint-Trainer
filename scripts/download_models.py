#!/usr/bin/env python3
"""Download the reference HRNet-W48 checkpoint used by this training recipe.

The weights (~542 MB) live as a GitHub Release asset on the sibling
inference project, so they are only stored once. Same file, same SHA256.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Hosted on Footprint-Creater's release to avoid duplicating a 542 MB asset
URL = "https://github.com/UVE1048/Footprint-Creater/releases/download/models-v1/hrnet_w48_iter_6000.pth"
SHA256 = "73db3f97998fc62cbbcb83897925d1c6ffaef2e1dadd78b5b7f90267dab5cf18"

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "work_dirs" / "hrnet_w48_footprint" / "iter_40000.pth"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--check", action="store_true", help="only verify the local file")
    args = parser.parse_args()

    if TARGET.exists() and not args.force:
        actual = sha256_of(TARGET)
        if actual == SHA256:
            print(f"OK   {TARGET} (sha256 verified)")
            return 0
        print(f"MISMATCH {TARGET}\n  expected {SHA256}\n  actual   {actual}")
        if args.check:
            return 1
        print("  re-downloading...")

    if args.check:
        print(f"MISS {TARGET}")
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_suffix(".pth.part")
    print(f"Downloading {URL}")

    def _progress(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(blocks * block_size, total)
        pct = done * 100 / total
        bar = "#" * int(pct // 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  {done / 1e6:7.1f}/{total / 1e6:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(URL, tmp, reporthook=_progress)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  x download failed: {exc}")
        tmp.unlink(missing_ok=True)
        return 1

    print()
    actual = sha256_of(tmp)
    if actual != SHA256:
        print(f"  x checksum mismatch\n    expected {SHA256}\n    actual   {actual}")
        tmp.unlink(missing_ok=True)
        return 1

    tmp.replace(TARGET)
    print(f"  + saved to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
