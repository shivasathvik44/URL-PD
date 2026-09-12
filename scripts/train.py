#!/usr/bin/env python3
"""Thin wrapper: `python scripts/train.py ...` == `python -m detector.train ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detector.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
