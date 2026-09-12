#!/usr/bin/env python3
"""Run the detector straight from a checkout, without installing it.

Identical to `python -m detector` - see `src/detector/cli.py` for the commands:

    python main.py analyze "https://example.com/login"
    python main.py analyze --json URL
    python main.py features --layer1-only URL
    python main.py train --sample
    python main.py evaluate --csv data/raw/sample/sample_dataset.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from detector.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
