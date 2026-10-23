"""Enable running the service via `python -m detector_service`."""

from __future__ import annotations

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
