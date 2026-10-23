"""Server entrypoint for the detector FastAPI service."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from .config import (
    DEFAULT_SERVICE_CONFIG,
    is_loopback_host,
    resolve_auth_token,
    service_config_from_env,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phishing URL Detector local FastAPI service.",
    )
    env_cfg = service_config_from_env()
    parser.add_argument(
        "--host",
        default=env_cfg.host,
        help=f"Host address to bind to (default: {DEFAULT_SERVICE_CONFIG.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=env_cfg.port,
        help=f"Port number to listen on (default: {DEFAULT_SERVICE_CONFIG.port})",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        default=env_cfg.allow_external_binding,
        help="Explicitly permit binding to a non-loopback interface (insecure)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args(argv)

    # Hard security boundary: refuse external binding unless explicitly allowed
    if not args.allow_external and not is_loopback_host(args.host):
        print(
            f"ERROR: Insecure host binding. Refusing to bind to '{args.host}'. "
            "The service must bind strictly to a loopback address (e.g. 127.0.0.1) "
            "unless the --allow-external flag is explicitly provided.",
            file=sys.stderr,
        )
        return 1

    if args.host in ("0.0.0.0", "::"):  # noqa: S104
        print(
            "WARNING: Binding to all network interfaces (0.0.0.0 / ::) exposes the service "
            "to external networks.",
            file=sys.stderr,
        )

    # Show auth status
    active_token = resolve_auth_token(env_cfg)
    if active_token:
        print(f"Authentication active. Token file: {env_cfg.token_file_path}")
    else:
        print("WARNING: Service authentication is disabled (require_auth=False).")

    print(f"Starting Phishing URL Detector Service on http://{args.host}:{args.port}")
    uvicorn.run(
        "detector_service.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
