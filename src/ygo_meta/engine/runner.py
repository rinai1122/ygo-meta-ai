"""
Subprocess manager for the ygoinf inference server.

Usage:
    # From CLI:
    python -m ygo_meta.engine.runner start

    # From Python:
    runner = YgoInfRunner()
    runner.start()
    # ... use the server ...
    runner.stop()

Environment variables:
    YGOAGENT_VENV       Path to a venv containing ygo-agent deps (JAX, fastapi, etc.)
                        If set, uvicorn is invoked from that venv's bin/. Useful when
                        JAX <= 0.4.28 conflicts with modern numpy in the main env.
    YGOAGENT_PORT       Port to listen on (default: 3000)
    YGOAGENT_CHECKPOINT Path to .flax_model or .tflite checkpoint file
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx


def _uvicorn_executable() -> str:
    venv = os.environ.get("YGOAGENT_VENV")
    if venv:
        # Unix: bin/uvicorn, Windows: Scripts/uvicorn.exe
        candidates = [
            Path(venv) / "bin" / "uvicorn",
            Path(venv) / "Scripts" / "uvicorn.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        raise FileNotFoundError(f"uvicorn not found in YGOAGENT_VENV={venv}")
    return "uvicorn"


def _ygoinf_dir() -> Path:
    here = Path(__file__).parent
    vendor = here.parent.parent.parent / "vendor" / "ygo-agent" / "ygoinf"
    if not vendor.exists():
        raise FileNotFoundError(
            f"vendor/ygo-agent not found at {vendor}. "
            "Run: git submodule update --init --recursive"
        )
    return vendor


class YgoInfRunner:
    def __init__(self) -> None:
        self._port = int(os.environ.get("YGOAGENT_PORT", "3000"))
        self._process: subprocess.Popen | None = None

    def start(self, wait: bool = True, timeout: float = 30.0) -> None:
        ygoinf_dir = _ygoinf_dir()
        checkpoint = os.environ.get("YGOAGENT_CHECKPOINT", "")

        env = os.environ.copy()
        if checkpoint:
            env["CHECKPOINT"] = checkpoint

        cmd = [
            _uvicorn_executable(),
            "ygoinf.server:app",
            "--host", "127.0.0.1",
            "--port", str(self._port),
        ]

        self._process = subprocess.Popen(
            cmd,
            cwd=str(ygoinf_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if wait:
            self._wait_healthy(timeout)
            print(f"ygoinf server running on port {self._port} (pid={self._process.pid})")

    def _wait_healthy(self, timeout: float) -> None:
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{self._port}/"
        while time.time() < deadline:
            try:
                r = httpx.get(url, timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"ygoinf server did not become healthy within {timeout}s")

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Start/stop the ygoinf inference server")
    parser.add_argument("action", choices=["start"], help="Action to perform")
    args = parser.parse_args()

    if args.action == "start":
        runner = YgoInfRunner()
        runner.start()
        try:
            # Block until Ctrl+C
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            runner.stop()


if __name__ == "__main__":
    main()
