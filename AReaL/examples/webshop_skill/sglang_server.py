# SPDX-License-Identifier: MIT

"""Start, reuse, and stop an OpenAI-compatible SGLang server for one model."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _server_ready(base_url: str, api_key: str, timeout_s: float = 3.0) -> bool:
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    try:
        response = requests.get(
            base_url.rstrip("/") + "/v1/models",
            headers=headers,
            timeout=timeout_s,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


@dataclass(frozen=True)
class ServerSpec:
    role: str
    model: str
    base_url: str
    host: str
    port: int
    gpus: str
    server_python: str
    tp_size: int
    dp_size: int
    dtype: str
    mem_fraction_static: float
    context_length: int
    startup_timeout_s: float
    api_key: str
    extra_args: tuple[str, ...]
    start: bool
    reuse_existing: bool
    log_path: Path


class ManagedSGLangServer:
    def __init__(self, spec: ServerSpec) -> None:
        self.spec = spec
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.base_url = spec.base_url

    def __enter__(self) -> str:
        if self.base_url and _server_ready(self.base_url, self.spec.api_key):
            if not self.spec.reuse_existing and self.spec.start:
                raise RuntimeError(
                    f"{self.spec.role} endpoint is already in use at {self.base_url}; "
                    "set reuse_existing=true only when this is the intended model server"
                )
            print(f"[server:{self.spec.role}] using {self.base_url}", flush=True)
            return self.base_url
        if self.base_url and not self.spec.start:
            raise RuntimeError(
                f"{self.spec.role} server is unreachable and managed startup is disabled: "
                f"{self.base_url}"
            )
        if not self.spec.start:
            raise RuntimeError(
                f"{self.spec.role} has neither a base_url nor managed startup"
            )

        port = self.spec.port or _free_port(self.spec.host)
        self.base_url = f"http://{self.spec.host}:{port}"
        cmd = [
            self.spec.server_python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.spec.model,
            "--host",
            self.spec.host,
            "--port",
            str(port),
            "--tp-size",
            str(self.spec.tp_size),
            "--dp-size",
            str(self.spec.dp_size),
            "--dtype",
            self.spec.dtype,
            "--mem-fraction-static",
            str(self.spec.mem_fraction_static),
            "--context-length",
            str(self.spec.context_length),
        ]
        cmd.extend(self.spec.extra_args)
        env = os.environ.copy()
        if self.spec.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.spec.gpus
        self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.spec.log_path.open("a", encoding="utf-8")
        print(
            f"[server:{self.spec.role}] starting model={self.spec.model} "
            f"gpus={self.spec.gpus or 'inherited'} endpoint={self.base_url}",
            flush=True,
        )
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            started = time.monotonic()
            while time.monotonic() - started < self.spec.startup_timeout_s:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"{self.spec.role} server exited with code "
                        f"{self.process.returncode}; see {self.spec.log_path}"
                    )
                if _server_ready(self.base_url, self.spec.api_key):
                    print(f"[server:{self.spec.role}] ready", flush=True)
                    return self.base_url
                time.sleep(2.0)
            raise TimeoutError(
                f"{self.spec.role} server did not become ready within "
                f"{self.spec.startup_timeout_s}s; see {self.spec.log_path}"
            )
        except BaseException:
            self._stop()
            raise

    def _stop(self, timeout_s: float = 30.0) -> None:
        try:
            if self.process is not None:
                process = self.process
                # start_new_session=True makes this PID the owned process group.
                # The leader can exit before its DP workers release their GPUs.
                pgid = process.pid
                print(f"[server:{self.spec.role}] stopping group={pgid}", flush=True)
                try:
                    os.killpg(pgid, signal.SIGTERM)
                    deadline = time.monotonic() + timeout_s
                    while time.monotonic() < deadline:
                        process.poll()  # Reap the leader without ignoring workers.
                        os.killpg(pgid, 0)
                        time.sleep(0.1)
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30.0)
                self.process = None
        finally:
            if self.log_handle is not None and not self.log_handle.closed:
                self.log_handle.close()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self._stop()


def server_spec(
    *,
    role: str,
    section: dict[str, Any],
    model: str,
    output_dir: Path,
    force_no_start: bool,
) -> ServerSpec:
    """Read a model section (base_url, api_key, server.*) into a server spec."""
    server = section.get("server") or {}
    if not isinstance(server, dict):
        raise ValueError(f"{role}.server must be a mapping")
    return ServerSpec(
        role=role,
        model=model,
        base_url=str(section.get("base_url") or "").rstrip("/"),
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 0)),
        gpus=str(server.get("gpus", "") or "").strip(),
        server_python=os.path.expandvars(
            os.path.expanduser(str(server.get("python", sys.executable)))
        ),
        tp_size=int(server.get("tp_size", 1)),
        dp_size=int(server.get("dp_size", 1)),
        dtype=str(server.get("dtype", "bfloat16")),
        mem_fraction_static=float(server.get("mem_fraction_static", 0.8)),
        context_length=int(server.get("context_length", 16384)),
        startup_timeout_s=float(server.get("startup_timeout_s", 1800.0)),
        api_key=str(section.get("api_key") or ""),
        extra_args=tuple(str(item) for item in server.get("extra_args", [])),
        start=bool(server.get("start", True)) and not force_no_start,
        reuse_existing=bool(server.get("reuse_existing", False)),
        log_path=output_dir / "server_logs" / f"{role}.log",
    )
