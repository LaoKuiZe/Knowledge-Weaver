# SPDX-License-Identifier: MIT

"""Start, or attach to, an OpenAI-compatible SGLang server for one run."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import requests

from areal.infra.utils.proc import kill_process_tree


@dataclass(frozen=True)
class ServerSpec:
    model_path: str
    host: str
    port: int
    gpus: str
    tp_size: int
    dp_size: int
    dtype: str
    mem_fraction_static: float
    context_length: int
    startup_timeout_s: float
    api_key: str
    extra_args: tuple[str, ...]
    enabled: bool

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def gpu_count(gpus: str) -> int:
    items = [item.strip() for item in str(gpus or "").split(",") if item.strip()]
    return max(1, len(items))


def _server_ready(base_url: str, api_key: str = "", timeout_s: float = 5.0) -> bool:
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


def _wait_for_server(spec: ServerSpec, process: Any) -> None:
    start = time.monotonic()
    while time.monotonic() - start < spec.startup_timeout_s:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "SGLang server exited before becoming ready "
                f"with code {return_code}: {spec.base_url}"
            )
        if _server_ready(spec.base_url, api_key=spec.api_key):
            return
        time.sleep(2.0)
    raise TimeoutError(
        f"SGLang server did not become ready within "
        f"{spec.startup_timeout_s}s: {spec.base_url}"
    )


@contextmanager
def managed_sglang_server(spec: ServerSpec):
    """Yield the base URL, reusing a reachable server or starting one for the block."""
    if _server_ready(spec.base_url, api_key=spec.api_key):
        print(f"[server] using existing server at {spec.base_url}", flush=True)
        yield spec.base_url
        return
    if not spec.enabled:
        raise RuntimeError(
            f"Server is disabled and no server is reachable at {spec.base_url}"
        )

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        spec.model_path,
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--tp-size",
        str(spec.tp_size),
        "--dp-size",
        str(spec.dp_size),
        "--dtype",
        spec.dtype,
        "--mem-fraction-static",
        str(spec.mem_fraction_static),
        "--context-length",
        str(spec.context_length),
    ]
    cmd.extend(spec.extra_args)

    env = os.environ.copy()
    if spec.gpus:
        env["CUDA_VISIBLE_DEVICES"] = spec.gpus
    print(f"[server] starting: {' '.join(cmd)}", flush=True)
    if spec.gpus:
        print(f"[server] CUDA_VISIBLE_DEVICES={spec.gpus}", flush=True)
    process = subprocess.Popen(cmd, env=env)
    try:
        _wait_for_server(spec, process)
        print(f"[server] ready at {spec.base_url}", flush=True)
        yield spec.base_url
    finally:
        print(f"[server] stopping pid={process.pid}", flush=True)
        kill_process_tree(process.pid)
