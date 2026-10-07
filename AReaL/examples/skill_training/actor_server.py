# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import requests

from examples.skill_training.configs import FrozenActorServerConfig

from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging

logger = logging.getLogger("SkillTrainingActorServer")


def _model_identifiers(value: str) -> set[str]:
    """Keep full model IDs and normalize local paths without basename matching."""

    identifiers = {value} if value else set()
    path = Path(value).expanduser()
    if value and (path.is_absolute() or value.startswith(".") or path.exists()):
        identifiers.add(str(path.resolve()))
    return identifiers


def _verify_server_model(config: FrozenActorServerConfig) -> None:
    expected = _model_identifiers(config.model_path)
    extra_args = list(config.extra_args)
    served_alias = None
    for index, argument in enumerate(extra_args):
        if argument == "--served-model-name":
            if index + 1 >= len(extra_args) or extra_args[index + 1].startswith("--"):
                raise ValueError(
                    "actor_server.extra_args needs a value after --served-model-name"
                )
            served_alias = extra_args[index + 1]
        elif argument.startswith("--served-model-name="):
            served_alias = argument.split("=", 1)[1]
    if served_alias is not None:
        if not served_alias:
            raise ValueError(
                "actor_server.extra_args needs a nonempty --served-model-name"
            )
        expected.add(served_alias)
    headers = {"Authorization": "Bearer " + config.api_key} if config.api_key else {}
    response = requests.get(
        config.base_url.rstrip("/") + "/v1/models", headers=headers, timeout=10.0
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = (
        {
            item["id"]
            for item in rows
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        }
        if isinstance(rows, list)
        else set()
    )
    served = set().union(*(_model_identifiers(value) for value in model_ids))
    if not served.intersection(expected):
        raise RuntimeError(
            f"Frozen actor server at {config.base_url} serves {sorted(model_ids)}, "
            f"expected {sorted(expected)}. Stop the server on this port or set "
            "actor_server.model_path to the intended frozen executor; for an "
            "explicit served alias, configure --served-model-name in actor_server.extra_args."
        )


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


def _wait_for_server(
    config: FrozenActorServerConfig, process: subprocess.Popen
) -> None:
    start = time.monotonic()
    while time.monotonic() - start < config.startup_timeout_s:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Frozen actor server exited with code {return_code} before "
                f"becoming ready: {config.base_url}"
            )
        if _server_ready(config.base_url, api_key=config.api_key):
            return
        time.sleep(2.0)
    raise TimeoutError(
        "Frozen actor server did not become ready within "
        f"{config.startup_timeout_s}s: {config.base_url}"
    )


def _start_actor_server(
    config: FrozenActorServerConfig,
) -> subprocess.Popen | None:
    if _server_ready(config.base_url, api_key=config.api_key):
        _verify_server_model(config)
        logger.info("Using existing frozen actor server at %s", config.base_url)
        return None

    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tp-size",
        str(config.tensor_parallel_size),
        "--dp-size",
        str(config.data_parallel_size),
        "--dtype",
        config.dtype,
        "--mem-fraction-static",
        str(config.mem_fraction_static),
        "--context-length",
        str(config.context_length),
        *config.extra_args,
    ]
    environment = os.environ.copy()
    if config.cuda_visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    logger.info("Starting frozen actor server: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_server(config, process)
        _verify_server_model(config)
    except Exception:
        kill_process_tree(process.pid)
        raise
    logger.info("Frozen actor server ready at %s", config.base_url)
    return process


@contextmanager
def frozen_actor_server(config: FrozenActorServerConfig) -> Iterator[str]:
    """Start or reuse the frozen actor server for any skill benchmark."""

    process = _start_actor_server(config)
    try:
        yield config.base_url
    finally:
        if process is not None:
            logger.info("Stopping frozen actor server pid=%s", process.pid)
            kill_process_tree(process.pid)
