# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from omegaconf import OmegaConf

AREAL_ROOT = Path(__file__).resolve().parents[2]
if str(AREAL_ROOT) not in sys.path:
    sys.path.insert(0, str(AREAL_ROOT))

from examples.webshop_skill.core import (  # noqa: E402
    DEFAULT_ACTOR_MODEL,
    OpenAIChatClient,
    WebShopRuntime,
    fixed_task_split,
)
from examples.webshop_skill.task_split import (  # noqa: E402
    official_train_task_split,
    validate_training_settings,
)


def _expanded_path(value: Any) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _load_config_in_process(
    path: Path, overrides: list[str] | None = None
) -> dict[str, Any]:
    # Match training's inherited YAML and CLI overrides without starting workers.
    from examples.webshop_skill.configs import WebShopSkillGRPOConfig

    from areal.api.cli_args import parse_cli_args, to_structured_cfg

    config, _ = parse_cli_args(["--config", str(path), *(overrides or [])])
    config = to_structured_cfg(config, WebShopSkillGRPOConfig)
    return {"webshop": OmegaConf.to_container(config.webshop, resolve=True)}


def _load_config(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    # Typed AReaL configuration imports distributed/model native runtimes. Resolve
    # it in a disposable interpreter before this process starts the Lucene JVM.
    with tempfile.TemporaryDirectory(prefix="webshop-config-") as directory:
        output = Path(directory) / "config.json"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(path),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--resolve-config-output",
                str(output),
                "--training-overrides",
                *(overrides or []),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError(f"WebShop config subprocess exited {result.returncode}")
        return json.loads(output.read_text())


def create_app(
    runtime: WebShopRuntime, *, task_split: str = "training_holdout"
) -> Flask:
    app = Flask(__name__)
    manifest_cache: dict[tuple[int, int, int, bool], dict[str, Any]] = {}
    manifest_lock = threading.Lock()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "goal_count": runtime.goal_count})

    @app.post("/task-manifest")
    def task_manifest():
        payload = request.get_json(force=True, silent=False) or {}
        requested_split = payload.get("task_split", "training_holdout")
        if requested_split != task_split:
            raise ValueError(
                "requested task_split differs from the environment service"
            )
        key = (
            int(payload.get("train_count", 300)),
            int(payload.get("eval_count", 100)),
            int(payload.get("seed", 1)),
            bool(payload.get("unique_asin", True)),
        )
        with manifest_lock:
            manifest = manifest_cache.get(key)
            if manifest is None:
                if task_split == "official_train":
                    validate_training_settings(
                        {
                            "task_split": task_split,
                            "human_goals": runtime.human_goals,
                            "num_products": runtime.num_products,
                            "environment_seed": runtime.environment_seed,
                            "train_task_count": key[0],
                            "eval_task_count": key[1],
                            "unique_asin_split": key[3],
                        }
                    )
                    manifest = official_train_task_split(
                        runtime.server.goals, eval_count=key[1], seed=key[2]
                    )
                else:
                    manifest = fixed_task_split(
                        runtime.server.goals,
                        train_count=key[0],
                        eval_count=key[1],
                        seed=key[2],
                        unique_asin=key[3],
                    )
                manifest_cache[key] = manifest
        return jsonify(manifest)

    @app.post("/task-contexts")
    def task_contexts():
        payload = request.get_json(force=True, silent=False) or {}
        task_indices = payload.get("task_indices") or []
        if not isinstance(task_indices, list):
            raise ValueError("task_indices must be a list")
        return jsonify(
            {
                "contexts": [
                    runtime.task_context(int(task_index)) for task_index in task_indices
                ]
            }
        )

    @app.post("/rollout")
    def rollout():
        payload = request.get_json(force=True, silent=False) or {}
        actor = OpenAIChatClient(
            base_url=str(payload["actor_base_url"]),
            model=str(payload.get("actor_model") or DEFAULT_ACTOR_MODEL),
            api_key=str(payload.get("actor_api_key") or ""),
            timeout_s=float(payload.get("actor_timeout_s", 120.0)),
            max_retries=int(payload.get("actor_max_retries", 3)),
        )
        episode = runtime.rollout(
            task_index=int(payload["task_index"]),
            actor=actor,
            skill=str(payload.get("skill") or ""),
            condition=str(payload.get("condition") or "webshop"),
            max_steps=int(payload.get("max_steps", 100)),
            memory_window=int(payload.get("memory_window", 5)),
            observation_char_limit=int(payload.get("observation_char_limit", 5000)),
            max_clickables=int(payload.get("max_clickables", 60)),
            actor_temperature=float(payload.get("actor_temperature", 0.0)),
            actor_max_tokens=int(payload.get("actor_max_tokens", 128)),
            actor_seed=int(payload.get("actor_seed", 1)),
            invalid_action_retries=int(payload.get("invalid_action_retries", 1)),
            success_threshold=float(payload.get("success_threshold", 0.999999)),
            session_key=str(payload.get("session_key") or ""),
        )
        return jsonify(episode)

    @app.errorhandler(Exception)
    def handle_error(exc: Exception):
        return (
            jsonify(
                {
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WebShop simulator service for AReaL training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--repo-root")
    parser.add_argument("--products-file")
    parser.add_argument("--attributes-file")
    parser.add_argument("--human-attributes-file")
    parser.add_argument("--search-index")
    parser.add_argument("--resolve-config-output", type=Path)
    parser.add_argument("--training-overrides", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolve_config_output:
        config = _load_config_in_process(
            _expanded_path(args.config), args.training_overrides
        )
        args.resolve_config_output.write_text(json.dumps(config))
        return
    print("[webshop-env] resolving typed config in isolated process", flush=True)
    config = _load_config(_expanded_path(args.config), args.training_overrides)
    print("[webshop-env] config resolved; initializing simulator/JVM", flush=True)
    webshop = config.get("webshop") or {}
    validate_training_settings(webshop)
    for key in (
        "repo_root",
        "products_file",
        "attributes_file",
        "human_attributes_file",
        "search_index",
    ):
        override = getattr(args, key)
        if override is not None:
            webshop[key] = override
    runtime = WebShopRuntime(
        repo_root=_expanded_path(webshop["repo_root"]),
        products_file=_expanded_path(webshop["products_file"]),
        attributes_file=_expanded_path(webshop["attributes_file"]),
        human_attributes_file=_expanded_path(webshop["human_attributes_file"]),
        search_index=_expanded_path(webshop["search_index"]),
        human_goals=bool(webshop.get("human_goals", False)),
        num_products=(
            int(webshop["num_products"])
            if webshop.get("num_products") is not None
            else None
        ),
        observation_mode=str(webshop.get("observation_mode", "text_rich")),
        environment_seed=webshop.get("environment_seed"),
    )
    print(
        f"[webshop-env] ready goals={runtime.goal_count} host={args.host} port={args.port}",
        flush=True,
    )
    create_app(runtime, task_split=webshop.get("task_split", "training_holdout")).run(
        host=args.host,
        port=args.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
