"""Collect genuine no-skill Qwen3.5-4B WebShop rollouts from the training split."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

from examples.webshop_skill.core import (
    NO_SKILL_TEXT,
    atomic_write_json,
    is_usable_webshop_source_episode,
)

from webshop_generality.run_semantic_skillbank_eval import (
    Environment,
    digest,
    output_lock,
    read_json,
    select_tasks,
    validate_episode,
)
from webshop_generality.skillbank_generation import (
    VERSION,
    guarded_contract,
    model_endpoint,
    model_location,
    path_from_root,
    pool_protocol,
    read_pool,
    request_model,
    workflow_adapter,
)


def collect_episodes(
    env, settings, config, output_dir, manifest, tasks, signature, url, key
):
    """Shared training payload builder; persist every completed task immediately."""
    workflow = workflow_adapter(config)
    actor = settings["trace_actor"]
    workflow.actor_model = request_model(actor)
    workflow.actor_base_url = url
    workflow.actor_api_key = key
    options = settings["collection"]
    workers = int(options["workers"])
    max_attempts = int(options.get("max_attempts_per_task", 1))
    if workers < 1 or max_attempts < 1:
        raise ValueError(
            "collection workers and max_attempts_per_task must be positive"
        )
    episodes = {}
    pending = []
    for position, task in enumerate(tasks):
        index = task["task_index"]
        cached = read_json(output_dir / "episodes" / f"task_{index:06d}.json")
        if cached is not None:
            validate_episode(cached, task, success_threshold=workflow.success_threshold)
            if cached.get("pool_signature") != signature:
                raise ValueError(
                    "Cached episode belongs to a different trace collection"
                )
            episodes[index] = cached
        else:
            pending.append((position, task))

    def publish():
        usable = [
            ep for ep in episodes.values() if is_usable_webshop_source_episode(ep)
        ]
        successes = sum(bool(ep["success"]) for ep in usable)
        summary = {
            "version": VERSION,
            "status": "complete" if len(episodes) == len(tasks) else "incomplete",
            "actor_model": actor["model"],
            "pool_signature": signature,
            "protocol": pool_protocol(config),
            "task_manifest": manifest,
            "tasks": tasks,
            "requested": len(tasks),
            "completed": len(episodes),
            "usable": len(usable),
            "usable_successes": successes,
            "usable_failures": len(usable) - successes,
            "episode_pattern": "episodes/task_{task_index:06d}.json",
        }
        atomic_write_json(output_dir / "pool.json", summary)
        atomic_write_json(
            output_dir / "summary.json",
            {
                key: value
                for key, value in summary.items()
                if key not in {"task_manifest", "tasks", "protocol"}
            },
        )

    def run_one(position, task):
        index = task["task_index"]
        # Matches workflow._run_episodes; WebShopRuntime also applies task/step offsets.
        seed = int(options["seed"]) + position * 997
        attempt_dir = output_dir / "errors" / f"task_{index:06d}"
        offset = len(list(attempt_dir.glob("attempt_*.json")))
        for attempt in range(max_attempts):
            try:
                episode = env.rollout(
                    workflow._rollout_payload(
                        task_index=index,
                        skill=NO_SKILL_TEXT,
                        condition="train_no_skill",
                        seed=seed,
                        session_key=f"trace_pool_{uuid4().hex}",
                    )
                )
                validate_episode(
                    episode, task, success_threshold=workflow.success_threshold
                )
                episode.update(
                    pool_signature=signature,
                    actor_model=actor["model"],
                    actor_seed_base=seed,
                )
                atomic_write_json(
                    output_dir / "episodes" / f"task_{index:06d}.json", episode
                )
                return episode
            except Exception as exc:
                atomic_write_json(
                    attempt_dir / f"attempt_{offset + attempt + 1:03d}.json",
                    {
                        "task_index": index,
                        "error_type": type(exc).__name__,
                        "actor_seed_base": seed,
                    },
                )
        return None

    publish()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in as_completed(
            [executor.submit(run_one, pos, task) for pos, task in pending]
        ):
            episode = future.result()
            if episode is not None:
                episodes[episode["task_index"]] = episode
            publish()
            print(f"[trace pool] completed={len(episodes)}/{len(tasks)}", flush=True)
    if len(episodes) != len(tasks):
        raise RuntimeError(
            "Trace pool incomplete; inspect errors/ and re-run collect to resume missing tasks"
        )
    print(f"Real trace pool: {output_dir / 'pool.json'}", flush=True)


def collect(settings, config):
    output_dir = path_from_root(settings["pool_dir"])
    actor = settings["trace_actor"]
    if actor["model"] != "Qwen/Qwen3.5-4B" or actor["backend"] == "remote_api":
        raise ValueError(
            "Trace collection requires a locally deployed Qwen/Qwen3.5-4B actor"
        )
    protocol = pool_protocol(config)
    environment = {**protocol, **settings.get("environment", {})}
    environment["env_service_url"] = os.path.expandvars(
        str(environment.get("env_service_url") or "")
    )
    if "$" in environment["env_service_url"]:
        raise ValueError(
            "Set the environment variable used by environment.env_service_url"
        )
    allowed = {
        "repo_root",
        "products_file",
        "attributes_file",
        "human_attributes_file",
        "search_index",
        "env_service_url",
        "rollout_timeout_s",
    }
    if set(settings.get("environment", {})) - allowed:
        raise ValueError(
            "Change task/rollout settings in training_overrides, not environment"
        )
    if not environment.get("env_service_url"):
        for name in allowed - {"env_service_url", "rollout_timeout_s"}:
            environment[name] = str(
                path_from_root(environment.get(name) or config.webshop[name])
            )
            if not Path(environment[name]).exists():
                raise FileNotFoundError(
                    f"Missing WebShop runtime/data path: {environment[name]}"
                )
    with output_lock(output_dir):
        env = Environment(environment)
        saved_manifest = settings["collection"].get("task_manifest")
        manifest = select_tasks(
            env, protocol, path_from_root(saved_manifest) if saved_manifest else None
        )
        # A saved manifest may select a different split; it must still match the YAML.
        if (
            manifest["seed"] != protocol["task_seed"]
            or len(manifest["train_indices"]) != protocol["train_task_count"]
            or len(manifest["eval_indices"]) != protocol["eval_task_count"]
            or manifest["unique_asin"] != protocol["unique_asin_split"]
        ):
            raise ValueError(
                "Supplied task manifest does not match the training config"
            )
        tasks = env.contexts(manifest["train_indices"])
        if [task["task_index"] for task in tasks] != manifest["train_indices"]:
            raise ValueError("Training task contexts are incomplete or reordered")
        contract = {
            "version": VERSION,
            "actor_model": actor["model"],
            "model_path": model_location(actor),
            "request_model": actor.get("request_model"),
            "protocol": protocol,
            "task_manifest": manifest,
            "tasks": tasks,
            "seed": int(settings["collection"]["seed"]),
        }
        guarded_contract(output_dir, contract)
        atomic_write_json(output_dir / "task_manifest.json", manifest)
        previous = read_json(output_dir / "pool.json")
        if previous and previous.get("status") == "complete":
            read_pool(output_dir, config)
            print(
                f"Trace pool already complete: {output_dir / 'pool.json'}", flush=True
            )
            return
        with model_endpoint(actor, output_dir, role="trace_actor") as (url, key):
            collect_episodes(
                env,
                settings,
                config,
                output_dir,
                manifest,
                tasks,
                digest(contract),
                url,
                key,
            )
