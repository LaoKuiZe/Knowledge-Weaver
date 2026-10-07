"""Frozen, task-paired top-k skillbank evaluation with WebShop's existing actor."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import subprocess
import time
import uuid
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests
import yaml

from alfworld_generality.semantic_skillbank import (
    build_retrieval_query,
    embed_texts,
    format_retrieved_skill_text,
    load_skillbank,
    retrieve_top_skills,
)
from examples.webshop_skill.core import (
    NO_SKILL_TEXT,
    OpenAIChatClient,
    WebShopRuntime,
    atomic_write_json,
    episode_metrics,
    fixed_task_split,
    paired_condition_metrics,
)
from examples.webshop_skill.sglang_server import ManagedSGLangServer, server_spec
from webshop_generality.official_test_split import (
    task_manifest as official_test_manifest,
    validate_settings as validate_official_test_settings,
)

VERSION = "webshop_semantic_skillbank_v1"
SCHEDULER = "flattened_k_task_queue_v1"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "config/webshop_semantic_skillbank_eval.yaml"
)


def digest(value: Any) -> str:
    """Hash the small evaluation contract, never scan a dataset or model tree."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parse_top_k(value: str, count: int) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError("top-k values must be nonempty and unique")
    if min(values) < 0 or max(values) > count:
        raise ValueError(f"top-k values must be between 0 and {count}")
    return values


def endpoint_root(value: str) -> str:
    return str(value).rstrip("/").removesuffix("/v1")


class Environment:
    """Share one simulator/index, or reuse the existing training env service."""

    def __init__(self, config: dict[str, Any]):
        self.settings = dict(config)
        if config.get("task_split") == "official_test_subset":
            validate_official_test_settings(config)
        self.url = str(config.get("env_service_url") or "").rstrip("/")
        self.timeout = float(config.get("rollout_timeout_s", 7200))
        self.runtime = None
        if not self.url:
            # WebShop samples prices before its fixed goal shuffle. Seed those
            # draws so every bank sees the same prices, without changing the
            # caller's RNG.
            state = random.getstate()
            environment_seed = config.get("environment_seed")
            if environment_seed is not None:
                random.seed(int(environment_seed))
            try:
                self.runtime = WebShopRuntime(
                    **{
                        key: Path(config[key]).expanduser().resolve()
                        for key in (
                            "repo_root",
                            "products_file",
                            "attributes_file",
                            "human_attributes_file",
                            "search_index",
                        )
                    },
                    human_goals=bool(config.get("human_goals", False)),
                    num_products=config.get("num_products"),
                    observation_mode=str(config.get("observation_mode", "text_rich")),
                )
            finally:
                if environment_seed is not None:
                    random.setstate(state)

    def post(self, route: str, payload: dict[str, Any]) -> Any:
        response = requests.post(self.url + route, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def manifest(self, settings: dict[str, Any]) -> dict[str, Any]:
        if self.url:
            return self.post("/task-manifest", settings)
        return fixed_task_split(self.runtime.server.goals, **settings)

    def contexts(self, indices: list[int]) -> list[dict[str, Any]]:
        if self.url:
            return self.post("/task-contexts", {"task_indices": indices})["contexts"]
        return [self.runtime.task_context(index) for index in indices]

    def rollout(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.url:
            return self.post("/rollout", payload)
        options = dict(payload)
        actor = OpenAIChatClient(
            base_url=options.pop("actor_base_url"),
            model=options.pop("actor_model"),
            api_key=options.pop("actor_api_key"),
            timeout_s=options.pop("actor_timeout_s"),
            enable_thinking=bool(options.pop("actor_enable_thinking", False)),
        )
        return self.runtime.rollout(actor=actor, **options)


def compare_saved_tasks(saved, current):
    """Require a saved training manifest to select the same tasks and goals."""
    for split in ("train", "eval"):
        key = f"{split}_indices"
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"saved training manifest differs from current environment at {key}"
            )
        old_rows, new_rows = saved.get(f"{split}_tasks", []), current[f"{split}_tasks"]
        if len(old_rows) != len(new_rows):
            raise ValueError(f"saved training manifest differs at {split} task count")
        for old, new in zip(old_rows, new_rows, strict=True):
            if old != new:
                raise ValueError(
                    f"saved training manifest differs from current environment at {split} task {new['index']}"
                )


def select_tasks(env: Environment, config: dict[str, Any], saved_path: Path | None):
    split = config.get("task_split", "training_holdout")
    if split == "official_test_subset":
        validate_official_test_settings(config)
        return official_test_manifest(env.runtime.server.goals, config, saved_path)
    if split != "training_holdout":
        raise ValueError(f"unknown task_split: {split}")
    saved = None
    if saved_path:
        saved = read_json(saved_path)
        if not isinstance(saved, dict) or not saved.get("eval_indices"):
            raise ValueError(
                "task manifest must be a WebShop training task_manifest.json"
            )
        settings = {
            "train_count": len(saved["train_indices"]),
            "eval_count": len(saved["eval_indices"]),
            "seed": int(saved["seed"]),
            "unique_asin": bool(saved["unique_asin"]),
        }
    else:
        settings = {
            "train_count": int(config["train_task_count"]),
            "eval_count": int(config["eval_task_count"]),
            "seed": int(config["task_seed"]),
            "unique_asin": bool(config["unique_asin_split"]),
        }
    manifest = env.manifest(settings)
    for split in ("train", "eval"):
        indices = manifest[f"{split}_indices"]
        records = manifest[f"{split}_tasks"]
        if len(indices) != settings[f"{split}_count"] or len(set(indices)) != len(
            indices
        ):
            raise ValueError(f"invalid {split} task count or duplicate task index")
        if [int(row["index"]) for row in records] != indices:
            raise ValueError(f"{split} task records do not match their indices")
    if set(manifest["train_indices"]) & set(manifest["eval_indices"]):
        raise ValueError("train/eval task overlap")
    if settings["unique_asin"]:
        train_asins = {row["asin"] for row in manifest["train_tasks"] if row["asin"]}
        eval_asins = [row["asin"] for row in manifest["eval_tasks"] if row["asin"]]
        if train_asins.intersection(eval_asins) or len(set(eval_asins)) != len(
            eval_asins
        ):
            raise ValueError("train/eval ASIN overlap or duplicate eval ASIN")
    if saved is not None:
        compare_saved_tasks(saved, manifest)
    return manifest


def prepare_retrieval(env, manifest, skills, config, output_dir):
    embedding = config["embedding"]
    # Store a full ranking of this small bank so extending k preserves all prefixes.
    signature = digest(
        {
            "version": VERSION,
            "tasks": manifest["eval_tasks"],
            "environment": config["webshop"],
            "skills": [skill.manifest() for skill in skills],
            "embedding_model": embedding["model"],
            "embedding_max_length": embedding["max_length"],
        }
    )
    path = output_dir / "retrieval_manifest.json"
    cached = read_json(path)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        rows = cached.get("tasks", [])
        valid_ids = {skill.skill_id for skill in skills}
        if len(rows) == len(manifest["eval_indices"]) and all(
            row.get("task_index") == index
            and len(row.get("retrieved_top_candidates", [])) == len(skills)
            and {item["skill_id"] for item in row["retrieved_top_candidates"]}
            == valid_ids
            for row, index in zip(rows, manifest["eval_indices"], strict=True)
        ):
            return cached

    contexts = env.contexts(manifest["eval_indices"])
    by_index = {int(row["task_index"]): row for row in contexts}
    if len(contexts) != len(by_index) or set(by_index) != set(manifest["eval_indices"]):
        raise RuntimeError(
            "task contexts contain missing, duplicate, or unexpected indices"
        )
    tasks = []
    for rollout_index, record in enumerate(manifest["eval_tasks"]):
        context = by_index[int(record["index"])]
        for key in ("instruction", "category"):
            if context[key] != record[key]:
                raise RuntimeError(f"task context differs from manifest at {key}")
        tasks.append(
            {
                **context,
                "rollout_index": rollout_index,
                "task_type": "WebShop/" + context["category"],
                "query": build_retrieval_query(
                    task_type="WebShop/" + context["category"],
                    task_description=context["instruction"],
                    initial_observation=context["initial_observation"],
                ),
            }
        )
    vectors = embed_texts(
        [skill.content for skill in skills] + [task["query"] for task in tasks],
        model_name=embedding["model"],
        device=embedding["device"],
        batch_size=int(embedding["batch_size"]),
        max_length=int(embedding["max_length"]),
        cache_dir=Path(embedding["cache_dir"]) if embedding.get("cache_dir") else None,
    )
    rankings = retrieve_top_skills(
        skills=skills,
        skill_embeddings=vectors[: len(skills)],
        query_embeddings=vectors[len(skills) :],
        top_k=len(skills),
    )
    for task, ranking in zip(tasks, rankings, strict=True):
        task["retrieved_top_candidates"] = ranking
    result = {
        "status": "complete",
        "signature": signature,
        "retrieval_once_per_episode": True,
        "retrieval_updated_during_episode": False,
        "embedding": {
            **embedding,
            "pooling": "attention_mask_mean_then_l2",
            "similarity": "cosine",
        },
        "tasks": tasks,
    }
    atomic_write_json(path, result)
    return result


def condition_guidance(tasks, skills_by_id, top_k, tokenizer):
    guidance = []
    for task in tasks:
        selected = task["retrieved_top_candidates"][:top_k]
        text = format_retrieved_skill_text(
            selected, skills_by_id, no_skill_text=NO_SKILL_TEXT
        )
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        guidance.append(
            {
                "selected_skills": selected,
                "selected_skill_ids": [row["skill_id"] for row in selected],
                "skill_text": text,
                "retrieved_skill_tokens": tokens,
                "per_skill_retrieved_tokens": tokens / top_k
                if top_k
                else float(tokens),
            }
        )
    return guidance


def validate_episode(episode, task, *, success_threshold):
    if not isinstance(episode, dict) or episode.get("status") not in {
        "complete",
        "max_steps",
    }:
        error = (
            episode.get("error", "invalid status")
            if isinstance(episode, dict)
            else "invalid record"
        )
        raise RuntimeError(f"unscorable episode: {error}")
    for key in ("task_index", "instruction", "category"):
        if episode.get(key) != task[key]:
            raise RuntimeError(f"episode has wrong task identity at {key}")
    trace = episode.get("trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError("episode has no complete trace")
    if not all(isinstance(row, dict) for row in trace):
        raise RuntimeError("episode trace contains an invalid step")
    if trace[0].get("observation") != task["initial_observation"]:
        raise RuntimeError(
            "episode initial observation differs from frozen retrieval context"
        )
    reward = float(episode["reward"])
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise RuntimeError("episode has an invalid WebShop reward")
    if episode.get("success") is not (reward >= success_threshold):
        raise RuntimeError(
            "episode success does not match the configured reward threshold"
        )
    terminal = episode["status"] == "complete"
    if episode.get("done") is not terminal or trace[-1].get("done") is not terminal:
        raise RuntimeError("episode trace does not match its terminal status")


def run_conditions(
    *,
    env,
    output_dir,
    config,
    run_signature,
    tasks,
    skills,
    tokenizer,
    top_k_values,
    actor_base_url,
    actor_api_key,
    on_update,
):
    evaluation, actor = config["evaluation"], config["actor"]
    skills_by_id = {skill.skill_id: skill for skill in skills}
    guidance = {
        k: condition_guidance(tasks, skills_by_id, k, tokenizer) for k in top_k_values
    }
    episodes = {k: {} for k in top_k_values}
    pending = deque()
    for index, task in enumerate(tasks):
        for k in top_k_values:
            info = guidance[k][index]
            signature = digest(
                {"run": run_signature, "task": task, "k": k, "guidance": info}
            )
            path = output_dir / "cache" / f"k_{k:02d}" / f"episode_{index:03d}.json"
            cached = read_json(path)
            if (
                isinstance(cached, dict)
                and cached.get("episode_signature") == signature
            ):
                try:
                    validate_episode(
                        cached, task, success_threshold=evaluation["success_threshold"]
                    )
                    if any(cached.get(key) != value for key, value in info.items()):
                        raise RuntimeError("cached retrieval metadata differs")
                    if cached.get("top_k") != k:
                        raise RuntimeError("cached top_k differs")
                    episodes[k][index] = cached
                    continue
                except (KeyError, TypeError, ValueError, RuntimeError):
                    pass
            pending.append((k, index, signature, path, 1))

    def completed():
        return {
            str(k): [episodes[k][index] for index in range(len(tasks))]
            for k in top_k_values
            if len(episodes[k]) == len(tasks)
        }

    def run_one(job):
        k, index, signature, _, attempt = job
        task, info = tasks[index], guidance[k][index]
        episode = env.rollout(
            {
                "task_index": task["task_index"],
                "skill": info["skill_text"],
                "condition": f"k_{k:02d}",
                "actor_base_url": actor_base_url,
                "actor_model": actor.get("request_model") or actor["model"],
                "actor_api_key": actor_api_key,
                "actor_timeout_s": actor["timeout_s"],
                "actor_temperature": actor["temperature"],
                "actor_max_tokens": actor["max_tokens"],
                "actor_enable_thinking": bool(
                    actor.get("enable_thinking", False)
                ),
                # WebShopRuntime adds task_index*1000 + step_index. k/attempt never change it.
                "actor_seed": evaluation["seed"],
                "max_steps": evaluation["max_steps"],
                "memory_window": evaluation["memory_window"],
                "observation_char_limit": evaluation["observation_char_limit"],
                "max_clickables": evaluation["max_clickables"],
                "invalid_action_retries": evaluation["invalid_action_retries"],
                "success_threshold": evaluation["success_threshold"],
                "session_key": f"eval_{uuid.uuid4().hex}_k{k}_t{task['task_index']}_a{attempt}",
            }
        )
        episode.update(
            {
                **info,
                "top_k": k,
                "episode_signature": signature,
                "retrieval_query": task["query"],
                "attempt": attempt,
                "actor_seed_base": evaluation["seed"],
            }
        )
        return episode

    total = len(top_k_values) * len(tasks)
    finished = sum(map(len, episodes.values()))
    print(
        f"[rollout] cached={finished}/{total} workers={evaluation['rollout_workers']}",
        flush=True,
    )
    on_update("running", completed(), {})
    failure = None
    started = time.monotonic()
    cached_episodes = finished
    fresh_actions = 0
    failed_attempts = 0

    def report_performance(active_count, pending_count):
        elapsed = max(time.monotonic() - started, 1e-9)
        atomic_write_json(
            output_dir / "performance.json",
            {
                "elapsed_seconds": elapsed,
                "cached_episodes": cached_episodes,
                "completed_new_episodes": finished - cached_episodes,
                "completed_actions": fresh_actions,
                "episodes_per_second": (finished - cached_episodes) / elapsed,
                "actions_per_second": fresh_actions / elapsed,
                "active_episodes": active_count,
                "pending_episodes": pending_count,
                "max_in_flight": evaluation["rollout_workers"],
                "failed_attempts": failed_attempts,
            },
        )

    try:
        with ThreadPoolExecutor(max_workers=evaluation["rollout_workers"]) as executor:
            active = {}
            while pending or active:
                while pending and len(active) < evaluation["rollout_workers"]:
                    job = pending.popleft()
                    active[executor.submit(run_one, job)] = job
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    job = active.pop(future)
                    k, index, signature, path, attempt = job
                    episode = None
                    try:
                        episode = future.result()
                        validate_episode(
                            episode,
                            tasks[index],
                            success_threshold=evaluation["success_threshold"],
                        )
                    except Exception as exc:
                        failed_attempts += 1
                        # Preserve diagnostics, but never put infrastructure failures in SR.
                        atomic_write_json(
                            output_dir
                            / "errors"
                            / f"k_{k:02d}_episode_{index:03d}_attempt_{attempt}.json",
                            {
                                "error": str(exc),
                                "episode": episode,
                                "episode_signature": signature,
                            },
                        )
                        if (
                            failure is None
                            and attempt < evaluation["episode_max_attempts"]
                        ):
                            pending.append((k, index, signature, path, attempt + 1))
                            continue
                        if failure is None:
                            failure = RuntimeError(
                                f"k={k} task={tasks[index]['task_index']} exhausted episode retries: {exc}"
                            )
                        # Drain and save already-running episodes before leaving.
                        pending.clear()
                        continue
                    atomic_write_json(path, episode)
                    episodes[k][index] = episode
                    finished += 1
                    fresh_actions += len(episode["trace"])
                    if len(episodes[k]) == len(tasks):
                        on_update("running", completed(), {})
                    if (
                        finished % evaluation["progress_every"] == 0
                        or finished == total
                    ):
                        print(f"[rollout] complete={finished}/{total}", flush=True)
                        atomic_write_json(
                            output_dir / "progress.json",
                            {
                                "completed_episodes": finished,
                                "total_episodes": total,
                                "completed_by_k": {
                                    str(key): len(value)
                                    for key, value in episodes.items()
                                },
                            },
                        )
                report_performance(len(active), len(pending))
        report_performance(0, len(pending))
        if failure is not None:
            raise failure
    except BaseException as exc:
        on_update("failed", completed(), {"error": str(exc)})
        raise
    on_update("complete", completed(), {})
    return completed()


def partition_metrics(episodes):
    return {
        **episode_metrics(episodes),
        "wins": sum(bool(row["success"]) for row in episodes),
        "sr": sum(bool(row["success"]) for row in episodes) / len(episodes),
    }


def write_results(
    *,
    output_dir,
    status,
    config,
    run_signature,
    skills,
    manifest,
    retrieval,
    top_k_values,
    episodes_by_k,
    extra,
):
    metrics, sr_by_k, rows = {}, {}, []
    expected_categories = Counter(row["category"] for row in manifest["eval_tasks"])
    for k in map(str, top_k_values):
        if k not in episodes_by_k:
            continue
        episodes = episodes_by_k[k]
        if [row["task_index"] for row in episodes] != manifest["eval_indices"]:
            raise RuntimeError(
                "cannot publish incomplete or misordered task denominator"
            )
        if Counter(row["category"] for row in episodes) != expected_categories:
            raise RuntimeError("episode category counts differ from task manifest")
        for episode, task in zip(episodes, retrieval["tasks"], strict=True):
            validate_episode(
                episode,
                task,
                success_threshold=config["evaluation"]["success_threshold"],
            )
        categories = {
            category: partition_metrics(
                [row for row in episodes if row["category"] == category]
            )
            for category in sorted(expected_categories)
        }
        aggregate = partition_metrics(episodes)
        assert sum(item["wins"] for item in categories.values()) == aggregate["wins"]
        assert sum(item["n"] for item in categories.values()) == aggregate["n"]
        token_stats = {
            "avg_total_retrieved_tokens": sum(
                row["retrieved_skill_tokens"] for row in episodes
            )
            / len(episodes),
            "avg_per_skill_retrieved_tokens": sum(
                row["per_skill_retrieved_tokens"] for row in episodes
            )
            / len(episodes),
        }
        headline = {
            "score": aggregate["mean_reward"],
            "sr": aggregate["sr"],
            **token_stats,
        }
        metrics[k] = {"all": aggregate, "per_category": categories, **headline}
        row = {
            "top_k": int(k),
            "all": {key: aggregate[key] for key in ("n", "wins", "sr", "mean_reward")},
            **headline,
            "per_category": {
                name: {key: item[key] for key in ("n", "wins", "sr", "mean_reward")}
                for name, item in categories.items()
            },
        }
        if "0" in episodes_by_k:
            paired = paired_condition_metrics(episodes_by_k["0"], episodes)
            metrics[k]["paired_vs_k0"] = paired
            row["delta_sr_vs_k0"] = paired["success_rate_delta"]
            row["delta_mean_reward_vs_k0"] = paired["mean_reward_delta"]
        sr_by_k[k] = {
            "all": aggregate["sr"],
            "mean_reward": aggregate["mean_reward"],
            "per_category": {name: item["sr"] for name, item in categories.items()},
            **headline,
        }
        rows.append(row)
        atomic_write_json(
            output_dir / "cache" / f"k_{int(k):02d}" / "metrics.json", metrics[k]
        )
    bank_path = Path(config.get("skillbank_path", "skillbank"))
    bank = {
        "name": f"{bank_path.parent.name}_{bank_path.stem}",
        "path": str(bank_path),
        "content_digest": digest([skill.manifest() for skill in skills]),
        "count": len(skills),
        "models": sorted({skill.model for skill in skills}),
        "skills": [skill.manifest() for skill in skills],
    }
    common = {
        "status": status,
        "mode": VERSION,
        "run_signature": run_signature,
        "rollout_scheduler": SCHEDULER,
        "updated_at": time.time(),
        **extra,
    }
    atomic_write_json(
        output_dir / "results.json",
        {
            **common,
            "run_config": config,
            "skillbank": bank,
            "task_manifest": manifest,
            "retrieval": {
                key: value for key, value in retrieval.items() if key != "tasks"
            },
            "retrieval_by_task": retrieval["tasks"],
            "sr_by_k": sr_by_k,
            "metrics_by_k": metrics,
            "episodes_by_k": episodes_by_k,
        },
    )
    best = (
        max(rows, key=lambda row: (row["all"]["sr"], -row["top_k"])) if rows else None
    )
    atomic_write_json(
        output_dir / "summary.json",
        {
            **common,
            "actor_model": config["actor"]["model"],
            "actor_seed_base": config["evaluation"]["seed"],
            "task_seed": manifest["seed"],
            "embedding_model": config["embedding"]["model"],
            "split": manifest.get("split", "held_out"),
            "test_subset": manifest.get("subset"),
            "skillbank": {key: value for key, value in bank.items() if key != "skills"},
            "task_count": len(manifest["eval_indices"]),
            "category_counts": dict(expected_categories),
            "requested_top_k_values": top_k_values,
            "completed_top_k_values": [int(k) for k in metrics],
            "pending_top_k_values": [k for k in top_k_values if str(k) not in metrics],
            "best_by_all_sr": {"top_k": best["top_k"], "all_sr": best["all"]["sr"]}
            if best
            else None,
            "results": rows,
        },
    )


@contextmanager
def output_lock(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"evaluation already active: {output_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_actor(base_url, api_key, expected_model, enable_thinking=False):
    response = requests.get(
        endpoint_root(base_url) + "/v1/models",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        timeout=10,
    )
    response.raise_for_status()
    ids = {row.get("id") for row in response.json().get("data", [])}
    expected = {expected_model}
    if Path(expected_model).exists():
        expected.add(str(Path(expected_model).resolve()))
    if not ids.intersection(expected):
        raise RuntimeError(
            f"served model IDs {sorted(ids)} do not match {sorted(expected)}"
        )
    # Metadata can be ready before the first prefill/decode compiles. Exercise
    # the same Chat Completions endpoint as the frozen executor before a batch.
    probe = requests.post(
        endpoint_root(base_url) + "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        json={
            "model": expected_model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0.0,
            "max_tokens": 8,
            "chat_template_kwargs": {
                "enable_thinking": bool(enable_thinking)
            },
        },
        timeout=300,
    )
    probe.raise_for_status()
    choices = probe.json().get("choices", [])
    if not choices or not choices[0].get("message", {}).get("content"):
        raise RuntimeError("model readiness completion returned no text")
    print("[preflight] real model completion passed", flush=True)


def check_gpus(spec):
    ids = [item.strip() for item in spec.gpus.split(",") if item.strip()]
    if not ids or len(set(ids)) != len(ids) or len(ids) != spec.tp_size * spec.dp_size:
        raise ValueError(
            "managed server requires unique GPU IDs and GPU count = tp_size * dp_size"
        )
    for gpu_id in ids:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu_id,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            raise RuntimeError(f"GPU {gpu_id} is occupied; select free GPU_IDS")


def load_config(args):
    raw = os.path.expandvars(args.config.read_text(encoding="utf-8"))
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    for section in ("webshop", "actor", "embedding", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration section: {section}")
    actor, evaluation = config["actor"], config["evaluation"]
    if args.model_path:
        actor["model"] = args.model_path
    if args.base_url:
        actor["base_url"] = args.base_url
    if args.actor_enable_thinking is not None:
        actor["enable_thinking"] = args.actor_enable_thinking
    actor["base_url"] = endpoint_root(actor.get("base_url", ""))
    if args.rollout_workers is not None:
        evaluation["rollout_workers"] = args.rollout_workers
    if args.seed is not None:
        evaluation["seed"] = args.seed
    if args.embedding_model:
        config["embedding"]["model"] = args.embedding_model
    server = actor.setdefault("server", {})
    if args.gpus:
        server["gpus"] = args.gpus
        server["dp_size"] = len(args.gpus.split(",")) // int(server.get("tp_size", 1))
    if args.max_running_requests is not None:
        if args.max_running_requests <= 0:
            raise ValueError("max-running-requests must be positive")
        extra = list(server.get("extra_args") or [])
        if "--max-running-requests" in extra:
            index = extra.index("--max-running-requests")
            extra[index + 1] = str(args.max_running_requests)
        else:
            extra.extend(["--max-running-requests", str(args.max_running_requests)])
        server["extra_args"] = extra
    for key in (
        "max_steps",
        "rollout_workers",
        "episode_max_attempts",
        "progress_every",
        "observation_char_limit",
        "max_clickables",
    ):
        if int(evaluation[key]) <= 0:
            raise ValueError(f"evaluation.{key} must be positive")
    if not 0 < float(evaluation["success_threshold"]) <= 1:
        raise ValueError("success_threshold must be in (0, 1]")
    if (
        int(evaluation["memory_window"]) < 0
        or int(evaluation["invalid_action_retries"]) < 0
    ):
        raise ValueError("memory_window and invalid_action_retries cannot be negative")
    if (
        float(actor["temperature"]) < 0
        or int(actor["max_tokens"]) <= 0
        or float(actor["timeout_s"]) <= 0
    ):
        raise ValueError("invalid actor generation settings")
    # Credentials never enter run_config, manifests, or cache signatures.
    api_key = os.environ.get(str(actor.pop("api_key_env", "ACTOR_API_KEY")), "")
    if actor.pop("api_key", ""):
        raise ValueError("use actor.api_key_env instead of a literal API key")
    return config, api_key


def execute(args):
    config, api_key = load_config(args)
    if args.validate_tasks_only:
        output_dir = args.output_dir.expanduser().resolve()
        with output_lock(output_dir):
            env = Environment(config["webshop"])
            manifest = select_tasks(env, config["webshop"], args.task_manifest)
            atomic_write_json(output_dir / "task_manifest.json", manifest)
            print(
                f"[task-preflight] train={len(manifest['train_indices'])} eval={len(manifest['eval_indices'])} validated",
                flush=True,
            )
        return
    skills = load_skillbank(args.skillbank, expected_count=args.expected_skill_count)
    top_k_values = parse_top_k(
        args.top_k_values or str(config["evaluation"]["top_k_values"]), len(skills)
    )
    output_dir = args.output_dir.expanduser().resolve()
    with output_lock(output_dir):
        env = Environment(config["webshop"])
        manifest = select_tasks(env, config["webshop"], args.task_manifest)
        # Worker count, endpoint placement and requested k do not change an episode.
        contract = {
            "version": VERSION,
            "task_manifest": manifest,
            "skills": [skill.manifest() for skill in skills],
            "webshop": {
                key: value
                for key, value in config["webshop"].items()
                if key not in {"env_service_url", "rollout_timeout_s"}
            },
            "actor": {
                key: value
                for key, value in config["actor"].items()
                if key not in {"base_url", "server", "timeout_s"}
            },
            "evaluation": {
                key: value
                for key, value in config["evaluation"].items()
                if key
                not in {
                    "top_k_values",
                    "rollout_workers",
                    "episode_max_attempts",
                    "progress_every",
                }
            },
            "embedding": {
                key: config["embedding"][key] for key in ("model", "max_length")
            },
        }
        signature = digest(contract)
        previous = read_json(output_dir / "run_config.json")
        if previous is not None and previous.get("run_signature") != signature:
            raise RuntimeError(
                "output directory has a different evaluation contract; choose a new OUTPUT_DIR"
            )
        config["top_k_values"] = top_k_values
        config["skillbank_path"] = str(args.skillbank.resolve())
        config["run_signature"] = signature
        atomic_write_json(output_dir / "run_config.json", config)
        atomic_write_json(output_dir / "task_manifest.json", manifest)
        retrieval = prepare_retrieval(env, manifest, skills, config, output_dir)
        print(
            f"[retrieval] tasks={len(manifest['eval_indices'])} bank={len(skills)} k={top_k_values}",
            flush=True,
        )
        if args.prepare_only:
            print(f"[prepared] {output_dir / 'retrieval_manifest.json'}", flush=True)
            return
        from transformers import AutoTokenizer

        actor = config["actor"]
        tokenizer = AutoTokenizer.from_pretrained(
            actor.get("tokenizer") or actor["model"]
        )
        server_section = {**actor, "api_key": api_key}
        spec = server_spec(
            role="actor",
            section=server_section,
            model=actor["model"],
            output_dir=output_dir,
            force_no_start=bool(actor["base_url"]),
        )
        if spec.start:
            check_gpus(spec)

        def on_update(status, episodes, extra):
            write_results(
                output_dir=output_dir,
                status=status,
                config=config,
                run_signature=signature,
                skills=skills,
                manifest=manifest,
                retrieval=retrieval,
                top_k_values=top_k_values,
                episodes_by_k=episodes,
                extra=extra,
            )

        previous_results = read_json(output_dir / "results.json") or {}
        previous_episodes = {
            key: value
            for key, value in previous_results.get("episodes_by_k", {}).items()
            if int(key) in top_k_values
        }
        on_update("running", previous_episodes, {})
        try:
            with ManagedSGLangServer(spec) as base_url:
                verify_actor(
                    base_url,
                    api_key,
                    actor.get("request_model") or actor["model"],
                    bool(actor.get("enable_thinking", False)),
                )
                run_conditions(
                    env=env,
                    output_dir=output_dir,
                    config=config,
                    run_signature=signature,
                    tasks=retrieval["tasks"],
                    skills=skills,
                    tokenizer=tokenizer,
                    top_k_values=top_k_values,
                    actor_base_url=endpoint_root(base_url),
                    actor_api_key=api_key,
                    on_update=on_update,
                )
        except BaseException as exc:
            # run_conditions already preserves completed conditions on rollout errors.
            latest_results = read_json(output_dir / "results.json") or {}
            if latest_results.get("status") != "failed":
                on_update(
                    "failed",
                    latest_results.get("episodes_by_k", previous_episodes),
                    {"error": str(exc)},
                )
            raise
        print(f"[complete] {output_dir / 'summary.json'}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skillbank", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="Saved test subset or preflight manifest (official test), or a training task_manifest.json (training_holdout)",
    )
    parser.add_argument(
        "--expected-skill-count",
        type=int,
        default=50,
        help="0 accepts any nonempty bank",
    )
    retrieval = parser.add_mutually_exclusive_group()
    retrieval.add_argument(
        "--top-k",
        type=int,
        help="Retrieved entries per task (default: 10; 0 disables retrieval).",
    )
    retrieval.add_argument("--top-k-values", help=argparse.SUPPRESS)
    parser.add_argument("--model-path")
    parser.add_argument("--base-url")
    parser.add_argument("--gpus")
    parser.add_argument("--max-running-requests", type=int)
    parser.add_argument("--rollout-workers", type=int)
    parser.add_argument(
        "--actor-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override actor.enable_thinking for hosted chat requests. The "
            "setting is included in the evaluation contract."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-tasks-only", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k is not None:
        if args.top_k < 0:
            parser.error("--top-k must be nonnegative")
        args.top_k_values = str(args.top_k)
    if args.skillbank is None and not args.validate_tasks_only:
        parser.error("--skillbank is required unless --validate-tasks-only is used")
    return args


def main():
    execute(parse_args())


if __name__ == "__main__":
    main()
