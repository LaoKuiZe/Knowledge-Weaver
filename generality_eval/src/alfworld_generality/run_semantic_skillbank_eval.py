"""Retrieve once per episode and keep guidance fixed; use identical tasks and seeds across top-k values."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import requests
import yaml
from transformers import AutoTokenizer

from areal.workflow.alfworld_runtime import (
    ALFWORLD_TASK_TYPES,
    _select_games_by_type,
)
from areal.workflow.alfworld_skill import (
    _NO_SKILL_BASELINE_TEXT,
    _atomic_write_json,
    _batch_item,
    _episode_metrics,
    _make_env,
    _per_task_type_metrics,
    _run_rollout_episode_process,
    _shorten,
)
from examples.alfworld_skill.sglang_server import (
    ServerSpec,
    gpu_count,
    managed_sglang_server,
)

from .rollout_executor import ProcessThreadExecutor, initialize_rollout_worker
from .semantic_skillbank import (
    DEFAULT_EMBEDDING_MODEL,
    SkillEntry,
    build_retrieval_query,
    embed_texts,
    format_retrieved_skill_text,
    load_skillbank,
    retrieve_top_skills,
)


EVALUATOR_VERSION = "semantic_skillbank_unseen_v1"
QUERY_VERSION = "episode_initial_state_v1"
ROLLOUT_SCHEDULER = "flattened_k_task_queue_v1"
EPISODE_CACHE_SIGNATURE_VERSION = "semantic_skillbank_episode_v2"
DEFAULT_TOP_K_VALUES = "10"

# Exclude orchestration fields so caches survive changed top-k requests or relocated identical banks.
EPISODE_CACHE_IGNORED_RUN_CONFIG_KEYS = frozenset(
    {
        "top_k_values",
        "skillbank_path",
        "episode_cache_signature",
        "episode_cache_signature_version",
    }
)

SCORABLE_EPISODE_STATUSES = {"won", "done", "max_steps"}


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized.strip("_.-") or "skillbank"


def _parse_top_k_values(value: str, *, skill_count: int) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("top-k values must be comma-separated integers") from exc
    if not values:
        raise ValueError("at least one top-k value is required")
    if len(values) != len(set(values)):
        raise ValueError("top-k values must not contain duplicates")
    if any(item < 0 or item > skill_count for item in values):
        raise ValueError(f"top-k values must be between 0 and {skill_count}")
    return values


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_cache_config(run_config: dict[str, Any]) -> dict[str, Any]:
    """Return only settings that can change a cached episode's semantics."""

    return {
        key: value
        for key, value in run_config.items()
        if key not in EPISODE_CACHE_IGNORED_RUN_CONFIG_KEYS
    }


def _episode_cache_run_signature(run_config: dict[str, Any]) -> str:
    return _json_sha256(
        {
            "signature_version": EPISODE_CACHE_SIGNATURE_VERSION,
            "run_config": _episode_cache_config(run_config),
        }
    )


def _probe_initial_observation_process(payload: dict[str, Any]) -> dict[str, Any]:
    """Reset one game and return the observation used by the retrieval query."""

    game = dict(payload["game"])
    env = _make_env(
        Path(str(payload["repo_root"])),
        str(game["gamefile"]),
        max_steps=int(payload["max_rollout_steps"]),
    )
    try:
        obs, _ = env.reset()
        return {
            "rollout_index": int(payload["rollout_index"]),
            "initial_observation": _shorten(str(_batch_item(obs, "")), limit=1200),
        }
    finally:
        env.close()


def _probe_initial_observations(
    *,
    repo_root: Path,
    games: Sequence[dict[str, Any]],
    max_rollout_steps: int,
    workers: int,
) -> list[str]:
    observations: list[str | None] = [None] * len(games)
    max_workers = max(1, min(int(workers), len(games)))
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(
                _probe_initial_observation_process,
                {
                    "repo_root": str(repo_root),
                    "game": dict(game),
                    "rollout_index": rollout_index,
                    "max_rollout_steps": max_rollout_steps,
                },
            ): rollout_index
            for rollout_index, game in enumerate(games)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rollout_index = int(result["rollout_index"])
            observations[rollout_index] = str(result["initial_observation"])
            if (
                completed == 1
                or completed % max_workers == 0
                or completed == len(games)
            ):
                print(
                    f"[retrieval] initial observations {completed}/{len(games)}",
                    flush=True,
                )
    if any(item is None for item in observations):
        raise RuntimeError("failed to collect every initial observation")
    return [str(item) for item in observations]


def _select_games(
    *, data_root: Path, split: str, tasks_per_type: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select games for every ALFWorld task type; 0 per type keeps all games."""

    return _select_games_by_type(
        data_root=data_root,
        split=split,
        task_types=ALFWORLD_TASK_TYPES,
        tasks_per_type=tasks_per_type,
        seed=seed,
    )


def _task_manifest(games: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rollout_index": index,
            "eval_task_index": game["eval_task_index"],
            "task_type": game["task_type"],
            "task_desc": game["task_desc"],
            "gamefile": game["gamefile"],
            "traj_json": game["traj_json"],
        }
        for index, game in enumerate(games)
    ]


def _prepare_retrieval_manifest(
    *,
    output_dir: Path,
    repo_root: Path,
    games: Sequence[dict[str, Any]],
    task_manifest: list[dict[str, Any]],
    skills: Sequence[SkillEntry],
    skillbank_sha256: str,
    embedding_model: str,
    embedding_device: str,
    embedding_batch_size: int,
    embedding_max_length: int,
    embedding_cache_dir: Path | None,
    max_top_k: int,
    max_rollout_steps: int,
    initial_observation_workers: int,
) -> dict[str, Any]:
    signature_payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "query_version": QUERY_VERSION,
        "skillbank_sha256": skillbank_sha256,
        "task_manifest_sha256": _json_sha256(task_manifest),
        "embedding_model": embedding_model,
        "embedding_max_length": embedding_max_length,
        "max_top_k": max_top_k,
    }
    signature = _json_sha256(signature_payload)
    path = output_dir / "retrieval_manifest.json"
    compatible_cached: dict[str, Any] | None = None
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("status") == "complete"
            and cached.get("signature") == signature
            and len(cached.get("tasks") or []) == len(games)
        ):
            print(f"[retrieval] reusing {path}", flush=True)
            return cached

        cached_signature_payload = (
            cached.get("signature_payload") if isinstance(cached, dict) else None
        )
        if isinstance(cached_signature_payload, dict):
            requested_core = {
                key: value
                for key, value in signature_payload.items()
                if key != "max_top_k"
            }
            cached_core = {
                key: value
                for key, value in cached_signature_payload.items()
                if key != "max_top_k"
            }
            cached_tasks = cached.get("tasks") or []
            if cached_core == requested_core and len(cached_tasks) == len(games):
                compatible_cached = cached
                print(
                    "[retrieval] extending compatible cached rankings "
                    f"from top_k={cached_signature_payload.get('max_top_k')} "
                    f"to top_k={max_top_k}",
                    flush=True,
                )

    if compatible_cached is not None:
        cached_tasks = compatible_cached["tasks"]
        initial_observations = [
            str(item["initial_observation"]) for item in cached_tasks
        ]
        queries = [str(item["query"]) for item in cached_tasks]
    else:
        initial_observations = _probe_initial_observations(
            repo_root=repo_root,
            games=games,
            max_rollout_steps=max_rollout_steps,
            workers=initial_observation_workers,
        )
        queries = [
            build_retrieval_query(
                task_type=str(game["task_type"]),
                task_description=str(game["task_desc"]),
                initial_observation=initial_observations[index],
            )
            for index, game in enumerate(games)
        ]
    all_embeddings = embed_texts(
        [skill.content for skill in skills] + queries,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        max_length=embedding_max_length,
        device=embedding_device,
        cache_dir=embedding_cache_dir,
    )
    skill_embeddings = all_embeddings[: len(skills)]
    query_embeddings = all_embeddings[len(skills) :]
    rankings = retrieve_top_skills(
        skills=skills,
        skill_embeddings=skill_embeddings,
        query_embeddings=query_embeddings,
        top_k=max_top_k,
    )
    if compatible_cached is not None:
        for index, cached_task in enumerate(compatible_cached["tasks"]):
            cached_ids = [
                str(item["skill_id"])
                for item in cached_task.get("retrieved_top_candidates") or []
            ]
            current_ids = [
                str(item["skill_id"]) for item in rankings[index][: len(cached_ids)]
            ]
            if cached_ids != current_ids:
                raise RuntimeError(
                    "cached retrieval ranking is not a prefix of the expanded "
                    f"ranking for rollout_index={index}"
                )
    tasks = [
        {
            **task_manifest[index],
            "initial_observation": initial_observations[index],
            "query": queries[index],
            "retrieved_top_candidates": rankings[index],
        }
        for index in range(len(games))
    ]
    payload = {
        "status": "complete",
        "mode": EVALUATOR_VERSION,
        "signature": signature,
        "signature_payload": signature_payload,
        "retrieval_once_per_episode": True,
        "retrieval_updated_during_episode": False,
        "query_version": QUERY_VERSION,
        "embedding": {
            "model": embedding_model,
            "device": embedding_device,
            "batch_size": embedding_batch_size,
            "max_length": embedding_max_length,
            "pooling": "attention_mask_mean_pooling_then_l2_normalization",
            "similarity": "cosine",
        },
        "tasks": tasks,
        "updated_at": time.time(),
    }
    _atomic_write_json(path, payload)
    print(f"[retrieval] wrote {path}", flush=True)
    return payload


def _retrieved_skill_token_stats(
    *,
    tasks: Sequence[dict[str, Any]],
    skills_by_id: dict[str, SkillEntry],
    top_k: int,
    tokenizer: Any,
) -> dict[str, float]:
    """Count tokens in the exact injected skill block, both total and per retrieved entry."""
    totals: list[int] = []
    per_skill: list[float] = []
    for task in tasks:
        selected = list(task["retrieved_top_candidates"][:top_k])
        text = format_retrieved_skill_text(
            selected, skills_by_id, no_skill_text=_NO_SKILL_BASELINE_TEXT
        )
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        totals.append(n_tokens)
        per_skill.append(n_tokens / len(selected) if selected else float(n_tokens))
    n = len(tasks)
    return {
        "avg_total_retrieved_tokens": sum(totals) / n if n else 0.0,
        "avg_per_skill_retrieved_tokens": sum(per_skill) / n if n else 0.0,
    }


def _endpoint_root(url: str) -> str:
    """Accept server URLs with or without the OpenAI-style /v1 suffix."""
    root = url.strip().rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def _verify_server_model(
    *, base_url: str, api_key: str, expected_model: str, timeout_s: float = 10.0
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(
        base_url.rstrip("/") + "/v1/models",
        headers=headers,
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    model_ids = {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }
    expected_ids = {expected_model}
    expected_path = Path(expected_model).expanduser()
    if expected_path.exists():
        expected_ids.add(str(expected_path.resolve()))
    if not model_ids or model_ids.isdisjoint(expected_ids):
        raise RuntimeError(
            f"served model IDs {sorted(model_ids)} do not match expected "
            f"{sorted(expected_ids)}"
        )


def _episode_signature(
    *, run_signature: str, top_k: int, rollout_index: int, selected_ids: list[str]
) -> str:
    return _json_sha256(
        {
            "run_signature": run_signature,
            "top_k": top_k,
            "rollout_index": rollout_index,
            "selected_skill_ids": selected_ids,
        }
    )


def _load_cached_episode(
    path: Path, *, expected_signature: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    episode = payload.get("episode")
    if (
        payload.get("status") != "complete"
        or payload.get("signature") != expected_signature
        or not isinstance(episode, dict)
        or episode.get("status") not in SCORABLE_EPISODE_STATUSES
    ):
        return None
    return payload


def _partition_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": _episode_metrics(episodes),
        "per_task_type": _per_task_type_metrics(episodes),
    }


def _selected_task_type_counts(run_config: dict[str, Any]) -> dict[str, int]:
    """Return the exact evaluation-manifest count for every concrete task type."""

    game_sampling = run_config.get("game_sampling") or {}
    task_types = game_sampling.get("task_types") or {}
    if not isinstance(task_types, dict):
        return {}
    counts: dict[str, int] = {}
    for task_type, metadata in task_types.items():
        if not isinstance(metadata, dict) or "selected" not in metadata:
            continue
        counts[str(task_type)] = int(metadata["selected"])
    return counts


def _validate_per_task_type_metrics(
    *, top_k: int, metrics: dict[str, Any], run_config: dict[str, Any]
) -> None:
    """Reject output that loses or misaggregates concrete task-type results."""

    per_task_type = metrics.get("per_task_type")
    if not isinstance(per_task_type, dict) or not per_task_type:
        raise ValueError(f"k={top_k} does not contain per-task-type metrics")

    expected_counts = _selected_task_type_counts(run_config)
    if expected_counts and set(per_task_type) != set(expected_counts):
        raise ValueError(
            f"k={top_k} task types {sorted(per_task_type)} do not match "
            f"evaluation manifest {sorted(expected_counts)}"
        )

    total_wins = 0
    total_n = 0
    for task_type, item in per_task_type.items():
        if not isinstance(item, dict):
            raise ValueError(f"k={top_k} task type {task_type!r} has invalid metrics")
        wins = int(item.get("wins", -1))
        n = int(item.get("n", -1))
        sr = float(item.get("sr", -1.0))
        if n < 0 or wins < 0 or wins > n:
            raise ValueError(
                f"k={top_k} task type {task_type!r} has invalid wins/n={wins}/{n}"
            )
        if abs(sr - wins / max(1, n)) > 1e-12:
            raise ValueError(
                f"k={top_k} task type {task_type!r} has inconsistent SR={sr}"
            )
        if expected_counts and n != expected_counts[task_type]:
            raise ValueError(
                f"k={top_k} task type {task_type!r} has n={n}, expected "
                f"{expected_counts[task_type]}"
            )
        total_wins += wins
        total_n += n

    aggregate = metrics.get("all") or {}
    if (
        int(aggregate.get("wins", -1)) != total_wins
        or int(aggregate.get("n", -1)) != total_n
    ):
        raise ValueError(f"k={top_k} per-task-type totals do not match ALL metrics")


def _validate_initial_observation(
    *, episode: dict[str, Any], expected_observation: str
) -> bool:
    steps = episode.get("steps") or []
    return (
        bool(steps) and str(steps[0].get("observation") or "") == expected_observation
    )


@dataclass
class _TopKConditionState:
    top_k: int
    condition_dir: Path
    records: list[dict[str, Any] | None]
    attempts_used: list[int]
    signatures: list[str]
    selected_by_index: list[list[dict[str, Any]]]


def _prepare_top_k_condition_state(
    *,
    top_k: int,
    cache_dir: Path,
    game_count: int,
    retrieval_tasks: Sequence[dict[str, Any]],
    run_signature: str,
) -> _TopKConditionState:
    condition_dir = cache_dir / f"k_{top_k:02d}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any] | None] = [None] * game_count
    attempts_used = [0] * game_count
    signatures: list[str] = []
    selected_by_index: list[list[dict[str, Any]]] = []

    for rollout_index, retrieval in enumerate(retrieval_tasks):
        selected = list(retrieval["retrieved_top_candidates"][:top_k])
        selected_ids = [str(item["skill_id"]) for item in selected]
        selected_by_index.append(selected)
        signature = _episode_signature(
            run_signature=run_signature,
            top_k=top_k,
            rollout_index=rollout_index,
            selected_ids=selected_ids,
        )
        signatures.append(signature)
        cached = _load_cached_episode(
            condition_dir / f"episode_{rollout_index:03d}.json",
            expected_signature=signature,
        )
        if cached is not None:
            records[rollout_index] = cached
            attempts_used[rollout_index] = int(cached.get("attempts_used", 1))

    return _TopKConditionState(
        top_k=top_k,
        condition_dir=condition_dir,
        records=records,
        attempts_used=attempts_used,
        signatures=signatures,
        selected_by_index=selected_by_index,
    )


def _rollout_payload(
    *,
    state: _TopKConditionState,
    rollout_index: int,
    game: dict[str, Any],
    skills_by_id: dict[str, SkillEntry],
    skillbank_name: str,
    repo_root: Path,
    actor_base_url: str,
    checkpoint_path: str,
    api_key: str,
    timeout_s: float,
    temperature: float,
    max_rollout_steps: int,
    memory_window: int,
    max_commands: int,
    seed: int,
    request_model: str = "",
    chat_completions: bool = False,
    actor_enable_thinking: bool = False,
) -> dict[str, Any]:
    selected = state.selected_by_index[rollout_index]
    return {
        "repo_root": str(repo_root),
        "sample_dir": str(state.condition_dir),
        "game": dict(game),
        "skill_name": f"semantic_skillbank/{skillbank_name}/top_{state.top_k}",
        "skill_text": format_retrieved_skill_text(
            selected,
            skills_by_id,
            no_skill_text=_NO_SKILL_BASELINE_TEXT,
        ),
        "rollout_index": rollout_index,
        "index_width": 3,
        "max_rollout_steps": max_rollout_steps,
        "memory_window": memory_window,
        "max_commands": max_commands,
        "actor_base_url": actor_base_url,
        "actor_model": request_model or checkpoint_path,
        "actor_api_key": api_key,
        "actor_timeout_s": timeout_s,
        "actor_temperature": temperature,
        "tokenizer_path": None if chat_completions else checkpoint_path,
        # Set remote thinking mode explicitly; local prompt completions render their own template.
        "actor_enable_thinking": (
            bool(actor_enable_thinking) if chat_completions else None
        ),
        # Deliberately omit top-k so every condition uses common random numbers
        # for the same task.
        "seed_base": seed * 1_000_000 + rollout_index * 100,
        "progress_metadata": {},
        "persist_progress": False,
    }


def _finalize_top_k_condition(
    *, state: _TopKConditionState, games: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if any(item is None for item in state.records):
        raise RuntimeError(f"cannot finalize incomplete k={state.top_k} condition")
    complete_records = [dict(item) for item in state.records if item is not None]
    episodes = [dict(item["episode"]) for item in complete_records]
    metrics = {
        "top_k": state.top_k,
        **_partition_metrics(episodes),
        "n_rollouts_expected": len(games),
        "n_rollouts_complete": len(episodes),
        "attempts_used": state.attempts_used,
        "updated_at": time.time(),
    }
    _atomic_write_json(state.condition_dir / "metrics.json", metrics)
    (state.condition_dir / "failed.json").unlink(missing_ok=True)
    return complete_records, metrics


def _run_flattened_top_k_conditions(
    *,
    executor: Executor,
    top_k_values: Sequence[int],
    cache_dir: Path,
    games: Sequence[dict[str, Any]],
    retrieval_tasks: Sequence[dict[str, Any]],
    skills_by_id: dict[str, SkillEntry],
    run_signature: str,
    skillbank_name: str,
    repo_root: Path,
    actor_base_url: str,
    checkpoint_path: str,
    api_key: str,
    timeout_s: float,
    temperature: float,
    max_rollout_steps: int,
    memory_window: int,
    max_commands: int,
    seed: int,
    episode_max_attempts: int,
    progress_every: int,
    request_model: str = "",
    chat_completions: bool = False,
    actor_enable_thinking: bool = False,
    max_in_flight: int | None = None,
    performance_path: Path | None = None,
    on_condition_complete: Callable[[int, list[dict[str, Any]], dict[str, Any]], None]
    | None = None,
) -> dict[int, tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Run every ``(top_k, task)`` job through one bounded executor queue."""

    if len(games) != len(retrieval_tasks):
        raise ValueError("games and retrieval_tasks must have the same length")
    ordered_top_k = list(top_k_values)
    if not ordered_top_k or len(ordered_top_k) != len(set(ordered_top_k)):
        raise ValueError("top_k_values must be non-empty and unique")
    total_jobs = len(games) * len(ordered_top_k)
    if max_in_flight is None:
        max_in_flight = max(1, total_jobs)
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")

    states = {
        top_k: _prepare_top_k_condition_state(
            top_k=top_k,
            cache_dir=cache_dir,
            game_count=len(games),
            retrieval_tasks=retrieval_tasks,
            run_signature=run_signature,
        )
        for top_k in ordered_top_k
    }
    completed_conditions: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}

    def finalize_if_ready(top_k: int) -> None:
        if top_k in completed_conditions:
            return
        state = states[top_k]
        if any(item is None for item in state.records):
            return
        result = _finalize_top_k_condition(state=state, games=games)
        completed_conditions[top_k] = result
        if on_condition_complete is not None:
            on_condition_complete(top_k, result[0], result[1])

    for top_k in ordered_top_k:
        finalize_if_ready(top_k)

    # Task-major ordering guarantees that the bounded executor sees different k
    # values immediately instead of draining all tasks for one condition first.
    job_queue = deque(
        (top_k, rollout_index, 1)
        for rollout_index in range(len(games))
        for top_k in ordered_top_k
        if states[top_k].records[rollout_index] is None
    )
    cached_jobs = total_jobs - len(job_queue)
    started = time.monotonic()
    fresh_completed = 0
    fresh_actions = 0
    retries = 0

    def report_performance():
        if performance_path is None:
            return
        elapsed = max(time.monotonic() - started, 1e-9)
        _atomic_write_json(
            performance_path,
            {
                "elapsed_seconds": elapsed,
                "cached_episodes": cached_jobs,
                "completed_new_episodes": fresh_completed,
                "completed_actions": fresh_actions,
                "episodes_per_second": fresh_completed / elapsed,
                "actions_per_second": fresh_actions / elapsed,
                "active_episodes": len(active),
                "pending_episodes": len(job_queue),
                "max_in_flight": max_in_flight,
                "failed_attempts": retries,
            },
        )

    print(
        f"[rollout] scheduler={ROLLOUT_SCHEDULER} total={total_jobs} "
        f"cached={cached_jobs} pending={len(job_queue)} "
        f"max_in_flight={max_in_flight}",
        flush=True,
    )

    active = {}
    exhausted: list[tuple[int, int]] = []
    stop_submitting = False
    while job_queue or active:
        while job_queue and len(active) < max_in_flight and not stop_submitting:
            top_k, rollout_index, attempt = job_queue.popleft()
            state = states[top_k]
            state.attempts_used[rollout_index] = attempt
            future = executor.submit(
                _run_rollout_episode_process,
                _rollout_payload(
                    state=state,
                    rollout_index=rollout_index,
                    game=dict(games[rollout_index]),
                    skills_by_id=skills_by_id,
                    skillbank_name=skillbank_name,
                    repo_root=repo_root,
                    actor_base_url=actor_base_url,
                    checkpoint_path=checkpoint_path,
                    api_key=api_key,
                    timeout_s=timeout_s,
                    temperature=temperature,
                    max_rollout_steps=max_rollout_steps,
                    memory_window=memory_window,
                    max_commands=max_commands,
                    seed=seed,
                    request_model=request_model,
                    chat_completions=chat_completions,
                    actor_enable_thinking=actor_enable_thinking,
                ),
            )
            active[future] = (top_k, rollout_index, attempt)

        if not active:
            break
        done, _ = wait(active, return_when=FIRST_COMPLETED)
        for future in done:
            top_k, rollout_index, attempt = active.pop(future)
            state = states[top_k]
            retry_required = False
            try:
                episode = future.result()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[rollout] k={top_k} index={rollout_index} "
                    f"attempt={attempt} future_error={exc!r}",
                    flush=True,
                )
                retry_required = True
                episode = None
            if (
                episode is not None
                and episode.get("status") not in SCORABLE_EPISODE_STATUSES
            ):
                print(
                    f"[rollout] k={top_k} index={rollout_index} "
                    f"attempt={attempt} retry_status={episode.get('status')}",
                    flush=True,
                )
                retry_required = True
            expected_observation = str(
                retrieval_tasks[rollout_index]["initial_observation"]
            )
            if (
                episode is not None
                and not retry_required
                and not _validate_initial_observation(
                    episode=episode,
                    expected_observation=expected_observation,
                )
            ):
                print(
                    f"[rollout] k={top_k} index={rollout_index} "
                    "initial observation mismatch; retrying",
                    flush=True,
                )
                retry_required = True

            if retry_required:
                retries += 1
                if attempt < episode_max_attempts and not stop_submitting:
                    # Prioritize diagnosis/recovery over spending the full fresh
                    # queue while the actor or environment may be unhealthy.
                    job_queue.appendleft((top_k, rollout_index, attempt + 1))
                elif attempt >= episode_max_attempts:
                    exhausted.append((top_k, rollout_index))
                    stop_submitting = True
                continue

            assert episode is not None
            selected = state.selected_by_index[rollout_index]
            payload = {
                "status": "complete",
                "signature": state.signatures[rollout_index],
                "signature_version": EPISODE_CACHE_SIGNATURE_VERSION,
                "rollout_index": rollout_index,
                "attempts_used": attempt,
                "retrieval": {
                    "retrieval_once_per_episode": True,
                    "retrieval_updated_during_episode": False,
                    "top_k": top_k,
                    "query": retrieval_tasks[rollout_index]["query"],
                    "selected_skill_ids": [str(item["skill_id"]) for item in selected],
                    "selected_skills": selected,
                },
                "episode": episode,
                "updated_at": time.time(),
            }
            _atomic_write_json(
                state.condition_dir / f"episode_{rollout_index:03d}.json",
                payload,
            )
            state.records[rollout_index] = payload
            fresh_completed += 1
            fresh_actions += int(
                episode.get("steps_taken", len(episode.get("steps", [])))
            )
            total_complete = sum(item is not None for item in state.records)
            if (
                total_complete == 1
                or total_complete % max(1, progress_every) == 0
                or total_complete == len(games)
            ):
                print(
                    f"[rollout] k={top_k} complete={total_complete}/{len(games)}",
                    flush=True,
                )
            finalize_if_ready(top_k)
        report_performance()

    report_performance()
    for top_k in ordered_top_k:
        finalize_if_ready(top_k)

    if exhausted:
        failures_by_k: dict[int, list[int]] = {}
        for top_k, rollout_index in exhausted:
            failures_by_k.setdefault(top_k, []).append(rollout_index)
        for top_k, failed_indices in failures_by_k.items():
            state = states[top_k]
            _atomic_write_json(
                state.condition_dir / "failed.json",
                {
                    "status": "failed",
                    "top_k": top_k,
                    "failed_rollout_indices": failed_indices,
                    "attempts_used": state.attempts_used,
                    "updated_at": time.time(),
                },
            )
        preview = [f"k={top_k}/index={index}" for top_k, index in exhausted[:10]]
        raise RuntimeError(
            f"flattened rollout failed for {len(exhausted)} (k, task) jobs after "
            f"{episode_max_attempts} attempts: {preview}"
        )

    return {top_k: completed_conditions[top_k] for top_k in ordered_top_k}


def _write_results(
    *,
    output_dir: Path,
    status: str,
    run_signature: str,
    run_config: dict[str, Any],
    skillbank_manifest: dict[str, Any],
    task_manifest: list[dict[str, Any]],
    retrieval_manifest: dict[str, Any],
    metrics_by_k: dict[str, dict[str, Any]],
    episodes_by_k: dict[str, list[dict[str, Any]]],
    token_stats_by_k: dict[str, dict[str, float]],
) -> dict[str, Any]:
    ordered_keys = [
        str(top_k)
        for top_k in run_config.get("top_k_values", [])
        if str(top_k) in metrics_by_k
    ]
    ordered_metrics_by_k = {key: metrics_by_k[key] for key in ordered_keys}
    ordered_episodes_by_k = {key: episodes_by_k[key] for key in ordered_keys}
    for key, metrics in ordered_metrics_by_k.items():
        _validate_per_task_type_metrics(
            top_k=int(key), metrics=metrics, run_config=run_config
        )
    sr_by_k = {
        key: {
            "all": value["all"]["sr"],
            **token_stats_by_k.get(key, {}),
            "per_task_type": {
                task_type: item["sr"]
                for task_type, item in value["per_task_type"].items()
            },
        }
        for key, value in ordered_metrics_by_k.items()
    }
    payload = {
        "status": status,
        "mode": EVALUATOR_VERSION,
        "rollout_scheduler": ROLLOUT_SCHEDULER,
        "run_signature": run_signature,
        "run_config": run_config,
        "skillbank": skillbank_manifest,
        "task_manifest": task_manifest,
        "retrieval": {
            key: value for key, value in retrieval_manifest.items() if key != "tasks"
        },
        "retrieval_by_task": retrieval_manifest["tasks"],
        "sr_by_k": sr_by_k,
        "metrics_by_k": ordered_metrics_by_k,
        "episodes_by_k": ordered_episodes_by_k,
        "updated_at": time.time(),
    }
    _atomic_write_json(output_dir / "results.json", payload)
    _write_compact_summary(
        output_dir=output_dir,
        status=status,
        run_signature=run_signature,
        run_config=run_config,
        skillbank_manifest=skillbank_manifest,
        metrics_by_k=ordered_metrics_by_k,
        token_stats_by_k=token_stats_by_k,
    )
    return payload


def _compact_partition(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "wins": int(metrics.get("wins", 0) or 0),
        "n": int(metrics.get("n", 0) or 0),
        "sr": float(metrics.get("sr", 0.0) or 0.0),
    }


def _write_compact_summary(
    *,
    output_dir: Path,
    status: str,
    run_signature: str,
    run_config: dict[str, Any],
    skillbank_manifest: dict[str, Any],
    metrics_by_k: dict[str, dict[str, Any]],
    token_stats_by_k: dict[str, dict[str, float]],
) -> dict[str, Any]:
    requested_top_k = [int(value) for value in run_config.get("top_k_values", [])]
    completed_top_k = [top_k for top_k in requested_top_k if str(top_k) in metrics_by_k]
    baseline = metrics_by_k.get("0")
    summary_rows: list[dict[str, Any]] = []
    for top_k in completed_top_k:
        metrics = metrics_by_k[str(top_k)]
        row = {
            "top_k": top_k,
            **token_stats_by_k.get(str(top_k), {}),
            "all": _compact_partition(metrics.get("all") or {}),
            "per_task_type": {
                task_type: _compact_partition(item)
                for task_type, item in sorted(
                    (metrics.get("per_task_type") or {}).items()
                )
            },
        }
        if baseline is not None:
            row["delta_sr_vs_k0"] = {
                "all": row["all"]["sr"]
                - float((baseline.get("all") or {}).get("sr", 0.0) or 0.0)
            }
        summary_rows.append(row)

    best_by_all_sr = None
    if summary_rows:
        best = max(summary_rows, key=lambda row: (row["all"]["sr"], -row["top_k"]))
        best_by_all_sr = {
            "top_k": best["top_k"],
            "all_sr": best["all"]["sr"],
        }

    payload = {
        "status": status,
        "mode": EVALUATOR_VERSION,
        "run_signature": run_signature,
        "skillbank": {
            "name": skillbank_manifest.get("name", ""),
            "path": skillbank_manifest.get("path", ""),
            "sha256": skillbank_manifest.get("sha256", ""),
            "count": int(skillbank_manifest.get("count", 0) or 0),
            "models": list(skillbank_manifest.get("models") or []),
        },
        "actor_model": run_config.get("request_model")
        or run_config.get("checkpoint_path", ""),
        "embedding_model": run_config.get("embedding_model", ""),
        "split": run_config.get("split", ""),
        "task_counts": dict(run_config.get("selected_game_counts") or {}),
        "task_type_counts": _selected_task_type_counts(run_config),
        "requested_top_k_values": requested_top_k,
        "completed_top_k_values": completed_top_k,
        "pending_top_k_values": [
            top_k for top_k in requested_top_k if top_k not in completed_top_k
        ],
        "best_by_all_sr": best_by_all_sr,
        "results": summary_rows,
        "updated_at": time.time(),
    }
    _atomic_write_json(output_dir / "summary.json", payload)
    return payload


def _resolve_rollout_worker_count(
    *, requested: int, game_count: int, condition_count: int
) -> int:
    if requested <= 0:
        raise ValueError("rollout worker count must be positive")
    total_jobs = game_count * condition_count
    if total_jobs <= 0:
        raise ValueError("at least one rollout job is required")
    return min(requested, total_jobs)


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config/alfworld_semantic_skillbank_eval.yaml"


def config_defaults(path):
    config = yaml.safe_load(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    aliases = {
        "actor": {"model": "checkpoint_path", "enable_thinking": "actor_enable_thinking"},
        "embedding": {},
        "alfworld": {},
        "evaluation": {},
    }
    defaults = {}
    for section, values in config.items():
        if section not in aliases or not isinstance(values, dict):
            raise ValueError(f"invalid configuration section: {section}")
        for key, value in values.items():
            if section == "actor" and key == "server":
                if not isinstance(value, dict):
                    raise ValueError("actor.server must be a mapping")
                defaults.update(value)
            else:
                dest = f"embedding_{key}" if section == "embedding" else aliases[section].get(key, key)
                defaults[dest] = value
    return defaults


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Evaluate one 50-skill natural-language bank with once-per-episode "
            "MPNet top-k retrieval on all ALFWorld valid_unseen tasks."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skillbank",
        required=True,
        help="Knowledge bank (.jsonl or .json) of id/model/content records.",
    )
    parser.add_argument(
        "--skillbank-name",
        default="",
        help="Output label; defaults to the skillbank parent directory name.",
    )
    parser.add_argument("--expected-skill-count", type=int, default=50)
    retrieval = parser.add_mutually_exclusive_group()
    retrieval.add_argument(
        "--top-k",
        type=int,
        help="Retrieved entries per task (default: 10; 0 disables retrieval).",
    )
    retrieval.add_argument(
        "--top-k-values", default=DEFAULT_TOP_K_VALUES, help=argparse.SUPPRESS
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-path", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--request-model",
        default="",
        help=(
            "Model name sent to the actor API; defaults to --checkpoint-path. "
            "This lets a local tokenizer path be paired with a hosted model ID."
        ),
    )
    parser.add_argument(
        "--chat-completions",
        action="store_true",
        help=(
            "Send structured messages to /v1/chat/completions instead of "
            "rendering a local prompt for /v1/completions."
        ),
    )
    parser.add_argument(
        "--actor-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable server-side thinking for remote chat completions. Defaults "
            "to disabled; use this only when the hosted model requires it."
        ),
    )
    parser.add_argument("--split", default="valid_unseen")
    parser.add_argument(
        "--tasks-per-type",
        type=int,
        default=0,
        help="Games sampled per task type; 0 keeps every game in the split.",
    )
    parser.add_argument("--expected-total-games", type=int, default=134)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--max-rollout-steps", type=int, default=50)
    parser.add_argument("--memory-window", type=int, default=5)
    parser.add_argument("--max-commands", type=int, default=140)
    parser.add_argument("--rollout-workers", type=int, default=384)
    parser.add_argument(
        "--rollout-backend", choices=("hybrid", "process"), default="process"
    )
    parser.add_argument(
        "--rollout-processes",
        type=int,
        default=32,
        help="Maximum processes in hybrid mode; I/O concurrency is rollout-workers",
    )
    parser.add_argument("--initial-observation-workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--episode-max-attempts", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")

    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--embedding-cache-dir", default="")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=34080)
    parser.add_argument(
        "--base-url",
        default="",
        help="Use an existing executor server (with or without the /v1 suffix).",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mem-fraction-static", type=float, default=0.86)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--startup-timeout-s", type=float, default=1800.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Read the actor API key from this environment variable.",
    )
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--no-start-server", action="store_true")
    raw_args = sys.argv[1:] if argv is None else list(argv)
    if any(arg == "--api-key" or arg.startswith("--api-key=") for arg in raw_args):
        # Reject before argparse, whose error message would echo the key.
        parser.error("pass the actor API key through an environment variable with --api-key-env")
    preliminary, _ = parser.parse_known_args(argv)
    try:
        defaults = config_defaults(preliminary.config)
        actions = {action.dest: action for action in parser._actions}
        for key, value in defaults.items():
            if key not in actions or key in {"config", "help"}:
                raise ValueError(f"unknown or unsupported configuration key: {key}")
            action = actions[key]
            if action.type is not None:
                value = action.type(value)
            if action.choices is not None and value not in action.choices:
                raise ValueError(f"invalid value for {key}: {value}")
            defaults[key] = value
        parser.set_defaults(**defaults)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    args = parser.parse_args(argv)
    args.base_url = _endpoint_root(args.base_url)
    if args.top_k is not None:
        if args.top_k < 0:
            parser.error("--top-k must be nonnegative")
        args.top_k_values = str(args.top_k)
    return args


def main() -> None:
    args = parse_args()
    if (
        args.rollout_backend == "hybrid"
        and os.environ.get("ALFWORLD_ISOLATE_ENV_PROCESS") != "1"
    ):
        raise ValueError(
            "Hybrid rollout requires ALFWORLD_ISOLATE_ENV_PROCESS=1; "
            "otherwise use the default process backend"
        )
    if args.episode_max_attempts <= 0:
        raise SystemExit("--episode-max-attempts must be positive")
    if args.actor_enable_thinking and not args.chat_completions:
        raise SystemExit("--actor-enable-thinking requires --chat-completions")
    api_key = ""
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise SystemExit(
                f"API key environment variable is unset: {args.api_key_env}"
            )

    skillbank_path = Path(args.skillbank).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not repo_root.is_dir() or not data_root.is_dir():
        raise SystemExit("--repo-root and --data-root must be existing directories")
    output_dir.mkdir(parents=True, exist_ok=True)

    skills = load_skillbank(
        skillbank_path,
        expected_count=args.expected_skill_count,
    )
    skills_by_id = {skill.skill_id: skill for skill in skills}
    top_k_values = _parse_top_k_values(args.top_k_values, skill_count=len(skills))
    skillbank_name = _slug(args.skillbank_name or skillbank_path.parent.name)
    skillbank_sha256 = _file_sha256(skillbank_path)

    games, sampling_metadata = _select_games(
        data_root=data_root,
        split=args.split,
        tasks_per_type=args.tasks_per_type,
        seed=args.seed,
    )
    counts = {"all": len(games)}
    if len(games) != args.expected_total_games:
        raise SystemExit(
            f"selected {len(games)} games, expected {args.expected_total_games}"
        )
    task_manifest = _task_manifest(games)
    task_manifest_sha256 = _json_sha256(task_manifest)

    run_config = {
        "mode": EVALUATOR_VERSION,
        "query_version": QUERY_VERSION,
        "skillbank_name": skillbank_name,
        "skillbank_path": str(skillbank_path),
        "skillbank_sha256": skillbank_sha256,
        "expected_skill_count": args.expected_skill_count,
        "top_k_values": top_k_values,
        "retrieval_once_per_episode": True,
        "retrieval_updated_during_episode": False,
        "checkpoint_path": args.checkpoint_path,
        "request_model": args.request_model or args.checkpoint_path,
        "actor_request_mode": (
            "chat_completions" if args.chat_completions else "prompt_completions"
        ),
        "actor_enable_thinking": (
            bool(args.actor_enable_thinking) if args.chat_completions else None
        ),
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "split": args.split,
        "selected_game_counts": counts,
        "task_manifest_sha256": task_manifest_sha256,
        "seed": args.seed,
        "max_rollout_steps": args.max_rollout_steps,
        "memory_window": args.memory_window,
        "max_commands": args.max_commands,
        "temperature": args.temperature,
        "embedding_model": args.embedding_model,
        "embedding_max_length": args.embedding_max_length,
        "game_sampling": sampling_metadata,
    }
    episode_cache_signature = _episode_cache_run_signature(run_config)
    run_config["episode_cache_signature_version"] = EPISODE_CACHE_SIGNATURE_VERSION
    run_config["episode_cache_signature"] = episode_cache_signature
    run_signature = _json_sha256(run_config)
    skillbank_manifest = {
        "name": skillbank_name,
        "path": str(skillbank_path),
        "sha256": skillbank_sha256,
        "count": len(skills),
        "models": sorted({skill.model for skill in skills}),
        "skills": [skill.manifest() for skill in skills],
    }
    _atomic_write_json(output_dir / "run_config.json", run_config)
    _atomic_write_json(output_dir / "task_manifest.json", task_manifest)

    print(
        f"[setup] bank={skillbank_name} skills={len(skills)} games={counts} "
        f"top_k={top_k_values} output={output_dir}",
        flush=True,
    )
    embedding_cache_dir = (
        Path(args.embedding_cache_dir).expanduser().resolve()
        if args.embedding_cache_dir
        else None
    )
    retrieval_manifest = _prepare_retrieval_manifest(
        output_dir=output_dir,
        repo_root=repo_root,
        games=games,
        task_manifest=task_manifest,
        skills=skills,
        skillbank_sha256=skillbank_sha256,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        embedding_cache_dir=embedding_cache_dir,
        max_top_k=max(top_k_values),
        max_rollout_steps=args.max_rollout_steps,
        initial_observation_workers=args.initial_observation_workers,
    )

    print(f"[tokens] loading tokenizer for {args.checkpoint_path}", flush=True)
    token_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    token_stats_by_k = {
        str(top_k): _retrieved_skill_token_stats(
            tasks=retrieval_manifest["tasks"],
            skills_by_id=skills_by_id,
            top_k=top_k,
            tokenizer=token_tokenizer,
        )
        for top_k in top_k_values
    }
    for top_k in top_k_values:
        stats = token_stats_by_k[str(top_k)]
        print(
            f"[tokens] k={top_k} avg_total={stats['avg_total_retrieved_tokens']:.1f} "
            f"avg_per_skill={stats['avg_per_skill_retrieved_tokens']:.1f}",
            flush=True,
        )

    if args.prepare_only:
        _write_results(
            output_dir=output_dir,
            status="prepared",
            run_signature=run_signature,
            run_config=run_config,
            skillbank_manifest=skillbank_manifest,
            task_manifest=task_manifest,
            retrieval_manifest=retrieval_manifest,
            metrics_by_k={},
            episodes_by_k={},
            token_stats_by_k=token_stats_by_k,
        )
        print("[done] retrieval prepared; actor rollout skipped", flush=True)
        return

    dp_size = args.dp_size or gpu_count(args.gpus)
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    server_spec = ServerSpec(
        model_path=args.checkpoint_path,
        host=args.host,
        port=args.port,
        gpus=args.gpus,
        tp_size=args.tp_size,
        dp_size=dp_size,
        dtype=args.dtype,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        startup_timeout_s=args.startup_timeout_s,
        api_key=api_key,
        extra_args=tuple(args.server_extra_arg),
        enabled=not args.no_start_server and not args.base_url,
    )
    server_context = (
        nullcontext(base_url) if args.base_url else managed_sglang_server(server_spec)
    )

    metrics_by_k: dict[str, dict[str, Any]] = {}
    episodes_by_k: dict[str, list[dict[str, Any]]] = {}
    try:
        max_workers = _resolve_rollout_worker_count(
            requested=args.rollout_workers,
            game_count=len(games),
            condition_count=len(top_k_values),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    def record_completed_condition(
        top_k: int,
        records: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        key = str(top_k)
        metrics_by_k[key] = metrics
        episodes_by_k[key] = records
        result = _write_results(
            output_dir=output_dir,
            status=(
                "complete" if len(metrics_by_k) == len(top_k_values) else "running"
            ),
            run_signature=run_signature,
            run_config=run_config,
            skillbank_manifest=skillbank_manifest,
            task_manifest=task_manifest,
            retrieval_manifest=retrieval_manifest,
            metrics_by_k=metrics_by_k,
            episodes_by_k=episodes_by_k,
            token_stats_by_k=token_stats_by_k,
        )
        sr = result["sr_by_k"][key]
        print(
            f"[summary] k={top_k} ALL={sr['all']:.4f}",
            flush=True,
        )
        task_type_metrics = {
            task_type: _compact_partition(item)
            for task_type, item in sorted(
                result["metrics_by_k"][key]["per_task_type"].items()
            )
        }
        print(
            f"[summary-task-types] k={top_k} "
            + json.dumps(
                task_type_metrics,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    with server_context as actor_base_url:
        _verify_server_model(
            base_url=actor_base_url,
            api_key=api_key,
            expected_model=args.request_model or args.checkpoint_path,
        )
        if args.rollout_processes <= 0:
            raise ValueError("rollout-processes must be positive")
        processes = min(args.rollout_processes, max_workers)
        threads = math.ceil(max_workers / processes)
        # These settings do not affect seeds, prompts, retrieval or cache identity.
        _atomic_write_json(
            output_dir / "execution_config.json",
            {
                "backend": args.rollout_backend,
                "environment_process_isolation": os.environ.get(
                    "ALFWORLD_ISOLATE_ENV_PROCESS"
                )
                == "1",
                "max_in_flight": max_workers,
                "processes": processes
                if args.rollout_backend == "hybrid"
                else max_workers,
                "threads_per_process": threads
                if args.rollout_backend == "hybrid"
                else 1,
            },
        )
        # Model inference is remote. Avoid BLAS/tokenizer thread multiplication
        # while the spawn workers import their CPU-only rollout dependencies.
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pool = (
            ProcessThreadExecutor(
                processes=processes,
                threads_per_process=threads,
                initializer=initialize_rollout_worker,
                initargs=(None if args.chat_completions else args.checkpoint_path,),
            )
            if args.rollout_backend == "hybrid"
            else ProcessThreadExecutor(
                processes=max_workers,
                threads_per_process=1,
                initializer=initialize_rollout_worker,
                initargs=(None if args.chat_completions else args.checkpoint_path,),
            )
        )
        with pool as executor:
            _run_flattened_top_k_conditions(
                executor=executor,
                top_k_values=top_k_values,
                cache_dir=output_dir / "cache",
                games=games,
                retrieval_tasks=retrieval_manifest["tasks"],
                skills_by_id=skills_by_id,
                run_signature=episode_cache_signature,
                skillbank_name=skillbank_name,
                repo_root=repo_root,
                actor_base_url=actor_base_url,
                checkpoint_path=args.checkpoint_path,
                api_key=api_key,
                timeout_s=args.timeout_s,
                temperature=args.temperature,
                max_rollout_steps=args.max_rollout_steps,
                memory_window=args.memory_window,
                max_commands=args.max_commands,
                seed=args.seed,
                episode_max_attempts=args.episode_max_attempts,
                progress_every=args.progress_every,
                request_model=args.request_model,
                chat_completions=args.chat_completions,
                actor_enable_thinking=args.actor_enable_thinking,
                max_in_flight=max_workers,
                performance_path=output_dir / "performance.json",
                on_condition_complete=record_completed_condition,
            )

    print(f"[done] {output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
