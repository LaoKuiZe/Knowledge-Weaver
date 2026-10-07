# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from examples.webshop_skill.core import NO_SKILL_TEXT

from areal.utils import logging
from areal.utils.http_session import get_thread_session
from areal.workflow.skill_eval_baseline import (
    BaselineRecordSchema,
    aggregate_repeated_baseline_runs,
)
from areal.workflow.skill_prompt import DEFAULT_SKILL_PROMPT_TOTAL_TOKEN_BUDGET

logger = logging.getLogger("WebShopSkillWorkflow")
_SERVICE_LIMIT_LOCK = threading.Lock()
_SERVICE_LIMITS: dict[str, threading.BoundedSemaphore] = {}
_GROUP_EPISODE_EXECUTOR_LOCK = threading.Lock()
_GROUP_EPISODE_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_GROUP_COORDINATOR_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_WEBSHOP_SKILL_PROMPT_TOTAL_TOKEN_BUDGET = DEFAULT_SKILL_PROMPT_TOTAL_TOKEN_BUDGET
_WEBSHOP_COMPLETE_TRACE_PROFILES = (
    ("tail8_all_result96", 8, 96, True, True),
    ("tail6_all_result64", 6, 64, True, True),
    ("tail6_sparse_result48", 6, 48, True, False),
    ("tail4_sparse_result32", 4, 32, False, False),
    ("tail2_sparse_result24", 2, 24, False, False),
    ("detail64_sparse_result16", 1, 16, True, False),
    ("actions_only", 0, 1, True, False),
)
# Names the group-reward artifact directory and the "mode" field of its index.
_SKILLBANK_MODE = "online_skillbank"
_RETRYABLE_ROLLOUT_STATUSES = frozenset({"error", "rollout_error"})
_WEBSHOP_EVAL_BASELINE_SCHEMA = BaselineRecordSchema(
    records_key="episodes",
    task_id_key="task_index",
    success_key="success",
    reward_key="reward",
    category_key="category",
)


def _aggregate_webshop_eval_no_skill_baseline_runs(
    runs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate repeated WebShop baselines under the shared eval contract."""

    aggregate = aggregate_repeated_baseline_runs(
        runs,
        schema=_WEBSHOP_EVAL_BASELINE_SCHEMA,
    )
    metrics = dict(aggregate.get("metrics", {}))
    metrics_std = dict(aggregate.get("metrics_std", {}))
    return {
        **aggregate,
        "sr": float(metrics.get("success_rate", 0.0)),
        "sr_std": float(metrics_std.get("success_rate", 0.0)),
        "mean_reward": float(metrics.get("mean_reward", 0.0)),
        "mean_reward_std": float(metrics_std.get("mean_reward", 0.0)),
        "per_category": dict(aggregate.get("categories", {})),
    }


def _group_episode_executor(max_workers: int) -> ThreadPoolExecutor:
    workers = max(1, int(max_workers))
    with _GROUP_EPISODE_EXECUTOR_LOCK:
        executor = _GROUP_EPISODE_EXECUTORS.get(workers)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="webshop-group-episode",
            )
            _GROUP_EPISODE_EXECUTORS[workers] = executor
        return executor


def _group_coordinator_executor(max_workers: int) -> ThreadPoolExecutor:
    workers = max(1, int(max_workers))
    with _GROUP_EPISODE_EXECUTOR_LOCK:
        executor = _GROUP_COORDINATOR_EXECUTORS.get(workers)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="webshop-group-coordinator",
            )
            _GROUP_COORDINATOR_EXECUTORS[workers] = executor
        return executor


def _normalized_skill_body(text: str) -> str:
    return " ".join(str(text or "").split())


def _skill_body_digest(text: str) -> str:
    return hashlib.sha256(_normalized_skill_body(text).encode("utf-8")).hexdigest()


def _format_skill_bodies(bodies: Sequence[str]) -> str:
    cleaned = [
        _normalized_skill_body(body) for body in bodies if _normalized_skill_body(body)
    ]
    if not cleaned:
        return NO_SKILL_TEXT
    if len(cleaned) == 1:
        return cleaned[0]
    return "Retrieved guidance skills:\n" + "\n\n".join(
        f"{index}. {body}" for index, body in enumerate(cleaned, start=1)
    )


def _service_semaphore(url: str, workers: int) -> threading.BoundedSemaphore:
    key = f"{url.rstrip('/')}::{max(1, workers)}"
    with _SERVICE_LIMIT_LOCK:
        semaphore = _SERVICE_LIMITS.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(max(1, workers))
            _SERVICE_LIMITS[key] = semaphore
        return semaphore


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


class WebShopEnvironmentClient:
    def __init__(self, base_url: str, *, timeout_s: float, workers: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = max(1.0, float(timeout_s))
        self.workers = max(1, int(workers))
        self.semaphore = _service_semaphore(self.base_url, self.workers)

    def _post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.semaphore:
            response = get_thread_session().post(
                self.base_url + route, json=payload, timeout=self.timeout_s
            )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError(
                f"benchmark service returned non-object JSON: {result!r}"
            )
        if result.get("status") == "error" and "task_index" not in result:
            raise RuntimeError(
                "benchmark service error: "
                + str(result.get("error") or result.get("traceback") or result)
            )
        return result

    def manifest(
        self,
        *,
        train_count: int,
        eval_count: int,
        seed: int,
        unique_asin: bool,
        task_split: str = "training_holdout",
    ) -> dict[str, Any]:
        return self._post(
            "/task-manifest",
            {
                "train_count": int(train_count),
                "eval_count": int(eval_count),
                "seed": int(seed),
                "unique_asin": bool(unique_asin),
                **(
                    {"task_split": task_split}
                    if task_split != "training_holdout"
                    else {}
                ),
            },
        )

    def rollout(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/rollout", payload)

    def task_contexts(self, task_indices: Sequence[int]) -> list[dict[str, Any]]:
        payload = self._post(
            "/task-contexts",
            {"task_indices": [int(index) for index in task_indices]},
        )
        contexts = payload.get("contexts")
        if not isinstance(contexts, list):
            raise RuntimeError("benchmark service returned malformed task contexts")
        return [dict(context) for context in contexts if isinstance(context, dict)]
