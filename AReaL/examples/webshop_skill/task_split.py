# SPDX-License-Identifier: MIT

"""Official WebShop goal partitions, after SimServer's canonical seed-233 shuffle."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

OFFICIAL_TRAIN_START = 1500
OFFICIAL_VALIDATION_START = 500
OFFICIAL_ENVIRONMENT_SEED = 233
OFFICIAL_GOAL_SHUFFLE_SEED = 233


def validate_training_settings(settings: Mapping[str, Any]) -> None:
    """Fail before simulator/GPU startup if official training would use a subset."""
    split = settings.get("task_split", "training_holdout")
    if split == "training_holdout":
        return
    if split != "official_train":
        raise ValueError(f"unknown WebShop training task_split: {split}")
    if settings.get("human_goals") is not True:
        raise ValueError("official_train requires human_goals=true")
    if settings.get("num_products") is not None:
        raise ValueError("official_train requires the full product catalog")
    if settings.get("environment_seed") != OFFICIAL_ENVIRONMENT_SEED:
        raise ValueError(
            f"official_train requires environment_seed={OFFICIAL_ENVIRONMENT_SEED}"
        )
    if int(settings.get("train_task_count", 0)) != -1:
        raise ValueError(
            "official_train requires train_task_count=-1 (all train goals)"
        )
    if settings.get("unique_asin_split") is not False:
        raise ValueError("official_train requires unique_asin_split=false")
    if not 0 < int(settings.get("eval_task_count", 0)) <= 1000:
        raise ValueError("official validation task count must be in [1, 1000]")


def official_train_task_split(
    goals: Sequence[dict[str, Any]], *, eval_count: int, seed: int
) -> dict[str, Any]:
    """Keep every official train goal; sample only the internal validation subset."""
    if len(goals) <= OFFICIAL_TRAIN_START:
        raise ValueError(
            f"official_train requires more than {OFFICIAL_TRAIN_START} human goals"
        )
    if not 0 < eval_count <= 1000:
        raise ValueError("official validation task count must be in [1, 1000]")
    train_indices = list(range(OFFICIAL_TRAIN_START, len(goals)))
    eval_indices = random.Random(seed).sample(
        range(OFFICIAL_VALIDATION_START, OFFICIAL_TRAIN_START), k=eval_count
    )

    def records(indices: list[int]) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "asin": str(goals[index].get("asin") or ""),
                "category": str(goals[index].get("category") or "unknown"),
                "instruction": str(goals[index].get("instruction_text") or ""),
            }
            for index in indices
        ]

    train_tasks, eval_tasks = records(train_indices), records(eval_indices)

    def summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(tasks),
            "unique_asins": len({row["asin"] for row in tasks}),
            "category_counts": dict(
                sorted(Counter(row["category"] for row in tasks).items())
            ),
        }

    manifest = {
        "version": 2,
        "task_split": "official_train",
        "split": "official_train_validation_subset",
        "human_goals": True,
        "num_products": None,
        "environment_seed": OFFICIAL_ENVIRONMENT_SEED,
        "goal_shuffle_seed": OFFICIAL_GOAL_SHUFFLE_SEED,
        "seed": int(seed),
        "unique_asin": False,
        "goal_count": len(goals),
        "candidate_count": len(train_indices),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
        "train_summary": summary(train_tasks),
        "eval_summary": summary(eval_tasks),
    }
    # Resume checks compare this signature instead of the full manifest.
    manifest["manifest_signature"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return manifest


def validate_official_manifest(
    manifest: Mapping[str, Any], *, eval_count: int, seed: int
) -> None:
    """Require all official train goals and the seeded validation sample."""
    goal_count = int(manifest.get("goal_count", 0))
    expected_eval = random.Random(seed).sample(
        range(OFFICIAL_VALIDATION_START, OFFICIAL_TRAIN_START), k=eval_count
    )
    if (
        manifest.get("task_split") != "official_train"
        or manifest.get("human_goals") is not True
        or manifest.get("num_products") is not None
        or manifest.get("environment_seed") != OFFICIAL_ENVIRONMENT_SEED
        or manifest.get("goal_shuffle_seed") != OFFICIAL_GOAL_SHUFFLE_SEED
        or manifest.get("unique_asin") is not False
        or manifest.get("seed") != seed
        or goal_count <= OFFICIAL_TRAIN_START
        or manifest.get("train_indices")
        != list(range(OFFICIAL_TRAIN_START, goal_count))
        or manifest.get("eval_indices") != expected_eval
    ):
        raise RuntimeError(
            "WebShop manifest does not cover the full official train split"
        )


def task_selection_contract(
    *, task_split: str, train_count: int, eval_count: int, seed: int, unique_asin: bool
) -> dict[str, Any]:
    return {
        "task_split": task_split,
        "train_task_count": train_count,
        "eval_task_count": eval_count,
        "task_seed": seed,
        "unique_asin_split": unique_asin,
    }
