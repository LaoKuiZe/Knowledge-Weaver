# SPDX-License-Identifier: MIT

"""Deterministic task planning, paired marginal scoring, and online-bank selection."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

TaskOutcomes: TypeAlias = Mapping[str, bool]


@dataclass(frozen=True)
class OnlineSkillbankStepPlan:
    """One optimizer step's input-type layout and bank-update type subset."""

    training_global_step: int
    group_input_task_types: tuple[str, ...]
    update_task_types: tuple[str, ...]

    @property
    def group_counts(self) -> dict[str, int]:
        """Return the number of GRPO groups assigned to every input type."""

        return dict(Counter(self.group_input_task_types))


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validated_task_types(
    task_types: Sequence[str], *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(task_types, (str, bytes)):
        raise TypeError("task_types must be a sequence of task-type strings")
    copied = tuple(task_types)
    if not copied and not allow_empty:
        raise ValueError("task_types must not be empty")
    if any(not isinstance(task_type, str) or not task_type for task_type in copied):
        raise ValueError("task types must be non-empty strings")
    if len(set(copied)) != len(copied):
        raise ValueError("task_types must be unique")
    return copied


def _derived_random_seed(
    *, seed: int, training_global_step: int, namespace: str
) -> int:
    material = f"{seed}\0{training_global_step}\0{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def build_online_skillbank_step_plan(
    task_types: Sequence[str],
    *,
    training_global_step: int,
    group_count: int,
    seed: int,
    warmup_steps: int = 20,
    warmup_update_type_count: int = 2,
    later_update_type_count: int = 1,
) -> OnlineSkillbankStepPlan:
    """Balance all task types across groups; independently sample bank-update types."""

    types = _validated_task_types(task_types)
    step = _validate_non_negative_int(
        training_global_step, name="training_global_step"
    )
    count = _validate_non_negative_int(group_count, name="group_count")
    if count < len(types):
        raise ValueError(
            "group_count must be at least the number of task types so every "
            "input type is trained"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    warmup_steps = _validate_non_negative_int(warmup_steps, name="warmup_steps")
    warmup_update_type_count = _validate_non_negative_int(
        warmup_update_type_count, name="warmup_update_type_count"
    )
    later_update_type_count = _validate_non_negative_int(
        later_update_type_count, name="later_update_type_count"
    )
    update_count = (
        warmup_update_type_count if step < warmup_steps else later_update_type_count
    )
    if update_count > len(types):
        raise ValueError("update type count must not exceed the task-type count")

    base_count, remainder = divmod(count, len(types))
    rotation_start = (seed + step) % len(types)
    rotated = types[rotation_start:] + types[:rotation_start]
    group_types = tuple(
        task_type for _ in range(base_count) for task_type in rotated
    ) + rotated[:remainder]

    rng = random.Random(
        _derived_random_seed(
            seed=seed,
            training_global_step=step,
            namespace="online-skillbank-update-types-v1",
        )
    )
    update_types = tuple(rng.sample(list(types), update_count))
    return OnlineSkillbankStepPlan(
        training_global_step=step,
        group_input_task_types=group_types,
        update_task_types=update_types,
    )


def _validate_task_outcomes(
    outcomes: TaskOutcomes,
    *,
    context: str,
    expected_tasks: set[str] | None = None,
) -> dict[str, bool]:
    if not isinstance(outcomes, Mapping):
        raise TypeError(f"{context} outcomes must be a task-to-bool mapping")
    copied = dict(outcomes)
    if not copied:
        raise ValueError(f"{context} outcomes must not be empty")
    if any(not isinstance(task_id, str) or not task_id for task_id in copied):
        raise ValueError(f"{context} task ids must be non-empty strings")
    invalid = [task_id for task_id, won in copied.items() if not isinstance(won, bool)]
    if invalid:
        raise TypeError(
            f"{context} outcomes must contain bool values; invalid tasks: {invalid!r}"
        )
    if expected_tasks is not None and set(copied) != expected_tasks:
        missing = sorted(expected_tasks - set(copied))
        extra = sorted(set(copied) - expected_tasks)
        raise ValueError(
            f"{context} task alignment mismatch: missing={missing!r}, extra={extra!r}"
        )
    return copied


def _validate_candidate_outcomes(
    outcomes: Mapping[str, TaskOutcomes],
    *,
    context: str,
    candidate_keys: set[str],
    expected_tasks: set[str],
) -> dict[str, dict[str, bool]]:
    if not isinstance(outcomes, Mapping):
        raise TypeError(f"{context} must be a sample-to-outcomes mapping")
    if set(outcomes) != candidate_keys:
        missing = sorted(candidate_keys - set(outcomes))
        extra = sorted(set(outcomes) - candidate_keys)
        raise ValueError(
            f"{context} sample alignment mismatch: missing={missing!r}, "
            f"extra={extra!r}"
        )
    return {
        sample_key: _validate_task_outcomes(
            outcomes[sample_key],
            context=f"{context} {sample_key!r}",
            expected_tasks=expected_tasks,
        )
        for sample_key in sorted(candidate_keys)
    }


def _validated_weight(value: float, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    copied = float(value)
    if not math.isfinite(copied) or copied < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return copied


def _mean_bool(outcomes: Mapping[str, bool]) -> float:
    return sum(float(won) for won in outcomes.values()) / len(outcomes)


def compute_online_skillbank_marginal_metrics(
    *,
    no_skill_outcomes: TaskOutcomes,
    retrieved_skill_outcomes: TaskOutcomes,
    singleton_outcomes: Mapping[str, TaskOutcomes],
    augmented_outcomes: Mapping[str, TaskOutcomes],
    bank_weight: float = 0.6,
    standalone_weight: float = 0.4,
) -> dict[str, dict[str, float]]:
    """Compute unclipped paired bank and standalone success deltas on identical tasks."""

    no_skill = _validate_task_outcomes(no_skill_outcomes, context="no-skill")
    expected_tasks = set(no_skill)
    retrieved = _validate_task_outcomes(
        retrieved_skill_outcomes,
        context="retrieved-skill",
        expected_tasks=expected_tasks,
    )
    if not isinstance(singleton_outcomes, Mapping):
        raise TypeError("singleton_outcomes must be a sample-to-outcomes mapping")
    candidate_keys = set(singleton_outcomes)
    if not candidate_keys:
        raise ValueError("singleton_outcomes must not be empty")
    if any(not isinstance(key, str) or not key for key in candidate_keys):
        raise ValueError("candidate sample keys must be non-empty strings")
    singletons = _validate_candidate_outcomes(
        singleton_outcomes,
        context="singleton",
        candidate_keys=candidate_keys,
        expected_tasks=expected_tasks,
    )
    augmented = _validate_candidate_outcomes(
        augmented_outcomes,
        context="augmented",
        candidate_keys=candidate_keys,
        expected_tasks=expected_tasks,
    )
    bank_weight = _validated_weight(bank_weight, name="bank_weight")
    standalone_weight = _validated_weight(
        standalone_weight, name="standalone_weight"
    )
    if bank_weight + standalone_weight <= 0.0:
        raise ValueError("at least one marginal weight must be positive")

    no_skill_sr = _mean_bool(no_skill)
    retrieved_sr = _mean_bool(retrieved)
    metrics: dict[str, dict[str, float]] = {}
    for sample_key in sorted(candidate_keys):
        singleton_sr = _mean_bool(singletons[sample_key])
        augmented_sr = _mean_bool(augmented[sample_key])
        bank_delta = augmented_sr - retrieved_sr
        standalone_delta = singleton_sr - no_skill_sr
        score = bank_weight * bank_delta + standalone_weight * standalone_delta
        metrics[sample_key] = {
            "no_skill_sr": no_skill_sr,
            "retrieved_skill_sr": retrieved_sr,
            "singleton_sr": singleton_sr,
            "augmented_sr": augmented_sr,
            "bank_delta": bank_delta,
            "standalone_delta": standalone_delta,
            "weighted_marginal_score": score,
        }
    return metrics


def select_positive_skillbank_updates(
    *,
    metrics: Mapping[str, Mapping[str, float]],
    candidate_task_types: Mapping[str, str],
    schema_valid: Mapping[str, bool],
    update_task_types: Sequence[str],
    zero_epsilon: float = 0.0,
) -> dict[str, str]:
    """Select positive, schema-valid winners per type; break ties by sample key."""

    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be a non-empty sample mapping")
    candidate_keys = set(metrics)
    if set(candidate_task_types) != candidate_keys:
        raise ValueError("candidate_task_types must align exactly with metrics")
    if set(schema_valid) != candidate_keys:
        raise ValueError("schema_valid must align exactly with metrics")
    task_types = _validated_task_types(update_task_types, allow_empty=True)
    epsilon = _validated_weight(zero_epsilon, name="zero_epsilon")

    scores: dict[str, float] = {}
    for sample_key in candidate_keys:
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError("candidate sample keys must be non-empty strings")
        task_type = candidate_task_types[sample_key]
        if not isinstance(task_type, str) or not task_type:
            raise ValueError("candidate task types must be non-empty strings")
        if not isinstance(schema_valid[sample_key], bool):
            raise TypeError("schema_valid values must be bool")
        raw_score = metrics[sample_key].get("weighted_marginal_score")
        if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
            raise TypeError("weighted_marginal_score must be numeric")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("weighted_marginal_score must be finite")
        scores[sample_key] = score

    selected: dict[str, str] = {}
    for task_type in task_types:
        eligible = [
            sample_key
            for sample_key in candidate_keys
            if candidate_task_types[sample_key] == task_type
            and schema_valid[sample_key]
            and scores[sample_key] > epsilon
        ]
        if eligible:
            selected[task_type] = min(
                eligible, key=lambda sample_key: (-scores[sample_key], sample_key)
            )
    return selected


def rank_online_skillbank_entries(
    *,
    entries: Sequence[Mapping[str, Any]],
    cosine_similarities: Sequence[float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve at most top_k entries; break score ties by skill ID, then source index."""

    if isinstance(entries, (str, bytes)):
        raise TypeError("entries must be a sequence of skill mappings")
    copied_entries = list(entries)
    scores = list(cosine_similarities)
    if len(copied_entries) != len(scores):
        raise ValueError("cosine similarities must align with skill entries")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    validated: list[tuple[str, int, float]] = []
    seen_ids: set[str] = set()
    aligned_entries = zip(copied_entries, scores, strict=True)
    for index, (entry, raw_score) in enumerate(aligned_entries):
        if not isinstance(entry, Mapping):
            raise TypeError("each skill entry must be a mapping")
        skill_id = entry.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError("each skill entry must have a non-empty skill_id")
        if skill_id in seen_ids:
            raise ValueError(f"duplicate skill id: {skill_id}")
        seen_ids.add(skill_id)
        source_index = entry.get("source_index", index)
        if not isinstance(source_index, int) or isinstance(source_index, bool):
            raise TypeError("source_index must be an integer")
        if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
            raise TypeError("cosine similarities must be numeric")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("cosine similarities must be finite")
        validated.append((skill_id, source_index, score))

    ordered = sorted(validated, key=lambda item: (-item[2], item[0], item[1]))[
        : min(top_k, len(validated))
    ]
    return [
        {
            "rank": rank,
            "skill_id": skill_id,
            "source_index": source_index,
            "cosine_similarity": score,
        }
        for rank, (skill_id, source_index, score) in enumerate(ordered, start=1)
    ]


__all__ = [
    "OnlineSkillbankStepPlan",
    "TaskOutcomes",
    "build_online_skillbank_step_plan",
    "compute_online_skillbank_marginal_metrics",
    "rank_online_skillbank_entries",
    "select_positive_skillbank_updates",
]
