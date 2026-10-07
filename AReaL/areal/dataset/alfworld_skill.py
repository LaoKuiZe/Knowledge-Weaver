# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from datasets import Dataset

ALFWORLD_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
)
# Every training round mixes all input task types under one artifact category.
MIXED_TASK_TYPE = "mixed_id_tasks"
MIXED_PROMPT_CATEGORY = "mixed_id_tasks"


def _as_task_types(value: Any | None) -> list[str]:
    if value is None:
        return list(ALFWORLD_TASK_TYPES)
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.replace(",", " ").split()]
    else:
        raw_items = [str(item).strip() for item in value]
    task_types = [item for item in raw_items if item]
    unknown = [item for item in task_types if item not in ALFWORLD_TASK_TYPES]
    if unknown:
        raise ValueError(
            "Unknown ALFWorld task type(s): "
            + ", ".join(unknown)
            + ". Expected one of: "
            + ", ".join(ALFWORLD_TASK_TYPES)
        )
    return task_types


def get_alfworld_skill_rl_dataset(
    path: str,
    split: str | None,
    tokenizer=None,
    max_length: int | None = None,
    *,
    task_types: list[str] | str | None = None,
    rounds_per_category: int = 5,
    groups_per_round: int = 16,
    mode: str = "train",
    eval_k: int = 6,
    seed: int = 1,
    **kwargs: Any,
) -> Dataset:
    """Build round descriptors; the workflow samples prompts from its trajectory pool."""

    _ = path, split, tokenizer, max_length
    num_rounds = int(kwargs.pop("num_rounds", rounds_per_category))
    mode = str(mode or "train")
    source_task_types = _as_task_types(task_types)
    if mode == "eval":
        if eval_k <= 0:
            raise ValueError(f"eval_k must be positive, got {eval_k}")
        return Dataset.from_list(
            [
                {
                    "mode": "eval",
                    "task_type": MIXED_TASK_TYPE,
                    "prompt_category": MIXED_PROMPT_CATEGORY,
                    "eval_index": index,
                    "eval_k": int(eval_k),
                    "source_task_types": source_task_types,
                    "seed": seed,
                }
                for index in range(int(eval_k))
            ]
        )

    if mode != "train":
        raise ValueError(f"Unsupported ALFWorld skill dataset mode: {mode!r}")

    if rounds_per_category <= 0:
        raise ValueError(
            f"rounds_per_category must be positive, got {rounds_per_category}"
        )
    if num_rounds <= 0:
        raise ValueError(f"num_rounds must be positive, got {num_rounds}")
    if groups_per_round <= 0:
        raise ValueError(f"groups_per_round must be positive, got {groups_per_round}")

    rows: list[dict[str, Any]] = []
    for round_index in range(num_rounds):
        for group_index in range(groups_per_round):
            rows.append(
                {
                    "task_type": MIXED_TASK_TYPE,
                    "prompt_category": MIXED_PROMPT_CATEGORY,
                    "category_index": 0,
                    "round_in_category": round_index,
                    "global_round": round_index,
                    "group_index": group_index,
                    "groups_per_round": groups_per_round,
                    "rounds_per_category": num_rounds,
                    "source_task_types": source_task_types,
                    "seed": seed,
                }
            )
    return Dataset.from_list(rows)
