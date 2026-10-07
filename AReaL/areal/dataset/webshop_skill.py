# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from datasets import Dataset

WEBSHOP_TASK_TYPE = "webshop_mixed_tasks"
WEBSHOP_PROMPT_CATEGORY = "webshop_product_search"


def get_webshop_skill_rl_dataset(
    path: str,
    split: str | None,
    tokenizer=None,
    max_length: int | None = None,
    *,
    rounds: int = 5,
    groups_per_round: int = 8,
    mode: str = "train",
    eval_k: int = 4,
    seed: int = 1,
    **kwargs: Any,
) -> Dataset:
    """Build round descriptors; the workflow samples goals from a fixed task manifest."""

    _ = path, split, tokenizer, max_length, kwargs
    mode = str(mode or "train")
    if mode == "eval":
        if eval_k <= 0:
            raise ValueError("eval_k must be positive")
        return Dataset.from_list(
            [
                {
                    "mode": "eval",
                    "task_type": WEBSHOP_TASK_TYPE,
                    "prompt_category": WEBSHOP_PROMPT_CATEGORY,
                    "eval_index": index,
                    "eval_k": int(eval_k),
                    "seed": int(seed),
                }
                for index in range(int(eval_k))
            ]
        )
    if mode != "train":
        raise ValueError(f"unsupported WebShop skill dataset mode: {mode!r}")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if groups_per_round <= 0:
        raise ValueError("groups_per_round must be positive")

    rows = []
    for round_index in range(int(rounds)):
        for group_index in range(int(groups_per_round)):
            rows.append(
                {
                    "mode": "train",
                    "task_type": WEBSHOP_TASK_TYPE,
                    "prompt_category": WEBSHOP_PROMPT_CATEGORY,
                    "round_in_category": round_index,
                    "global_round": round_index,
                    "group_index": group_index,
                    "groups_per_round": int(groups_per_round),
                    "rounds": int(rounds),
                    "seed": int(seed),
                }
            )
    return Dataset.from_list(rows)
