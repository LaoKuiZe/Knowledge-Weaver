"""The frozen 100-task subset of WebShop's official human-goal test split."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

SUBSET_FILE = (
    Path(__file__).resolve().parents[2]
    / "task_sets/webshop_official_test100_seed42.json"
)
BUNDLED_MANIFEST_FILE = (
    Path(__file__).resolve().parents[3] / "eval_data/webshop/tasks.json"
)
SUBSET_SCHEMA = "webshop_official_test_subset_v1"
MANIFEST_SCHEMA = "webshop_official_test_manifest_v1"


def load_subset():
    subset = json.loads(SUBSET_FILE.read_text())
    expected = random.Random(42).sample(range(500), 100)
    if (
        subset.get("schema") != SUBSET_SCHEMA
        or subset.get("split") != "test"
        or subset.get("human_goals") is not True
        or subset.get("num_products") is not None
        or subset.get("environment_seed") != 233
        or subset.get("goal_shuffle_seed") != 233
        or subset.get("sampling_seed") != 42
        or subset.get("official_test_range") != [0, 500]
        or subset.get("task_indices") != expected
    ):
        raise ValueError("invalid frozen official test100 subset")
    return subset


def validate_settings(config):
    if (
        config.get("human_goals") is not True
        or config.get("num_products") is not None
        or config.get("environment_seed") != 233
        or config.get("env_service_url")
    ):
        raise ValueError(
            "official test requires local human_goals=true, the full product "
            "catalog (num_products=null), and environment_seed=233"
        )
    if config.get("eval_task_count", 100) != 100 or config.get("task_seed", 42) != 42:
        raise ValueError("official test100 has fixed count=100 and sampling seed=42")


def task_manifest(goals, config, saved_path=None):
    """Validate bundled goals against official seed-233 test indices without deduplicating or re-splitting."""
    validate_settings(config)
    subset = load_subset()
    if len(goals) <= 1500:
        raise ValueError("official test requires the full human-goal catalog")
    indices = subset["task_indices"]
    records = []
    for index in indices:
        goal = goals[index]
        if not goal.get("asin") or not goal.get("instruction_text"):
            raise ValueError(f"invalid official test goal at index {index}")
        records.append(
            {
                "index": index,
                "asin": str(goal["asin"]),
                "category": str(goal.get("category") or "unknown"),
                "instruction": str(goal["instruction_text"]),
                "attributes": goal["attributes"],
                "goal_options": goal["goal_options"],
                "price_upper": goal["price_upper"],
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "version": 1,
        "split": "official_test_subset",
        "subset": subset,
        "seed": subset["sampling_seed"],
        "unique_asin": False,
        "goal_count": len(goals),
        "candidate_count": 500,
        "train_indices": [],
        "train_tasks": [],
        "train_summary": {"count": 0, "unique_asins": 0, "category_counts": {}},
        "eval_indices": indices,
        "eval_tasks": records,
        "eval_summary": {
            "count": len(indices),
            "unique_asins": len({row["asin"] for row in records}),
            "category_counts": dict(
                sorted(Counter(r["category"] for r in records).items())
            ),
        },
    }
    bundled = json.loads(BUNDLED_MANIFEST_FILE.read_text())
    if manifest != bundled:
        raise ValueError(
            "official test tasks differ from the bundled evaluation definitions; "
            "verify the full human-goal catalog and environment_seed=233"
        )
    if saved_path is not None:
        saved = json.loads(Path(saved_path).read_text())
        if saved.get("schema") == SUBSET_SCHEMA:
            if saved != subset:
                raise ValueError(
                    "saved subset differs from the frozen official test100"
                )
        elif saved.get("schema") == MANIFEST_SCHEMA:
            if saved != manifest:
                raise ValueError(
                    "official test tasks differ from the saved preflight manifest"
                )
        else:
            raise ValueError(
                "official test cannot reuse a synthetic training task manifest"
            )
    return manifest
