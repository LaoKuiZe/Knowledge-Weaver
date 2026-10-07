# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import io
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from areal.workflow.alfworld_runtime import (
    MIXED_TASK_TYPE,
    _acquire_json_lock,
    _atomic_write_json,
    _atomic_write_text,
    _read_json,
    _release_json_lock,
)


def _checkpoint_model_path(checkpoint_model_root: Path | None, global_step: int) -> str:
    if checkpoint_model_root is None or global_step < 0:
        return ""
    matches = sorted(checkpoint_model_root.glob(f"epoch*globalstep{global_step}"))
    return str(matches[-1]) if matches else ""


def _mean_skill_sr(skill_metrics: list[dict[str, Any]], split: str) -> float:
    srs = [float(item.get(split, {}).get("sr", 0.0)) for item in skill_metrics]
    return mean(srs) if srs else 0.0


def _schema_valid_rate(skill_metrics: list[dict[str, Any]]) -> float:
    if not skill_metrics:
        return 0.0
    valid_count = sum(1 for item in skill_metrics if item.get("schema_valid"))
    return valid_count / len(skill_metrics)


def _mean_eval_metric(
    skill_metrics: list[dict[str, Any]],
    path: Sequence[str],
    *,
    default: float = 0.0,
) -> float:
    values: list[float] = []
    for item in skill_metrics:
        current: Any = item
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is None:
            continue
        try:
            values.append(float(current))
        except (TypeError, ValueError):
            continue
    return mean(values) if values else default


def _eval_history_rows(
    eval_root: Path, task_types: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(eval_root.glob("globalstep_*/summary.json")):
        try:
            summary = _read_json(summary_path)
        except Exception:
            continue
        if summary.get("status") != "complete":
            continue
        skill_metrics = summary.get("skills", [])
        if not isinstance(skill_metrics, list):
            skill_metrics = []
        global_step = int(summary.get("checkpoint_global_step", -1))
        all_mean_sr = summary.get("all_mean_sr")
        if all_mean_sr is None:
            all_mean_sr = _mean_skill_sr(skill_metrics, "overall")
        schema_valid_count = int(
            summary.get(
                "schema_valid_count",
                sum(1 for item in skill_metrics if item.get("schema_valid")),
            )
        )
        schema_valid_rate = summary.get("schema_valid_rate")
        if schema_valid_rate is None:
            schema_valid_rate = _schema_valid_rate(skill_metrics)
        matched_task_type_srs = summary.get("matched_task_type_srs") or {}
        rows.append(
            {
                "global_step": global_step,
                "all_sr": float(all_mean_sr),
                **{
                    f"matched_{task_type}_sr": float(
                        matched_task_type_srs.get(task_type, 0.0)
                    )
                    for task_type in task_types
                },
                "all_delta_sr": float(summary.get("all_mean_baseline_delta_sr", 0.0)),
                "all_mean_0_to_1": float(summary.get("all_mean_0_to_1", 0.0)),
                "all_mean_1_to_0": float(summary.get("all_mean_1_to_0", 0.0)),
                "schema_valid_rate": float(schema_valid_rate),
                "schema_valid_count": schema_valid_count,
                "completed_skill_count": int(summary.get("completed_skill_count", 0)),
                "expected_skill_count": int(summary.get("expected_skill_count", 0)),
                "status": str(summary.get("status", "")),
                "checkpoint_model_path": str(summary.get("checkpoint_model_path", "")),
                "eval_dir": str(summary_path.parent),
            }
        )
    return sorted(rows, key=lambda row: (row["global_step"], row["eval_dir"]))


def _write_eval_metrics_csv(
    eval_root: Path, rows: list[dict[str, Any]], task_types: Sequence[str]
) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "global_step",
            "all_sr",
            *(f"matched_{task_type}_sr" for task_type in task_types),
            "all_delta_sr",
            "all_mean_0_to_1",
            "all_mean_1_to_0",
            "schema_valid_rate",
            "schema_valid_count",
            "completed_skill_count",
            "expected_skill_count",
            "status",
            "checkpoint_model_path",
            "eval_dir",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _atomic_write_text(eval_root / "metrics_history.csv", output.getvalue())


def _write_eval_metrics_history(eval_root: Path, task_types: Sequence[str]) -> None:
    rows = _eval_history_rows(eval_root, task_types)
    _write_eval_metrics_csv(eval_root, rows, task_types)


def _write_eval_summary(
    *,
    eval_dir: Path,
    artifact_dir: Path,
    expected_skill_count: int,
    checkpoint_model_root: Path | None,
    task_types: Sequence[str],
) -> dict[str, Any]:
    lock_path = eval_dir / "summary.lock"
    fd = _acquire_json_lock(lock_path)
    try:
        skill_metrics: list[dict[str, Any]] = []
        for metrics_path in sorted(eval_dir.glob("skills/*/metrics.json")):
            try:
                metrics = _read_json(metrics_path)
            except Exception:
                continue
            metrics["metrics_path"] = str(metrics_path)
            skill_metrics.append(metrics)

        all_srs = [
            float(item.get("overall", {}).get("sr", 0.0)) for item in skill_metrics
        ]
        matched_task_type_srs: dict[str, float] = {}
        for item in skill_metrics:
            input_task_type = str(item.get("input_task_type") or "")
            if not input_task_type or input_task_type == MIXED_TASK_TYPE:
                continue
            matched_task_type_srs[input_task_type] = float(
                item.get("matched_task_type_sr", 0.0) or 0.0
            )
        all_delta_sr = _mean_eval_metric(
            skill_metrics, ("baseline_delta", "overall", "delta_sr")
        )
        all_mean_0_to_1 = _mean_eval_metric(
            skill_metrics, ("baseline_delta", "overall", "improved_0_to_1")
        )
        all_mean_1_to_0 = _mean_eval_metric(
            skill_metrics, ("baseline_delta", "overall", "regressed_1_to_0")
        )
        checkpoint_global_step = -1
        if skill_metrics:
            checkpoint_global_step = max(
                int(item.get("checkpoint_global_step", -1)) for item in skill_metrics
            )
        checkpoint_path = _checkpoint_model_path(
            checkpoint_model_root, checkpoint_global_step
        )
        complete = len(skill_metrics) >= expected_skill_count
        schema_valid_count = sum(
            1 for item in skill_metrics if item.get("schema_valid")
        )
        schema_valid_rate = _schema_valid_rate(skill_metrics)
        summary = {
            "status": "complete" if complete else "partial",
            "checkpoint_global_step": checkpoint_global_step,
            "checkpoint_model_path": checkpoint_path,
            "expected_skill_count": expected_skill_count,
            "completed_skill_count": len(skill_metrics),
            "schema_valid_count": schema_valid_count,
            "schema_valid_rate": schema_valid_rate,
            "all_mean_sr": mean(all_srs) if all_srs else 0.0,
            "matched_task_type_srs": matched_task_type_srs,
            "all_mean_baseline_delta_sr": all_delta_sr,
            "all_mean_0_to_1": all_mean_0_to_1,
            "all_mean_1_to_0": all_mean_1_to_0,
            "skills": skill_metrics,
            "updated_at": time.time(),
        }
        _atomic_write_json(eval_dir / "summary.json", summary)
        _write_eval_metrics_history(artifact_dir / "eval", task_types)

        if complete:
            best_path = artifact_dir / "eval" / "best_checkpoint.json"
            best_summary: dict[str, Any] | None = None
            best_summary_dir: Path | None = None
            best_score = -1.0
            best_step = -1
            for summary_path in sorted(
                (artifact_dir / "eval").glob("globalstep_*/summary.json")
            ):
                try:
                    candidate = _read_json(summary_path)
                except Exception:
                    continue
                if candidate.get("status") != "complete":
                    continue
                try:
                    candidate_score = float(candidate.get("all_mean_sr", -1.0))
                except (TypeError, ValueError):
                    continue
                candidate_step = int(candidate.get("checkpoint_global_step", -1))
                if (
                    best_summary is None
                    or candidate_score > best_score
                    or (candidate_score == best_score and candidate_step > best_step)
                ):
                    best_summary = candidate
                    best_summary_dir = summary_path.parent
                    best_score = candidate_score
                    best_step = candidate_step
            if best_summary is not None:
                best = {
                    **best_summary,
                    "selection_metric": "alfworld_eval_all_sr",
                    "selection_summary_key": "all_mean_sr",
                    "selection_value": best_score,
                    "eval_dir": str(best_summary_dir)
                    if best_summary_dir is not None
                    else "",
                    "selected_at": time.time(),
                }
                _atomic_write_json(best_path, best)
        return summary
    finally:
        _release_json_lock(fd, lock_path)
