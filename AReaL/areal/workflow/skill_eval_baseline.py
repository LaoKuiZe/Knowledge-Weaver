# SPDX-License-Identifier: MIT

"""Cache, resume, aggregate, and persist repeated no-skill baseline evaluations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

_REPEAT_SEED_STRIDE = 100_000


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_stale_lock(
    lock_path: Path, *, foreign_host_stale_after_s: float | None = None
) -> bool:
    try:
        payload = _read_json(lock_path)
    except Exception:
        return False
    lock_host = payload.get("host")
    if lock_host not in (None, "", socket.gethostname()):
        if foreign_host_stale_after_s is None:
            return False
        try:
            claimed_at = float(payload.get("claimed_at", 0.0))
        except (TypeError, ValueError):
            return False
        if time.time() - claimed_at < max(0.0, foreign_host_stale_after_s):
            return False
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return False
        return True
    try:
        pid = int(payload.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if _process_is_alive(pid):
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return True


def _acquire_lock(
    lock_path: Path,
    *,
    timeout_s: float,
    foreign_host_stale_after_s: float | None,
) -> tuple[int, str]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    owner_token = uuid.uuid4().hex
    while True:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            _remove_stale_lock(
                lock_path,
                foreign_host_stale_after_s=foreign_host_stale_after_s,
            )
            if time.monotonic() - started >= timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(0.2)
            continue
        with os.fdopen(descriptor, "w") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "claimed_at": time.time(),
                    "owner_token": owner_token,
                },
                handle,
            )
        return os.open(str(lock_path), os.O_RDONLY), owner_token


def _release_lock(descriptor: int, lock_path: Path, owner_token: str) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass
    try:
        payload = _read_json(lock_path)
    except Exception:
        payload = {}
    if payload.get("owner_token") == owner_token:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def stable_baseline_signature(payload: Mapping[str, Any]) -> str:
    """Return the canonical signature used by all benchmark baseline caches."""

    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _numeric_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


@dataclass(frozen=True)
class BaselineRecordSchema:
    """Map environment episode fields into the shared aggregation contract."""

    records_key: str
    task_id_key: str
    success_key: str
    reward_key: str | None = None
    split_key: str | None = None
    category_key: str | None = None


def aggregate_repeated_baseline_runs(
    runs: Sequence[dict[str, Any]],
    *,
    schema: BaselineRecordSchema,
) -> dict[str, Any]:
    """Aggregate aligned per-task outcomes and per-repeat benchmark metrics."""

    if not runs:
        raise ValueError("at least one baseline run is required")

    records_by_run: list[list[dict[str, Any]]] = []
    identities: list[Any] | None = None
    for run in runs:
        records = run.get(schema.records_key)
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ValueError(
                f"baseline run {schema.records_key} must be a list of objects"
            )
        current_identities = [record.get(schema.task_id_key) for record in records]
        if identities is None:
            identities = current_identities
        elif current_identities != identities:
            raise ValueError("baseline runs must contain the same aligned tasks")
        records_by_run.append(records)

    metric_rows = [_numeric_metrics(run.get("metrics")) for run in runs]
    for row, records in zip(metric_rows, records_by_run, strict=True):
        successes = [float(bool(record.get(schema.success_key))) for record in records]
        rewards = [
            (
                float(record.get(schema.reward_key, 0.0))
                if schema.reward_key is not None
                else success
            )
            for record, success in zip(records, successes, strict=True)
        ]
        row.setdefault("success_rate", mean(successes) if successes else 0.0)
        row.setdefault("mean_reward", mean(rewards) if rewards else 0.0)

    metric_keys = sorted({key for row in metric_rows for key in row})
    metrics = {
        key: mean(float(row.get(key, 0.0)) for row in metric_rows)
        for key in metric_keys
    }
    metrics_std = {
        key: (
            pstdev(float(row.get(key, 0.0)) for row in metric_rows)
            if len(metric_rows) > 1
            else 0.0
        )
        for key in metric_keys
    }

    task_summaries: list[dict[str, Any]] = []
    for position, task_id in enumerate(identities or []):
        repeated = [records[position] for records in records_by_run]
        successes = [bool(record.get(schema.success_key)) for record in repeated]
        rewards = [
            (
                float(record.get(schema.reward_key, 0.0))
                if schema.reward_key is not None
                else float(success)
            )
            for record, success in zip(repeated, successes, strict=True)
        ]
        first = repeated[0]
        summary = {
            "task_id": task_id,
            "repeat_count": len(repeated),
            "success_rate": mean(float(value) for value in successes),
            "success_std": (
                pstdev(float(value) for value in successes)
                if len(successes) > 1
                else 0.0
            ),
            "mean_reward": mean(rewards),
            "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "repeat_successes": successes,
            "repeat_rewards": rewards,
            "representative": first,
        }
        if schema.split_key is not None:
            summary["split"] = str(first.get(schema.split_key, ""))
        if schema.category_key is not None:
            summary["category"] = str(first.get(schema.category_key, ""))
        task_summaries.append(summary)

    def group_metrics(
        summaries: Sequence[dict[str, Any]],
        *,
        run_rates: Sequence[float],
        run_rewards: Sequence[float],
    ) -> dict[str, Any]:
        return {
            "task_count": len(summaries),
            "success_rate": (
                mean(float(item["success_rate"]) for item in summaries)
                if summaries
                else 0.0
            ),
            "success_rate_std": (pstdev(run_rates) if len(run_rates) > 1 else 0.0),
            "mean_reward": (
                mean(float(item["mean_reward"]) for item in summaries)
                if summaries
                else 0.0
            ),
            "mean_reward_std": (pstdev(run_rewards) if len(run_rewards) > 1 else 0.0),
            "repeat_success_rates": list(run_rates),
            "repeat_mean_rewards": list(run_rewards),
        }

    def grouped(field: str, source_key: str | None) -> dict[str, dict[str, Any]]:
        if source_key is None:
            return {}
        values = sorted(
            {
                str(record.get(source_key, ""))
                for records in records_by_run
                for record in records
                if str(record.get(source_key, ""))
            }
        )
        result: dict[str, dict[str, Any]] = {}
        for value in values:
            summaries = [
                summary for summary in task_summaries if summary.get(field) == value
            ]
            run_rates = []
            run_rewards = []
            for records in records_by_run:
                selected = [
                    record
                    for record in records
                    if str(record.get(source_key, "")) == value
                ]
                outcomes = [
                    float(bool(record.get(schema.success_key))) for record in selected
                ]
                rewards = [
                    (
                        float(record.get(schema.reward_key, 0.0))
                        if schema.reward_key is not None
                        else outcome
                    )
                    for record, outcome in zip(selected, outcomes, strict=True)
                ]
                run_rates.append(mean(outcomes) if outcomes else 0.0)
                run_rewards.append(mean(rewards) if rewards else 0.0)
            result[value] = group_metrics(
                summaries, run_rates=run_rates, run_rewards=run_rewards
            )
        return result

    return {
        "metrics": metrics,
        "metrics_std": metrics_std,
        "task_summaries": task_summaries,
        "splits": grouped("split", schema.split_key),
        "categories": grouped("category", schema.category_key),
    }


class RepeatedNoSkillBaselinePipeline:
    """Execute and cache fixed repeated baseline rollouts for one task set."""

    def __init__(
        self,
        *,
        cache_root: Path,
        signature_payload: Mapping[str, Any],
        repeat_count: int,
        seed_base: int,
        task_count: int,
        final_filename: str,
        metadata: Mapping[str, Any] | None = None,
        lock_timeout_s: float = 7200.0,
        foreign_host_stale_after_s: float = 120.0,
        lock_heartbeat_interval_s: float = 30.0,
    ) -> None:
        if repeat_count <= 0:
            raise ValueError("repeat_count must be positive")
        if task_count <= 0:
            raise ValueError("task_count must be positive")
        self.repeat_count = int(repeat_count)
        self.seed_base = int(seed_base)
        self.task_count = int(task_count)
        self.final_filename = final_filename
        self.lock_timeout_s = float(lock_timeout_s)
        self.foreign_host_stale_after_s = float(foreign_host_stale_after_s)
        self.lock_heartbeat_interval_s = float(lock_heartbeat_interval_s)
        self.metadata = dict(metadata or {})
        canonical_payload = {
            "pipeline": "repeated_no_skill_baseline_v1",
            "repeat_count": self.repeat_count,
            "seed_base": self.seed_base,
            **dict(signature_payload),
        }
        self.signature = stable_baseline_signature(canonical_payload)
        self.baseline_dir = Path(cache_root) / self.signature

    def repeat_seed(self, repeat_index: int) -> int:
        return self.seed_base + int(repeat_index) * _REPEAT_SEED_STRIDE

    def _read_complete(self) -> dict[str, Any] | None:
        result_path = self.baseline_dir / self.final_filename
        checkpoint_path = self.baseline_dir / "checkpoint.json"
        if not result_path.exists() or not checkpoint_path.exists():
            return None
        try:
            result = _read_json(result_path)
            checkpoint = _read_json(checkpoint_path)
        except Exception:
            return None
        if (
            checkpoint.get("status") == "complete"
            and result.get("signature") == self.signature
            and int(result.get("repeat_count", 0)) == self.repeat_count
        ):
            return result
        return None

    def _read_repeat(self, repeat_index: int) -> dict[str, Any] | None:
        repeat_dir = self.baseline_dir / "repeats" / f"repeat_{repeat_index:02d}"
        result_path = repeat_dir / "rollouts.json"
        checkpoint_path = repeat_dir / "checkpoint.json"
        if not result_path.exists() or not checkpoint_path.exists():
            return None
        try:
            result = _read_json(result_path)
            checkpoint = _read_json(checkpoint_path)
        except Exception:
            return None
        if (
            checkpoint.get("status") == "complete"
            and result.get("signature") == self.signature
            and int(result.get("repeat_index", -1)) == repeat_index
        ):
            return result
        return None

    def load_or_run(
        self,
        *,
        run_repeat: Callable[[Path, str, int], dict[str, Any]],
        aggregate_runs: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
        omit_from_run_summary: Sequence[str] = ("episodes", "entries"),
    ) -> dict[str, Any]:
        """Reuse a complete result or resume and aggregate missing repeats."""

        cached = self._read_complete()
        if cached is not None:
            return cached

        lock_path = self.baseline_dir / "baseline.lock"
        descriptor, owner_token = _acquire_lock(
            lock_path,
            timeout_s=self.lock_timeout_s,
            foreign_host_stale_after_s=self.foreign_host_stale_after_s,
        )
        heartbeat_stop = threading.Event()

        def refresh_lock() -> None:
            while not heartbeat_stop.wait(self.lock_heartbeat_interval_s):
                try:
                    current = _read_json(lock_path)
                    if current.get("owner_token") != owner_token:
                        return
                    _atomic_write_json(
                        lock_path,
                        {
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "claimed_at": time.time(),
                            "owner_token": owner_token,
                        },
                    )
                except Exception:
                    return

        heartbeat_thread = threading.Thread(
            target=refresh_lock,
            name=f"skill-eval-baseline-{self.signature}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            cached = self._read_complete()
            if cached is not None:
                return cached
            common = {
                **self.metadata,
                "mode": "eval_no_skill_baseline",
                "signature": self.signature,
                "baseline_dir": str(self.baseline_dir),
                "repeat_count": self.repeat_count,
                "seed_base": self.seed_base,
                "n_tasks": self.task_count,
                "total_rollouts_expected": self.task_count * self.repeat_count,
            }
            _atomic_write_json(
                self.baseline_dir / "checkpoint.json",
                {**common, "status": "rollout_in_progress", "updated_at": time.time()},
            )
            runs: list[dict[str, Any]] = []
            for repeat_index in range(self.repeat_count):
                run = self._read_repeat(repeat_index)
                if run is None:
                    repeat_dir = (
                        self.baseline_dir / "repeats" / f"repeat_{repeat_index:02d}"
                    )
                    repeat_metadata = {
                        "signature": self.signature,
                        "repeat_index": repeat_index,
                        "seed_base": self.repeat_seed(repeat_index),
                    }
                    _atomic_write_json(
                        repeat_dir / "checkpoint.json",
                        {
                            **repeat_metadata,
                            "status": "rollout_in_progress",
                            "updated_at": time.time(),
                        },
                    )
                    payload = run_repeat(
                        self.baseline_dir, self.signature, repeat_index
                    )
                    if not isinstance(payload, dict):
                        raise TypeError("run_repeat must return a dictionary")
                    run = {
                        **payload,
                        **repeat_metadata,
                        "status": "complete",
                        "updated_at": time.time(),
                    }
                    _atomic_write_json(repeat_dir / "rollouts.json", run)
                    record_count = max(
                        (
                            len(run.get(key, []))
                            for key in ("episodes", "entries")
                            if isinstance(run.get(key), list)
                        ),
                        default=0,
                    )
                    _atomic_write_json(
                        repeat_dir / "checkpoint.json",
                        {
                            **repeat_metadata,
                            "status": "complete",
                            "n_rollouts": record_count,
                            "updated_at": time.time(),
                        },
                    )
                runs.append(run)

            aggregate = aggregate_runs(runs)
            omitted = set(omit_from_run_summary)
            result = {
                **common,
                "status": "complete",
                "total_rollouts": self.task_count * len(runs),
                "runs": [
                    {key: value for key, value in run.items() if key not in omitted}
                    for run in runs
                ],
                **aggregate,
                "updated_at": time.time(),
            }
            _atomic_write_json(self.baseline_dir / self.final_filename, result)
            success_rate = float(
                result.get(
                    "sr",
                    result.get("metrics", {}).get("success_rate", 0.0),
                )
            )
            success_rate_std = float(
                result.get(
                    "sr_std",
                    result.get("metrics_std", {}).get("success_rate", 0.0),
                )
            )
            _atomic_write_json(
                self.baseline_dir / "checkpoint.json",
                {
                    **common,
                    "status": "complete",
                    "total_rollouts": result["total_rollouts"],
                    "sr": success_rate,
                    "sr_std": success_rate_std,
                    "updated_at": time.time(),
                },
            )
            return result
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5.0)
            _release_lock(descriptor, lock_path, owner_token)
