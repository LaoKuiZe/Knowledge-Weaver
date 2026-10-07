# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import time
from typing import Any

from areal.utils.saver import Saver

_METRIC_SUMMARY_KEYS = {
    "alfworld_eval_all_sr": "all_mean_sr",
    "webshop_eval_sr": "mean_sr",
    "webshop_eval_skillbank_policy_sr": "skillbank_policy_sr",
}


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp_path, path)


def _checkpoint_global_step(path: pathlib.Path) -> int | None:
    match = re.search(r"globalstep(\d+)$", path.name)
    return int(match.group(1)) if match else None


def _checkpoint_dirs(checkpoint_model_root: pathlib.Path) -> list[pathlib.Path]:
    if not checkpoint_model_root.exists():
        return []
    checkpoints: list[tuple[int, pathlib.Path]] = []
    for child in checkpoint_model_root.iterdir():
        if not child.is_dir():
            continue
        step = _checkpoint_global_step(child)
        if step is not None:
            checkpoints.append((step, child))
    checkpoints.sort(key=lambda item: (item[0], str(item[1])))
    return [path for _, path in checkpoints]


class EvalCheckpointRetention:
    """Keep latest, best-eval, and forced model checkpoints; leave the resumable checkpoint untouched."""

    def __init__(
        self,
        *,
        artifact_dir: str,
        checkpoint_model_root: pathlib.Path,
        eval_config: Any,
        logger: Any,
        forced_global_steps: list[int] | None = None,
    ) -> None:
        self.artifact_dir = pathlib.Path(artifact_dir).expanduser()
        self.checkpoint_model_root = checkpoint_model_root.expanduser()
        self.eval_config = eval_config
        self.forced_global_steps = frozenset(forced_global_steps or [])
        self.logger = logger
        self.manifest_path = self.artifact_dir / "checkpoint_retention_manifest.json"

    @classmethod
    def from_training_config(
        cls,
        config: Any,
        *,
        artifact_dir: str,
        eval_config: Any,
        logger: Any,
    ) -> EvalCheckpointRetention:
        checkpoint_model_root = pathlib.Path(
            Saver.get_model_save_root(
                config.saver.experiment_name,
                config.saver.trial_name,
                config.saver.fileroot,
                "default",
            )
        )
        return cls(
            artifact_dir=artifact_dir,
            checkpoint_model_root=checkpoint_model_root,
            eval_config=eval_config,
            forced_global_steps=list(getattr(config.saver, "force_steps", []) or []),
            logger=logger,
        )

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.eval_config, "checkpoint_retention_enabled", False))

    def initialize(self) -> None:
        if not self.enabled or self.manifest_path.exists():
            return
        protected_paths: list[str] = []
        if getattr(self.eval_config, "checkpoint_retention_protect_existing", False):
            protected_paths = [
                str(path) for path in _checkpoint_dirs(self.checkpoint_model_root)
            ]
        _write_json(
            self.manifest_path,
            {
                "version": 2,
                "mode": "keep_latest_and_best_by_eval_metric",
                "metric": self._metric_name(),
                "summary_key": self._summary_key(),
                "forced_global_steps": sorted(self.forced_global_steps),
                "protected_checkpoint_paths": protected_paths,
                "created_at": time.time(),
            },
        )
        self.logger.info(
            "Initialized shared checkpoint retention manifest at %s with %d "
            "protected checkpoint(s).",
            self.manifest_path,
            len(protected_paths),
        )

    def _metric_name(self) -> str:
        return str(getattr(self.eval_config, "checkpoint_retention_metric", "mean_sr"))

    def _summary_key(self) -> str:
        metric = self._metric_name()
        return _METRIC_SUMMARY_KEYS.get(metric, metric)

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            return _read_json(self.manifest_path)
        except Exception:
            self.logger.exception(
                "Failed to read checkpoint retention manifest %s",
                self.manifest_path,
            )
            return {}

    def _checkpoint_for_summary(self, summary: dict[str, Any]) -> pathlib.Path | None:
        raw_path = str(summary.get("checkpoint_model_path") or "")
        if raw_path:
            checkpoint_path = pathlib.Path(raw_path)
            if checkpoint_path.exists():
                return checkpoint_path
        try:
            step = int(summary.get("checkpoint_global_step", -1))
        except (TypeError, ValueError):
            return None
        matches = sorted(self.checkpoint_model_root.glob(f"epoch*globalstep{step}"))
        return matches[-1] if matches else None

    def _best_eval_checkpoint(
        self,
    ) -> tuple[pathlib.Path | None, float | None, int | None]:
        summary_key = self._summary_key()
        best_path: pathlib.Path | None = None
        best_score: float | None = None
        best_step: int | None = None
        eval_root = self.artifact_dir / "eval"
        for summary_path in sorted(eval_root.glob("globalstep_*/summary.json")):
            try:
                summary = _read_json(summary_path)
            except Exception:
                continue
            if summary.get("status") != "complete":
                continue
            raw_score = summary.get(summary_key)
            if raw_score is None:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            checkpoint_path = self._checkpoint_for_summary(summary)
            if checkpoint_path is None or not checkpoint_path.exists():
                continue
            step = _checkpoint_global_step(checkpoint_path)
            if step is None:
                continue
            if (
                best_score is None
                or score > best_score
                or (score == best_score and step > (best_step or -1))
            ):
                best_path = checkpoint_path
                best_score = score
                best_step = step
        return best_path, best_score, best_step

    def prune_after_eval(self, global_step: int) -> None:
        if not self.enabled:
            return
        self.initialize()
        keep_latest = max(
            1,
            int(
                getattr(
                    self.eval_config,
                    "checkpoint_retention_keep_latest",
                    1,
                )
            ),
        )
        checkpoint_paths = _checkpoint_dirs(self.checkpoint_model_root)
        latest = set(checkpoint_paths[-keep_latest:])
        forced = {
            path
            for path in checkpoint_paths
            if _checkpoint_global_step(path) in self.forced_global_steps
        }
        best_path, best_score, best_step = self._best_eval_checkpoint()
        best_summary_path: pathlib.Path | None = None
        best_summary: dict[str, Any] = {}
        if best_step is not None:
            candidate = (
                self.artifact_dir
                / "eval"
                / f"globalstep_{best_step:06d}"
                / "summary.json"
            )
            if candidate.is_file():
                best_summary_path = candidate
                best_summary = _read_json(candidate)

        manifest = self._load_manifest()
        protected: set[pathlib.Path] = set()
        if getattr(self.eval_config, "checkpoint_retention_protect_existing", False):
            protected = {
                pathlib.Path(item)
                for item in manifest.get("protected_checkpoint_paths", [])
                if isinstance(item, str) and item
            }
        keep = protected | forced | latest
        if best_path is not None:
            keep.add(best_path)

        pruned: list[str] = []
        for path in checkpoint_paths:
            if path in keep:
                continue
            try:
                shutil.rmtree(path)
                pruned.append(str(path))
                self.logger.info(
                    "Pruned checkpoint %s after global_step=%s.",
                    path,
                    global_step,
                )
            except FileNotFoundError:
                continue
            except Exception:
                self.logger.exception("Failed to prune checkpoint %s", path)

        manifest.update(
            {
                "version": 2,
                "mode": "keep_latest_and_best_by_eval_metric",
                "metric": self._metric_name(),
                "summary_key": self._summary_key(),
                "latest_checkpoint_paths": [str(path) for path in sorted(latest)],
                "forced_global_steps": sorted(self.forced_global_steps),
                "forced_checkpoint_paths": [str(path) for path in sorted(forced)],
                "best_checkpoint_path": (
                    str(best_path) if best_path is not None else ""
                ),
                "best_global_step": best_step,
                "best_score": best_score,
                "best_eval_summary_path": (
                    str(best_summary_path) if best_summary_path is not None else ""
                ),
                "best_skillbank_snapshot_step": best_summary.get("bank_snapshot_step"),
                "best_skillbank_snapshot_path": str(
                    best_summary.get("bank_snapshot_path") or ""
                ),
                "best_skillbank_snapshot_digest": str(
                    best_summary.get("bank_snapshot_digest") or ""
                ),
                "last_pruned_checkpoint_paths": pruned,
                "last_pruned_at_global_step": int(global_step),
                "updated_at": time.time(),
            }
        )
        _write_json(self.manifest_path, manifest)
