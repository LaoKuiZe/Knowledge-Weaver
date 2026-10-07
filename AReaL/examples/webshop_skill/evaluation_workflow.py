# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from examples.webshop_skill.core import (
    category_metrics,
    episode_metrics,
    select_trajectory_bundles,
)
from examples.webshop_skill.runtime import (
    _RETRYABLE_ROLLOUT_STATUSES,
    _aggregate_webshop_eval_no_skill_baseline_runs,
    logger,
)

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, ModelResponse
from areal.utils import stats_tracker
from areal.utils.hf_utils import apply_chat_template
from areal.workflow.alfworld_skill import (
    _acquire_json_lock,
    _atomic_write_json,
    _complete_rollout_paths,
    _read_json,
    _release_json_lock,
    skill_payload_to_prompt_text,
)
from areal.workflow.skill_eval_baseline import RepeatedNoSkillBaselinePipeline


class WebShopEvaluationMixin:
    def __init__(
        self,
        *args: Any,
        eval_k: int = 4,
        eval_task_count: int = 100,
        eval_seed: int = 1001,
        eval_max_parallel_rollouts_per_skill: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, eval_task_count=eval_task_count, **kwargs)
        if eval_k <= 0 or eval_task_count <= 0:
            raise ValueError("eval_k and eval_task_count must be positive")
        self.eval_k = int(eval_k)
        self.eval_task_count = int(eval_task_count)
        self.eval_seed = int(eval_seed)
        self.eval_max_parallel_rollouts_per_skill = max(
            1, int(eval_max_parallel_rollouts_per_skill)
        )

    @staticmethod
    def _eval_step_name(global_step: int) -> str:
        return (
            "globalstep_pretrain"
            if global_step < 0
            else f"globalstep_{global_step:06d}"
        )

    def _latest_complete_training_round(self) -> tuple[Path, int]:
        category_dir = self.artifact_dir / "categories" / self.TASK_TYPE
        candidates = []
        canonical_names = {
            f"group_{group_index:02d}_sample_{sample_index:02d}"
            for group_index in range(self.groups_per_round)
            for sample_index in range(self.samples_per_round)
        }
        for round_dir in category_dir.glob("round_*"):
            try:
                index = int(round_dir.name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            complete_names = {
                path.parent.name
                for path in _complete_rollout_paths(round_dir)
                if path.parent.name in canonical_names
            }
            if complete_names == canonical_names:
                candidates.append((index, round_dir))
        if not candidates:
            raise RuntimeError(
                f"no complete {self.BENCHMARK_NAME} training round is available for eval"
            )
        index, path = max(candidates, key=lambda item: item[0])
        return path, index

    def _prepare_eval_sample(
        self, data: dict[str, Any], *, global_step: int
    ) -> dict[str, Any]:
        eval_index = int(data.get("eval_index", 0))
        eval_dir = self.artifact_dir / "eval" / self._eval_step_name(global_step)
        sample_dir = eval_dir / "skills" / f"skill_{eval_index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        round_dir, source_round = self._latest_complete_training_round()
        pool, sampling = self._source_pool_from_rounds([round_dir])
        selected = select_trajectory_bundles(
            pool,
            num_skills=1,
            trajectories_per_skill=self.source_trajectories_per_prompt,
            seed=self.eval_seed + global_step * 101 + eval_index,
            success_threshold=self.success_threshold,
        )[0]
        messages, prompt_render = self._fit_skill_messages(selected)
        prepared = {
            "eval_index": eval_index,
            "eval_k": self.eval_k,
            "global_step": global_step,
            "eval_dir": str(eval_dir),
            "sample_dir": str(sample_dir),
            "source_round": source_round,
            "sampling": {**sampling, "pool_size": len(pool)},
            "messages": messages,
            "prompt_render": prompt_render,
        }
        _atomic_write_json(
            sample_dir / "prompt.json",
            {**prepared, "source_episodes": selected, "created_at": time.time()},
        )
        return prepared

    def _eval_no_skill_baseline_pipeline(
        self, task_indices: Sequence[int]
    ) -> RepeatedNoSkillBaselinePipeline:
        indices = list(map(int, task_indices))
        manifest = self._task_manifest()
        signature_payload = {
            "version": 1,
            "benchmark": self.BENCHMARK_NAME,
            "task_indices": indices,
            "task_manifest_signature": manifest.get("manifest_signature"),
            "actor_model": self.actor_model,
            "actor_temperature": self.actor_temperature,
            "actor_max_tokens": self.actor_max_tokens,
            "max_rollout_steps": self.max_rollout_steps,
            "memory_window": self.memory_window,
            "observation_char_limit": self.observation_char_limit,
            "max_clickables": self.max_clickables,
            "invalid_action_retries": self.invalid_action_retries,
            "success_threshold": self.success_threshold,
        }
        return RepeatedNoSkillBaselinePipeline(
            cache_root=self.artifact_dir / "eval" / "no_skill_baselines",
            signature_payload=signature_payload,
            repeat_count=self.eval_no_skill_baseline_repeats,
            seed_base=self.eval_no_skill_baseline_seed,
            task_count=len(indices),
            final_filename="baseline_index.json",
            metadata={
                "benchmark": self.BENCHMARK_NAME,
                "task_indices": indices,
                "task_manifest_signature": manifest.get("manifest_signature"),
            },
        )

    def _run_eval_no_skill_baseline_repeat(
        self,
        *,
        task_indices: Sequence[int],
        baseline_dir: Path,
        signature: str,
        repeat_index: int,
    ) -> dict[str, Any]:
        del baseline_dir, signature
        seed_base = self.eval_no_skill_baseline_seed + int(repeat_index) * 100_000
        episodes = self._run_episodes(
            task_indices,
            skill="",
            condition="eval_no_skill",
            seed_base=seed_base,
            max_workers=self.eval_max_parallel_rollouts_per_skill,
        )
        if len(episodes) != len(task_indices):
            raise RuntimeError("WebShop eval baseline returned an incomplete task set")
        if [int(episode.get("task_index", -1)) for episode in episodes] != list(
            map(int, task_indices)
        ):
            raise RuntimeError(
                "WebShop eval baseline returned the wrong task identity or order"
            )
        retryable = [
            episode
            for episode in episodes
            if str(episode.get("status") or "") in _RETRYABLE_ROLLOUT_STATUSES
        ]
        if retryable:
            raise RuntimeError(
                "WebShop eval baseline contains retryable infrastructure failures: "
                f"{[item.get('task_index') for item in retryable[:8]]}"
            )
        return {
            "episodes": episodes,
            "task_indices": list(map(int, task_indices)),
            "metrics": episode_metrics(episodes),
            "per_category": category_metrics(episodes),
        }

    def _load_or_create_eval_no_skill_baseline(
        self, task_indices: Sequence[int]
    ) -> dict[str, Any]:
        indices = list(map(int, task_indices))
        pipeline = self._eval_no_skill_baseline_pipeline(indices)
        return pipeline.load_or_run(
            run_repeat=lambda baseline_dir, signature, repeat_index: (
                self._run_eval_no_skill_baseline_repeat(
                    task_indices=indices,
                    baseline_dir=baseline_dir,
                    signature=signature,
                    repeat_index=repeat_index,
                )
            ),
            aggregate_runs=_aggregate_webshop_eval_no_skill_baseline_runs,
        )

    def _eval_baseline(
        self, eval_dir: Path, task_indices: Sequence[int]
    ) -> dict[str, Any]:
        result = self._load_or_create_eval_no_skill_baseline(task_indices)
        reference_path = eval_dir / "no_skill_baseline" / "reference.json"
        reference = {
            "status": "complete",
            "signature": result["signature"],
            "baseline_dir": result["baseline_dir"],
            "repeat_count": result["repeat_count"],
            "total_rollouts": result["total_rollouts"],
            "sr": result["sr"],
            "sr_std": result["sr_std"],
            "updated_at": time.time(),
        }
        if reference_path.exists():
            existing = _read_json(reference_path)
            if existing.get("signature") != result["signature"]:
                raise RuntimeError(
                    "WebShop eval step mixes incompatible no-skill baselines"
                )
        else:
            _atomic_write_json(reference_path, reference)
        return result

    def _skillbank_policy_eval_signature(
        self,
        *,
        checkpoint_global_step: int,
        snapshot: dict[str, Any],
        task_indices: Sequence[int],
        baseline_signature: str,
    ) -> str:
        manifest = self._task_manifest()
        payload = {
            "version": 1,
            "mode": "webshop_skillbank_policy_eval",
            "checkpoint_global_step": int(checkpoint_global_step),
            "bank_snapshot_step": int(snapshot["snapshot_step"]),
            "bank_snapshot_digest": str(snapshot["snapshot_digest"]),
            "bank_snapshot_size": int(snapshot["snapshot_size"]),
            "task_indices": list(map(int, task_indices)),
            "task_manifest_signature": manifest.get("manifest_signature"),
            "top_k": self.skillbank_top_k,
            "embedding_model": self.skillbank_embedding_model,
            "embedding_batch_size": self.skillbank_embedding_batch_size,
            "embedding_max_length": self.skillbank_embedding_max_length,
            "embedding_device": self.skillbank_embedding_device,
            "query_contract": "task_initial_state_topk_fixed_episode_v1",
            "actor_model": self.actor_model,
            "actor_temperature": self.actor_temperature,
            "actor_max_tokens": self.actor_max_tokens,
            "max_rollout_steps": self.max_rollout_steps,
            "memory_window": self.memory_window,
            "observation_char_limit": self.observation_char_limit,
            "max_clickables": self.max_clickables,
            "invalid_action_retries": self.invalid_action_retries,
            "success_threshold": self.success_threshold,
            "actor_seed_base": self.eval_no_skill_baseline_seed,
            "baseline_signature": str(baseline_signature),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _load_or_create_skillbank_policy_eval(
        self,
        *,
        eval_dir: Path,
        checkpoint_global_step: int,
        task_indices: Sequence[int],
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot = self._load_eval_skillbank_snapshot(checkpoint_global_step)
        indices = list(map(int, task_indices))
        signature = self._skillbank_policy_eval_signature(
            checkpoint_global_step=checkpoint_global_step,
            snapshot=snapshot,
            task_indices=indices,
            baseline_signature=str(baseline.get("signature") or ""),
        )
        root = eval_dir / "skillbank_policy"
        protocol_path = root / "protocol.json"
        result_path = root / "rollouts.json"
        checkpoint_path = root / "checkpoint.json"

        def read_complete() -> dict[str, Any] | None:
            if not result_path.exists() or not checkpoint_path.exists():
                return None
            result = _read_json(result_path)
            checkpoint = _read_json(checkpoint_path)
            if (
                result.get("signature") != signature
                or checkpoint.get("signature") != signature
                or result.get("baseline_signature") != baseline.get("signature")
            ):
                raise RuntimeError(
                    "cached WebShop skillbank policy eval differs; use a new trial"
                )
            if (
                result.get("status") == "complete"
                and checkpoint.get("status") == "complete"
            ):
                return result
            return None

        cached = read_complete()
        if cached is not None:
            return cached
        lock_path = root / "policy.lock"
        fd = _acquire_json_lock(
            lock_path,
            timeout_s=172800.0,
            foreign_host_stale_after_s=120.0,
        )
        heartbeat_stop = threading.Event()

        def refresh_policy_lock() -> None:
            while not heartbeat_stop.wait(30.0):
                try:
                    _atomic_write_json(
                        lock_path,
                        {
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "claimed_at": time.time(),
                            "mode": "skillbank_policy_eval",
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "failed to refresh WebShop skillbank policy lock %s",
                        lock_path,
                    )

        heartbeat_thread = threading.Thread(
            target=refresh_policy_lock,
            name=f"webshop-skillbank-policy-{checkpoint_global_step}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            cached = read_complete()
            if cached is not None:
                return cached
            protocol = {
                "version": 1,
                "signature": signature,
                "checkpoint_global_step": int(checkpoint_global_step),
                "bank_snapshot_step": int(snapshot["snapshot_step"]),
                "bank_snapshot_path": str(snapshot["snapshot_path"]),
                "bank_snapshot_digest": str(snapshot["snapshot_digest"]),
                "bank_snapshot_size": int(snapshot["snapshot_size"]),
                "task_indices": indices,
                "top_k": self.skillbank_top_k,
                "baseline_signature": str(baseline.get("signature") or ""),
            }
            if protocol_path.exists():
                if _read_json(protocol_path) != protocol:
                    raise RuntimeError(
                        "persisted WebShop skillbank policy protocol differs"
                    )
            else:
                _atomic_write_json(protocol_path, protocol)

            retrieval = self._retrieve_skillbank_for_tasks(
                task_indices=indices,
                bank=snapshot["skills"],
                bank_digest=str(snapshot["snapshot_digest"]),
            )
            jobs: list[dict[str, Any]] = []
            for position, task_index in enumerate(indices):
                job = {
                    "job_id": f"task_{position:04d}",
                    "condition": "eval_skillbank_policy",
                    "task_index": task_index,
                    "skill": retrieval["skill_texts"][position],
                    "actor_seed": self.eval_no_skill_baseline_seed + position * 997,
                    "retrieved_skill_ids": [
                        str(item["skill_id"])
                        for item in retrieval["selected"][position]
                    ],
                }
                job["signature"] = hashlib.sha256(
                    json.dumps(job, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                jobs.append(job)
            job_results, attempted_count, cached_count = self._run_group_episode_jobs(
                jobs,
                cache_dir=root / "episodes",
                group_signature=signature,
                max_workers=self.eval_max_parallel_rollouts_per_skill,
            )
            episodes = [
                job_results[f"task_{position:04d}"] for position in range(len(indices))
            ]
            metrics = episode_metrics(episodes)
            baseline_sr = float(baseline.get("sr", 0.0))
            retrieval_scores = [
                float(item["cosine_similarity"])
                for row in retrieval["selected"]
                for item in row
            ]
            retrieved_ids = [
                str(item["skill_id"]) for row in retrieval["selected"] for item in row
            ]
            result = {
                **protocol,
                "status": "complete",
                "mode": "skillbank_retrieval_policy",
                "baseline_signature": protocol["baseline_signature"],
                "n_rollouts": len(episodes),
                "attempted_rollout_count": attempted_count,
                "cached_rollout_count": cached_count,
                "sr": float(metrics["success_rate"]),
                "mean_env_reward": float(metrics["mean_reward"]),
                "baseline_sr": baseline_sr,
                "baseline_delta_sr": float(metrics["success_rate"]) - baseline_sr,
                "metrics": metrics,
                "per_category": category_metrics(episodes),
                "retrieval_mean_cosine": (
                    mean(retrieval_scores) if retrieval_scores else 0.0
                ),
                "retrieval_min_cosine": (
                    min(retrieval_scores) if retrieval_scores else 0.0
                ),
                "retrieval_max_cosine": (
                    max(retrieval_scores) if retrieval_scores else 0.0
                ),
                "retrieved_unique_skill_ids": sorted(set(retrieved_ids)),
                "retrieval": retrieval["manifest"],
                "episodes": episodes,
                "updated_at": time.time(),
            }
            _atomic_write_json(
                root / "retrieval.json",
                {
                    **protocol,
                    "status": "complete",
                    "retrieval": result["retrieval"],
                    "updated_at": result["updated_at"],
                },
            )
            _atomic_write_json(result_path, result)
            _atomic_write_json(
                root / "metrics.json",
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"episodes", "retrieval"}
                },
            )
            _atomic_write_json(
                checkpoint_path,
                {
                    **protocol,
                    "status": "complete",
                    "n_rollouts": len(episodes),
                    "sr": result["sr"],
                    "updated_at": result["updated_at"],
                },
            )
            return result
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5.0)
            _release_json_lock(fd, lock_path)

    def _maybe_write_eval_summary(self, eval_dir: Path) -> dict[str, Any] | None:
        result_paths = sorted((eval_dir / "skills").glob("skill_*/rollouts.json"))
        if len(result_paths) < self.eval_k:
            return None
        summary_path, lock_path = eval_dir / "summary.json", eval_dir / "summary.lock"
        fd = _acquire_json_lock(lock_path, timeout_s=7200.0)
        try:
            if summary_path.exists():
                return _read_json(summary_path)
            results = [_read_json(path) for path in result_paths[: self.eval_k]]
            srs = [float(item.get("sr", 0.0)) for item in results]
            rewards = [float(item.get("mean_env_reward", 0.0)) for item in results]
            baseline_signatures = {
                str(item.get("baseline_signature") or "") for item in results
            }
            if len(baseline_signatures) != 1 or not next(iter(baseline_signatures), ""):
                raise RuntimeError(
                    "WebShop eval skills do not share one repeated baseline"
                )
            policies = [item.get("skillbank_policy") for item in results]
            if not all(isinstance(item, dict) for item in policies):
                raise RuntimeError(
                    "skillbank eval summary is missing a frozen policy result"
                )
            policy_signatures = {str(item.get("signature") or "") for item in policies}
            if len(policy_signatures) != 1 or not next(iter(policy_signatures), ""):
                raise RuntimeError(
                    "WebShop eval rows reference different skillbank policies"
                )
            policy = policies[0]
            if policy.get("baseline_signature") != results[0]["baseline_signature"]:
                raise RuntimeError(
                    "WebShop skillbank policy and singleton probes use different "
                    "baselines"
                )
            categories: dict[str, list[float]] = {}
            for result in results:
                for category, metrics in result.get("per_category", {}).items():
                    categories.setdefault(str(category), []).append(
                        float(metrics.get("success_rate", 0.0))
                    )
            payload = {
                "version": 2,
                "status": "complete",
                "checkpoint_global_step": int(results[0].get("global_step", -1)),
                "n_skills": len(results),
                "task_count": self.eval_task_count,
                "max_rollout_steps": self.max_rollout_steps,
                "mean_sr": mean(srs),
                "min_sr": min(srs),
                "max_sr": max(srs),
                "mean_env_reward": mean(rewards),
                "schema_valid_rate": mean(
                    float(bool(item.get("schema_valid"))) for item in results
                ),
                "baseline_signature": results[0]["baseline_signature"],
                "baseline_dir": results[0]["baseline_dir"],
                "baseline_sr": float(results[0]["baseline_sr"]),
                "baseline_sr_std": float(results[0]["baseline_sr_std"]),
                "baseline_mean_reward": float(results[0]["baseline_mean_reward"]),
                "baseline_mean_reward_std": float(
                    results[0]["baseline_mean_reward_std"]
                ),
                "baseline_repeat_count": int(results[0]["baseline_repeat_count"]),
                "baseline_total_rollouts": int(results[0]["baseline_total_rollouts"]),
                "per_category_mean_sr": {
                    key: mean(values) for key, values in sorted(categories.items())
                },
                "skill_results": [
                    {
                        "eval_index": item.get("eval_index"),
                        "sr": item.get("sr"),
                        "mean_env_reward": item.get("mean_env_reward"),
                        "baseline_delta_sr": item.get("baseline_delta_sr"),
                    }
                    for item in results
                ],
                "evaluation_mode": "skillbank_retrieval_with_singleton_probe",
                "updated_at": time.time(),
                "skillbank_policy_sr": float(policy["sr"]),
                "skillbank_policy_mean_env_reward": float(policy["mean_env_reward"]),
                "skillbank_policy_baseline_sr": float(policy["baseline_sr"]),
                "skillbank_policy_delta_sr": float(policy["baseline_delta_sr"]),
                "skillbank_policy_signature": policy["signature"],
                "skillbank_policy_metrics_path": policy["metrics_path"],
                "skillbank_policy_rollouts_path": policy["rollouts_path"],
                "bank_snapshot_step": int(policy["bank_snapshot_step"]),
                "bank_snapshot_path": policy["bank_snapshot_path"],
                "bank_snapshot_digest": policy["bank_snapshot_digest"],
                "bank_snapshot_size": int(policy["bank_snapshot_size"]),
                "skillbank_top_k": int(policy["top_k"]),
                "singleton_probe": {
                    "n_skills": len(results),
                    "mean_sr": mean(srs),
                    "min_sr": min(srs),
                    "max_sr": max(srs),
                    "mean_env_reward": mean(rewards),
                    "schema_valid_rate": mean(
                        float(bool(item.get("schema_valid"))) for item in results
                    ),
                },
            }
            _atomic_write_json(summary_path, payload)
            return payload
        finally:
            _release_json_lock(fd, lock_path)

    def _evaluate_eval_and_save(
        self,
        prepared: dict[str, Any],
        raw: str,
        resp: ModelResponse,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sample_dir, eval_dir = Path(prepared["sample_dir"]), Path(prepared["eval_dir"])
        final = attempts[-1]
        parsed = final.get("parsed") if isinstance(final.get("parsed"), dict) else None
        valid = bool(final.get("schema_valid")) and parsed is not None
        skill_text = skill_payload_to_prompt_text(parsed) if valid else ""
        task_indices = list(map(int, self._task_manifest()["eval_indices"]))
        if valid:
            episodes = self._run_episodes(
                task_indices,
                skill=skill_text,
                condition="eval_skill",
                seed_base=1_000_000 + int(prepared["eval_index"]) * 100_000,
                max_workers=self.eval_max_parallel_rollouts_per_skill,
            )
        else:
            episodes = [
                {
                    "task_index": index,
                    "status": "invalid_skill",
                    "reward": 0.0,
                    "success": False,
                    "trace": [],
                }
                for index in task_indices
            ]
        baseline = self._eval_baseline(eval_dir, task_indices)
        metrics = episode_metrics(episodes)
        policy_result = self._load_or_create_skillbank_policy_eval(
            eval_dir=eval_dir,
            checkpoint_global_step=int(prepared["global_step"]),
            task_indices=task_indices,
            baseline=baseline,
        )
        result = {
            "status": "complete",
            "eval_index": int(prepared["eval_index"]),
            "global_step": int(prepared["global_step"]),
            "source_round": int(prepared["source_round"]),
            "schema_valid": valid,
            "schema_error": str(final.get("schema_error") or ""),
            "attempt_count": len(attempts),
            "raw": raw,
            "parsed": parsed,
            "input_token_count": len(resp.input_tokens),
            "output_token_count": len(resp.output_tokens),
            "task_indices": task_indices,
            "n_rollouts": len(episodes),
            "sr": float(metrics["success_rate"]),
            "mean_env_reward": float(metrics["mean_reward"]),
            "baseline_signature": str(baseline["signature"]),
            "baseline_dir": str(baseline["baseline_dir"]),
            "baseline_sr": float(baseline["sr"]),
            "baseline_sr_std": float(baseline["sr_std"]),
            "baseline_mean_reward": float(baseline["mean_reward"]),
            "baseline_mean_reward_std": float(baseline["mean_reward_std"]),
            "baseline_repeat_count": int(baseline["repeat_count"]),
            "baseline_total_rollouts": int(baseline["total_rollouts"]),
            "baseline_delta_sr": float(metrics["success_rate"]) - float(baseline["sr"]),
            "metrics": metrics,
            "per_category": category_metrics(episodes),
            "episodes": episodes,
            "updated_at": time.time(),
            "skillbank_policy": {
                "signature": policy_result["signature"],
                "baseline_signature": policy_result["baseline_signature"],
                "sr": policy_result["sr"],
                "mean_env_reward": policy_result["mean_env_reward"],
                "baseline_sr": policy_result["baseline_sr"],
                "baseline_delta_sr": policy_result["baseline_delta_sr"],
                "bank_snapshot_step": policy_result["bank_snapshot_step"],
                "bank_snapshot_path": policy_result["bank_snapshot_path"],
                "bank_snapshot_digest": policy_result["bank_snapshot_digest"],
                "bank_snapshot_size": policy_result["bank_snapshot_size"],
                "top_k": policy_result["top_k"],
                "n_rollouts": policy_result["n_rollouts"],
                "retrieval_mean_cosine": policy_result["retrieval_mean_cosine"],
                "metrics_path": str(eval_dir / "skillbank_policy" / "metrics.json"),
                "rollouts_path": str(eval_dir / "skillbank_policy" / "rollouts.json"),
            },
        }
        _atomic_write_json(
            sample_dir / "generation.json",
            {
                "raw": raw,
                "parsed": parsed,
                "schema_valid": valid,
                "schema_error": result["schema_error"],
                "attempts": attempts,
                "output_tokens": list(map(int, resp.output_tokens)),
                "updated_at": time.time(),
            },
        )
        _atomic_write_json(sample_dir / "rollouts.json", result)
        _atomic_write_json(
            sample_dir / "metrics.json",
            {key: value for key, value in result.items() if key != "episodes"},
        )
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "complete",
                "sr": result["sr"],
                "primary_sr": policy_result["sr"],
                "updated_at": time.time(),
            },
        )
        result["summary"] = self._maybe_write_eval_summary(eval_dir)
        return result

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        global_step = max(-1, int(engine.get_version()) - 1)
        prepared = await asyncio.to_thread(
            self._prepare_eval_sample, data, global_step=global_step
        )
        input_ids = apply_chat_template(
            self.tokenizer,
            prepared["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.skill_generation_enable_thinking,
        )
        attempts: list[dict[str, Any]] = []
        resp: ModelResponse | None = None
        raw = ""
        for attempt_index in range(self._max_skill_generation_retries() + 1):
            resp = await engine.agenerate(
                ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=input_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
            )
            raw = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=False)
            attempts.append(self._generation_attempt(raw, resp, attempt_index))
            if attempts[-1]["schema_valid"]:
                break
        if resp is None:
            raise RuntimeError(
                f"{self.BENCHMARK_NAME} eval skill generator returned no response"
            )
        result = await asyncio.to_thread(
            self._evaluate_eval_and_save, prepared, raw, resp, attempts
        )
        policy = result["skillbank_policy"]
        primary_sr = float(policy["sr"])
        primary_mean_reward = float(policy["mean_env_reward"])
        primary_delta_sr = float(policy["baseline_delta_sr"])
        eval_metrics = {
            "eval_sr": primary_sr,
            "eval_mean_env_reward": primary_mean_reward,
            "eval_baseline_sr": float(result["baseline_sr"]),
            "eval_delta_sr": primary_delta_sr,
            "eval_baseline_sr_std": float(result["baseline_sr_std"]),
            "eval_baseline_mean_reward": float(result["baseline_mean_reward"]),
            "eval_baseline_mean_reward_std": float(result["baseline_mean_reward_std"]),
            "eval_baseline_repeats": int(result["baseline_repeat_count"]),
            "eval_baseline_total_rollouts": int(result["baseline_total_rollouts"]),
            "eval_schema_valid": float(bool(result["schema_valid"])),
            "eval_rollouts": int(policy["n_rollouts"]),
            "eval_skillbank_policy_sr": primary_sr,
            "eval_skillbank_policy_delta_sr": primary_delta_sr,
            "eval_skillbank_policy_mean_env_reward": primary_mean_reward,
            "eval_singleton_probe_sr": float(result["sr"]),
            "eval_singleton_probe_mean_env_reward": float(result["mean_env_reward"]),
            "eval_skillbank_size": int(policy["bank_snapshot_size"]),
            "eval_skillbank_retrieval_mean_cosine": float(
                policy["retrieval_mean_cosine"]
            ),
        }
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            **{
                f"{self.METRIC_PREFIX}_{key}": value
                for key, value in eval_metrics.items()
            }
        )
        return self._tensor_result(resp, primary_sr)
