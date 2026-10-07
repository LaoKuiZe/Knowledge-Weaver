# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import threading
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import Future, as_completed
from pathlib import Path
from typing import Any

import requests
import torch

from examples.skill_training.semantic_skillbank import (
    SkillEntry,
    build_retrieval_query,
    compute_skillbank_counterfactual_reward,
    embed_texts,
    format_retrieved_skill_text,
    load_skillbank,
    retrieve_top_skills,
    select_online_skillbank_admission,
    skillbank_snapshot_digest,
    write_skillbank,
)
from examples.webshop_skill.core import (
    NO_SKILL_TEXT,
    episode_metrics,
    paired_condition_metrics,
)
from examples.webshop_skill.runtime import (
    _RETRYABLE_ROLLOUT_STATUSES,
    _SKILLBANK_MODE,
    _format_skill_bodies,
    _group_coordinator_executor,
    _group_episode_executor,
    _normalized_skill_body,
    _skill_body_digest,
    logger,
)

from areal.workflow.alfworld_skill import (
    _acquire_json_lock,
    _atomic_write_json,
    _read_json,
    _release_json_lock,
    skill_payload_to_prompt_text,
)


class WebShopBankMixin:
    @property
    def _skillbank_root(self) -> Path:
        return self.artifact_dir / "skillbank"

    def _skillbank_snapshot_path(self, training_global_step: int) -> Path:
        return (
            self._skillbank_root
            / "snapshots"
            / f"globalstep_{int(training_global_step):06d}.jsonl"
        )

    def _initial_skillbank(self) -> list[SkillEntry]:
        return []

    def _read_skillbank(self, path: Path) -> list[SkillEntry]:
        return load_skillbank(path, allow_empty=True)

    def _initialize_skillbank(self) -> None:
        initial_skills = self._initial_skillbank()
        digest = skillbank_snapshot_digest(initial_skills)
        root = self._skillbank_root
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / "initialize.lock"
        fd = _acquire_json_lock(
            lock_path,
            timeout_s=7200.0,
            foreign_host_stale_after_s=120.0,
        )
        try:
            manifest_path = root / "manifest.json"
            expected_manifest = {
                "version": 1,
                "initial_digest": digest,
                "initial_size": len(initial_skills),
                "top_k": self.skillbank_top_k,
                "embedding_model": self.skillbank_embedding_model,
                "embedding_max_length": self.skillbank_embedding_max_length,
                "embedding_device": self.skillbank_embedding_device,
                "standalone_weight": self.skillbank_standalone_weight,
                "retrieval_weight": self.skillbank_retrieval_weight,
                "update_min_marginal": self.skillbank_update_min_marginal,
                "allow_noop_update": self.skillbank_allow_noop_update,
                "max_size": self.skillbank_max_size,
                "reward_weight": self.skillbank_reward_weight,
            }
            if manifest_path.exists():
                if _read_json(manifest_path) != expected_manifest:
                    raise RuntimeError(
                        "persisted WebShop skillbank manifest differs; use a new trial"
                    )
            else:
                _atomic_write_json(manifest_path, expected_manifest)
            initial_path = self._skillbank_snapshot_path(0)
            if initial_path.exists():
                persisted = self._read_skillbank(initial_path)
                if skillbank_snapshot_digest(persisted) != digest:
                    raise RuntimeError("persisted initial skillbank snapshot differs")
            else:
                write_skillbank(initial_path, initial_skills)
            with self._skillbank_snapshot_lock:
                self._skillbank_snapshot_cache[0] = initial_skills
        finally:
            _release_json_lock(fd, lock_path)

    def _load_skillbank_snapshot(
        self, training_global_step: int, *, timeout_s: float = 7200.0
    ) -> list[SkillEntry]:
        step = int(training_global_step)
        with self._skillbank_snapshot_lock:
            cached = self._skillbank_snapshot_cache.get(step)
            if cached is not None:
                return cached
        path = self._skillbank_snapshot_path(step)
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        while not path.exists():
            if step > 0:
                self._maybe_update_skillbank(step - 1)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for WebShop skillbank snapshot {path}"
                )
            time.sleep(0.5)
        skills = self._read_skillbank(path)
        with self._skillbank_snapshot_lock:
            self._skillbank_snapshot_cache[step] = skills
        return skills

    def _load_eval_skillbank_snapshot(
        self, checkpoint_global_step: int
    ) -> dict[str, Any]:
        """Load the checkpoint-aligned bank without creating or repairing state."""

        snapshot_step = max(0, int(checkpoint_global_step) + 1)
        manifest_path = self._skillbank_root / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("WebShop skillbank manifest is missing during eval")
        manifest = _read_json(manifest_path)
        initial_path = self._skillbank_snapshot_path(0)
        if not initial_path.is_file():
            raise RuntimeError("WebShop initial skillbank snapshot is missing")
        previous = self._read_skillbank(initial_path)
        previous_digest = skillbank_snapshot_digest(previous)
        if manifest.get("initial_digest") != previous_digest or int(
            manifest.get("initial_size", -1)
        ) != len(previous):
            raise RuntimeError("WebShop initial skillbank snapshot fails its manifest")

        for step in range(1, snapshot_step + 1):
            update_path = (
                self._skillbank_root / "updates" / f"globalstep_{step - 1:06d}.json"
            )
            next_path = self._skillbank_snapshot_path(step)
            if not update_path.is_file() or not next_path.is_file():
                raise RuntimeError(
                    "checkpoint-aligned WebShop skillbank chain is incomplete: "
                    f"snapshot_step={step}"
                )
            update = _read_json(update_path)
            current = self._read_skillbank(next_path)
            current_digest = skillbank_snapshot_digest(current)
            if (
                update.get("status") != "complete"
                or int(update.get("training_global_step", -1)) != step - 1
                or update.get("previous_snapshot")
                != str(self._skillbank_snapshot_path(step - 1))
                or update.get("previous_digest") != previous_digest
                or int(update.get("previous_size", -1)) != len(previous)
                or update.get("next_snapshot") != str(next_path)
                or update.get("next_digest") != current_digest
                or int(update.get("next_size", -1)) != len(current)
            ):
                raise RuntimeError(
                    f"WebShop skillbank update {step - 1} fails chain validation"
                )
            admitted = update.get("admitted")
            if admitted is None:
                if not bool(update.get("noop")) or current_digest != previous_digest:
                    raise RuntimeError(
                        f"WebShop skillbank no-op update {step - 1} changed the bank"
                    )
            else:
                if (
                    bool(update.get("noop"))
                    or not isinstance(admitted, dict)
                    or len(current) != len(previous) + 1
                    or skillbank_snapshot_digest(current[:-1]) != previous_digest
                    or current[-1].skill_id != str(admitted.get("id") or "")
                    or current[-1].model != str(admitted.get("model") or "")
                    or current[-1].content
                    != _normalized_skill_body(admitted.get("content") or "")
                ):
                    raise RuntimeError(
                        f"WebShop skillbank admission {step - 1} is inconsistent"
                    )
            previous = current
            previous_digest = current_digest

        snapshot_path = self._skillbank_snapshot_path(snapshot_step)
        return {
            "checkpoint_global_step": int(checkpoint_global_step),
            "snapshot_step": snapshot_step,
            "snapshot_path": str(snapshot_path),
            "snapshot_digest": previous_digest,
            "snapshot_size": len(previous),
            "skills": previous,
        }

    def _canonical_group_generation_records(
        self, *, round_index: int, group_index: int
    ) -> list[dict[str, Any]]:
        round_dir = self._round_dir(self.TASK_TYPE, round_index)
        records: list[dict[str, Any]] = []
        expected_tasks: list[int] | None = None
        for sample_index in range(self.samples_per_round):
            sample_key = f"group_{group_index:02d}_sample_{sample_index:02d}"
            path = round_dir / "skills" / sample_key / "generation.json"
            if not path.exists():
                raise RuntimeError(f"missing group generation artifact: {path}")
            payload = _read_json(path)
            tasks = list(map(int, payload.get("selected_games") or []))
            if expected_tasks is None:
                expected_tasks = tasks
            elif tasks != expected_tasks:
                raise RuntimeError(
                    f"group {group_index} candidates do not share ordered reward tasks"
                )
            parsed = payload.get("parsed")
            valid = bool(payload.get("schema_valid")) and isinstance(parsed, dict)
            records.append(
                {
                    "sample_key": sample_key,
                    "sample_index": sample_index,
                    "path": str(path),
                    "schema_valid": valid,
                    "schema_error": str(payload.get("schema_error") or ""),
                    "content": skill_payload_to_prompt_text(parsed) if valid else "",
                    "task_indices": tasks,
                }
            )
        if not expected_tasks or len(expected_tasks) != self.rollouts_per_skill:
            raise RuntimeError(
                f"group {group_index} expected {self.rollouts_per_skill} reward tasks"
            )
        return records

    def _group_reward_signature(
        self,
        *,
        round_index: int,
        group_index: int,
        records: Sequence[dict[str, Any]],
        bank_digest: str = "",
    ) -> str:
        payload = {
            "version": 1,
            "mode": _SKILLBANK_MODE,
            "round_index": round_index,
            "group_index": group_index,
            "records": [
                {
                    "sample_key": record["sample_key"],
                    "schema_valid": record["schema_valid"],
                    "content_digest": _skill_body_digest(record["content"]),
                    "task_indices": record["task_indices"],
                }
                for record in records
            ],
            "actor_model": self.actor_model,
            "actor_temperature": self.actor_temperature,
            "task_seed": self.task_seed,
            "actor_max_tokens": self.actor_max_tokens,
            "max_rollout_steps": self.max_rollout_steps,
            "memory_window": self.memory_window,
            "observation_char_limit": self.observation_char_limit,
            "max_clickables": self.max_clickables,
            "invalid_action_retries": self.invalid_action_retries,
            "success_threshold": self.success_threshold,
            "bank_digest": bank_digest,
            "bank_top_k": self.skillbank_top_k,
            "bank_embedding_model": self.skillbank_embedding_model,
            "bank_embedding_max_length": self.skillbank_embedding_max_length,
            "bank_retrieval_prompt_version": "task_initial_state_topk_fixed_episode_v1",
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _run_group_episode_jobs(
        self,
        jobs: Sequence[dict[str, Any]],
        *,
        cache_dir: Path,
        group_signature: str,
        max_workers: int | None = None,
    ) -> tuple[dict[str, dict[str, Any]], int, int]:
        """Run all group conditions through one shared bounded HTTP executor."""

        cache_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, dict[str, Any]] = {}
        cached_count = 0
        attempted_count = 0

        def cache_path(job: dict[str, Any]) -> Path:
            return cache_dir / f"{job['job_id']}.json"

        def read_cached(job: dict[str, Any]) -> dict[str, Any] | None:
            path = cache_path(job)
            if not path.exists():
                return None
            try:
                wrapper = _read_json(path)
            except Exception:
                return None
            episode = wrapper.get("episode")
            if (
                wrapper.get("status") != "complete"
                or wrapper.get("group_signature") != group_signature
                or wrapper.get("job_signature") != job["signature"]
                or not isinstance(episode, dict)
                or int(episode.get("task_index", -1)) != int(job["task_index"])
                or str(episode.get("status") or "") in _RETRYABLE_ROLLOUT_STATUSES
            ):
                return None
            return episode

        missing: list[dict[str, Any]] = []
        for job in jobs:
            cached = read_cached(job)
            if cached is None:
                missing.append(dict(job))
            else:
                results[str(job["job_id"])] = cached
                cached_count += 1

        executor = _group_episode_executor(
            int(max_workers or self.episode_rollout_workers)
        )

        def run(job: dict[str, Any], attempt: int) -> dict[str, Any]:
            try:
                episode = self._env_client.rollout(
                    self._rollout_payload(
                        task_index=int(job["task_index"]),
                        skill=str(job["skill"]),
                        condition=str(job["condition"]),
                        seed=int(job["actor_seed"]),
                        session_key=(
                            f"{group_signature[:12]}-{job['job_id']}-attempt{attempt}"
                        ),
                    )
                )
                if episode.get("status") == "error":
                    episode["status"] = "rollout_error"
                return episode
            except requests.Timeout as exc:
                return {
                    "task_index": int(job["task_index"]),
                    "condition": str(job["condition"]),
                    "status": "rollout_timeout",
                    "reward": 0.0,
                    "success": False,
                    "trace": [],
                    "error": repr(exc),
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "task_index": int(job["task_index"]),
                    "condition": str(job["condition"]),
                    "status": "rollout_error",
                    "reward": 0.0,
                    "success": False,
                    "trace": [],
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }

        future_jobs: dict[Future[dict[str, Any]], tuple[dict[str, Any], int]] = {}
        for job in missing:
            attempted_count += 1
            future_jobs[executor.submit(run, job, 1)] = (job, 1)
        failures: list[str] = []
        while future_jobs:
            for future in as_completed(list(future_jobs)):
                job, attempt = future_jobs.pop(future)
                episode = future.result()
                status = str(episode.get("status") or "")
                if status == "rollout_timeout":
                    failures.append(f"{job['job_id']}: rollout_timeout")
                    continue
                if status in _RETRYABLE_ROLLOUT_STATUSES:
                    if attempt < 3:
                        attempted_count += 1
                        future_jobs[executor.submit(run, job, attempt + 1)] = (
                            job,
                            attempt + 1,
                        )
                    else:
                        failures.append(f"{job['job_id']}: {status}")
                    continue
                wrapper = {
                    "status": "complete",
                    "group_signature": group_signature,
                    "job_signature": job["signature"],
                    "job_id": job["job_id"],
                    "condition": job["condition"],
                    "task_index": job["task_index"],
                    "actor_seed": job["actor_seed"],
                    "attempts_used": attempt,
                    "episode": episode,
                    "updated_at": time.time(),
                }
                _atomic_write_json(cache_path(job), wrapper)
                results[str(job["job_id"])] = episode
        if failures:
            raise RuntimeError(
                f"{self.BENCHMARK_NAME} group rollout jobs failed: {failures[:8]}"
            )
        return results, attempted_count, cached_count

    def _skillbank_embeddings(
        self, skills: Sequence[SkillEntry], digest: str
    ) -> torch.Tensor:
        with self._skillbank_embedding_lock:
            cached = self._skillbank_embedding_cache.get(digest)
            if cached is not None:
                return cached
            embeddings = embed_texts(
                [skill.content for skill in skills],
                model_name=self.skillbank_embedding_model,
                batch_size=self.skillbank_embedding_batch_size,
                max_length=self.skillbank_embedding_max_length,
                device=self.skillbank_embedding_device,
            )
            self._skillbank_embedding_cache[digest] = embeddings
            return embeddings

    def _retrieve_skillbank_for_tasks(
        self,
        *,
        task_indices: Sequence[int],
        bank: Sequence[SkillEntry],
        bank_digest: str,
    ) -> dict[str, Any]:
        """Retrieve one frozen skill bundle per task for an entire episode."""

        indices = list(map(int, task_indices))
        contexts = self._env_client.task_contexts(indices)
        contexts_by_task = {int(context["task_index"]): context for context in contexts}
        if set(contexts_by_task) != set(indices):
            raise RuntimeError("WebShop task-context response is incomplete")
        queries = [
            build_retrieval_query(
                task_type=(
                    f"WebShop/{contexts_by_task[task_index].get('category', 'unknown')}"
                ),
                task_description=str(
                    contexts_by_task[task_index].get("instruction") or ""
                ),
                initial_observation=str(
                    contexts_by_task[task_index].get("initial_observation") or ""
                ),
            )
            for task_index in indices
        ]
        if bank:
            with self._skillbank_embedding_lock:
                bank_embeddings = self._skillbank_embeddings(bank, bank_digest)
                query_embeddings = embed_texts(
                    queries,
                    model_name=self.skillbank_embedding_model,
                    batch_size=self.skillbank_embedding_batch_size,
                    max_length=self.skillbank_embedding_max_length,
                    device=self.skillbank_embedding_device,
                )
            selected_by_task = retrieve_top_skills(
                skills=bank,
                skill_embeddings=bank_embeddings,
                query_embeddings=query_embeddings,
                top_k=min(self.skillbank_top_k, len(bank)),
            )
        else:
            selected_by_task = [[] for _ in indices]
        bank_by_id = {skill.skill_id: skill for skill in bank}
        skill_texts = [
            format_retrieved_skill_text(
                selected, bank_by_id, no_skill_text=NO_SKILL_TEXT
            )
            for selected in selected_by_task
        ]
        selected_bodies = [
            [bank_by_id[str(item["skill_id"])].content for item in selected]
            for selected in selected_by_task
        ]
        manifest = [
            {
                "task_index": task_index,
                "task_type": (
                    f"WebShop/{contexts_by_task[task_index].get('category', 'unknown')}"
                ),
                "task_description": str(
                    contexts_by_task[task_index].get("instruction") or ""
                ),
                "initial_observation": str(
                    contexts_by_task[task_index].get("initial_observation") or ""
                ),
                "query": queries[position],
                "selected": selected_by_task[position],
            }
            for position, task_index in enumerate(indices)
        ]
        return {
            "queries": queries,
            "selected": selected_by_task,
            "skill_texts": skill_texts,
            "selected_bodies": selected_bodies,
            "manifest": manifest,
        }

    def _build_skillbank_group_index(
        self,
        *,
        round_index: int,
        group_index: int,
        records: Sequence[dict[str, Any]],
        root: Path,
        signature: str,
        bank: Sequence[SkillEntry],
        bank_digest: str,
    ) -> dict[str, Any]:
        task_indices = list(map(int, records[0]["task_indices"]))
        retrieval = self._retrieve_skillbank_for_tasks(
            task_indices=task_indices,
            bank=bank,
            bank_digest=bank_digest,
        )
        bank_skills = list(retrieval["skill_texts"])
        retrieved_bodies = list(retrieval["selected_bodies"])
        jobs: list[dict[str, Any]] = []

        def add_job(
            job_id: str,
            condition: str,
            task_position: int,
            skill: str,
        ) -> None:
            payload = {
                "job_id": job_id,
                "condition": condition,
                "task_index": task_indices[task_position],
                "task_position": task_position,
                "skill": skill,
                "actor_seed": self._group_actor_seed(
                    round_index=round_index,
                    group_index=group_index,
                    task_position=task_position,
                ),
            }
            payload["signature"] = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            jobs.append(payload)

        for position in range(len(task_indices)):
            add_job(
                f"no_skill_{position:02d}",
                "skillbank_no_skill",
                position,
                NO_SKILL_TEXT,
            )
            if bank:
                add_job(
                    f"bank_only_{position:02d}",
                    "skillbank_bank_only",
                    position,
                    bank_skills[position],
                )
        for record in records:
            if not record["schema_valid"]:
                continue
            for position in range(len(task_indices)):
                add_job(
                    f"singleton_{record['sample_index']:02d}_{position:02d}",
                    "skillbank_singleton",
                    position,
                    str(record["content"]),
                )
                if bank:
                    add_job(
                        f"augmented_{record['sample_index']:02d}_{position:02d}",
                        "skillbank_augmented",
                        position,
                        _format_skill_bodies(
                            [*retrieved_bodies[position], str(record["content"])]
                        ),
                    )
        job_results, attempted_count, cached_count = self._run_group_episode_jobs(
            jobs,
            cache_dir=root / "episodes",
            group_signature=signature,
        )
        no_skill = [
            job_results[f"no_skill_{position:02d}"]
            for position in range(len(task_indices))
        ]
        bank_only = [
            job_results[
                f"bank_only_{position:02d}" if bank else f"no_skill_{position:02d}"
            ]
            for position in range(len(task_indices))
        ]
        sample_metrics: dict[str, dict[str, Any]] = {}
        singleton_episodes: dict[str, list[dict[str, Any]]] = {}
        augmented_episodes: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            key = str(record["sample_key"])
            if record["schema_valid"]:
                singleton = [
                    job_results[
                        f"singleton_{record['sample_index']:02d}_{position:02d}"
                    ]
                    for position in range(len(task_indices))
                ]
                augmented = [
                    job_results[
                        f"augmented_{record['sample_index']:02d}_{position:02d}"
                        if bank
                        else f"singleton_{record['sample_index']:02d}_{position:02d}"
                    ]
                    for position in range(len(task_indices))
                ]
            else:
                singleton = [
                    {
                        "task_index": task_index,
                        "condition": "skillbank_singleton",
                        "skill": "",
                        "trace": [],
                        "reward": 0.0,
                        "success": False,
                        "done": False,
                        "status": "invalid_skill",
                        "error": str(record.get("schema_error") or "invalid skill"),
                    }
                    for task_index in task_indices
                ]
                augmented = [dict(episode) for episode in singleton]
            singleton_episodes[key], augmented_episodes[key] = singleton, augmented
            singleton_summary = episode_metrics(singleton)
            no_skill_summary = episode_metrics(no_skill)
            augmented_summary = episode_metrics(augmented)
            bank_only_summary = episode_metrics(bank_only)
            reward_metrics = (
                compute_skillbank_counterfactual_reward(
                    singleton_sr=float(singleton_summary["success_rate"]),
                    no_skill_sr=float(no_skill_summary["success_rate"]),
                    bank_plus_candidate_sr=float(augmented_summary["success_rate"]),
                    bank_only_sr=float(bank_only_summary["success_rate"]),
                    standalone_weight=self.skillbank_standalone_weight,
                    retrieval_weight=self.skillbank_retrieval_weight,
                )
                if record["schema_valid"]
                else {
                    "singleton_sr": 0.0,
                    "no_skill_sr": float(no_skill_summary["success_rate"]),
                    "standalone_delta": 0.0,
                    "bank_plus_candidate_sr": 0.0,
                    "bank_only_sr": float(bank_only_summary["success_rate"]),
                    "bank_marginal_delta": 0.0,
                    "combined_sr_reward": 0.0,
                    "combined_reward": 0.0,
                }
            )
            reward_metrics.update(
                {
                    "schema_valid": bool(record["schema_valid"]),
                    "marginal_eligible": float(bool(record["schema_valid"])),
                    "standalone_mean_reward_delta": paired_condition_metrics(
                        no_skill, singleton
                    )["mean_reward_delta"],
                    "bank_mean_reward_delta": paired_condition_metrics(
                        bank_only, augmented
                    )["mean_reward_delta"],
                }
            )
            sample_metrics[key] = reward_metrics
        return {
            "status": "complete",
            "mode": _SKILLBANK_MODE,
            "signature": signature,
            "round_index": round_index,
            "group_index": group_index,
            "training_global_step": self._training_step_index(round_index, group_index),
            "task_indices": task_indices,
            "bank_size": len(bank),
            "bank_digest": bank_digest,
            "bank_top_k": self.skillbank_top_k,
            "retrieval": retrieval["manifest"],
            "planned_environment_episode_budget": len(jobs),
            "attempted_environment_episode_count": attempted_count,
            "cached_environment_episode_count": cached_count,
            "no_skill_episodes": no_skill,
            "bank_only_episodes": bank_only,
            "singleton_episodes": singleton_episodes,
            "augmented_episodes": augmented_episodes,
            "sample_metrics": sample_metrics,
            "updated_at": time.time(),
        }

    def _load_or_create_group_reward_index(
        self, *, round_index: int, group_index: int
    ) -> dict[str, Any]:
        records = self._canonical_group_generation_records(
            round_index=round_index, group_index=group_index
        )
        bank = self._load_skillbank_snapshot(
            self._training_step_index(round_index, group_index)
        )
        bank_digest = skillbank_snapshot_digest(bank)
        signature = self._group_reward_signature(
            round_index=round_index,
            group_index=group_index,
            records=records,
            bank_digest=bank_digest,
        )
        root = (
            self._round_dir(self.TASK_TYPE, round_index)
            / "group_reward"
            / _SKILLBANK_MODE
            / f"group_{group_index:02d}"
        )
        index_path = root / "index.json"

        def read_complete() -> dict[str, Any] | None:
            if not index_path.exists():
                return None
            payload = _read_json(index_path)
            if payload.get("status") != "complete":
                return None
            if payload.get("signature") != signature:
                raise RuntimeError(
                    f"cached {_SKILLBANK_MODE} group index differs; use a new trial"
                )
            return payload

        current = read_complete()
        if current is not None:
            return current
        lock_path = root / "group.lock"
        fd = _acquire_json_lock(
            lock_path,
            timeout_s=172800.0,
            foreign_host_stale_after_s=120.0,
        )
        heartbeat_stop = threading.Event()

        def refresh_group_lock() -> None:
            while not heartbeat_stop.wait(30.0):
                try:
                    _atomic_write_json(
                        lock_path,
                        {
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "claimed_at": time.time(),
                            "mode": _SKILLBANK_MODE,
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.warning("failed to refresh WebShop group lock %s", lock_path)

        heartbeat_thread = threading.Thread(
            target=refresh_group_lock,
            name=f"webshop-{_SKILLBANK_MODE}-lock-{round_index:04d}-{group_index:02d}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            current = read_complete()
            if current is not None:
                return current
            payload = self._build_skillbank_group_index(
                round_index=round_index,
                group_index=group_index,
                records=records,
                root=root,
                signature=signature,
                bank=bank,
                bank_digest=bank_digest,
            )
            _atomic_write_json(index_path, payload)
            return payload
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5.0)
            _release_json_lock(fd, lock_path)

    async def _shared_group_reward_index(
        self, *, round_index: int, group_index: int
    ) -> dict[str, Any]:
        key = (int(round_index), int(group_index))
        async with self._group_reward_task_lock:
            cached = self._group_reward_results.get(key)
            if cached is not None:
                return cached
            task = self._group_reward_tasks.get(key)
            if task is None:
                loop = asyncio.get_running_loop()
                concurrent_future = _group_coordinator_executor(
                    self.outcome_condition_workers
                ).submit(
                    self._load_or_create_group_reward_index,
                    round_index=round_index,
                    group_index=group_index,
                )
                task = asyncio.ensure_future(
                    asyncio.wrap_future(concurrent_future, loop=loop)
                )
                self._group_reward_tasks[key] = task
        try:
            payload = await asyncio.shield(task)
        except BaseException:
            if task.done():
                async with self._group_reward_task_lock:
                    if self._group_reward_tasks.get(key) is task:
                        self._group_reward_tasks.pop(key, None)
            raise
        async with self._group_reward_task_lock:
            self._group_reward_results[key] = payload
            self._group_reward_tasks.pop(key, None)
            current_step = self._training_step_index(round_index, group_index)
            stale = [
                cached_key
                for cached_key in self._group_reward_results
                if self._training_step_index(*cached_key)
                < current_step - 1
            ]
            for cached_key in stale:
                self._group_reward_results.pop(cached_key, None)
        return payload

    def _maybe_update_skillbank(
        self, training_global_step: int
    ) -> dict[str, Any] | None:
        step = int(training_global_step)
        root = self._skillbank_root
        update_path = root / "updates" / f"globalstep_{step:06d}.json"
        lock_path = root / "updates" / f"globalstep_{step:06d}.lock"
        fd = _acquire_json_lock(
            lock_path,
            timeout_s=7200.0,
            foreign_host_stale_after_s=120.0,
        )
        try:
            current = self._load_skillbank_snapshot(step)
            current_digest = skillbank_snapshot_digest(current)
            if update_path.exists():
                payload = _read_json(update_path)
                if (
                    payload.get("status") != "complete"
                    or int(payload.get("training_global_step", -1)) != step
                    or payload.get("previous_digest") != current_digest
                    or int(payload.get("previous_size", -1)) != len(current)
                ):
                    raise RuntimeError(
                        f"persisted WebShop skillbank update {step} is inconsistent"
                    )
                admitted = payload.get("admitted")
                expected_next = list(current)
                if isinstance(admitted, dict):
                    admitted_id = str(admitted.get("id") or "").strip()
                    admitted_model = str(admitted.get("model") or "").strip()
                    admitted_content = _normalized_skill_body(
                        admitted.get("content") or ""
                    )
                    if not admitted_id or not admitted_model or not admitted_content:
                        raise RuntimeError(
                            f"persisted WebShop skillbank update {step} has an "
                            "invalid admitted skill"
                        )
                    expected_next.append(
                        SkillEntry(
                            source_index=len(expected_next),
                            skill_id=admitted_id,
                            model=admitted_model,
                            content=admitted_content,
                        )
                    )
                expected_next_digest = skillbank_snapshot_digest(expected_next)
                next_path = self._skillbank_snapshot_path(step + 1)
                if (
                    payload.get("next_snapshot") != str(next_path)
                    or payload.get("next_digest") != expected_next_digest
                    or int(payload.get("next_size", -1)) != len(expected_next)
                    or bool(payload.get("noop")) != (admitted is None)
                ):
                    raise RuntimeError(
                        f"persisted WebShop skillbank update {step} has invalid output"
                    )
                if next_path.exists():
                    persisted_digest = skillbank_snapshot_digest(
                        self._read_skillbank(next_path)
                    )
                    if persisted_digest != expected_next_digest:
                        raise RuntimeError(
                            "persisted next WebShop skillbank snapshot differs"
                        )
                else:
                    write_skillbank(next_path, expected_next)
                with self._skillbank_snapshot_lock:
                    self._skillbank_snapshot_cache[step + 1] = expected_next
                return payload
            sample_paths = self._training_step_sample_paths(step)
            if any(
                not generation_path.exists() or not metrics_path.exists()
                for _, generation_path, metrics_path in sample_paths
            ):
                return None
            candidates: list[dict[str, Any]] = []
            for sample_key, generation_path, metrics_path in sample_paths:
                generation = _read_json(generation_path)
                metrics = _read_json(metrics_path)
                bank_metrics = metrics.get("skillbank_counterfactual")
                bank_payload = metrics.get("skillbank")
                if not isinstance(bank_metrics, dict):
                    return None
                if (
                    not isinstance(bank_payload, dict)
                    or bank_payload.get("bank_digest") != current_digest
                ):
                    raise RuntimeError(
                        f"WebShop sample {sample_key} was scored against a different "
                        "skillbank snapshot"
                    )
                parsed = generation.get("parsed")
                valid = bool(generation.get("schema_valid")) and isinstance(
                    parsed, dict
                )
                candidates.append(
                    {
                        "sample_key": sample_key,
                        "content": (
                            skill_payload_to_prompt_text(parsed) if valid else ""
                        ),
                        "model": self.actor_model,
                        "schema_valid": valid,
                        "bank_marginal_delta": float(
                            bank_metrics.get("bank_marginal_delta", 0.0)
                        ),
                        "combined_sr_reward": float(
                            bank_metrics.get("combined_sr_reward", 0.0)
                        ),
                        "standalone_delta": float(
                            bank_metrics.get("standalone_delta", 0.0)
                        ),
                        "mean_reward_delta": float(
                            bank_metrics.get("bank_mean_reward_delta", 0.0)
                        ),
                    }
                )
            winner = select_online_skillbank_admission(
                candidates=candidates,
                current_skills=current,
                min_marginal=self.skillbank_update_min_marginal,
                allow_noop=self.skillbank_allow_noop_update,
                max_size=self.skillbank_max_size,
            )
            next_skills = list(current)
            admitted: dict[str, Any] | None = None
            if winner is not None:
                entry = SkillEntry(
                    source_index=len(next_skills),
                    skill_id=(
                        f"train_step_{step:06d}_"
                        f"{str(winner['sample_key']).replace('/', '_')}"
                    ),
                    model=str(winner.get("model") or self.actor_model),
                    content=str(winner["content"]),
                )
                next_skills.append(entry)
                admitted = {
                    "id": entry.skill_id,
                    "model": entry.model,
                    "content": entry.content,
                    "source_sample_key": winner["sample_key"],
                    "bank_marginal_delta": winner["bank_marginal_delta"],
                    "combined_sr_reward": winner["combined_sr_reward"],
                    "standalone_delta": winner["standalone_delta"],
                }
            next_path = self._skillbank_snapshot_path(step + 1)
            next_digest = skillbank_snapshot_digest(next_skills)
            if next_path.exists():
                if (
                    skillbank_snapshot_digest(self._read_skillbank(next_path))
                    != next_digest
                ):
                    raise RuntimeError(
                        "persisted next WebShop skillbank snapshot differs"
                    )
            else:
                write_skillbank(next_path, next_skills)
            payload = {
                "status": "complete",
                "training_global_step": step,
                "previous_snapshot": str(self._skillbank_snapshot_path(step)),
                "previous_digest": current_digest,
                "previous_size": len(current),
                "next_snapshot": str(next_path),
                "next_digest": next_digest,
                "next_size": len(next_skills),
                "candidate_count": len(candidates),
                "admitted": admitted,
                "noop": admitted is None,
                "updated_at": time.time(),
            }
            _atomic_write_json(update_path, payload)
            with self._skillbank_snapshot_lock:
                self._skillbank_snapshot_cache[step + 1] = next_skills
            return payload
        finally:
            _release_json_lock(fd, lock_path)
