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
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future
from concurrent.futures import wait as wait_futures
from functools import partial
from pathlib import Path
from typing import Any

from areal.reward.alfworld_online_skillbank import (
    build_online_skillbank_step_plan,
    compute_online_skillbank_marginal_metrics,
    rank_online_skillbank_entries,
    select_positive_skillbank_updates,
)
from areal.workflow.alfworld_environment import (
    _drain_outcome_condition_futures,
    _get_outcome_episode_dispatcher,
    _invalid_skill_episode,
    _rollout_timeout_episode,
    _run_rollout_episode_process,
    _write_rollout_episode_file,
)
from areal.workflow.alfworld_runtime import (
    _NO_SKILL_BASELINE_TEXT,
    _ONLINE_SKILLBANK_MODE,
    _ONLINE_SKILLBANK_QUERY_VERSION,
    _OUTCOME_RETRYABLE_INFRA_STATUSES,
    _ROUND_WAIT_LOG_INTERVAL_S,
    _acquire_json_lock,
    _atomic_write_json,
    _baseline_index_from_episodes,
    _embed_group_similarity_texts,
    _episodes_as_outcomes,
    _get_episode_rollout_executor,
    _get_outcome_condition_executor,
    _online_skillbank_initial_observation,
    _online_skillbank_prompt_text,
    _online_skillbank_retrieval_query,
    _online_skillbank_skills_signature,
    _read_json,
    _release_json_lock,
    _unlink_if_exists,
    logger,
)
from areal.workflow.skill_prompts import (
    skill_payload_to_prompt_text,
)


class ALFWorldBankMixin:
    def _online_skillbank_root(self) -> Path:
        return self.artifact_dir / "online_skillbank"

    def _online_skillbank_step_dir(self, training_global_step: int) -> Path:
        return (
            self._online_skillbank_root()
            / "steps"
            / (f"globalstep_{int(training_global_step):06d}")
        )

    def _online_skillbank_step_plan(self, training_global_step: int):
        return build_online_skillbank_step_plan(
            self.task_types,
            training_global_step=int(training_global_step),
            group_count=self.train_batch_size,
            seed=self.online_skillbank_seed,
            warmup_steps=self.online_skillbank_warmup_steps,
            warmup_update_type_count=(self.online_skillbank_warmup_update_type_count),
            later_update_type_count=(self.online_skillbank_later_update_type_count),
        )

    @staticmethod
    def _read_online_skillbank_payload(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = _read_json(path)
        except Exception:
            return None
        skills = payload.get("skills")
        if payload.get("status") != "complete" or not isinstance(skills, list):
            return None
        if any(not isinstance(skill, dict) for skill in skills):
            return None
        try:
            skill_count = int(payload.get("skill_count", -1))
        except (TypeError, ValueError):
            return None
        if skill_count != len(skills):
            return None
        skill_ids = [skill.get("skill_id") for skill in skills]
        if any(not isinstance(skill_id, str) or not skill_id for skill_id in skill_ids):
            return None
        if len(set(skill_ids)) != len(skill_ids):
            return None
        if any(
            skill.get("source_index") != source_index
            for source_index, skill in enumerate(skills)
        ):
            return None
        signature = _online_skillbank_skills_signature(skills)
        if payload.get("bank_signature") != signature:
            return None
        return payload

    def _validate_online_skillbank_snapshot_plan(
        self, payload: dict[str, Any], training_global_step: int
    ) -> None:
        """Reject a resume that would reinterpret an existing step snapshot."""

        step = int(training_global_step)
        if int(payload.get("training_global_step", -1)) != step:
            raise RuntimeError("online skillbank snapshot step mismatch")
        plan = self._online_skillbank_step_plan(step)
        observed_groups = payload.get("group_input_task_types")
        observed_updates = payload.get("update_task_types")
        if observed_groups != list(
            plan.group_input_task_types
        ) or observed_updates != list(plan.update_task_types):
            raise RuntimeError(
                "online skillbank snapshot plan does not match the current seed, "
                "task types, or batch layout; use a new trial name"
            )

    def _load_or_create_online_skillbank_snapshot(
        self, training_global_step: int
    ) -> dict[str, Any]:
        """Return the immutable pre-update bank used by one optimizer step."""

        step = int(training_global_step)
        step_dir = self._online_skillbank_step_dir(step)
        snapshot_path = step_dir / "bank_before.json"
        current = self._read_online_skillbank_payload(snapshot_path)
        if current is not None:
            self._validate_online_skillbank_snapshot_plan(current, step)
            return current

        previous_payload: dict[str, Any] | None = None
        if step > 0:
            previous_after = (
                self._online_skillbank_step_dir(step - 1) / "bank_after.json"
            )
            wait_started = time.monotonic()
            last_log = 0.0
            wait_timeout = max(
                172800.0,
                float(self.rollout_timeout_s or 7200.0) * self.train_batch_size,
            )
            while previous_payload is None:
                previous_payload = self._read_online_skillbank_payload(previous_after)
                if previous_payload is not None:
                    break
                # Recover a crash between the last group index and step finalization.
                self._maybe_finalize_online_skillbank_step(step - 1)
                previous_payload = self._read_online_skillbank_payload(previous_after)
                if previous_payload is not None:
                    break
                elapsed = time.monotonic() - wait_started
                if elapsed >= wait_timeout:
                    raise TimeoutError(
                        "timed out waiting for previous online skillbank update: "
                        f"{previous_after}"
                    )
                now = time.time()
                if now - last_log >= _ROUND_WAIT_LOG_INTERVAL_S:
                    logger.info(
                        "waiting for online skillbank globalstep %s update before "
                        "preparing globalstep %s",
                        step - 1,
                        step,
                    )
                    last_log = now
                time.sleep(2.0)

        step_dir.mkdir(parents=True, exist_ok=True)
        lock_path = step_dir / "bank_before.lock"
        fd = _acquire_json_lock(
            lock_path,
            timeout_s=1800.0,
            foreign_host_stale_after_s=120.0,
        )
        try:
            current = self._read_online_skillbank_payload(snapshot_path)
            if current is not None:
                self._validate_online_skillbank_snapshot_plan(current, step)
                return current
            skills = (
                [dict(skill) for skill in previous_payload.get("skills", [])]
                if previous_payload is not None
                else []
            )
            for source_index, skill in enumerate(skills):
                skill["source_index"] = source_index
            plan = self._online_skillbank_step_plan(step)
            payload = {
                "status": "complete",
                "schema_version": 1,
                "mode": _ONLINE_SKILLBANK_MODE,
                "training_global_step": step,
                "source_training_global_step": step - 1,
                "source_bank_signature": (
                    str(previous_payload.get("bank_signature", ""))
                    if previous_payload is not None
                    else ""
                ),
                "bank_signature": _online_skillbank_skills_signature(skills),
                "skill_count": len(skills),
                "skills": skills,
                "group_input_task_types": list(plan.group_input_task_types),
                "update_task_types": list(plan.update_task_types),
                "updated_at": time.time(),
            }
            _atomic_write_json(snapshot_path, payload)
            return payload
        finally:
            _release_json_lock(fd, lock_path)

    async def _shared_online_skillbank_group_index(
        self,
        *,
        task_type: str,
        round_in_category: int,
        group_index: int,
        games: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run one in-process online-bank coordinator for all group samples."""

        key = (task_type, round_in_category, group_index)
        self._prune_shared_training_tasks(
            self._training_step_index(round_in_category, group_index)
        )
        tasks = getattr(self, "_online_skillbank_group_tasks", None)
        if tasks is None:
            tasks = {}
            self._online_skillbank_group_tasks = tasks
        task = tasks.get(key)
        if task is None:
            build = partial(
                self._load_or_create_online_skillbank_group_index,
                task_type=task_type,
                round_in_category=round_in_category,
                group_index=group_index,
                games=[dict(game) for game in games],
            )
            task = asyncio.create_task(self._run_skill_eval_blocking(build))
            tasks[key] = task

            def remove_failed(done_task: asyncio.Task[dict[str, Any]]) -> None:
                if done_task.cancelled() or done_task.exception() is not None:
                    if tasks.get(key) is done_task:
                        tasks.pop(key, None)

            task.add_done_callback(remove_failed)
        return await asyncio.shield(task)

    @staticmethod
    def _read_complete_outcome_condition(
        condition_dir: Path, *, signature: str
    ) -> dict[str, Any] | None:
        index_path = condition_dir / "condition_index.json"
        checkpoint_path = condition_dir / "checkpoint.json"
        if not index_path.exists() or not checkpoint_path.exists():
            return None
        try:
            index = _read_json(index_path)
            checkpoint = _read_json(checkpoint_path)
        except Exception:
            return None
        if checkpoint.get("status") != "complete":
            return None
        if index.get("status") != "complete" or index.get("signature") != signature:
            return None
        episodes = index.get("episodes")
        if not isinstance(episodes, list) or any(
            not isinstance(episode, dict)
            or str(episode.get("status", "")) in _OUTCOME_RETRYABLE_INFRA_STATUSES
            for episode in episodes
        ):
            return None
        return index

    def _run_outcome_condition_rollouts(
        self,
        *,
        condition_dir: Path,
        signature: str,
        condition_name: str,
        skill_name: str,
        skill_texts: Sequence[str],
        games: Sequence[dict[str, Any]],
        task_indices: Sequence[int],
        round_in_category: int,
        group_index: int,
        synthetic_episodes: Sequence[dict[str, Any]] | None = None,
        infra_retries_remaining: int = 2,
        attempted_rollout_count: int | None = None,
        condition_mode: str = "online_skillbank",
    ) -> dict[str, Any]:
        """Run one resume-safe outcome condition, filling only missing episodes."""

        cached = self._read_complete_outcome_condition(
            condition_dir, signature=signature
        )
        if cached is not None:
            return cached
        selected_games = [dict(games[int(index)]) for index in task_indices]
        if len(skill_texts) != len(selected_games):
            raise ValueError("outcome condition skill texts must align with tasks")
        if synthetic_episodes is not None and len(synthetic_episodes) != len(
            selected_games
        ):
            raise ValueError("synthetic outcome episodes must align with tasks")

        condition_dir.mkdir(parents=True, exist_ok=True)
        progress_metadata = {
            "mode": str(condition_mode),
            "condition": condition_name,
            "skill_name": skill_name,
            "round_in_category": round_in_category,
            "group_index": group_index,
            "signature": signature,
            "task_indices": [int(index) for index in task_indices],
        }
        episodes: list[dict[str, Any] | None] = [None] * len(selected_games)
        for local_index, game in enumerate(selected_games):
            episode_path = condition_dir / f"episode_{local_index:02d}.json"
            if not episode_path.exists():
                continue
            try:
                payload = _read_json(episode_path)
                episode = payload.get("episode")
            except Exception:
                continue
            if payload.get("signature") != signature or not isinstance(episode, dict):
                continue
            if str(episode.get("gamefile", "")) != str(game.get("gamefile", "")):
                continue
            if str(episode.get("status", "")) in _OUTCOME_RETRYABLE_INFRA_STATUSES:
                # A reconnect/resume must rerun infrastructure failures rather
                # than caching them as scientific task failures.
                continue
            episodes[local_index] = episode

        if attempted_rollout_count is None:
            attempted_rollout_count = sum(episode is not None for episode in episodes)
            checkpoint_path = condition_dir / "checkpoint.json"
            if checkpoint_path.exists():
                try:
                    previous_checkpoint = _read_json(checkpoint_path)
                    if previous_checkpoint.get("signature") == signature:
                        attempted_rollout_count = max(
                            attempted_rollout_count,
                            int(
                                previous_checkpoint.get(
                                    "attempted_rollout_count",
                                    attempted_rollout_count,
                                )
                            ),
                        )
                except Exception:
                    pass

        _atomic_write_json(
            condition_dir / "checkpoint.json",
            {
                **progress_metadata,
                "status": "rollout_in_progress",
                "n_rollouts": sum(episode is not None for episode in episodes),
                "n_rollouts_expected": len(selected_games),
                "attempted_rollout_count": attempted_rollout_count,
                "updated_at": time.time(),
            },
        )

        if synthetic_episodes is not None:
            for local_index, episode in enumerate(synthetic_episodes):
                episodes[local_index] = dict(episode)
                _write_rollout_episode_file(
                    sample_dir=condition_dir,
                    rollout_index=local_index,
                    episode=episodes[local_index],
                    final=True,
                    metadata=progress_metadata,
                )
        else:
            missing_indices = [
                index for index, episode in enumerate(episodes) if episode is None
            ]
            attempted_rollout_count += len(missing_indices)
            _atomic_write_json(
                condition_dir / "checkpoint.json",
                {
                    **progress_metadata,
                    "status": "rollout_in_progress",
                    "n_rollouts": sum(episode is not None for episode in episodes),
                    "n_rollouts_expected": len(selected_games),
                    "attempted_rollout_count": attempted_rollout_count,
                    "updated_at": time.time(),
                },
            )
            flattened = bool(getattr(self, "outcome_flattened_scheduler", False))
            executor = (
                _get_outcome_episode_dispatcher(self.episode_rollout_workers)
                if flattened
                else _get_episode_rollout_executor(self.episode_rollout_workers)
            )
            if executor is None:
                raise RuntimeError(
                    "outcome rollouts require episode_rollout_workers > 0"
                )
            futures: dict[Future, int] = {}
            for local_index in missing_indices:
                worker_payload = {
                    "repo_root": str(self.repo_root),
                    "sample_dir": str(condition_dir),
                    "game": selected_games[local_index],
                    "skill_name": skill_name,
                    "skill_text": str(skill_texts[local_index]),
                    "rollout_index": local_index,
                    "max_rollout_steps": self.max_rollout_steps,
                    "memory_window": self.memory_window,
                    "max_commands": self.max_commands,
                    "actor_base_url": self.actor_base_url,
                    "actor_model": self.actor_model,
                    "actor_api_key": self.actor_api_key,
                    "actor_timeout_s": self.actor_timeout_s,
                    "actor_temperature": self.actor_temperature,
                    "tokenizer_path": self.tokenizer_path,
                    # The seed depends only on the group and original task,
                    # so every outcome condition uses common randomness.
                    "seed_base": (
                        900000
                        + round_in_category * 100000
                        + group_index * 1000
                        + int(task_indices[local_index]) * 100
                    ),
                    # Only the coordinator publishes canonical artifacts; late workers must not overwrite retries.
                    "persist_progress": False,
                    "progress_metadata": progress_metadata,
                }
                if flattened:
                    future = executor.submit(
                        worker_payload,
                        timeout_s=self.rollout_timeout_s,
                        priority=(0 if infra_retries_remaining < 2 else 1),
                    )
                else:
                    future = executor.submit(
                        _run_rollout_episode_process, worker_payload
                    )
                futures[future] = local_index
            pending = set(futures)
            deadline = (
                time.monotonic() + self.rollout_timeout_s
                if self.rollout_timeout_s is not None
                else None
            )
            abort_flattened_condition = False
            while pending:
                wait_timeout = None
                if deadline is not None:
                    wait_timeout = max(0.0, deadline - time.monotonic())
                    if wait_timeout <= 0.0:
                        break
                done, pending = wait_futures(
                    pending,
                    timeout=wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    break
                for future in done:
                    local_index = futures[future]
                    try:
                        episodes[local_index] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        episode = _invalid_skill_episode(
                            game=selected_games[local_index],
                            skill_name=skill_name,
                            error="outcome_rollout_exception: " + repr(exc),
                        )
                        episode["status"] = "outcome_rollout_error"
                        episode["traceback"] = traceback.format_exc()
                        episodes[local_index] = episode
                    if (
                        flattened
                        and str(episodes[local_index].get("status", ""))
                        == "rollout_timeout"
                    ):
                        abort_flattened_condition = True
                if abort_flattened_condition:
                    # Cancel queued siblings; running tasks retain dispatcher slots until they exit.
                    for future in pending:
                        local_index = futures[future]
                        future.cancel()
                        episodes[local_index] = _rollout_timeout_episode(
                            game=selected_games[local_index],
                            skill_name=skill_name,
                            error=(
                                "outcome condition cancelled after a sibling "
                                "rollout timed out"
                            ),
                            timeout_s=self.rollout_timeout_s,
                        )
                    pending.clear()
                    break
            if pending:
                timeout_text = (
                    f"{self.rollout_timeout_s:.1f}s"
                    if self.rollout_timeout_s is not None
                    else "the configured deadline"
                )
                for future in pending:
                    local_index = futures[future]
                    future.cancel()
                    episodes[local_index] = _rollout_timeout_episode(
                        game=selected_games[local_index],
                        skill_name=skill_name,
                        error=("outcome rollout timed out after " + timeout_text),
                        timeout_s=self.rollout_timeout_s,
                    )

        complete_episodes = [
            episode
            if episode is not None
            else _invalid_skill_episode(
                game=selected_games[index],
                skill_name=skill_name,
                error="outcome rollout missing",
            )
            for index, episode in enumerate(episodes)
        ]
        status_counts = Counter(
            str(episode.get("status")) for episode in complete_episodes
        )
        if synthetic_episodes is None:
            bad_indices = [
                index
                for index, episode in enumerate(complete_episodes)
                if str(episode.get("status", "")) in _OUTCOME_RETRYABLE_INFRA_STATUSES
            ]
            if bad_indices:
                # Keep failed attempts outside the episode files: retries must not
                # reuse them, but the original cause must survive the retry.
                failure_details = [
                    {
                        "rollout_index": index,
                        "task_index": int(task_indices[index]),
                        "episode": complete_episodes[index],
                    }
                    for index in bad_indices
                ]
                failure_path = (
                    condition_dir
                    / "diagnostics"
                    / f"rollout_failure_{uuid.uuid4().hex}.json"
                )
                _atomic_write_json(
                    failure_path,
                    {
                        **progress_metadata,
                        "status": "retryable_infra_error",
                        "infra_retries_remaining": infra_retries_remaining,
                        "attempted_rollout_count": attempted_rollout_count,
                        "failures": failure_details,
                        "updated_at": time.time(),
                    },
                )
                error_summary = [
                    {
                        "rollout_index": item["rollout_index"],
                        "status": item["episode"].get("status"),
                        "error": item["episode"].get("error"),
                    }
                    for item in failure_details
                ]
                logger.error(
                    "outcome condition %s infrastructure errors: %s; diagnostics=%s",
                    condition_name,
                    error_summary,
                    failure_path,
                )
                bad_index_set = set(bad_indices)
                for local_index, episode in enumerate(complete_episodes):
                    if local_index in bad_index_set:
                        continue
                    _write_rollout_episode_file(
                        sample_dir=condition_dir,
                        rollout_index=local_index,
                        episode=episode,
                        final=True,
                        metadata=progress_metadata,
                    )
                for local_index in bad_indices:
                    _unlink_if_exists(condition_dir / f"episode_{local_index:02d}.json")
                    _unlink_if_exists(
                        condition_dir / f"current_episode_{local_index:02d}.json"
                    )
                _atomic_write_json(
                    condition_dir / "checkpoint.json",
                    {
                        **progress_metadata,
                        "status": "retryable_infra_error",
                        "retryable_rollout_indices": bad_indices,
                        "status_counts": dict(status_counts),
                        "attempted_rollout_count": attempted_rollout_count,
                        "updated_at": time.time(),
                    },
                )
                timed_out = any(
                    str(complete_episodes[index].get("status", "")) == "rollout_timeout"
                    for index in bad_indices
                )
                if infra_retries_remaining > 0 and not timed_out:
                    logger.warning(
                        "retrying outcome condition %s for infra-failed indices %s "
                        "(%d retries remaining)",
                        condition_name,
                        bad_indices,
                        infra_retries_remaining,
                    )
                    return self._run_outcome_condition_rollouts(
                        condition_dir=condition_dir,
                        signature=signature,
                        condition_name=condition_name,
                        skill_name=skill_name,
                        skill_texts=skill_texts,
                        games=games,
                        task_indices=task_indices,
                        round_in_category=round_in_category,
                        group_index=group_index,
                        synthetic_episodes=synthetic_episodes,
                        infra_retries_remaining=infra_retries_remaining - 1,
                        attempted_rollout_count=attempted_rollout_count,
                        condition_mode=condition_mode,
                    )
                raise RuntimeError(
                    "ALFWorld outcome condition infrastructure failure: "
                    f"condition={condition_name} retryable_indices={bad_indices} "
                    f"status_counts={dict(status_counts)} errors={error_summary} "
                    f"diagnostics={failure_path}"
                )

        for local_index, episode in enumerate(complete_episodes):
            _write_rollout_episode_file(
                sample_dir=condition_dir,
                rollout_index=local_index,
                episode=episode,
                final=True,
                metadata=progress_metadata,
            )

        wins = sum(int(bool(episode.get("won"))) for episode in complete_episodes)
        payload = {
            **progress_metadata,
            "status": "complete",
            "synthetic": synthetic_episodes is not None,
            "wins": wins,
            "n_rollouts": len(complete_episodes),
            "attempted_rollout_count": attempted_rollout_count,
            "sr": wins / max(1, len(complete_episodes)),
            "status_counts": dict(status_counts),
            "episodes": complete_episodes,
            "updated_at": time.time(),
        }
        _atomic_write_json(condition_dir / "condition_index.json", payload)
        _atomic_write_json(
            condition_dir / "checkpoint.json",
            {
                **progress_metadata,
                "status": "complete",
                "synthetic": synthetic_episodes is not None,
                "wins": wins,
                "n_rollouts": len(complete_episodes),
                "attempted_rollout_count": attempted_rollout_count,
                "sr": payload["sr"],
                "updated_at": time.time(),
            },
        )
        return payload

    def _run_outcome_condition_jobs(
        self,
        jobs: Sequence[tuple[str, Callable[[], dict[str, Any]]]],
    ) -> dict[str, dict[str, Any]]:
        """Execute independent condition coordinators, preserving input ordering."""

        if not bool(getattr(self, "outcome_flattened_scheduler", False)):
            return {name: job() for name, job in jobs}
        executor = _get_outcome_condition_executor(self.outcome_condition_workers)
        futures = [(name, executor.submit(job)) for name, job in jobs]
        return _drain_outcome_condition_futures(futures)

    def _online_skillbank_records(
        self,
        *,
        task_type: str,
        round_in_category: int,
        group_index: int,
        games: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        round_dir = self._round_dir(task_type, round_in_category)
        generation_paths = sorted(
            (round_dir / "skills").glob(
                f"group_{group_index:02d}_sample_*/generation.json"
            )
        )
        if len(generation_paths) != self.samples_per_round:
            raise RuntimeError(
                "online_skillbank expected "
                f"{self.samples_per_round} generated skills for group {group_index}, "
                f"found {len(generation_paths)}"
            )
        expected_gamefiles = [str(game.get("gamefile", "")) for game in games]
        records: list[dict[str, Any]] = []
        for generation_path in generation_paths:
            generation = _read_json(generation_path)
            selected_games = generation.get("selected_games")
            if not isinstance(selected_games, list):
                raise RuntimeError(
                    "online_skillbank generation has no selected games: "
                    f"{generation_path}"
                )
            gamefiles = [str(game.get("gamefile", "")) for game in selected_games]
            if gamefiles != expected_gamefiles:
                raise RuntimeError(
                    "online_skillbank requires identical ordered games within one "
                    f"group: {generation_path}"
                )
            parsed = generation.get("parsed")
            valid = bool(generation.get("schema_valid")) and isinstance(parsed, dict)
            skill_text = skill_payload_to_prompt_text(parsed) if valid else ""
            records.append(
                {
                    "sample_key": generation_path.parent.name,
                    "generation_path": str(generation_path),
                    "schema_valid": valid,
                    "skill_text": skill_text,
                }
            )
        return records

    def _online_skillbank_retrieval_manifest(
        self,
        *,
        bank_snapshot: dict[str, Any],
        games: Sequence[dict[str, Any]],
        baseline_episodes: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        skills = [dict(skill) for skill in bank_snapshot.get("skills", [])]
        queries = [
            _online_skillbank_retrieval_query(
                task_type=str(game.get("task_type") or ""),
                task_description=str(game.get("task_desc") or ""),
                initial_observation=_online_skillbank_initial_observation(
                    dict(baseline_episodes[index])
                ),
            )
            for index, game in enumerate(games)
        ]
        rankings: list[list[dict[str, Any]]] = [[] for _ in games]
        if skills:
            embeddings = _embed_group_similarity_texts(
                [str(skill.get("text") or "") for skill in skills] + queries,
                model_name=self.online_skillbank_embedding_model,
                batch_size=self.online_skillbank_embedding_batch_size,
                max_length=self.online_skillbank_embedding_max_length,
            )
            skill_embeddings = embeddings[: len(skills)]
            query_embeddings = embeddings[len(skills) :]
            similarities = query_embeddings.float() @ skill_embeddings.float().T
            rankings = [
                rank_online_skillbank_entries(
                    entries=skills,
                    cosine_similarities=similarities[index].tolist(),
                    top_k=self.online_skillbank_top_k,
                )
                for index in range(len(games))
            ]

        skills_by_id = {str(skill["skill_id"]): skill for skill in skills}
        tasks: list[dict[str, Any]] = []
        retrieved_skill_texts: list[str] = []
        for index, (game, query, ranking) in enumerate(
            zip(games, queries, rankings, strict=True)
        ):
            selected_entries = [skills_by_id[str(item["skill_id"])] for item in ranking]
            retrieved_skill_texts.append(
                _online_skillbank_prompt_text(selected_entries)
            )
            tasks.append(
                {
                    "rollout_index": index,
                    "task_type": str(game.get("task_type") or ""),
                    "gamefile": str(game.get("gamefile") or ""),
                    "task_description": str(game.get("task_desc") or ""),
                    "initial_observation": _online_skillbank_initial_observation(
                        dict(baseline_episodes[index])
                    ),
                    "query": query,
                    "retrieved_top_skills": ranking,
                }
            )
        signature_payload = {
            "mode": _ONLINE_SKILLBANK_MODE,
            "query_version": _ONLINE_SKILLBANK_QUERY_VERSION,
            "bank_signature": bank_snapshot["bank_signature"],
            "embedding_model": self.online_skillbank_embedding_model,
            "embedding_max_length": self.online_skillbank_embedding_max_length,
            "top_k": self.online_skillbank_top_k,
            "tasks": tasks,
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        return {
            "status": "complete",
            "signature": signature,
            **signature_payload,
            "retrieved_skill_texts": retrieved_skill_texts,
            "updated_at": time.time(),
        }

    def _load_or_create_online_skillbank_group_index(
        self,
        *,
        task_type: str,
        round_in_category: int,
        group_index: int,
        games: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate one GRPO group against a shared immutable bank snapshot."""

        training_global_step = self._training_step_index(round_in_category, group_index)
        global_group_index = round_in_category * self.groups_per_round + group_index
        training_step_group_index = global_group_index % self.train_batch_size
        bank_snapshot = self._load_or_create_online_skillbank_snapshot(
            training_global_step
        )
        records = self._online_skillbank_records(
            task_type=task_type,
            round_in_category=round_in_category,
            group_index=group_index,
            games=games,
        )
        signature_payload = {
            "schema_version": 1,
            "mode": _ONLINE_SKILLBANK_MODE,
            "training_global_step": training_global_step,
            "training_step_group_index": training_step_group_index,
            "task_type": task_type,
            "input_task_type": str(games[0].get("task_type") or "") if games else "",
            "round_in_category": round_in_category,
            "group_index": group_index,
            "actor_model": self.actor_model,
            "actor_temperature": self.actor_temperature,
            "max_rollout_steps": self.max_rollout_steps,
            "memory_window": self.memory_window,
            "max_commands": self.max_commands,
            "bank_signature": bank_snapshot["bank_signature"],
            "bank_weight": self.online_skillbank_bank_weight,
            "standalone_weight": self.online_skillbank_standalone_weight,
            "top_k": self.online_skillbank_top_k,
            "embedding_model": self.online_skillbank_embedding_model,
            "embedding_max_length": self.online_skillbank_embedding_max_length,
            "records": records,
            "games": [
                {
                    "task_type": game.get("task_type", ""),
                    "gamefile": game.get("gamefile", ""),
                    "traj_json": game.get("traj_json", ""),
                }
                for game in games
            ],
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        group_dir = (
            self._online_skillbank_step_dir(training_global_step)
            / "groups"
            / f"group_{training_step_group_index:02d}"
        )
        index_path = group_dir / "index.json"

        def read_current() -> dict[str, Any] | None:
            if not index_path.exists():
                return None
            try:
                current = _read_json(index_path)
            except Exception:
                return None
            if current.get("status") != "complete":
                return None
            if current.get("signature") != signature:
                raise RuntimeError(
                    "online skillbank group signature changed inside an existing "
                    f"training step: {index_path}"
                )
            return current

        current = read_current()
        if current is not None:
            self._maybe_finalize_online_skillbank_step(training_global_step)
            return current

        group_dir.mkdir(parents=True, exist_ok=True)
        condition_waves = 2 + 2 * self.samples_per_round
        per_wave_timeout = float(self.rollout_timeout_s or 7200.0)
        lock_timeout = max(172800.0, condition_waves * per_wave_timeout + 3600.0)
        lock_path = group_dir / "group.lock"
        fd = _acquire_json_lock(
            lock_path,
            timeout_s=lock_timeout,
            foreign_host_stale_after_s=120.0,
        )
        heartbeat_stop = threading.Event()

        def refresh_lock_heartbeat() -> None:
            while not heartbeat_stop.wait(30.0):
                try:
                    _atomic_write_json(
                        lock_path,
                        {
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "claimed_at": time.time(),
                            "mode": _ONLINE_SKILLBANK_MODE,
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.warning("failed to refresh online bank lock %s", lock_path)

        heartbeat_thread = threading.Thread(
            target=refresh_lock_heartbeat,
            name=f"online-skillbank-lock-{training_step_group_index:02d}",
            daemon=True,
        )
        heartbeat_thread.start()
        condition_futures: list[tuple[str, Future]] = []
        try:
            current = read_current()
            if current is not None:
                return current
            _atomic_write_json(
                group_dir / "checkpoint.json",
                {
                    "status": "rollout_in_progress",
                    "signature": signature,
                    "training_global_step": training_global_step,
                    "training_step_group_index": training_step_group_index,
                    "updated_at": time.time(),
                },
            )
            full_task_indices = list(range(len(games)))
            no_skill_dir = group_dir / "conditions" / "no_skill"
            no_skill_signature = hashlib.sha256(
                f"{signature}:no_skill".encode()
            ).hexdigest()
            no_skill_name = (
                f"online_skillbank/no_skill/globalstep_{training_global_step:06d}/"
                f"group_{training_step_group_index:02d}"
            )
            singleton_jobs: list[tuple[str, Callable[[], dict[str, Any]]]] = []
            for record in records:
                sample_key = str(record["sample_key"])
                singleton_dir = group_dir / "conditions" / "singletons" / sample_key
                singleton_signature = hashlib.sha256(
                    f"{signature}:singleton:{sample_key}".encode()
                ).hexdigest()
                singleton_name = (
                    "online_skillbank/singleton/"
                    f"globalstep_{training_global_step:06d}/"
                    f"group_{training_step_group_index:02d}/{sample_key}"
                )
                singleton_kwargs: dict[str, Any] = {
                    "condition_dir": singleton_dir,
                    "signature": singleton_signature,
                    "condition_name": f"singleton/{sample_key}",
                    "skill_name": singleton_name,
                    "skill_texts": (
                        [str(record["skill_text"])] * len(games)
                        if record.get("schema_valid")
                        else [""] * len(games)
                    ),
                    "games": games,
                    "task_indices": full_task_indices,
                    "round_in_category": round_in_category,
                    "group_index": group_index,
                    "condition_mode": _ONLINE_SKILLBANK_MODE,
                }
                if not record.get("schema_valid"):
                    singleton_kwargs["synthetic_episodes"] = [
                        _invalid_skill_episode(
                            game=dict(game),
                            skill_name=singleton_name,
                            error="online_skillbank schema invalid",
                        )
                        for game in games
                    ]
                singleton_job = f"singleton/{sample_key}"
                singleton_jobs.append(
                    (
                        singleton_job,
                        partial(
                            self._run_outcome_condition_rollouts,
                            **singleton_kwargs,
                        ),
                    )
                )

            no_skill_job = partial(
                self._run_outcome_condition_rollouts,
                condition_dir=no_skill_dir,
                signature=no_skill_signature,
                condition_name="no_skill",
                skill_name=no_skill_name,
                skill_texts=[_NO_SKILL_BASELINE_TEXT] * len(games),
                games=games,
                task_indices=full_task_indices,
                round_in_category=round_in_category,
                group_index=group_index,
                condition_mode=_ONLINE_SKILLBANK_MODE,
            )
            if bool(getattr(self, "outcome_flattened_scheduler", False)):
                executor = _get_outcome_condition_executor(
                    self.outcome_condition_workers
                )
                baseline_future = executor.submit(no_skill_job)
                condition_futures.append(("no_skill", baseline_future))
                for name, job in singleton_jobs:
                    condition_futures.append((name, executor.submit(job)))
                # Retrieval still consumes the original completed baseline, but
                # independent singleton episodes can already use the executor.
                no_skill_condition = baseline_future.result()
            else:
                no_skill_condition = no_skill_job()
            baseline_episodes = [
                dict(episode)
                for episode in no_skill_condition.get("episodes", [])
                if isinstance(episode, dict)
            ]
            if len(baseline_episodes) != len(games):
                raise RuntimeError("online skillbank no-skill baseline is incomplete")
            baseline_index = _baseline_index_from_episodes(
                baseline_episodes,
                games,
                baseline_dir=no_skill_dir,
                signature=no_skill_signature,
                skill_name=no_skill_name,
            )
            _atomic_write_json(no_skill_dir / "baseline_index.json", baseline_index)

            retrieval = self._online_skillbank_retrieval_manifest(
                bank_snapshot=bank_snapshot,
                games=games,
                baseline_episodes=baseline_episodes,
            )
            _atomic_write_json(group_dir / "retrieval.json", retrieval)
            retrieved_skill_texts = [
                str(text) for text in retrieval["retrieved_skill_texts"]
            ]
            bank_skills_by_id = {
                str(skill["skill_id"]): dict(skill)
                for skill in bank_snapshot.get("skills", [])
            }
            retrieved_entries_by_task = [
                [
                    bank_skills_by_id[str(item["skill_id"])]
                    for item in task["retrieved_top_skills"]
                ]
                for task in retrieval["tasks"]
            ]
            bank_is_empty = not bank_snapshot.get("skills")
            actual_environment_episode_budget = int(
                no_skill_condition.get("attempted_rollout_count", 0) or 0
            )
            condition_jobs: list[tuple[str, Callable[[], dict[str, Any]]]] = []
            if not bank_is_empty:
                retrieved_dir = group_dir / "conditions" / "retrieved"
                retrieved_signature = hashlib.sha256(
                    f"{signature}:retrieved:{retrieval['signature']}".encode()
                ).hexdigest()
                condition_jobs.append(
                    (
                        "retrieved",
                        partial(
                            self._run_outcome_condition_rollouts,
                            condition_dir=retrieved_dir,
                            signature=retrieved_signature,
                            condition_name="retrieved_top3",
                            skill_name=(
                                "online_skillbank/retrieved/"
                                f"globalstep_{training_global_step:06d}/"
                                f"group_{training_step_group_index:02d}"
                            ),
                            skill_texts=retrieved_skill_texts,
                            games=games,
                            task_indices=full_task_indices,
                            round_in_category=round_in_category,
                            group_index=group_index,
                            condition_mode=_ONLINE_SKILLBANK_MODE,
                        ),
                    )
                )

            candidate_specs: dict[str, dict[str, Any]] = {}
            for record in records:
                sample_key = str(record["sample_key"])
                candidate_id = f"group_{training_step_group_index:02d}/{sample_key}"
                singleton_dir = group_dir / "conditions" / "singletons" / sample_key
                singleton_job = f"singleton/{sample_key}"
                if not bool(getattr(self, "outcome_flattened_scheduler", False)):
                    condition_jobs.append(
                        (singleton_job, dict(singleton_jobs)[singleton_job])
                    )

                augmented_job = ""
                augmented_dir = singleton_dir
                if not bank_is_empty:
                    augmented_dir = group_dir / "conditions" / "augmented" / sample_key
                    augmented_signature = hashlib.sha256(
                        f"{signature}:augmented:{sample_key}:"
                        f"{retrieval['signature']}".encode()
                    ).hexdigest()
                    augmented_name = (
                        "online_skillbank/augmented/"
                        f"globalstep_{training_global_step:06d}/"
                        f"group_{training_step_group_index:02d}/{sample_key}"
                    )
                    augmented_kwargs: dict[str, Any] = {
                        "condition_dir": augmented_dir,
                        "signature": augmented_signature,
                        "condition_name": f"augmented/{sample_key}",
                        "skill_name": augmented_name,
                        "skill_texts": [],
                        "games": games,
                        "task_indices": full_task_indices,
                        "round_in_category": round_in_category,
                        "group_index": group_index,
                        "condition_mode": _ONLINE_SKILLBANK_MODE,
                    }
                    augmented_kwargs["skill_texts"] = [
                        _online_skillbank_prompt_text(
                            retrieved_entries_by_task[index],
                            candidate_text=str(record["skill_text"]),
                        )
                        for index in range(len(games))
                    ]
                    if not record.get("schema_valid"):
                        augmented_kwargs["synthetic_episodes"] = [
                            _invalid_skill_episode(
                                game=dict(game),
                                skill_name=augmented_name,
                                error="online_skillbank schema invalid",
                            )
                            for game in games
                        ]
                    augmented_job = f"augmented/{sample_key}"
                    condition_jobs.append(
                        (
                            augmented_job,
                            partial(
                                self._run_outcome_condition_rollouts,
                                **augmented_kwargs,
                            ),
                        )
                    )
                candidate_specs[candidate_id] = {
                    **record,
                    "candidate_id": candidate_id,
                    "singleton_job": singleton_job,
                    "singleton_dir": str(singleton_dir),
                    "augmented_job": augmented_job,
                    "augmented_dir": str(augmented_dir),
                }

            if condition_futures:
                for name, job in condition_jobs:
                    condition_futures.append((name, executor.submit(job)))
                condition_results = _drain_outcome_condition_futures(condition_futures)
                condition_futures.clear()
            else:
                condition_results = self._run_outcome_condition_jobs(condition_jobs)
            if bank_is_empty:
                retrieved_condition = no_skill_condition
                retrieved_dir = no_skill_dir
            else:
                retrieved_condition = condition_results["retrieved"]
                actual_environment_episode_budget += int(
                    retrieved_condition.get("attempted_rollout_count", 0) or 0
                )

            no_skill_outcomes = _episodes_as_outcomes(
                baseline_episodes, games, full_task_indices
            )
            retrieved_outcomes = _episodes_as_outcomes(
                [dict(episode) for episode in retrieved_condition["episodes"]],
                games,
                full_task_indices,
            )
            singleton_outcomes: dict[str, dict[str, bool]] = {}
            augmented_outcomes: dict[str, dict[str, bool]] = {}
            for candidate_id, spec in candidate_specs.items():
                singleton = condition_results[str(spec["singleton_job"])]
                singleton_episodes = [
                    dict(episode) for episode in singleton["episodes"]
                ]
                if spec.get("schema_valid"):
                    actual_environment_episode_budget += int(
                        singleton.get("attempted_rollout_count", 0) or 0
                    )
                singleton_outcomes[candidate_id] = _episodes_as_outcomes(
                    singleton_episodes, games, full_task_indices
                )
                if bank_is_empty:
                    augmented = singleton
                    augmented_episodes = singleton_episodes
                else:
                    augmented = condition_results[str(spec["augmented_job"])]
                    augmented_episodes = [
                        dict(episode) for episode in augmented["episodes"]
                    ]
                    if spec.get("schema_valid"):
                        actual_environment_episode_budget += int(
                            augmented.get("attempted_rollout_count", 0) or 0
                        )
                augmented_outcomes[candidate_id] = _episodes_as_outcomes(
                    augmented_episodes, games, full_task_indices
                )

            raw_metrics = compute_online_skillbank_marginal_metrics(
                no_skill_outcomes=no_skill_outcomes,
                retrieved_skill_outcomes=retrieved_outcomes,
                singleton_outcomes=singleton_outcomes,
                augmented_outcomes=augmented_outcomes,
                bank_weight=self.online_skillbank_bank_weight,
                standalone_weight=self.online_skillbank_standalone_weight,
            )
            input_task_type = str(games[0].get("task_type") or "") if games else ""
            sample_metrics: dict[str, dict[str, Any]] = {}
            for candidate_id, spec in candidate_specs.items():
                metrics = dict(raw_metrics[candidate_id])
                if not spec.get("schema_valid"):
                    metrics.update(
                        {
                            "bank_delta": 0.0,
                            "standalone_delta": 0.0,
                            "weighted_marginal_score": 0.0,
                        }
                    )
                score = float(metrics["weighted_marginal_score"])
                sample_metrics[candidate_id] = {
                    **metrics,
                    "status": (
                        "complete" if spec.get("schema_valid") else "schema_invalid"
                    ),
                    "candidate_id": candidate_id,
                    "sample_key": str(spec["sample_key"]),
                    "training_step_group_index": training_step_group_index,
                    "input_task_type": input_task_type,
                    "schema_valid": bool(spec.get("schema_valid")),
                    "marginal_eligible": float(bool(spec.get("schema_valid"))),
                    "skill_text": str(spec.get("skill_text") or ""),
                    "generation_path": str(spec.get("generation_path") or ""),
                    "positive_marginal_indicator": float(
                        bool(spec.get("schema_valid"))
                        and score > self.online_skillbank_zero_epsilon
                    ),
                    "singleton_condition_dir": str(spec["singleton_dir"]),
                    "augmented_condition_dir": str(spec["augmented_dir"]),
                }

            base_episode_budget = self.samples_per_round * self.rollouts_per_skill
            planned_episode_budget = (
                base_episode_budget + self.rollouts_per_skill
                if bank_is_empty
                else 2 * self.rollouts_per_skill + 2 * base_episode_budget
            )
            group_result_signature = hashlib.sha256(
                json.dumps(
                    {
                        "signature": signature,
                        "retrieval_signature": retrieval["signature"],
                        "sample_metrics": sample_metrics,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            payload = {
                "status": "complete",
                "schema_version": 1,
                "mode": _ONLINE_SKILLBANK_MODE,
                "signature": signature,
                "group_result_signature": group_result_signature,
                "training_global_step": training_global_step,
                "training_step_group_index": training_step_group_index,
                "round_in_category": round_in_category,
                "group_index": group_index,
                "task_type": task_type,
                "input_task_type": input_task_type,
                "bank_before_signature": bank_snapshot["bank_signature"],
                "bank_before_size": int(bank_snapshot["skill_count"]),
                "bank_snapshot_path": str(
                    self._online_skillbank_step_dir(training_global_step)
                    / "bank_before.json"
                ),
                "retrieval": {
                    "signature": retrieval["signature"],
                    "path": str(group_dir / "retrieval.json"),
                    "query_version": _ONLINE_SKILLBANK_QUERY_VERSION,
                    "top_k": self.online_skillbank_top_k,
                    "retrieved_count_by_task": [
                        len(task["retrieved_top_skills"]) for task in retrieval["tasks"]
                    ],
                },
                "baseline": baseline_index,
                "retrieved_condition_dir": str(retrieved_dir),
                "sample_metrics": sample_metrics,
                "base_episode_budget": base_episode_budget,
                "planned_episode_budget": planned_episode_budget,
                "planned_compute_multiplier": (
                    planned_episode_budget / base_episode_budget
                ),
                "actual_environment_episode_budget": (
                    actual_environment_episode_budget
                ),
                "actual_environment_compute_multiplier": (
                    actual_environment_episode_budget / base_episode_budget
                ),
                "max_compute_multiplier": (
                    self.online_skillbank_max_compute_multiplier
                ),
                "paired_task_alignment": "same_eight_matched_games_common_rng",
                "updated_at": time.time(),
            }
            _atomic_write_json(index_path, payload)
            _atomic_write_json(
                group_dir / "checkpoint.json",
                {
                    "status": "complete",
                    "signature": signature,
                    "group_result_signature": group_result_signature,
                    "training_global_step": training_global_step,
                    "training_step_group_index": training_step_group_index,
                    "planned_episode_budget": planned_episode_budget,
                    "planned_compute_multiplier": payload["planned_compute_multiplier"],
                    "updated_at": time.time(),
                },
            )
            self._maybe_finalize_online_skillbank_step(training_global_step)
            return payload
        finally:
            for _, future in condition_futures:
                future.cancel()
            for _, future in condition_futures:
                try:
                    future.result()
                except BaseException:  # noqa: BLE001
                    pass  # Preserve the original failure after draining writers.
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5.0)
            _release_json_lock(fd, lock_path)

    def _maybe_finalize_online_skillbank_step(
        self, training_global_step: int
    ) -> dict[str, Any] | None:
        """Append positive per-type winners once all batch groups are complete."""

        step = int(training_global_step)
        step_dir = self._online_skillbank_step_dir(step)
        groups_dir = step_dir / "groups"
        expected_slots = set(range(self.train_batch_size))

        def complete_groups() -> dict[int, dict[str, Any]]:
            groups: dict[int, dict[str, Any]] = {}
            for index_path in sorted(groups_dir.glob("group_*/index.json")):
                try:
                    payload = _read_json(index_path)
                    slot = int(payload.get("training_step_group_index", -1))
                except Exception:
                    continue
                if (
                    payload.get("status") == "complete"
                    and int(payload.get("training_global_step", -1)) == step
                    and slot in expected_slots
                ):
                    groups[slot] = payload
            return groups

        groups = complete_groups()
        if set(groups) != expected_slots:
            return None
        bank_before_path = step_dir / "bank_before.json"
        bank_before = self._read_online_skillbank_payload(bank_before_path)
        if bank_before is None:
            return None
        self._validate_online_skillbank_snapshot_plan(bank_before, step)
        before_signature = str(bank_before["bank_signature"])
        mismatched = [
            slot
            for slot, group in groups.items()
            if str(group.get("bank_before_signature") or "") != before_signature
        ]
        if mismatched:
            raise RuntimeError(
                "online skillbank groups used inconsistent bank snapshots: "
                f"step={step}, slots={mismatched}"
            )
        plan = self._online_skillbank_step_plan(step)
        layout_mismatched = [
            slot
            for slot, group in groups.items()
            if str(group.get("input_task_type") or "")
            != plan.group_input_task_types[slot]
        ]
        if layout_mismatched:
            raise RuntimeError(
                "online skillbank groups do not match the balanced input layout: "
                f"step={step}, slots={layout_mismatched}"
            )
        group_signatures = [
            str(groups[slot].get("group_result_signature") or "")
            for slot in sorted(groups)
        ]
        finalize_signature = hashlib.sha256(
            json.dumps(
                {
                    "mode": _ONLINE_SKILLBANK_MODE,
                    "training_global_step": step,
                    "bank_before_signature": before_signature,
                    "group_result_signatures": group_signatures,
                    "update_task_types": list(plan.update_task_types),
                    "zero_epsilon": self.online_skillbank_zero_epsilon,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        bank_after_path = step_dir / "bank_after.json"
        current_after = self._read_online_skillbank_payload(bank_after_path)
        if current_after is not None:
            if current_after.get("update_signature") != finalize_signature:
                raise RuntimeError(
                    "online skillbank finalized state does not match current group "
                    f"artifacts: {bank_after_path}"
                )
            return current_after

        lock_path = step_dir / "finalize.lock"
        fd = _acquire_json_lock(
            lock_path,
            timeout_s=1800.0,
            foreign_host_stale_after_s=120.0,
        )
        try:
            current_after = self._read_online_skillbank_payload(bank_after_path)
            if current_after is not None:
                if current_after.get("update_signature") != finalize_signature:
                    raise RuntimeError(
                        "online skillbank update signature mismatch after lock"
                    )
                return current_after
            groups = complete_groups()
            if set(groups) != expected_slots:
                return None

            candidates: dict[str, dict[str, Any]] = {}
            candidate_metrics: dict[str, dict[str, float]] = {}
            candidate_task_types: dict[str, str] = {}
            schema_valid: dict[str, bool] = {}
            existing_texts = {
                " ".join(str(skill.get("text") or "").split())
                for skill in bank_before.get("skills", [])
            }
            for slot in sorted(groups):
                group = groups[slot]
                input_task_type = str(group.get("input_task_type") or "")
                raw_sample_metrics = group.get("sample_metrics")
                if not isinstance(raw_sample_metrics, dict):
                    raise RuntimeError(
                        f"online skillbank group {slot} has invalid sample metrics"
                    )
                for candidate_id, raw_metrics in raw_sample_metrics.items():
                    if not isinstance(raw_metrics, dict):
                        raise RuntimeError(
                            f"online skillbank candidate is malformed: {candidate_id}"
                        )
                    candidate_id = str(candidate_id)
                    if candidate_id in candidates:
                        raise RuntimeError(
                            f"duplicate online skillbank candidate id: {candidate_id}"
                        )
                    candidate = dict(raw_metrics)
                    candidate["input_task_type"] = input_task_type
                    candidates[candidate_id] = candidate
                    candidate_metrics[candidate_id] = {
                        "weighted_marginal_score": float(
                            candidate.get("weighted_marginal_score", 0.0) or 0.0
                        )
                    }
                    candidate_task_types[candidate_id] = input_task_type
                    normalized_text = " ".join(
                        str(candidate.get("skill_text") or "").split()
                    )
                    schema_valid[candidate_id] = (
                        bool(candidate.get("schema_valid"))
                        and normalized_text not in existing_texts
                    )

            selected = select_positive_skillbank_updates(
                metrics=candidate_metrics,
                candidate_task_types=candidate_task_types,
                schema_valid=schema_valid,
                update_task_types=plan.update_task_types,
                zero_epsilon=self.online_skillbank_zero_epsilon,
            )
            skills = [dict(skill) for skill in bank_before.get("skills", [])]
            additions: list[dict[str, Any]] = []
            for task_type in plan.update_task_types:
                candidate_id = selected.get(task_type)
                if candidate_id is None:
                    continue
                candidate = candidates[candidate_id]
                entry = {
                    "skill_id": f"globalstep_{step:06d}/{candidate_id}",
                    "source_index": len(skills),
                    "text": str(candidate.get("skill_text") or ""),
                    "input_task_type": task_type,
                    "added_training_global_step": step,
                    "candidate_id": candidate_id,
                    "sample_key": str(candidate.get("sample_key") or ""),
                    "training_step_group_index": int(
                        candidate.get("training_step_group_index", -1)
                    ),
                    "generation_path": str(candidate.get("generation_path") or ""),
                    "weighted_marginal_score": float(
                        candidate.get("weighted_marginal_score", 0.0) or 0.0
                    ),
                    "bank_delta": float(candidate.get("bank_delta", 0.0) or 0.0),
                    "standalone_delta": float(
                        candidate.get("standalone_delta", 0.0) or 0.0
                    ),
                    "bank_before_signature": before_signature,
                }
                skills.append(entry)
                additions.append(entry)

            after_signature = _online_skillbank_skills_signature(skills)
            update_payload = {
                "status": "complete",
                "schema_version": 1,
                "mode": _ONLINE_SKILLBANK_MODE,
                "training_global_step": step,
                "update_signature": finalize_signature,
                "bank_before_signature": before_signature,
                "bank_after_signature": after_signature,
                "bank_size_before": int(bank_before["skill_count"]),
                "bank_size_after": len(skills),
                "selected_task_types": list(plan.update_task_types),
                "selected_candidates": selected,
                "addition_count": len(additions),
                "additions": additions,
                "group_result_signatures": group_signatures,
                "positive_only": True,
                "zero_epsilon": self.online_skillbank_zero_epsilon,
                "updated_at": time.time(),
            }
            _atomic_write_json(step_dir / "update.json", update_payload)
            bank_after = {
                "status": "complete",
                "schema_version": 1,
                "mode": _ONLINE_SKILLBANK_MODE,
                "training_global_step": step,
                "source_training_global_step": step,
                "source_bank_signature": before_signature,
                "update_signature": finalize_signature,
                "bank_signature": after_signature,
                "skill_count": len(skills),
                "skills": skills,
                "update_path": str(step_dir / "update.json"),
                "updated_at": time.time(),
            }
            # This complete state is the marker consumed by the next step.
            _atomic_write_json(bank_after_path, bank_after)
            return bank_after
        finally:
            _release_json_lock(fd, lock_path)
