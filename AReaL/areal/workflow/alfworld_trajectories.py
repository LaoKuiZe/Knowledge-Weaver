# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import itertools
import json
import random
import shutil
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from areal.workflow.alfworld_environment import (
    _invalid_skill_episode,
    _run_no_skill_result_episode_process,
)
from areal.workflow.alfworld_runtime import (
    _NO_SKILL_BASELINE_TEXT,
    _ONLINE_SKILLBANK_MODE,
    _ONLINE_SKILLBANK_QUERY_VERSION,
    _PROMPT_CONTRACT_VERSION,
    _ROUND_WAIT_LOG_INTERVAL_S,
    _TRAJECTORY_RENDER_VERSION,
    MIXED_TASK_TYPE,
    PROMPT_CATEGORY_BY_TASK_TYPE,
    _acquire_json_lock,
    _atomic_write_json,
    _get_episode_rollout_executor,
    _is_usable_prompt_episode,
    _read_json,
    _release_json_lock,
    _safe_path_component,
    _sample_trajectory_pool_episodes,
    _sanitize_trajectory_episode,
    _stable_seed,
    logger,
)
from areal.workflow.skill_prompts import (
    _fit_skill_prompt_messages,
    _skill_prompt_budget,
    _trajectory_pool_episode_ref,
)


class ALFWorldTrajectoryMixin:
    def _trajectory_pool_root(self) -> Path:
        return self.artifact_dir / "trajectory_pool"

    def _trajectory_pool_step_dir(self, training_global_step: int) -> Path:
        return (
            self._trajectory_pool_root()
            / "steps"
            / f"globalstep_{training_global_step:06d}"
        )

    def _select_initial_trajectory_pool_games(self) -> list[dict[str, str]]:
        task_types = [item for item in self.task_types if item != MIXED_TASK_TYPE]
        if not task_types:
            raise RuntimeError(
                "trajectory pool requires at least one concrete task type"
            )
        target = self.trajectory_pool_initial_size
        base_count, remainder = divmod(target, len(task_types))
        selected: list[dict[str, str]] = []
        selected_paths: set[str] = set()
        leftovers: list[dict[str, str]] = []
        for type_index, input_task_type in enumerate(task_types):
            games = [dict(game) for game in self._games_for_task(input_task_type)]
            rng = random.Random(_stable_seed(13000000, f"initial:{input_task_type}"))
            rng.shuffle(games)
            requested = base_count + int(type_index < remainder)
            picked = games[:requested]
            selected.extend(picked)
            selected_paths.update(str(game["gamefile"]) for game in picked)
            leftovers.extend(games[requested:])
        if len(selected) < target:
            remaining = [
                game
                for game in leftovers
                if str(game["gamefile"]) not in selected_paths
            ]
            random.Random(13000000).shuffle(remaining)
            selected.extend(remaining[: target - len(selected)])
        if len(selected) < target:
            raise RuntimeError(
                "not enough ALFWorld games for initial trajectory pool: "
                f"requested={target}, available={len(selected)}"
            )
        return selected[:target]

    def _initial_trajectory_pool_signature(
        self, games: Sequence[dict[str, Any]]
    ) -> str:
        payload = {
            "schema_version": 1,
            "actor_model": self.actor_model,
            "train_split": self.train_split,
            "task_types": list(self.task_types),
            "initial_size": self.trajectory_pool_initial_size,
            "max_rollout_steps": self.max_rollout_steps,
            "memory_window": self.memory_window,
            "max_commands": self.max_commands,
            "actor_temperature": self.actor_temperature,
            "skill_text": _NO_SKILL_BASELINE_TEXT,
            "games": [str(game.get("gamefile") or "") for game in games],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        return hashlib.sha1(encoded).hexdigest()[:16]

    def _read_complete_initial_trajectory_pool(
        self, *, signature: str
    ) -> dict[str, Any] | None:
        index_path = self._trajectory_pool_root() / "initial" / "index.json"
        if not index_path.exists():
            return None
        try:
            payload = _read_json(index_path)
        except Exception:
            return None
        if payload.get("status") != "complete":
            return None
        if payload.get("signature") != signature:
            return None
        return payload

    def _ensure_initial_trajectory_pool(self) -> dict[str, Any]:
        games = self._select_initial_trajectory_pool_games()
        signature = self._initial_trajectory_pool_signature(games)
        cached = self._read_complete_initial_trajectory_pool(signature=signature)
        if cached is not None:
            return cached

        initial_dir = self._trajectory_pool_root() / "initial"
        lock_path = initial_dir / "build.lock"
        fd = _acquire_json_lock(lock_path, timeout_s=21600.0)
        try:
            cached = self._read_complete_initial_trajectory_pool(signature=signature)
            if cached is not None:
                return cached
            episodes_dir = initial_dir / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            run_config = {
                "status": "building",
                "mode": "fixed_initial_no_skill_trajectory_pool",
                "signature": signature,
                "initial_size": len(games),
                "task_types": list(self.task_types),
                "skill_text": _NO_SKILL_BASELINE_TEXT,
                "max_rollout_steps": self.max_rollout_steps,
                "memory_window": self.memory_window,
                "max_commands": self.max_commands,
                "actor_temperature": self.actor_temperature,
                "selected_games": games,
                "updated_at": time.time(),
            }
            _atomic_write_json(initial_dir / "run_config.json", run_config)

            episode_paths: list[Path] = []
            missing: list[tuple[int, dict[str, Any], Path]] = []
            for pool_index, game in enumerate(games):
                task_component = _safe_path_component(game.get("task_type"))
                episode_path = (
                    episodes_dir / task_component / f"episode_{pool_index:04d}.json"
                )
                episode_paths.append(episode_path)
                valid_existing = False
                if episode_path.exists():
                    try:
                        existing = _read_json(episode_path)
                        valid_existing = (
                            existing.get("signature") == signature
                            and isinstance(existing.get("episodes"), list)
                            and len(existing["episodes"]) == 1
                        )
                    except Exception:
                        valid_existing = False
                if not valid_existing:
                    missing.append((pool_index, game, episode_path))

            if missing:
                max_workers = min(
                    len(missing),
                    max(1, self.trajectory_pool_initial_workers),
                )
                executor = _get_episode_rollout_executor(
                    max(max_workers, self.episode_rollout_workers)
                )
                futures = {
                    executor.submit(
                        _run_no_skill_result_episode_process,
                        {
                            "repo_root": str(self.repo_root),
                            "game": game,
                            "skill_name": "trajectory_pool/initial/no_skill",
                            "skill_text": _NO_SKILL_BASELINE_TEXT,
                            "rollout_index": pool_index,
                            "max_rollout_steps": self.max_rollout_steps,
                            "memory_window": self.memory_window,
                            "max_commands": self.max_commands,
                            "actor_base_url": self.actor_base_url,
                            "actor_model": self.actor_model,
                            "actor_api_key": self.actor_api_key,
                            "actor_timeout_s": self.actor_timeout_s,
                            "actor_temperature": self.actor_temperature,
                            "tokenizer_path": self.tokenizer_path,
                            "seed_base": 13000000 + pool_index * 100,
                            "return_episode": True,
                        },
                    ): (pool_index, game, episode_path)
                    for pool_index, game, episode_path in missing
                }
                completed = len(games) - len(missing)
                for future in futures:
                    pool_index, game, episode_path = futures[future]
                    try:
                        episode = future.result()
                    except Exception as exc:  # noqa: BLE001
                        episode = _invalid_skill_episode(
                            game=game,
                            skill_name="trajectory_pool/initial/no_skill",
                            error="initial_pool_exception: " + repr(exc),
                        )
                        episode["status"] = "rollout_error"
                    clean = _sanitize_trajectory_episode(episode)
                    _atomic_write_json(
                        episode_path,
                        {
                            "mode": "fixed_initial_no_skill_trajectory",
                            "signature": signature,
                            "pool_index": pool_index,
                            "episodes": [clean],
                            "updated_at": time.time(),
                        },
                    )
                    completed += 1
                    if completed % 10 == 0 or completed == len(games):
                        _atomic_write_json(
                            initial_dir / "checkpoint.json",
                            {
                                **run_config,
                                "status": "building",
                                "completed": completed,
                                "expected": len(games),
                                "updated_at": time.time(),
                            },
                        )

            refs: list[dict[str, Any]] = []
            status_counts: Counter[str] = Counter()
            per_task_type: Counter[str] = Counter()
            wins = 0
            for episode_path in episode_paths:
                payload = _read_json(episode_path)
                episode = payload["episodes"][0]
                ref = _trajectory_pool_episode_ref(
                    episode,
                    source_path=episode_path,
                    rollout_index=0,
                    pool_source="initial_no_skill",
                )
                refs.append(ref)
                status_counts[str(episode.get("status") or "missing")] += 1
                per_task_type[str(episode.get("task_type") or "unknown")] += 1
                wins += int(bool(episode.get("won")))
            usable_count = sum(_is_usable_prompt_episode(ref) for ref in refs)
            for input_task_type in self.task_types:
                usable_for_type = sum(
                    _is_usable_prompt_episode(ref)
                    and ref.get("task_type") == input_task_type
                    for ref in refs
                )
                if usable_for_type < self.trajectory_pool_prompt_episodes:
                    raise RuntimeError(
                        "initial trajectory pool has too few usable trajectories for "
                        f"{input_task_type}: {usable_for_type}"
                    )
            index = {
                **run_config,
                "status": "complete",
                "episode_count": len(refs),
                "usable_episode_count": usable_count,
                "wins": wins,
                "sr": wins / max(1, len(refs)),
                "per_task_type_count": dict(per_task_type),
                "status_counts": dict(status_counts),
                "refs": refs,
                "updated_at": time.time(),
            }
            _atomic_write_json(initial_dir / "index.json", index)
            _atomic_write_json(
                initial_dir / "checkpoint.json",
                {
                    **run_config,
                    "status": "complete",
                    "episode_count": len(refs),
                    "usable_episode_count": usable_count,
                    "updated_at": time.time(),
                },
            )
            return index
        finally:
            _release_json_lock(fd, lock_path)

    def _finalize_training_step_trajectory_pool(
        self, training_global_step: int
    ) -> dict[str, Any] | None:
        step_dir = self._trajectory_pool_step_dir(training_global_step)
        expected_samples = self.train_batch_size * self.samples_per_round
        sample_paths = sorted((step_dir / "samples").glob("*.json"))
        if len(sample_paths) < expected_samples:
            return None
        index_path = step_dir / "index.json"
        if index_path.exists():
            try:
                existing = _read_json(index_path)
                if existing.get("status") == "complete":
                    return existing
            except Exception:
                pass
        lock_path = step_dir / "finalize.lock"
        fd = _acquire_json_lock(lock_path, timeout_s=1800.0)
        try:
            if index_path.exists():
                try:
                    existing = _read_json(index_path)
                    if existing.get("status") == "complete":
                        return existing
                except Exception:
                    pass
            sample_paths = sorted((step_dir / "samples").glob("*.json"))
            if len(sample_paths) < expected_samples:
                return None
            refs: list[dict[str, Any]] = []
            status_counts: Counter[str] = Counter()
            for sample_path in sample_paths:
                payload = _read_json(sample_path)
                if int(payload.get("training_global_step", -1)) != training_global_step:
                    continue
                episodes = payload.get("episodes") or []
                for rollout_index, episode in enumerate(episodes):
                    if not isinstance(episode, dict):
                        continue
                    refs.append(
                        _trajectory_pool_episode_ref(
                            episode,
                            source_path=sample_path,
                            rollout_index=rollout_index,
                            pool_source="previous_step_skill_rollout",
                            source_training_global_step=training_global_step,
                        )
                    )
                    status_counts[str(episode.get("status") or "missing")] += 1
            index = {
                "status": "complete",
                "mode": "sanitized_skill_rollout_trajectory_pool",
                "training_global_step": training_global_step,
                "expected_sample_count": expected_samples,
                "completed_sample_count": len(sample_paths),
                "episode_count": len(refs),
                "usable_episode_count": sum(
                    _is_usable_prompt_episode(ref) for ref in refs
                ),
                "status_counts": dict(status_counts),
                "refs": refs,
                "updated_at": time.time(),
            }
            _atomic_write_json(index_path, index)
            return index
        finally:
            _release_json_lock(fd, lock_path)

    def _restore_consumed_training_step_trajectory_pool(
        self, training_global_step: int
    ) -> dict[str, Any] | None:
        """Rebuild a consumed predecessor pool from preserved canonical rollouts when replaying an interrupted step."""
        consumer_step = training_global_step + 1
        marker_path = (
            self._trajectory_pool_root()
            / "consumption"
            / f"globalstep_{consumer_step:06d}.json"
        )
        if not marker_path.exists():
            return None
        try:
            marker = _read_json(marker_path)
        except Exception:
            return None
        if (
            marker.get("status") != "consumed"
            or int(marker.get("deleted_training_global_step", -1))
            != training_global_step
            or marker.get("canonical_rollouts_preserved") is not True
        ):
            return None

        step_dir = self._trajectory_pool_step_dir(training_global_step)
        restore_lock_path = step_dir / "restore.lock"
        fd = _acquire_json_lock(restore_lock_path, timeout_s=1800.0)
        try:
            existing = self._finalize_training_step_trajectory_pool(
                training_global_step
            )
            if existing is not None:
                return existing

            sources: list[tuple[int, int, str, Path, list[dict[str, Any]]]] = []
            start_group = training_global_step * self.train_batch_size
            for global_group_index in range(
                start_group, start_group + self.train_batch_size
            ):
                round_index, group_index = divmod(
                    global_group_index, self.groups_per_round
                )
                round_dir = self._round_dir(MIXED_TASK_TYPE, round_index)
                for sample_index in range(self.samples_per_round):
                    sample_key = f"group_{group_index:02d}_sample_{sample_index:02d}"
                    sample_dir = round_dir / "skills" / sample_key
                    prompt_path = sample_dir / "prompt.json"
                    rollout_path = sample_dir / "rollouts.json"
                    if not prompt_path.exists() or not rollout_path.exists():
                        return None
                    try:
                        prompt = _read_json(prompt_path)
                        rollout = _read_json(rollout_path)
                    except Exception:
                        return None
                    if (
                        int(prompt.get("training_global_step", -1))
                        != training_global_step
                    ):
                        return None
                    episodes = rollout.get("episodes")
                    if not isinstance(episodes, list):
                        return None
                    sources.append(
                        (
                            round_index,
                            group_index,
                            sample_key,
                            rollout_path,
                            [dict(item) for item in episodes if isinstance(item, dict)],
                        )
                    )

            expected_samples = self.train_batch_size * self.samples_per_round
            if len(sources) != expected_samples:
                return None
            for (
                round_index,
                group_index,
                sample_key,
                rollout_path,
                episodes,
            ) in sources:
                sample_path = (
                    step_dir / "samples" / f"round_{round_index:04d}__{sample_key}.json"
                )
                _atomic_write_json(
                    sample_path,
                    {
                        "status": "complete",
                        "mode": "sanitized_skill_rollout_trajectories_recovered",
                        "training_global_step": training_global_step,
                        "round_in_category": round_index,
                        "group_index": group_index,
                        "sample_key": sample_key,
                        "skill_fields_removed": True,
                        "recovered_from_canonical_rollouts": str(rollout_path),
                        "episodes": [
                            _sanitize_trajectory_episode(episode)
                            for episode in episodes
                        ],
                        "updated_at": time.time(),
                    },
                )
            logger.warning(
                "restored consumed trajectory pool globalstep=%s from %s "
                "canonical skill rollouts",
                training_global_step,
                len(sources),
            )
            return self._finalize_training_step_trajectory_pool(training_global_step)
        finally:
            _release_json_lock(fd, restore_lock_path)

    def _record_training_step_trajectories(
        self,
        *,
        prepared: dict[str, Any],
        episodes: Sequence[dict[str, Any]],
    ) -> None:
        if not bool(getattr(self, "trajectory_pool_enabled", False)):
            return
        training_global_step = int(prepared["training_global_step"])
        step_dir = self._trajectory_pool_step_dir(training_global_step)
        round_index = int(prepared["round_in_category"])
        sample_key = _safe_path_component(prepared["sample_key"])
        sample_path = (
            step_dir / "samples" / f"round_{round_index:04d}__{sample_key}.json"
        )
        clean_episodes = [
            _sanitize_trajectory_episode(dict(episode)) for episode in episodes
        ]
        _atomic_write_json(
            sample_path,
            {
                "status": "complete",
                "mode": "sanitized_skill_rollout_trajectories",
                "training_global_step": training_global_step,
                "round_in_category": round_index,
                "group_index": int(prepared.get("group_index", 0)),
                "sample_key": prepared["sample_key"],
                "skill_fields_removed": True,
                "episodes": clean_episodes,
                "updated_at": time.time(),
            },
        )
        self._finalize_training_step_trajectory_pool(training_global_step)

    def _wait_for_training_step_trajectory_pool(
        self, training_global_step: int
    ) -> dict[str, Any]:
        step_dir = self._trajectory_pool_step_dir(training_global_step)
        index_path = step_dir / "index.json"
        deadline = self._round_barrier_deadline()
        last_log = 0.0
        while True:
            if index_path.exists():
                try:
                    payload = _read_json(index_path)
                    if payload.get("status") == "complete":
                        return payload
                except Exception:
                    pass
            payload = self._finalize_training_step_trajectory_pool(training_global_step)
            if payload is not None:
                return payload
            payload = self._restore_consumed_training_step_trajectory_pool(
                training_global_step
            )
            if payload is not None:
                return payload
            completed = len(list((step_dir / "samples").glob("*.json")))
            expected = self.train_batch_size * self.samples_per_round
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for ALFWorld trajectory pool: "
                    f"globalstep={training_global_step}, "
                    f"complete={completed}/{expected}, step_dir={step_dir}"
                )
            now = time.time()
            if now - last_log >= _ROUND_WAIT_LOG_INTERVAL_S:
                logger.info(
                    "waiting for trajectory pool globalstep=%s: %s/%s samples complete",
                    training_global_step,
                    completed,
                    expected,
                )
                last_log = now
            time.sleep(5.0)

    def _trajectory_pool_refs_for_task(
        self, *, input_task_type: str, training_global_step: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        initial = self._ensure_initial_trajectory_pool()
        initial_refs = [
            dict(ref)
            for ref in initial.get("refs", [])
            if isinstance(ref, dict) and ref.get("task_type") == input_task_type
        ]
        previous_refs: list[dict[str, Any]] = []
        previous_step = training_global_step - 1
        if self.trajectory_pool_use_previous_step and previous_step >= 0:
            previous = self._wait_for_training_step_trajectory_pool(previous_step)
            previous_refs = [
                dict(ref)
                for ref in previous.get("refs", [])
                if isinstance(ref, dict) and ref.get("task_type") == input_task_type
            ]
        refs = initial_refs + previous_refs
        return refs, {
            "input_task_type": input_task_type,
            "training_global_step": training_global_step,
            "initial_candidate_count": len(initial_refs),
            "previous_step": previous_step if previous_refs else None,
            "previous_step_candidate_count": len(previous_refs),
            "candidate_count": len(refs),
        }

    def _select_trajectory_pool_prompt(
        self,
        *,
        input_task_type: str,
        training_global_step: int,
        seed: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        refs, pool_metadata = self._trajectory_pool_refs_for_task(
            input_task_type=input_task_type,
            training_global_step=training_global_step,
        )
        episodes, sampling = _sample_trajectory_pool_episodes(
            refs,
            seed=seed,
            count=self.trajectory_pool_prompt_episodes,
        )
        return episodes, {**pool_metadata, **sampling}

    def _prepared_prompt_payload(
        self, payload: dict[str, Any], sample_dir: Path
    ) -> dict[str, Any]:
        return {
            "task_type": str(payload["task_type"]),
            "input_task_type": str(
                payload.get("input_task_type") or payload["task_type"]
            ),
            "prompt_category": str(payload["prompt_category"]),
            "round_in_category": int(payload["round_in_category"]),
            "group_index": int(payload.get("group_index", 0)),
            "groups_per_round": int(
                payload.get("groups_per_round", self.groups_per_round)
            ),
            "training_global_step": int(payload["training_global_step"]),
            "training_step_group_index": int(payload["training_step_group_index"]),
            "sample_key": str(payload["sample_key"]),
            "sample_dir": str(sample_dir),
            "selected_games": list(payload.get("selected_games") or []),
            "messages": list(payload.get("messages") or []),
            "sampling": dict(payload.get("sampling") or {}),
            "skill_prompt_version": str(payload.get("skill_prompt_version") or ""),
            "trajectory_render_version": str(
                payload.get("trajectory_render_version") or ""
            ),
            "online_skillbank_snapshot_path": str(
                payload.get("online_skillbank_snapshot_path") or ""
            ),
            "online_skillbank_before_signature": str(
                payload.get("online_skillbank_before_signature") or ""
            ),
            "online_skillbank_before_size": int(
                payload.get("online_skillbank_before_size", 0) or 0
            ),
        }

    def _prepared_prompt_matches_current(self, payload: dict[str, Any]) -> bool:
        prompt_render = dict(payload.get("prompt_render") or {})
        cached_prompt_version = str(
            payload.get("skill_prompt_version")
            or prompt_render.get("skill_prompt_version")
            or ""
        )
        cached_render_version = str(
            payload.get("trajectory_render_version")
            or prompt_render.get("trajectory_render_version")
            or ""
        )
        online_layout_matches = True
        if str(payload.get("mode") or "train") != "eval":
            try:
                training_global_step = int(payload["training_global_step"])
                training_step_group_index = int(payload["training_step_group_index"])
                plan = self._online_skillbank_step_plan(training_global_step)
                expected_input_task_type = plan.group_input_task_types[
                    training_step_group_index
                ]
            except (KeyError, TypeError, ValueError, IndexError):
                online_layout_matches = False
            else:
                online_layout_matches = (
                    str(payload.get("input_task_type") or "")
                    == expected_input_task_type
                )
        return (
            cached_prompt_version == str(self.skill_prompt_version)
            and cached_render_version == _TRAJECTORY_RENDER_VERSION
            and online_layout_matches
        )

    def _current_prompt_contract(self) -> dict[str, Any]:
        return {
            "contract_version": _PROMPT_CONTRACT_VERSION,
            "skill_prompt_version": str(self.skill_prompt_version),
            "trajectory_render_version": _TRAJECTORY_RENDER_VERSION,
            "skill_output_format": str(self.skill_output_format),
            "skill_description_max_words": int(self.skill_description_max_words),
            "online_skillbank_mode": _ONLINE_SKILLBANK_MODE,
            "online_skillbank_query_version": _ONLINE_SKILLBANK_QUERY_VERSION,
            "online_skillbank_task_types": list(self.task_types),
            "online_skillbank_top_k": self.online_skillbank_top_k,
            "online_skillbank_embedding_model": self.online_skillbank_embedding_model,
            "online_skillbank_embedding_max_length": (
                self.online_skillbank_embedding_max_length
            ),
            "online_skillbank_seed": self.online_skillbank_seed,
            "online_skillbank_bank_weight": self.online_skillbank_bank_weight,
            "online_skillbank_standalone_weight": (
                self.online_skillbank_standalone_weight
            ),
            "online_skillbank_reward_weight": self.reward_online_skillbank_weight,
            "online_skillbank_warmup_steps": self.online_skillbank_warmup_steps,
            "online_skillbank_warmup_update_type_count": (
                self.online_skillbank_warmup_update_type_count
            ),
            "online_skillbank_later_update_type_count": (
                self.online_skillbank_later_update_type_count
            ),
            "online_skillbank_zero_epsilon": self.online_skillbank_zero_epsilon,
            "online_skillbank_train_batch_size": self.train_batch_size,
        }

    def _ensure_prompt_contract(self) -> None:
        """Prevent a resumed trial from mixing prompt or render protocols."""

        if bool(getattr(self, "_prompt_contract_verified", False)):
            return
        artifact_dir = Path(self.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        contract_path = artifact_dir / "prompt_contract.json"
        lock_path = artifact_dir / "prompt_contract.lock"
        current = self._current_prompt_contract()
        fd = _acquire_json_lock(lock_path, timeout_s=1800.0)
        try:
            if contract_path.exists():
                existing = _read_json(contract_path)
                mismatches = {
                    key: {"existing": existing.get(key), "current": value}
                    for key, value in current.items()
                    if existing.get(key) != value
                }
                if mismatches:
                    raise RuntimeError(
                        "prompt contract mismatch for existing ALFWorld trial; "
                        "use a new trial name instead of mixing prompt protocols: "
                        f"{mismatches}"
                    )
            else:
                existing_prompt_path = next(
                    itertools.chain(
                        artifact_dir.glob("categories/*/round_*/skills/*/prompt.json"),
                        artifact_dir.glob("eval/*/skills/*/prompt.json"),
                    ),
                    None,
                )
                if existing_prompt_path is not None:
                    existing_prompt = _read_json(existing_prompt_path)
                    prompt_render = dict(existing_prompt.get("prompt_render") or {})
                    observed = {
                        "skill_prompt_version": str(
                            existing_prompt.get("skill_prompt_version")
                            or prompt_render.get("skill_prompt_version")
                            or ""
                        ),
                        "trajectory_render_version": str(
                            existing_prompt.get("trajectory_render_version")
                            or prompt_render.get("trajectory_render_version")
                            or ""
                        ),
                    }
                    mismatches = {
                        key: {"existing": value, "current": current[key]}
                        for key, value in observed.items()
                        if value != current[key]
                    }
                    if mismatches:
                        raise RuntimeError(
                            "existing ALFWorld prompts predate the requested prompt "
                            "contract; use a new trial name instead of mixing prompt "
                            f"protocols: path={existing_prompt_path}, {mismatches}"
                        )
                _atomic_write_json(
                    contract_path,
                    {**current, "status": "active", "created_at": time.time()},
                )
            self._prompt_contract_verified = True
        finally:
            _release_json_lock(fd, lock_path)

    def _maybe_delete_consumed_previous_step_pool(
        self, training_global_step: int
    ) -> None:
        if not self.trajectory_pool_delete_consumed_step or training_global_step <= 0:
            return
        previous_step = training_global_step - 1
        expected = self.train_batch_size * self.samples_per_round
        start_group = training_global_step * self.train_batch_size
        prompt_paths: list[Path] = []
        for global_group_index in range(
            start_group, start_group + self.train_batch_size
        ):
            round_index, group_index = divmod(global_group_index, self.groups_per_round)
            prompt_paths.extend(
                self._round_dir(MIXED_TASK_TYPE, round_index).glob(
                    f"skills/group_{group_index:02d}_*/prompt.json"
                )
            )
        if len(prompt_paths) < expected:
            return

        expected_eval_prompts = int(
            getattr(self, "trajectory_pool_eval_prompts_per_step", 0)
        )
        eval_prompt_paths: list[Path] = []
        if expected_eval_prompts > 0:
            eval_prompt_paths = list(
                (
                    self.artifact_dir
                    / "eval"
                    / f"globalstep_{previous_step:06d}"
                    / "skills"
                ).glob("*/prompt.json")
            )
            if len(eval_prompt_paths) < expected_eval_prompts:
                return

        consumption_dir = self._trajectory_pool_root() / "consumption"
        marker_path = consumption_dir / f"globalstep_{training_global_step:06d}.json"
        if marker_path.exists():
            return
        lock_path = consumption_dir / f"globalstep_{training_global_step:06d}.lock"
        fd = _acquire_json_lock(lock_path, timeout_s=1800.0)
        try:
            if marker_path.exists():
                return
            # Write the marker before deleting so a crash mid-delete can still be
            # restored from the preserved canonical rollouts.
            _atomic_write_json(
                marker_path,
                {
                    "status": "consumed",
                    "consumer_training_global_step": training_global_step,
                    "deleted_training_global_step": previous_step,
                    "prepared_prompt_count": len(prompt_paths),
                    "expected_prompt_count": expected,
                    "prepared_eval_prompt_count": len(eval_prompt_paths),
                    "expected_eval_prompt_count": expected_eval_prompts,
                    "canonical_rollouts_preserved": True,
                    "updated_at": time.time(),
                },
            )
            previous_dir = self._trajectory_pool_step_dir(previous_step)
            (previous_dir / "index.json").unlink(missing_ok=True)
            if previous_dir.exists():
                shutil.rmtree(previous_dir)
        finally:
            _release_json_lock(fd, lock_path)

    def _prepare_sample(self, data: dict[str, Any]) -> dict[str, Any]:
        """Prepare one sample's prompt; the caller already waited for the previous round."""

        self._ensure_prompt_contract()
        task_type = str(data.get("task_type") or MIXED_TASK_TYPE)
        trajectory_pool_enabled = bool(getattr(self, "trajectory_pool_enabled", False))
        round_in_category = int(data["round_in_category"])
        group_index = int(data.get("group_index", 0))
        groups_per_round = int(data.get("groups_per_round", self.groups_per_round))
        seed = int(data.get("seed", 1))
        round_dir = self._round_dir(task_type, round_in_category)
        global_group_index = round_in_category * groups_per_round + group_index
        training_global_step = global_group_index // self.train_batch_size
        training_step_group_index = global_group_index % self.train_batch_size
        online_skillbank_snapshot = self._load_or_create_online_skillbank_snapshot(
            training_global_step
        )
        # Wait for the predecessor bank before locking a sample slot, so cancellation cannot strand it.
        sample_key, sample_dir = self._claim_sample_slot(
            task_type, round_in_category, group_index
        )

        prompt_path = sample_dir / "prompt.json"
        if trajectory_pool_enabled and prompt_path.exists():
            try:
                existing_prompt = _read_json(prompt_path)
                if existing_prompt.get(
                    "trajectory_pool_enabled"
                ) is True and self._prepared_prompt_matches_current(existing_prompt):
                    if str(
                        existing_prompt.get("online_skillbank_before_signature") or ""
                    ) != str(online_skillbank_snapshot["bank_signature"]):
                        raise RuntimeError(
                            "cached prompt uses a different online skillbank snapshot"
                        )
                    prepared = self._prepared_prompt_payload(
                        existing_prompt, sample_dir
                    )
                    self._maybe_delete_consumed_previous_step_pool(training_global_step)
                    return prepared
            except RuntimeError:
                # A bank snapshot mismatch is a scientific resume error, not a
                # cache miss that may be repaired by overwriting one prompt.
                raise
            except Exception:
                logger.warning(
                    "failed to reuse prepared trajectory-pool prompt %s",
                    prompt_path,
                    exc_info=True,
                )

        sample_seed = seed * 100000 + round_in_category * 1000 + group_index * 100
        # The step plan balances input task types across the optimizer batch.
        plan = self._online_skillbank_step_plan(training_global_step)
        input_task_type = plan.group_input_task_types[training_step_group_index]
        prompt_category = PROMPT_CATEGORY_BY_TASK_TYPE[input_task_type]
        sampled_episodes, sampling = self._select_trajectory_pool_prompt(
            input_task_type=input_task_type,
            training_global_step=training_global_step,
            seed=sample_seed,
        )

        # All policy samples in a GRPO group share the same reward games.
        selected_games = self._select_reward_games(
            input_task_type, round_in_category, group_index
        )
        messages, prompt_render = _fit_skill_prompt_messages(
            tokenizer=self.tokenizer,
            prompt_category=prompt_category,
            sampled_episodes=sampled_episodes,
            max_prompt_tokens=_skill_prompt_budget(self.gconfig),
            prompt_observation_char_limit=self.prompt_observation_char_limit,
            prompt_result_char_limit=self.prompt_result_char_limit,
            enable_thinking=self.skill_generation_enable_thinking,
            skill_output_format=self.skill_output_format,
            skill_prompt_version=self.skill_prompt_version,
        )
        prompt_payload = {
            "task_type": task_type,
            "input_task_type": input_task_type,
            "prompt_category": prompt_category,
            "round_in_category": round_in_category,
            "group_index": group_index,
            "groups_per_round": groups_per_round,
            "training_global_step": training_global_step,
            "training_step_group_index": training_step_group_index,
            "sample_key": sample_key,
            "trajectory_pool_enabled": trajectory_pool_enabled,
            "skill_prompt_version": self.skill_prompt_version,
            "trajectory_render_version": _TRAJECTORY_RENDER_VERSION,
            "online_skillbank_snapshot_path": str(
                self._online_skillbank_step_dir(training_global_step)
                / "bank_before.json"
            ),
            "online_skillbank_before_signature": str(
                online_skillbank_snapshot["bank_signature"]
            ),
            "online_skillbank_before_size": int(
                online_skillbank_snapshot["skill_count"]
            ),
            "selected_games": selected_games,
            "messages": messages,
            "prompt_render": prompt_render,
            "sampled_episodes": [
                {
                    "task_type": episode.get("task_type"),
                    "gamefile": episode.get("gamefile"),
                    "won": episode.get("won"),
                    "label": int(bool(episode.get("won"))),
                    "status": episode.get("status"),
                    "steps_taken": episode.get("steps_taken"),
                    "source_rollouts_path": episode.get("_source_rollouts_path"),
                    "source_episode_path": episode.get("_source_episode_path"),
                }
                for episode in sampled_episodes
            ],
            "sampling": sampling,
            "created_at": time.time(),
        }
        _atomic_write_json(prompt_path, prompt_payload)
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "prompt_prepared",
                "task_type": task_type,
                "input_task_type": input_task_type,
                "round_in_category": round_in_category,
                "group_index": group_index,
                "sample_key": sample_key,
                "training_global_step": training_global_step,
                "updated_at": time.time(),
            },
        )
        _atomic_write_json(
            round_dir / "round_state.json",
            {
                "task_type": task_type,
                "round_in_category": round_in_category,
                "groups_per_round": groups_per_round,
                "samples_per_group": self.samples_per_round,
                "expected_samples_per_round": self.samples_per_round * groups_per_round,
                "trajectory_pool_enabled": trajectory_pool_enabled,
                "updated_at": time.time(),
            },
        )
        if trajectory_pool_enabled:
            self._maybe_delete_consumed_previous_step_pool(training_global_step)
        return self._prepared_prompt_payload(prompt_payload, sample_dir)
