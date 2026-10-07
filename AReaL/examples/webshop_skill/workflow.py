# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import math
import random
import re
import threading
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from examples.skill_training.semantic_skillbank import SkillEntry
from examples.webshop_skill.bank import WebShopBankMixin
from examples.webshop_skill.core import (
    compact_webshop_source_episode,
    is_usable_webshop_source_episode,
    render_webshop_trajectories,
    select_trajectory_bundles,
    webshop_skill_messages,
)
from examples.webshop_skill.evaluation_workflow import WebShopEvaluationMixin
from examples.webshop_skill.rollout import WebShopRolloutMixin
from examples.webshop_skill.runtime import (
    _SKILLBANK_MODE,
    _WEBSHOP_COMPLETE_TRACE_PROFILES,
    _WEBSHOP_SKILL_PROMPT_TOTAL_TOKEN_BUDGET,
    WebShopEnvironmentClient,
    logger,
)
from examples.webshop_skill.task_split import (
    task_selection_contract,
    validate_official_manifest,
)

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, ModelResponse
from areal.api.cli_args import GenerationHyperparameters
from areal.dataset.webshop_skill import WEBSHOP_PROMPT_CATEGORY, WEBSHOP_TASK_TYPE
from areal.utils import stats_tracker
from areal.utils.hf_utils import apply_chat_template
from areal.workflow.alfworld_skill import (
    ALFWorldSkillGRPOWorkflow,
    _acquire_json_lock,
    _atomic_write_json,
    _complete_rollout_paths,
    _output_token_nll_stats,
    _parse_skill_generation,
    _read_json,
    _release_json_lock,
    _split_skill_generation_sections,
)
from areal.workflow.skill_prompt import skill_prompt_budgets

_FATAL_ERROR_FILE = "fatal_workflow_error.json"


class WebShopSkillGRPOWorkflow(
    WebShopBankMixin, WebShopRolloutMixin, ALFWorldSkillGRPOWorkflow
):
    """Train a skill generator against a frozen actor on fixed WebShop tasks."""

    BENCHMARK_NAME = "WebShop"
    TASK_TYPE = WEBSHOP_TASK_TYPE
    PROMPT_CATEGORY = WEBSHOP_PROMPT_CATEGORY
    METRIC_PREFIX = "webshop"
    ENV_CLIENT_CLASS = WebShopEnvironmentClient

    def _fit_skill_messages(
        self, episodes: Sequence[dict[str, Any]]
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Render every source action, compressing page text until the prompt fits."""
        total_budget, max_prompt_tokens = skill_prompt_budgets(
            self.gconfig,
            total_token_budget=_WEBSHOP_SKILL_PROMPT_TOTAL_TOKEN_BUDGET,
        )

        def token_count(messages: list[dict[str, str]]) -> int:
            return len(
                apply_chat_template(
                    self.tokenizer,
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.skill_generation_enable_thinking,
                )
            )

        render_metadata = {
            "max_prompt_tokens": max_prompt_tokens,
            "total_token_budget": total_budget,
            "skill_prompt_version": "evidence_discovery",
            "skill_output_format": "skill_xml",
            "trajectory_render_version": "webshop_complete_history_v1",
            "full_history_preserved": True,
            "dropped_history": False,
            "prompt_observation_char_limit": self.prompt_observation_char_limit,
        }
        if not episodes:
            messages = webshop_skill_messages(
                [], max_words=self.skill_description_max_words
            )
            current_count = token_count(messages)
            if current_count > max_prompt_tokens:
                raise ValueError(
                    "zero-shot WebShop evidence prompt exceeds the token budget: "
                    f"{current_count} > {max_prompt_tokens}"
                )
            return messages, {
                **render_metadata,
                "prompt_token_count": current_count,
                "was_compacted": False,
                "detail_profile": "zero_shot",
                "source_step_count": 0,
                "rendered_action_count": 0,
                "trace_token_count": 0,
            }

        source_step_count = sum(
            len(
                [
                    row
                    for row in list(episode.get("trace") or [])
                    if isinstance(row, dict) and str(row.get("action") or "").strip()
                ]
            )
            for episode in episodes
        )
        smallest_count: int | None = None
        for profile_index, (
            profile_name,
            detail_tail_steps,
            summary_limit,
            include_state_metadata,
            include_intermediate_results,
        ) in enumerate(_WEBSHOP_COMPLETE_TRACE_PROFILES):
            compact_episodes = [
                compact_webshop_source_episode(
                    episode,
                    observation_char_limit=(
                        min(self.prompt_observation_char_limit, 64)
                        if profile_name == "detail64_sparse_result16"
                        else self.prompt_observation_char_limit
                    ),
                    detail_tail_steps=detail_tail_steps,
                    summary_observation_char_limit=summary_limit,
                    include_state_metadata=include_state_metadata,
                    include_intermediate_results=include_intermediate_results,
                    actions_only=profile_name == "actions_only",
                )
                for episode in episodes
            ]
            messages = webshop_skill_messages(
                compact_episodes, max_words=self.skill_description_max_words
            )
            current_count = token_count(messages)
            smallest_count = (
                current_count
                if smallest_count is None
                else min(smallest_count, current_count)
            )
            if current_count > max_prompt_tokens:
                continue
            return messages, {
                **render_metadata,
                "prompt_token_count": current_count,
                "was_compacted": True,
                "rendered_max_steps": max(
                    (len(episode.get("trace") or []) for episode in compact_episodes),
                    default=0,
                ),
                "source_step_count": source_step_count,
                "rendered_action_count": source_step_count,
                "detail_profile_index": profile_index,
                "detail_profile": profile_name,
                "observations_omitted": profile_name == "actions_only",
                "trace_token_count": token_count(
                    [
                        {
                            "role": "user",
                            "content": render_webshop_trajectories(compact_episodes),
                        }
                    ]
                ),
            }
        raise ValueError(
            "complete WebShop action history exceeds the actual prompt budget even "
            "after observation compression; refusing to drop actions: "
            f"smallest_prompt={smallest_count}, max_prompt={max_prompt_tokens}, "
            f"source_steps={source_step_count}"
        )

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: Any,
        *,
        artifact_dir: str,
        actor_base_url: str,
        env_service_url: str,
        actor_model: str = "Qwen/Qwen3.5-4B",
        actor_api_key: str = "",
        actor_timeout_s: float = 120.0,
        actor_temperature: float = 0.0,
        task_split: str = "training_holdout",
        train_task_count: int = 300,
        eval_task_count: int = 100,
        task_seed: int = 1,
        unique_asin_split: bool = True,
        rollouts_per_skill: int = 8,
        max_rollout_steps: int = 100,
        memory_window: int = 5,
        observation_char_limit: int = 5000,
        max_clickables: int = 60,
        invalid_action_retries: int = 1,
        success_threshold: float = 0.999999,
        actor_max_tokens: int = 128,
        rollout_timeout_s: float = 7200.0,
        episode_rollout_workers: int = 64,
        source_trajectories_per_prompt: int = 4,
        prompt_observation_char_limit: int = 400,
        samples_per_round: int = 8,
        rounds: int = 5,
        groups_per_round: int = 8,
        train_batch_size: int | None = None,
        parallelize_same_step_rounds: bool = False,
        skill_generation_enable_thinking: bool = False,
        skill_generation_max_retries: int = 0,
        outcome_condition_workers: int = 32,
        skill_description_max_words: int = 120,
        reward_sr_weight: float = 1.0,
        reward_mutual_information_weight: float = 0.0,
        reward_mutual_information_scale: float = 1.0,
        reward_mutual_information_token_margin_clip: float | None = 1.0,
        reward_mutual_information_token_reward_soft_cap: float | None = None,
        reward_mutual_information_token_reward_clip: float | None = None,
        reward_schema_valid_bonus: float = 0.0,
        reward_schema_invalid_penalty: float = -0.1,
        skillbank_top_k: int = 3,
        skillbank_embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        skillbank_embedding_batch_size: int = 64,
        skillbank_embedding_max_length: int = 256,
        skillbank_embedding_device: str = "cpu",
        skillbank_standalone_weight: float = 0.5,
        skillbank_retrieval_weight: float = 0.5,
        skillbank_reward_weight: float = 1.0,
        skillbank_update_min_marginal: float = 0.0,
        skillbank_allow_noop_update: bool = True,
        skillbank_max_size: int = 0,
        eval_no_skill_baseline_repeats: int = 1,
        eval_no_skill_baseline_seed: int = 900000,
    ) -> None:
        if task_split not in {"official_train", "training_holdout"}:
            raise ValueError(f"unknown WebShop training task_split: {task_split}")
        if task_split == "official_train":
            if train_task_count != -1 or unique_asin_split:
                raise ValueError(
                    "official_train requires all goals: count=-1, unique_asin=false"
                )
        elif train_task_count <= 0:
            raise ValueError("train_task_count must be positive for training_holdout")
        if eval_task_count <= 0:
            raise ValueError("train_task_count and eval_task_count must be positive")
        if source_trajectories_per_prompt <= 0:
            raise ValueError("source_trajectories_per_prompt must be positive")
        if samples_per_round != 8 or rollouts_per_skill != 8:
            raise ValueError(
                f"{_SKILLBANK_MODE} requires exactly 8 samples and 8 reward tasks"
            )
        if train_batch_size is None:
            train_batch_size = groups_per_round
        if parallelize_same_step_rounds and (
            train_batch_size <= 0 or train_batch_size % groups_per_round != 0
        ):
            raise ValueError(
                "parallelize_same_step_rounds requires an integer number of rounds per batch"
            )
        if eval_no_skill_baseline_repeats <= 0:
            raise ValueError("eval_no_skill_baseline_repeats must be positive")
        if skillbank_top_k <= 0:
            raise ValueError("skillbank_top_k must be positive")
        if abs(skillbank_standalone_weight + skillbank_retrieval_weight - 1.0) > 1e-09:
            raise ValueError("skillbank reward weights must sum to 1")
        if min(skillbank_standalone_weight, skillbank_retrieval_weight) < 0.0:
            raise ValueError("skillbank reward weights must be non-negative")
        if not math.isfinite(skillbank_reward_weight) or skillbank_reward_weight < 0:
            raise ValueError("skillbank_reward_weight must be finite and non-negative")
        super().__init__(
            gconfig,
            tokenizer,
            artifact_dir=artifact_dir,
            actor_base_url=actor_base_url,
            actor_model=actor_model,
            actor_api_key=actor_api_key,
            actor_timeout_s=actor_timeout_s,
            actor_temperature=actor_temperature,
            rollouts_per_skill=rollouts_per_skill,
            max_rollout_steps=max_rollout_steps,
            memory_window=memory_window,
            prompt_observation_char_limit=prompt_observation_char_limit,
            prompt_result_char_limit=prompt_observation_char_limit,
            rollout_timeout_s=rollout_timeout_s,
            episode_rollout_workers=episode_rollout_workers,
            max_commands=max_clickables,
            samples_per_round=samples_per_round,
            rounds_per_category=rounds,
            groups_per_round=groups_per_round,
            train_batch_size=train_batch_size,
            # Satisfies the parent's parallel-round check; the ALFWorld pool is unused.
            trajectory_pool_enabled=bool(parallelize_same_step_rounds),
            trajectory_pool_prompt_episodes=source_trajectories_per_prompt,
            parallelize_same_step_rounds=parallelize_same_step_rounds,
            skill_generation_enable_thinking=skill_generation_enable_thinking,
            skill_generation_max_retries=skill_generation_max_retries,
            outcome_condition_workers=outcome_condition_workers,
            skill_description_max_words=skill_description_max_words,
            task_types=[self.TASK_TYPE],
            reward_sr_weight=reward_sr_weight,
            reward_mutual_information_weight=reward_mutual_information_weight,
            reward_mutual_information_scale=reward_mutual_information_scale,
            reward_mutual_information_token_margin_clip=reward_mutual_information_token_margin_clip,
            reward_mutual_information_token_reward_soft_cap=reward_mutual_information_token_reward_soft_cap,
            reward_mutual_information_token_reward_clip=reward_mutual_information_token_reward_clip,
            reward_schema_valid_bonus=reward_schema_valid_bonus,
            reward_schema_invalid_penalty=reward_schema_invalid_penalty,
        )
        self.env_service_url = str(env_service_url).rstrip("/")
        self.task_split = task_split
        self.train_task_count = int(train_task_count)
        self.eval_task_count = int(eval_task_count)
        self.task_seed = int(task_seed)
        self.unique_asin_split = bool(unique_asin_split)
        self.observation_char_limit = int(observation_char_limit)
        self.max_clickables = int(max_clickables)
        self.invalid_action_retries = int(invalid_action_retries)
        self.success_threshold = float(success_threshold)
        self.actor_max_tokens = int(actor_max_tokens)
        self.source_trajectories_per_prompt = int(source_trajectories_per_prompt)
        self.eval_no_skill_baseline_repeats = int(eval_no_skill_baseline_repeats)
        self.eval_no_skill_baseline_seed = int(eval_no_skill_baseline_seed)
        self.skillbank_top_k = int(skillbank_top_k)
        self.skillbank_embedding_model = str(skillbank_embedding_model)
        self.skillbank_embedding_batch_size = int(skillbank_embedding_batch_size)
        self.skillbank_embedding_max_length = int(skillbank_embedding_max_length)
        self.skillbank_embedding_device = str(skillbank_embedding_device)
        self.skillbank_standalone_weight = float(skillbank_standalone_weight)
        self.skillbank_retrieval_weight = float(skillbank_retrieval_weight)
        self.skillbank_reward_weight = float(skillbank_reward_weight)
        self.skillbank_update_min_marginal = float(skillbank_update_min_marginal)
        self.skillbank_allow_noop_update = bool(skillbank_allow_noop_update)
        self.skillbank_max_size = int(skillbank_max_size)
        self._group_reward_tasks: dict[
            tuple[int, int], asyncio.Future[dict[str, Any]]
        ] = {}
        self._group_reward_results: dict[tuple[int, int], dict[str, Any]] = {}
        self._group_reward_task_lock = asyncio.Lock()
        self._skillbank_snapshot_cache: dict[int, list[SkillEntry]] = {}
        self._skillbank_snapshot_lock = threading.RLock()
        self._skillbank_embedding_cache: dict[str, torch.Tensor] = {}
        self._skillbank_embedding_lock = threading.RLock()
        self._manifest_cache: dict[str, Any] | None = None
        self._manifest_lock = threading.Lock()
        self._env_client = self.ENV_CLIENT_CLASS(
            self.env_service_url,
            timeout_s=float(rollout_timeout_s or 7200.0),
            workers=max(1, int(episode_rollout_workers or rollouts_per_skill)),
        )
        self._ensure_webshop_run_contract()
        self._initialize_skillbank()

    def _validate_online_skillbank_settings(self) -> None:
        # The parent __init__ checks its ALFWorld bank here; WebShop checks its own.
        pass

    def _raise_if_workflow_failed(self) -> None:
        # Called by the inherited generation and round barriers.
        artifact_dir = getattr(self, "artifact_dir", None)
        if (
            artifact_dir is not None
            and (Path(artifact_dir) / _FATAL_ERROR_FILE).is_file()
        ):
            raise RuntimeError(
                f"WEBSHOP_WORKFLOW_FATAL: a peer failed; see {_FATAL_ERROR_FILE}"
            )
        super()._raise_if_workflow_failed()

    def _ensure_webshop_run_contract(self) -> None:
        """Reject resume into artifacts produced by a different prompt/reward config."""
        payload = {
            "version": 2,
            "benchmark": self.BENCHMARK_NAME,
            "parallelize_same_step_rounds": self.parallelize_same_step_rounds,
            "samples_per_round": self.samples_per_round,
            "groups_per_round": self.groups_per_round,
            "train_batch_size": self.train_batch_size,
            "rollouts_per_skill": self.rollouts_per_skill,
            "source_trajectories_per_prompt": self.source_trajectories_per_prompt,
            "prompt_observation_char_limit": self.prompt_observation_char_limit,
            "skill_description_max_words": self.skill_description_max_words,
            "skill_generation_enable_thinking": self.skill_generation_enable_thinking,
            "skill_generation_max_retries": self.skill_generation_max_retries,
            "reward_sr_weight": self.reward_sr_weight,
            "reward_schema_valid_bonus": self.reward_schema_valid_bonus,
            "reward_schema_invalid_penalty": self.reward_schema_invalid_penalty,
            "mutual_information_mode": "tokenwise_abs_aligned",
            "mutual_information_weight": self.reward_mutual_information_weight,
            "mutual_information_scale": self.reward_mutual_information_scale,
            "mutual_information_token_margin_clip": self.reward_mutual_information_token_margin_clip,
            "mutual_information_token_reward_soft_cap": self.reward_mutual_information_token_reward_soft_cap,
            "mutual_information_token_reward_clip": self.reward_mutual_information_token_reward_clip,
            "actor_model": self.actor_model,
            "actor_temperature": self.actor_temperature,
            "task_seed": self.task_seed,
            "max_rollout_steps": self.max_rollout_steps,
            "memory_window": self.memory_window,
            "actor_max_tokens": self.actor_max_tokens,
            "observation_char_limit": self.observation_char_limit,
            "max_clickables": self.max_clickables,
            "invalid_action_retries": self.invalid_action_retries,
            "success_threshold": self.success_threshold,
            "skillbank_top_k": self.skillbank_top_k,
            "skillbank_embedding_model": self.skillbank_embedding_model,
            "skillbank_embedding_batch_size": self.skillbank_embedding_batch_size,
            "skillbank_embedding_max_length": self.skillbank_embedding_max_length,
            "skillbank_embedding_device": self.skillbank_embedding_device,
            "skillbank_standalone_weight": self.skillbank_standalone_weight,
            "skillbank_retrieval_weight": self.skillbank_retrieval_weight,
            "skillbank_reward_weight": self.skillbank_reward_weight,
            "skillbank_update_min_marginal": self.skillbank_update_min_marginal,
            "skillbank_allow_noop_update": self.skillbank_allow_noop_update,
            "skillbank_max_size": self.skillbank_max_size,
            "recovery_protocol": "archive_future_rows_v1",
        }
        if self.task_split != "training_holdout":
            payload["task_selection"] = task_selection_contract(
                task_split=self.task_split,
                train_count=self.train_task_count,
                eval_count=self.eval_task_count,
                seed=self.task_seed,
                unique_asin=self.unique_asin_split,
            )
        path = self.artifact_dir / "webshop_training_contract.json"
        lock_path = self.artifact_dir / "webshop_training_contract.lock"
        fd = _acquire_json_lock(
            lock_path, timeout_s=7200.0, foreign_host_stale_after_s=120.0
        )
        try:
            if path.exists():
                existing = _read_json(path)
                if existing != payload:
                    raise RuntimeError(
                        "persisted WebShop training contract differs from the current prompt/reward configuration; use a new trial"
                    )
                return
            _atomic_write_json(path, payload)
        finally:
            _release_json_lock(fd, lock_path)

    def _task_manifest(self) -> dict[str, Any]:
        with self._manifest_lock:
            if self._manifest_cache is not None:
                return self._manifest_cache
            manifest = self._env_client.manifest(
                train_count=self.train_task_count,
                eval_count=self.eval_task_count,
                seed=self.task_seed,
                unique_asin=self.unique_asin_split,
                task_split=self.task_split,
            )
            train_indices = list(map(int, manifest.get("train_indices", [])))
            eval_indices = list(map(int, manifest.get("eval_indices", [])))
            if self.task_split == "official_train":
                validate_official_manifest(
                    manifest, eval_count=self.eval_task_count, seed=self.task_seed
                )
            elif len(train_indices) != self.train_task_count:
                raise RuntimeError(
                    f"{self.BENCHMARK_NAME} manifest train task count mismatch"
                )
            if len(eval_indices) != self.eval_task_count:
                raise RuntimeError(
                    f"{self.BENCHMARK_NAME} manifest eval task count mismatch"
                )
            if set(train_indices) & set(eval_indices):
                raise RuntimeError(
                    f"{self.BENCHMARK_NAME} train/eval manifest overlaps"
                )
            persisted = self.artifact_dir / "task_manifest.json"
            if persisted.exists():
                existing = _read_json(persisted)
                for key in (
                    "manifest_signature",
                    "seed",
                    "unique_asin",
                    "train_indices",
                    "eval_indices",
                ):
                    if existing.get(key) != manifest.get(key):
                        raise RuntimeError(
                            f"persisted {self.BENCHMARK_NAME} manifest differs at {key}; use a new trial"
                        )
            else:
                _atomic_write_json(persisted, manifest)
            self._manifest_cache = manifest
            return manifest

    def _group_task_indices(self, round_index: int, group_index: int) -> list[int]:
        pool = list(map(int, self._task_manifest()["train_indices"]))
        count = min(self.rollouts_per_skill, len(pool))
        seed = self.task_seed * 1_000_003 + round_index * 10_007 + group_index * 101
        return random.Random(seed).sample(pool, k=count)

    def _rollout_payload(
        self,
        *,
        task_index: int,
        skill: str,
        condition: str,
        seed: int,
        session_key: str,
    ) -> dict[str, Any]:
        return {
            "task_index": int(task_index),
            "skill": skill,
            "condition": condition,
            "max_steps": self.max_rollout_steps,
            "memory_window": self.memory_window,
            "observation_char_limit": self.observation_char_limit,
            "max_clickables": self.max_clickables,
            "actor_temperature": self.actor_temperature,
            "actor_max_tokens": self.actor_max_tokens,
            "actor_seed": int(seed),
            "invalid_action_retries": self.invalid_action_retries,
            "success_threshold": self.success_threshold,
            "actor_base_url": self.actor_base_url,
            "actor_model": self.actor_model,
            "actor_api_key": self.actor_api_key,
            "actor_timeout_s": self.actor_timeout_s,
            "session_key": session_key,
        }

    def _run_episodes(
        self,
        task_indices: Sequence[int],
        *,
        skill: str,
        condition: str,
        seed_base: int,
        max_workers: int | None = None,
    ) -> list[dict[str, Any]]:
        indices = list(map(int, task_indices))
        workers = min(
            len(indices),
            max(1, int(max_workers or self.episode_rollout_workers or len(indices))),
        )
        episodes: list[dict[str, Any] | None] = [None] * len(indices)

        def run(position: int, task_index: int) -> tuple[int, dict[str, Any]]:
            try:
                result = self._env_client.rollout(
                    self._rollout_payload(
                        task_index=task_index,
                        skill=skill,
                        condition=condition,
                        seed=seed_base + position * 997,
                        session_key=(
                            f"{condition}-{task_index}-{seed_base}-{position}-"
                            f"{uuid.uuid4().hex[:8]}"
                        ),
                    )
                )
                if result.get("status") == "error":
                    result["status"] = "rollout_error"
            except Exception as exc:  # noqa: BLE001
                result = {
                    "task_index": task_index,
                    "condition": condition,
                    "skill": skill,
                    "trace": [],
                    "reward": 0.0,
                    "success": False,
                    "done": False,
                    "status": "rollout_error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            return position, result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run, position, task_index)
                for position, task_index in enumerate(indices)
            ]
            for future in as_completed(futures):
                position, episode = future.result()
                episodes[position] = episode
        return [episode for episode in episodes if episode is not None]

    @staticmethod
    def _episodes_from_round(round_dir: Path) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        for rollouts_path in _complete_rollout_paths(round_dir):
            if (
                re.fullmatch(r"group_\d{2}_sample_\d{2}", rollouts_path.parent.name)
                is None
            ):
                continue
            try:
                payload = _read_json(rollouts_path)
            except Exception:
                continue
            for episode in payload.get("episodes", []):
                if isinstance(episode, dict):
                    copied = dict(episode)
                    copied["_source_rollouts_path"] = str(rollouts_path)
                    episodes.append(copied)
        return episodes

    def _source_pool_from_rounds(
        self, source_paths: Sequence[Path]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Use candidate traces, or real train controls if every candidate is invalid."""
        candidates = [
            episode
            for path in source_paths
            for episode in self._episodes_from_round(path)
        ]
        metadata: dict[str, Any] = {
            "source": ",".join(map(str, source_paths)),
            "candidate_status_counts": dict(
                Counter(str(item.get("status")) for item in candidates)
            ),
            "fallback": None,
        }
        if any(is_usable_webshop_source_episode(item) for item in candidates):
            return candidates, metadata

        # These controls already ran for the same training rounds. Never borrow
        # eval episodes or treat invalid-skill placeholders as real evidence.
        manifest = self._task_manifest()
        train_indices = set(map(int, manifest["train_indices"]))
        train_indices.difference_update(map(int, manifest.get("eval_indices", [])))
        controls: dict[str, list[dict[str, Any]]] = {
            "no_skill_episodes": [],
            "bank_only_episodes": [],
        }
        for path in source_paths:
            for index_path in sorted(
                (path / "group_reward" / _SKILLBANK_MODE).glob("group_*/index.json")
            ):
                payload = _read_json(index_path)
                if (
                    payload.get("status") != "complete"
                    or payload.get("mode") != _SKILLBANK_MODE
                ):
                    continue
                if (
                    int(payload.get("round_index", -1))
                    != int(path.name.rsplit("_", 1)[1])
                    or index_path.parent.name
                    != f"group_{int(payload.get('group_index', -1)):02d}"
                ):
                    continue
                allowed = train_indices.intersection(
                    map(int, payload.get("task_indices", []))
                )
                for condition, episodes in controls.items():
                    for item in payload.get(condition, []):
                        if (
                            isinstance(item, dict)
                            and item.get("task_index") in allowed
                            and is_usable_webshop_source_episode(item)
                        ):
                            episodes.append(
                                {
                                    **item,
                                    "_source_rollouts_path": str(index_path),
                                    "_source_episode_field": condition,
                                }
                            )
        for condition, episodes in controls.items():
            if episodes:
                metadata["fallback"] = condition
                metadata["fallback_reason"] = "no_usable_candidate_trajectories"
                logger.warning(
                    "Using %s source evidence from %s; candidate statuses=%s",
                    condition,
                    metadata["source"],
                    metadata["candidate_status_counts"],
                )
                return episodes, metadata
        raise RuntimeError(
            f"no usable {self.BENCHMARK_NAME} training source trajectories from "
            f"{metadata['source']}; candidate_status_counts="
            f"{metadata['candidate_status_counts']}; no usable completed train controls"
        )

    def _training_step_round_indices(self, training_global_step: int) -> list[int]:
        start_group = int(training_global_step) * self.train_batch_size
        stop_group = start_group + self.train_batch_size
        return sorted(
            {
                global_group // self.groups_per_round
                for global_group in range(start_group, stop_group)
            }
        )

    async def _wait_for_training_step_rollouts_async(
        self, training_global_step: int
    ) -> None:
        deadline = time.monotonic() + max(1.0, self.rollout_timeout_s)
        for source_round in self._training_step_round_indices(training_global_step):
            round_dir = self._round_dir(self.TASK_TYPE, source_round)
            canonical_names = {
                f"group_{group_index:02d}_sample_{sample_index:02d}"
                for group_index in range(self.groups_per_round)
                for sample_index in range(self.samples_per_round)
            }
            while True:
                self._raise_if_workflow_failed()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"WebShop training-step rollout barrier timed out: step={training_global_step}"
                    )
                complete = {
                    path.parent.name
                    for path in _complete_rollout_paths(round_dir)
                    if path.parent.name in canonical_names
                }
                if complete == canonical_names:
                    break
                await asyncio.sleep(1.0)

    def _source_episodes(
        self,
        *,
        round_index: int,
        group_index: int,
        task_indices: Sequence[int],
        seed: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        parallel = bool(getattr(self, "parallelize_same_step_rounds", False))
        training_global_step = (
            self._training_step_index(round_index, group_index) if parallel else -1
        )
        if round_index == 0 or (parallel and training_global_step == 0):
            return [], {
                "source": "round0_zero_shot",
                "pool_size": 0,
                "selected_task_indices": [],
                "selected_successes": 0,
            }

        if parallel:
            source_rounds = self._training_step_round_indices(training_global_step - 1)
            source_paths = [
                self._round_dir(self.TASK_TYPE, source_round)
                for source_round in source_rounds
            ]
        else:
            source_paths = [self._round_dir(self.TASK_TYPE, round_index - 1)]
        pool, metadata = self._source_pool_from_rounds(source_paths)
        selected = select_trajectory_bundles(
            pool,
            num_skills=1,
            trajectories_per_skill=self.source_trajectories_per_prompt,
            seed=seed,
            success_threshold=self.success_threshold,
        )[0]
        return selected, {
            **metadata,
            "pool_size": len(pool),
            "selected_task_indices": [
                int(item.get("task_index", -1)) for item in selected
            ],
            "selected_successes": sum(bool(item.get("success")) for item in selected),
        }

    def _prepare_sample(self, data: dict[str, Any]) -> dict[str, Any]:
        round_index = int(data["round_in_category"])
        group_index = int(data.get("group_index", 0))
        groups_per_round = int(data.get("groups_per_round", self.groups_per_round))
        seed = int(data.get("seed", self.task_seed))
        sample_key, sample_dir = self._claim_sample_slot(
            self.TASK_TYPE, round_index, group_index
        )
        task_indices = self._group_task_indices(round_index, group_index)
        # Samples in a group share source trajectories, as the MI reward requires.
        source_seed = seed * 100_000 + round_index * 1_000 + group_index * 100
        source_episodes, sampling = self._source_episodes(
            round_index=round_index,
            group_index=group_index,
            task_indices=task_indices,
            seed=source_seed,
        )
        messages, prompt_render = self._fit_skill_messages(source_episodes)
        global_group_index = round_index * groups_per_round + group_index
        prepared = {
            "task_type": self.TASK_TYPE,
            "prompt_category": self.PROMPT_CATEGORY,
            "round_in_category": round_index,
            "group_index": group_index,
            "groups_per_round": groups_per_round,
            "training_global_step": global_group_index // self.train_batch_size,
            "training_step_group_index": global_group_index % self.train_batch_size,
            "sample_key": sample_key,
            "sample_dir": str(sample_dir),
            "selected_games": list(task_indices),
            "task_indices": list(task_indices),
            "messages": messages,
            "prompt_render": prompt_render,
            "sampling": sampling,
        }
        _atomic_write_json(
            sample_dir / "prompt.json",
            {**prepared, "source_episodes": source_episodes, "created_at": time.time()},
        )
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "prompt_prepared",
                "task_type": self.TASK_TYPE,
                "round_in_category": round_index,
                "group_index": group_index,
                "sample_key": sample_key,
                "updated_at": time.time(),
            },
        )
        return prepared

    def _generation_attempt(
        self, raw: str, resp: ModelResponse, attempt_index: int
    ) -> dict[str, Any]:
        parsed, valid, error, repair_notes = _parse_skill_generation(
            raw,
            self.PROMPT_CATEGORY,
            max_words=self.skill_description_max_words,
        )
        thinking, answer, section_notes = _split_skill_generation_sections(raw)
        entropy, entropy_count = _output_token_nll_stats(resp.output_logprobs)
        return {
            "attempt": attempt_index + 1,
            "raw": raw,
            "thinking": thinking,
            "answer": answer,
            "parsed": parsed,
            "schema_valid": valid,
            "schema_error": error,
            "repair_notes": section_notes + repair_notes,
            "input_token_count": len(resp.input_tokens),
            "output_token_count": len(resp.output_tokens),
            "output_entropy_proxy": entropy,
            "output_entropy_token_count": entropy_count,
            "updated_at": time.time(),
        }

    def _group_actor_seed(
        self, *, round_index: int, group_index: int, task_position: int
    ) -> int:
        return (
            self.task_seed * 10_000_019
            + round_index * 100_003
            + group_index * 1_009
            + task_position * 17
        )

    def _training_step_sample_paths(
        self, training_global_step: int
    ) -> list[tuple[str, Path, Path]]:
        paths: list[tuple[str, Path, Path]] = []
        start_group = int(training_global_step) * self.train_batch_size
        for global_group in range(start_group, start_group + self.train_batch_size):
            round_index, group_index = divmod(global_group, self.groups_per_round)
            round_dir = self._round_dir(self.TASK_TYPE, round_index)
            for sample_index in range(self.samples_per_round):
                sample_key = f"group_{group_index:02d}_sample_{sample_index:02d}"
                sample_dir = round_dir / "skills" / sample_key
                paths.append(
                    (
                        f"round_{round_index:04d}/{sample_key}",
                        sample_dir / "generation.json",
                        sample_dir / "metrics.json",
                    )
                )
        return paths

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        try:
            return await self._arun_episode(engine, data)
        except Exception as exc:
            # AsyncTaskRunner otherwise consumes this error and refills later groups,
            # leaving peers waiting for samples which can never become complete.
            logger.error(
                "WEBSHOP_WORKFLOW_FATAL round=%s group=%s type=%s",
                data.get("round_in_category"),
                data.get("group_index"),
                type(exc).__name__,
            )
            failure_path = self.artifact_dir / _FATAL_ERROR_FILE
            try:
                # Keep the first failure; peers then fail on this marker.
                if not failure_path.exists():
                    await asyncio.to_thread(
                        _atomic_write_json,
                        failure_path,
                        {
                            "status": "failed",
                            "round_in_category": data.get("round_in_category"),
                            "group_index": data.get("group_index"),
                            "error_type": type(exc).__name__,
                            "error": self._diagnostic_text(str(exc)),
                            "updated_at": time.time(),
                        },
                    )
            except Exception:
                logger.error("could not persist WebShop failure diagnostic")
            raise

    async def _arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        round_index = int(data["round_in_category"])
        groups_per_round = int(data.get("groups_per_round", self.groups_per_round))
        group_index = int(data.get("group_index", 0))
        training_step = self._training_step_index(round_index, group_index)
        if self.parallelize_same_step_rounds and training_step > 0:
            await self._wait_for_training_step_rollouts_async(training_step - 1)
        elif not self.parallelize_same_step_rounds and round_index > 0:
            await self._wait_for_previous_round_async(
                self._round_dir(self.TASK_TYPE, round_index - 1),
                expected_count=self.samples_per_round * groups_per_round,
            )
        prepared = await asyncio.to_thread(self._prepare_sample, data)
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
            await asyncio.to_thread(self._write_generation_attempts, prepared, attempts)
            if attempts[-1]["schema_valid"]:
                break
        if resp is None:
            raise RuntimeError(
                f"{self.BENCHMARK_NAME} skill generator returned no response"
            )
        await asyncio.to_thread(
            self._write_training_generation,
            prepared,
            raw,
            list(resp.input_tokens),
            list(resp.output_tokens),
            attempts,
        )
        await self._wait_for_group_generation_async(
            self._round_dir(self.TASK_TYPE, round_index),
            group_index=group_index,
            expected_count=self.samples_per_round,
        )
        group_waiter = asyncio.create_task(
            self._shared_group_reward_index(
                round_index=round_index,
                group_index=group_index,
            )
        )
        await self._wait_for_round_generation_async(
            self._round_dir(self.TASK_TYPE, round_index),
            expected_count=self.samples_per_round * groups_per_round,
            canonical_group_count=groups_per_round,
        )
        await asyncio.to_thread(
            self._write_training_step_skill_manifest,
            self.TASK_TYPE,
            round_index,
            int(prepared["group_index"]),
        )
        group_reward_index = await group_waiter
        result = await asyncio.to_thread(
            self._evaluate_and_save,
            prepared,
            raw,
            list(resp.input_tokens),
            list(resp.output_tokens),
            attempts,
            engine,
            group_reward_index,
        )
        entropy = float(attempts[-1].get("output_entropy_proxy", 0.0))
        metric_values = {
            "reward": float(result["reward"]),
            "task_reward": float(result["sr"]),
            "env_reward": float(result["mean_env_reward"]),
            "sr": float(result["sr"]),
            "sr_reward": float(result["sr_reward"]),
            "baseline_sr": float(result["baseline_sr"]),
            "baseline_delta_sr": float(result["baseline_delta_sr"]),
            "mutual_information_score": float(result["mutual_information_score"]),
            "mutual_information_raw_score": float(
                result["mutual_information_raw_score"]
            ),
            "mutual_information_reward": float(result["mutual_information_reward"]),
            "mi_reward_abs_fraction": float(
                result["mutual_information_reward_abs_fraction"]
            ),
            "mi_token_reward_body_mean": float(
                result.get("mi_token_reward_body_mean", 0.0)
            ),
            "mi_token_reward_body_count": float(
                result.get("mi_token_reward_body_count", 0)
            ),
            "mi_token_reward_body_clipped_fraction": float(
                result.get("mi_token_reward_body_clipped_fraction", 0.0)
            ),
            "mi_token_reward_mean": float(
                result["mutual_information_token_reward_mean"]
            ),
            "mi_token_reward_min": float(result["mutual_information_token_reward_min"]),
            "mi_token_reward_max": float(result["mutual_information_token_reward_max"]),
            "mi_token_reward_count": float(
                result["mutual_information_token_reward_count"]
            ),
            "mi_token_reward_clipped_fraction": float(
                result["mutual_information_token_reward_clipped_fraction"]
            ),
            "schema_reward": float(result["schema_reward"]),
            "schema_valid": float(bool(result["schema_valid"])),
            "output_entropy_proxy": entropy,
        }
        scoped_metrics: dict[str, float] = {
            f"{self.METRIC_PREFIX}_{key}": value for key, value in metric_values.items()
        }
        bank_metrics = result["skillbank_counterfactual"]
        bank_payload = result["skillbank"]
        retrieval_scores = [
            float(selected["cosine_similarity"])
            for item in bank_payload.get("retrieval", [])
            for selected in item.get("selected", [])
        ]
        scoped_metrics.update(
            {
                "skillbank/singleton_sr": float(bank_metrics["singleton_sr"]),
                "skillbank/no_skill_sr": float(bank_metrics["no_skill_sr"]),
                "skillbank/standalone_delta": float(bank_metrics["standalone_delta"]),
                "skillbank/bank_only_sr": float(bank_metrics["bank_only_sr"]),
                "skillbank/bank_plus_candidate_sr": float(
                    bank_metrics["bank_plus_candidate_sr"]
                ),
                "skillbank/bank_marginal_delta": float(
                    bank_metrics["bank_marginal_delta"]
                ),
                "skillbank/combined_sr_reward": float(
                    bank_metrics["combined_sr_reward"]
                ),
                "skillbank/reward": float(result["skillbank_reward"]),
                "skillbank/reward_weight": self.skillbank_reward_weight,
                "skillbank/bank_size": float(bank_payload["bank_size"]),
                "skillbank/retrieval_mean_cosine": mean(retrieval_scores)
                if retrieval_scores
                else 0.0,
            }
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=float(result["reward"]), **scoped_metrics
        )
        return self._tensor_result(
            resp,
            float(result["terminal_reward"]),
            token_rewards=result["token_rewards"],
            sequence_reward=float(result["sequence_reward"]),
        )


class WebShopSkillEvalWorkflow(WebShopEvaluationMixin, WebShopSkillGRPOWorkflow):
    pass
