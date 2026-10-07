# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import wait as wait_futures
from functools import partial
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, ModelResponse, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import stats_tracker
from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer
from areal.workflow.alfworld_bank import ALFWorldBankMixin
from areal.workflow.alfworld_env_process import EnvironmentProcessError
from areal.workflow.alfworld_environment import (
    _batch_item,
    _invalid_skill_episode,
    _make_env,
    _rollout_timeout_episode,
    _run_rollout_episode_process,
)
from areal.workflow.alfworld_evaluation import ALFWorldEvaluationMixin
from areal.workflow.alfworld_rollout import ALFWorldRolloutMixin
from areal.workflow.alfworld_runtime import (
    _DEFAULT_GROUP_SIMILARITY_MODEL,
    _DEFAULT_SKILL_GENERATION_MAX_RETRIES,
    _NO_SKILL_BASELINE_TEXT,
    _PROMPT_OBSERVATION_CHAR_LIMIT,
    _PROMPT_RESULT_CHAR_LIMIT,
    _ROUND_WAIT_LOG_INTERVAL_S,
    ALFWORLD_TASK_TYPES,
    MIXED_TASK_TYPE,
    PROMPT_CATEGORY_BY_TASK_TYPE,
    _acquire_json_lock,
    _atomic_write_json,
    _collect_games,
    _complete_rollout_paths,
    _default_data_root,
    _default_repo_root,
    _embed_group_similarity_texts,
    _episode_metrics,
    _generation_failure_messages,
    _generation_ready_checkpoint_paths,
    _get_skill_eval_executor,
    _partial_rollout_failure_messages,
    _per_task_type_metrics,
    _read_json,
    _release_json_lock,
    _remove_stale_lock,
    _rollout_worker_tokenizer,
    _round_prompt_prepared_stall_messages,
    _sample_trajectory_pool_episodes,
    _shorten,
    logger,
)
from areal.workflow.alfworld_trajectories import ALFWorldTrajectoryMixin
from areal.workflow.skill_feedback import SkillFeedbackMixin
from areal.workflow.skill_prompts import (
    _normalize_skill_prompt_version,
    _parse_skill_generation,
    _render_trajectory_bundle,
    _skill_generation_messages,
    _skill_xml_token_reward_mask,
    _split_skill_generation_sections,
    _write_skill_manifest,
    skill_payload_to_prompt_text,
)
from areal.workflow.skill_token_feedback import (
    _clip_weighted_tokenwise_mi_rewards,
    _output_token_nll_stats,
)

# Besides the workflows, these names are imported from this module by the
# WebShop workflow, the evaluation and bank-building scripts, and generality_eval.
__all__ = [
    "ALFWorldSkillGRPOWorkflow",
    "ALFWorldSkillEvalWorkflow",
    "PROMPT_CATEGORY_BY_TASK_TYPE",
    "_NO_SKILL_BASELINE_TEXT",
    "_acquire_json_lock",
    "_atomic_write_json",
    "_batch_item",
    "_clip_weighted_tokenwise_mi_rewards",
    "_complete_rollout_paths",
    "_embed_group_similarity_texts",
    "_episode_metrics",
    "_make_env",
    "_output_token_nll_stats",
    "_parse_skill_generation",
    "_per_task_type_metrics",
    "_read_json",
    "_release_json_lock",
    "_render_trajectory_bundle",
    "_rollout_worker_tokenizer",
    "_run_rollout_episode_process",
    "_sample_trajectory_pool_episodes",
    "_shorten",
    "_skill_generation_messages",
    "skill_payload_to_prompt_text",
    "_split_skill_generation_sections",
]


class ALFWorldSkillGRPOWorkflow(
    ALFWorldBankMixin,
    ALFWorldTrajectoryMixin,
    SkillFeedbackMixin,
    ALFWorldRolloutMixin,
    RolloutWorkflow,
):
    """Generate guidance skills, score them with a frozen executor, and train the curator.

    The reward combines standalone success, marginal success against an online
    skillbank, a schema term, and token-wise input/skill mutual information.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        *,
        artifact_dir: str,
        actor_base_url: str,
        actor_model: str = "Qwen/Qwen3.5-4B",
        actor_api_key: str = "",
        actor_timeout_s: float = 120.0,
        actor_temperature: float = 0.0,
        repo_root: str | None = None,
        data_root: str | None = None,
        train_split: str = "train",
        rollouts_per_skill: int = 8,
        max_rollout_steps: int = 50,
        memory_window: int = 5,
        prompt_observation_char_limit: int = _PROMPT_OBSERVATION_CHAR_LIMIT,
        prompt_result_char_limit: int = _PROMPT_RESULT_CHAR_LIMIT,
        rollout_timeout_s: float | None = None,
        skill_eval_workers: int = 0,
        episode_rollout_workers: int = 128,
        progress_summary_interval_s: float = 10.0,
        max_commands: int = 140,
        samples_per_round: int = 8,
        rounds_per_category: int = 5,
        groups_per_round: int = 16,
        train_batch_size: int | None = None,
        trajectory_pool_enabled: bool = True,
        trajectory_pool_initial_size: int = 300,
        trajectory_pool_prompt_episodes: int = 4,
        trajectory_pool_initial_workers: int = 64,
        trajectory_pool_use_previous_step: bool = True,
        trajectory_pool_delete_consumed_step: bool = True,
        trajectory_pool_eval_prompts_per_step: int = 0,
        parallelize_same_step_rounds: bool = False,
        round_barrier_timeout_s: float = 21600.0,
        skill_generation_enable_thinking: bool = False,
        skill_prompt_version: str = "evidence_discovery",
        skill_generation_max_retries: int = _DEFAULT_SKILL_GENERATION_MAX_RETRIES,
        outcome_flattened_scheduler: bool = False,
        outcome_condition_workers: int = 32,
        skill_output_format: str = "skill_xml",
        skill_description_max_words: int = 0,
        fail_fast_actor_error_rate: float = 0.5,
        fail_fast_infra_error_rate: float = 0.5,
        task_types: Sequence[str] | None = None,
        reward_sr_weight: float = 0.2,
        reward_mutual_information_weight: float = 0.0,
        reward_mutual_information_scale: float = 1.0,
        reward_mutual_information_token_margin_clip: float | None = 1.0,
        reward_mutual_information_token_reward_soft_cap: float | None = None,
        reward_mutual_information_token_reward_clip: float | None = None,
        reward_schema_valid_bonus: float = 0.0,
        reward_schema_invalid_penalty: float = -0.1,
        reward_online_skillbank_weight: float = 0.0,
        online_skillbank_bank_weight: float = 0.6,
        online_skillbank_standalone_weight: float = 0.4,
        online_skillbank_top_k: int = 3,
        online_skillbank_seed: int = 23000,
        online_skillbank_warmup_steps: int = 20,
        online_skillbank_warmup_update_type_count: int = 2,
        online_skillbank_later_update_type_count: int = 1,
        online_skillbank_zero_epsilon: float = 0.0,
        online_skillbank_embedding_model: str = _DEFAULT_GROUP_SIMILARITY_MODEL,
        online_skillbank_embedding_batch_size: int = 64,
        online_skillbank_embedding_max_length: int = 256,
        online_skillbank_max_compute_multiplier: float = 2.25,
    ) -> None:
        tokenizer_path = (
            tokenizer
            if isinstance(tokenizer, str)
            else str(getattr(tokenizer, "name_or_path", "") or "")
        )
        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        if rollouts_per_skill <= 0:
            raise ValueError("rollouts_per_skill must be positive")
        if max_rollout_steps <= 0:
            raise ValueError("max_rollout_steps must be positive")
        if samples_per_round <= 0:
            raise ValueError("samples_per_round must be positive")
        if rounds_per_category <= 0:
            raise ValueError("rounds_per_category must be positive")
        if groups_per_round <= 0:
            raise ValueError("groups_per_round must be positive")
        if train_batch_size is None:
            train_batch_size = groups_per_round
        if train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if trajectory_pool_enabled:
            if trajectory_pool_initial_size <= 0:
                raise ValueError("trajectory_pool_initial_size must be positive")
            if trajectory_pool_prompt_episodes <= 0:
                raise ValueError("trajectory_pool_prompt_episodes must be positive")
            if trajectory_pool_initial_workers <= 0:
                raise ValueError("trajectory_pool_initial_workers must be positive")
            if trajectory_pool_eval_prompts_per_step < 0:
                raise ValueError(
                    "trajectory_pool_eval_prompts_per_step must be non-negative"
                )
        if parallelize_same_step_rounds and (not trajectory_pool_enabled):
            raise ValueError(
                "parallelize_same_step_rounds requires trajectory_pool_enabled=true"
            )
        if round_barrier_timeout_s < 0:
            raise ValueError("round_barrier_timeout_s must be non-negative")
        if skill_generation_max_retries < 0:
            raise ValueError("skill_generation_max_retries must be non-negative")
        skill_prompt_version = _normalize_skill_prompt_version(skill_prompt_version)
        if skill_description_max_words < 0:
            raise ValueError("skill_description_max_words must be non-negative")
        skill_output_format = str(skill_output_format or "").strip().lower()
        if skill_output_format != "skill_xml":
            raise ValueError("skill_output_format must be 'skill_xml'")
        if outcome_flattened_scheduler and episode_rollout_workers <= 0:
            raise ValueError(
                "outcome_flattened_scheduler requires episode_rollout_workers > 0"
            )
        if outcome_flattened_scheduler and outcome_condition_workers <= 0:
            raise ValueError("outcome_condition_workers must be positive")
        if skill_eval_workers < 0:
            raise ValueError("skill_eval_workers must be non-negative")
        if episode_rollout_workers < 0:
            raise ValueError("episode_rollout_workers must be non-negative")
        if progress_summary_interval_s < 0:
            raise ValueError("progress_summary_interval_s must be non-negative")
        if prompt_observation_char_limit <= 0:
            raise ValueError("prompt_observation_char_limit must be positive")
        if prompt_result_char_limit <= 0:
            raise ValueError("prompt_result_char_limit must be positive")
        if reward_sr_weight < 0:
            raise ValueError("reward_sr_weight must be non-negative")
        if reward_mutual_information_weight < 0:
            raise ValueError("reward_mutual_information_weight must be non-negative")
        if reward_mutual_information_weight > 0 and groups_per_round < 2:
            # The marginal compares each skill against the other groups' inputs.
            raise ValueError(
                "tokenwise mutual information requires at least two input groups"
            )
        if reward_mutual_information_scale < 0:
            raise ValueError("reward_mutual_information_scale must be non-negative")
        if (
            reward_mutual_information_token_margin_clip is not None
            and reward_mutual_information_token_margin_clip <= 0
        ):
            raise ValueError(
                "reward_mutual_information_token_margin_clip must be positive when enabled"
            )
        if (
            reward_mutual_information_token_reward_soft_cap is not None
            and reward_mutual_information_token_reward_soft_cap <= 0
        ):
            raise ValueError(
                "reward_mutual_information_token_reward_soft_cap must be positive when enabled"
            )
        if (
            reward_mutual_information_token_reward_clip is not None
            and reward_mutual_information_token_reward_clip <= 0
        ):
            raise ValueError(
                "reward_mutual_information_token_reward_clip must be positive when enabled"
            )
        if fail_fast_actor_error_rate < 0:
            raise ValueError("fail_fast_actor_error_rate must be non-negative")
        if fail_fast_infra_error_rate < 0:
            raise ValueError("fail_fast_infra_error_rate must be non-negative")
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path or actor_model
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.artifact_dir = Path(artifact_dir).expanduser()
        self.actor_base_url = actor_base_url
        self.actor_model = actor_model
        self.actor_api_key = actor_api_key
        self.actor_timeout_s = actor_timeout_s
        self.actor_temperature = actor_temperature
        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root
            else _default_repo_root()
        )
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root
            else _default_data_root(self.repo_root)
        )
        self.train_split = train_split
        self.rollouts_per_skill = rollouts_per_skill
        self.max_rollout_steps = max_rollout_steps
        self.memory_window = memory_window
        self.prompt_observation_char_limit = int(prompt_observation_char_limit)
        self.prompt_result_char_limit = int(prompt_result_char_limit)
        self.rollout_timeout_s = (
            float(rollout_timeout_s)
            if rollout_timeout_s is not None and float(rollout_timeout_s) > 0.0
            else None
        )
        self.skill_eval_workers = int(skill_eval_workers)
        self.episode_rollout_workers = int(episode_rollout_workers)
        self.progress_summary_interval_s = float(progress_summary_interval_s)
        self.max_commands = max_commands
        self.samples_per_round = samples_per_round
        self.rounds_per_category = rounds_per_category
        self.groups_per_round = groups_per_round
        self.train_batch_size = int(train_batch_size)
        self.trajectory_pool_enabled = bool(trajectory_pool_enabled)
        self.trajectory_pool_initial_size = int(trajectory_pool_initial_size)
        self.trajectory_pool_prompt_episodes = int(trajectory_pool_prompt_episodes)
        self.trajectory_pool_initial_workers = int(trajectory_pool_initial_workers)
        self.trajectory_pool_use_previous_step = bool(trajectory_pool_use_previous_step)
        self.trajectory_pool_delete_consumed_step = bool(
            trajectory_pool_delete_consumed_step
        )
        self.parallelize_same_step_rounds = bool(parallelize_same_step_rounds)
        self.round_barrier_timeout_s = float(round_barrier_timeout_s)
        self.trajectory_pool_eval_prompts_per_step = int(
            trajectory_pool_eval_prompts_per_step
        )
        self.skill_generation_enable_thinking = bool(skill_generation_enable_thinking)
        self.skill_prompt_version = skill_prompt_version
        self.skill_generation_max_retries = int(skill_generation_max_retries)
        self.outcome_flattened_scheduler = bool(outcome_flattened_scheduler)
        self.outcome_condition_workers = int(outcome_condition_workers)
        self.skill_output_format = skill_output_format
        self.skill_description_max_words = int(skill_description_max_words)
        self.fail_fast_actor_error_rate = float(fail_fast_actor_error_rate)
        self.fail_fast_infra_error_rate = float(fail_fast_infra_error_rate)
        self.task_types = tuple(task_types or ALFWORLD_TASK_TYPES)
        self.reward_sr_weight = float(reward_sr_weight)
        self.reward_mutual_information_weight = float(reward_mutual_information_weight)
        self.reward_mutual_information_scale = float(reward_mutual_information_scale)
        self.reward_mutual_information_token_margin_clip = (
            None
            if reward_mutual_information_token_margin_clip is None
            else float(reward_mutual_information_token_margin_clip)
        )
        self.reward_mutual_information_token_reward_soft_cap = (
            None
            if reward_mutual_information_token_reward_soft_cap is None
            else float(reward_mutual_information_token_reward_soft_cap)
        )
        self.reward_mutual_information_token_reward_clip = (
            None
            if reward_mutual_information_token_reward_clip is None
            else float(reward_mutual_information_token_reward_clip)
        )
        self.reward_schema_valid_bonus = float(reward_schema_valid_bonus)
        self.reward_schema_invalid_penalty = float(reward_schema_invalid_penalty)
        self.reward_online_skillbank_weight = float(reward_online_skillbank_weight)
        self.online_skillbank_bank_weight = float(online_skillbank_bank_weight)
        self.online_skillbank_standalone_weight = float(
            online_skillbank_standalone_weight
        )
        self.online_skillbank_top_k = int(online_skillbank_top_k)
        self.online_skillbank_seed = int(online_skillbank_seed)
        self.online_skillbank_warmup_steps = int(online_skillbank_warmup_steps)
        self.online_skillbank_warmup_update_type_count = int(
            online_skillbank_warmup_update_type_count
        )
        self.online_skillbank_later_update_type_count = int(
            online_skillbank_later_update_type_count
        )
        self.online_skillbank_zero_epsilon = float(online_skillbank_zero_epsilon)
        self.online_skillbank_embedding_model = str(online_skillbank_embedding_model)
        self.online_skillbank_embedding_batch_size = int(
            online_skillbank_embedding_batch_size
        )
        self.online_skillbank_embedding_max_length = int(
            online_skillbank_embedding_max_length
        )
        self.online_skillbank_max_compute_multiplier = float(
            online_skillbank_max_compute_multiplier
        )
        self._games_cache: dict[str, list[dict[str, str]]] = {}
        self._games_cache_lock = threading.RLock()
        self._validate_online_skillbank_settings()

    def _validate_online_skillbank_settings(self) -> None:
        """Check the ALFWorld online-skillbank reward contract.

        Subclasses that reuse this workflow with their own skillbank (WebShop)
        override this with their own checks.
        """

        task_types = self.task_types
        if self.episode_rollout_workers <= 0:
            raise ValueError("episode_rollout_workers must be positive")
        if not self.trajectory_pool_enabled:
            raise ValueError(
                "online_skillbank requires trajectory_pool_enabled=true so every "
                "optimizer batch can use concrete balanced input types"
            )
        if MIXED_TASK_TYPE in task_types or len(set(task_types)) != len(task_types):
            raise ValueError(
                "online_skillbank task_types must be unique concrete task types"
            )
        if self.train_batch_size < len(task_types):
            raise ValueError(
                "online_skillbank train batch must contain every input task type"
            )
        if self.reward_online_skillbank_weight < 0.0:
            raise ValueError("reward_online_skillbank_weight must be non-negative")
        if self.online_skillbank_bank_weight < 0.0:
            raise ValueError("online_skillbank_bank_weight must be non-negative")
        if self.online_skillbank_standalone_weight < 0.0:
            raise ValueError("online_skillbank_standalone_weight must be non-negative")
        if (
            self.online_skillbank_bank_weight + self.online_skillbank_standalone_weight
            <= 0.0
        ):
            raise ValueError("online skillbank marginal weights must be positive")
        if self.online_skillbank_top_k <= 0:
            raise ValueError("online_skillbank_top_k must be positive")
        if self.online_skillbank_warmup_steps < 0:
            raise ValueError("online_skillbank_warmup_steps must be non-negative")
        if not 0 <= self.online_skillbank_warmup_update_type_count <= len(task_types):
            raise ValueError(
                "online_skillbank_warmup_update_type_count is out of range"
            )
        if not 0 <= self.online_skillbank_later_update_type_count <= len(task_types):
            raise ValueError("online_skillbank_later_update_type_count is out of range")
        if self.online_skillbank_zero_epsilon < 0.0:
            raise ValueError("online_skillbank_zero_epsilon must be non-negative")
        if not self.online_skillbank_embedding_model.strip():
            raise ValueError("online_skillbank_embedding_model must not be empty")
        if self.online_skillbank_embedding_batch_size <= 0:
            raise ValueError("online_skillbank_embedding_batch_size must be positive")
        if self.online_skillbank_embedding_max_length <= 0:
            raise ValueError("online_skillbank_embedding_max_length must be positive")
        if self.online_skillbank_max_compute_multiplier <= 0.0:
            raise ValueError("online_skillbank_max_compute_multiplier must be positive")
        # Non-empty bank: no-skill + retrieved conditions plus a singleton and an
        # augmented condition per sample, relative to one rollout set per sample.
        base_episodes = self.samples_per_round * self.rollouts_per_skill
        online_episodes = 2 * self.rollouts_per_skill + 2 * base_episodes
        compute_multiplier = online_episodes / base_episodes
        if compute_multiplier > self.online_skillbank_max_compute_multiplier + 1e-12:
            raise ValueError(
                "online_skillbank environment cost exceeds configured maximum: "
                f"{online_episodes}/{base_episodes}={compute_multiplier:.4f} > "
                f"{self.online_skillbank_max_compute_multiplier:.4f}"
            )

    def _requires_previous_round_barrier(self, data: dict[str, Any]) -> bool:
        round_in_category = int(data["round_in_category"])
        if round_in_category <= 0:
            return False
        if not (
            bool(getattr(self, "trajectory_pool_enabled", False))
            and bool(getattr(self, "parallelize_same_step_rounds", False))
        ):
            return True
        groups_per_round = int(data.get("groups_per_round", self.groups_per_round))
        group_index = int(data.get("group_index", 0))
        current_global_group = round_in_category * groups_per_round + group_index
        previous_round_last_group = round_in_category * groups_per_round - 1
        return (
            current_global_group // self.train_batch_size
            != previous_round_last_group // self.train_batch_size
        )

    def _prune_shared_training_tasks(self, current_step: int) -> None:
        """Bound completed single-flight tasks to the current and previous step."""

        keep_from_step = max(0, int(current_step) - 1)
        tasks = getattr(self, "_online_skillbank_group_tasks", None)
        if not tasks:
            return
        for key, task in list(tasks.items()):
            if not task.done():
                continue
            _, round_in_category, group_index = key
            task_step = self._training_step_index(
                int(round_in_category), int(group_index)
            )
            if task_step < keep_from_step:
                tasks.pop(key, None)

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor] | None:
        self._raise_if_workflow_failed()
        try:
            return await self._arun_episode(engine, data)
        except Exception as exc:
            error = self._diagnostic_text(repr(exc))
            try:
                failure_path = self.artifact_dir / "diagnostics/workflow_failure.json"
                if not failure_path.exists():
                    await asyncio.to_thread(
                        _atomic_write_json,
                        failure_path,
                        {
                            "status": "fatal",
                            "error": error,
                            "traceback": self._diagnostic_text(traceback.format_exc()),
                            "updated_at": time.time(),
                        },
                    )
            except Exception:
                logger.error("could not persist ALFWorld failure diagnostic")
            logger.error("ALFWORLD_WORKFLOW_FATAL: %s", error)
            raise

    def _diagnostic_text(self, value: str) -> str:
        # Exceptions may embed request URLs or credentials; publish only bounded,
        # redacted text, never the request payload, environment, or actor config.
        for secret in (
            getattr(self, "actor_api_key", ""),
            os.environ.get("WANDB_API_KEY", ""),
        ):
            if secret:
                value = value.replace(secret, "<redacted>")
        value = re.sub(r"https?://\S+", "<url>", value)
        value = re.sub(r"wandb_v1_[A-Za-z0-9_-]+", "<redacted>", value)
        value = re.sub(
            r"(?i)(authorization|api[_-]?key|token|password)([\"']?\s*[:=]\s*)[^\s,}]+",
            r"\1\2<redacted>",
            value,
        )
        return value[-8192:]

    def _raise_if_workflow_failed(self) -> None:
        artifact_dir = getattr(self, "artifact_dir", None)
        if (
            artifact_dir is not None
            and (Path(artifact_dir) / "diagnostics/workflow_failure.json").exists()
        ):
            raise RuntimeError(
                "ALFWORLD_WORKFLOW_FATAL: a peer failed; see diagnostics/workflow_failure.json"
            )

    async def _arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        task_type = str(data.get("task_type") or MIXED_TASK_TYPE)
        round_in_category = int(data["round_in_category"])
        if self._requires_previous_round_barrier(data):
            groups_per_round = int(data.get("groups_per_round", self.groups_per_round))
            await self._wait_for_previous_round_async(
                self._round_dir(task_type, round_in_category - 1),
                expected_count=self.samples_per_round * groups_per_round,
            )
        try:
            prepared = await asyncio.to_thread(self._prepare_sample, data)
        except Exception as exc:
            try:
                await asyncio.to_thread(self._write_preparation_failure, data, exc)
            except Exception:
                logger.exception(
                    "failed to publish ALFWorld prompt preparation failure"
                )
            raise
        try:
            input_ids = apply_chat_template(
                self.tokenizer,
                prepared["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.skill_generation_enable_thinking,
            )
            generation_attempts: list[dict[str, Any]] = []
            resp: ModelResponse | None = None
            raw = ""
            max_retries = self._max_skill_generation_retries()
            max_attempts = max_retries + 1
            for attempt_index in range(max_attempts):
                req = ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=input_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
                resp = await engine.agenerate(req)
                raw = self.tokenizer.decode(
                    resp.output_tokens, skip_special_tokens=False
                )
                parsed, schema_valid, schema_error, repair_notes = (
                    _parse_skill_generation(
                        raw,
                        str(prepared["prompt_category"]),
                        skill_output_format=self.skill_output_format,
                        max_words=int(getattr(self, "skill_description_max_words", 0)),
                    )
                )
                thinking_text, answer_text, section_notes = (
                    _split_skill_generation_sections(raw)
                )
                output_entropy_proxy, output_entropy_token_count = (
                    _output_token_nll_stats(resp.output_logprobs)
                )
                generation_attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "raw": raw,
                        "thinking": thinking_text,
                        "answer": answer_text,
                        "parsed": parsed,
                        "schema_valid": schema_valid,
                        "schema_error": schema_error,
                        "repair_notes": section_notes + repair_notes,
                        "input_token_count": len(resp.input_tokens),
                        "output_token_count": len(resp.output_tokens),
                        "output_entropy_proxy": output_entropy_proxy,
                        "output_entropy_token_count": output_entropy_token_count,
                        "updated_at": time.time(),
                    }
                )
                await asyncio.to_thread(
                    self._write_generation_attempts, prepared, generation_attempts
                )
                if schema_valid:
                    break
            if resp is None:
                raise RuntimeError("skill generator did not return a response")
            await asyncio.to_thread(
                self._write_training_generation,
                prepared,
                raw,
                list(resp.input_tokens),
                list(resp.output_tokens),
                generation_attempts,
            )
        except Exception as exc:
            await asyncio.to_thread(self._write_generation_failure, prepared, exc)
            raise
        optimized_outcome = bool(getattr(self, "outcome_flattened_scheduler", False))
        outcome_waiter: asyncio.Task | None = None
        if optimized_outcome:
            await self._wait_for_group_generation_async(
                self._round_dir(task_type, round_in_category),
                group_index=int(prepared.get("group_index", 0)),
                expected_count=self.samples_per_round,
            )
            outcome_waiter = asyncio.create_task(
                self._shared_online_skillbank_group_index(
                    task_type=task_type,
                    round_in_category=round_in_category,
                    group_index=int(prepared.get("group_index", 0)),
                    games=list(prepared.get("selected_games") or []),
                )
            )
        round_generation_kwargs: dict[str, Any] = {
            "expected_count": self.samples_per_round
            * int(prepared.get("groups_per_round", self.groups_per_round))
        }
        if optimized_outcome:
            round_generation_kwargs["canonical_group_count"] = int(
                prepared.get("groups_per_round", self.groups_per_round)
            )
        await self._wait_for_round_generation_async(
            self._round_dir(task_type, round_in_category), **round_generation_kwargs
        )
        await asyncio.to_thread(
            self._write_training_step_skill_manifest,
            task_type,
            round_in_category,
            int(prepared.get("group_index", 0)),
        )
        precomputed_outcome_index = (
            await outcome_waiter if outcome_waiter is not None else None
        )
        rollout_result = await self._run_skill_eval_blocking(
            self._evaluate_and_save,
            prepared,
            raw,
            list(resp.input_tokens),
            list(resp.output_tokens),
            generation_attempts,
            engine,
            precomputed_outcome_index,
        )
        reward = float(rollout_result["reward"])
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=reward,
            alfworld_reward=reward,
            alfworld_task_reward=float(rollout_result["sr"]),
            alfworld_baseline_sr=float(rollout_result["baseline_sr"]),
            alfworld_baseline_delta_sr=float(rollout_result["baseline_delta_sr"]),
            alfworld_sr_reward=float(rollout_result["sr_reward"]),
            alfworld_mutual_information_score=float(
                rollout_result.get("mutual_information_score", 0.0)
            ),
            alfworld_mutual_information_raw_score=float(
                rollout_result.get(
                    "mutual_information_raw_score",
                    rollout_result.get("mutual_information_score", 0.0),
                )
            ),
            alfworld_mutual_information_reward=float(
                rollout_result.get("mutual_information_reward", 0.0)
            ),
            alfworld_mi_reward_abs_fraction=float(
                rollout_result.get("mutual_information_reward_abs_fraction", 0.0)
            ),
            alfworld_mi_token_reward_mean=float(
                rollout_result.get("mutual_information_token_reward_mean", 0.0)
            ),
            alfworld_mi_token_reward_count=float(
                rollout_result.get("mutual_information_token_reward_count", 0.0)
            ),
            alfworld_mi_token_reward_min=float(
                rollout_result.get("mutual_information_token_reward_min", 0.0)
            ),
            alfworld_mi_token_reward_max=float(
                rollout_result.get("mutual_information_token_reward_max", 0.0)
            ),
            alfworld_mi_token_unsmoothed_reward_mean=float(
                rollout_result.get(
                    "mutual_information_token_unsmoothed_reward_mean", 0.0
                )
            ),
            alfworld_mi_token_reward_compression_ratio=float(
                rollout_result.get(
                    "mutual_information_token_reward_compression_ratio", 1.0
                )
            ),
            alfworld_mi_token_reward_soft_cap=float(
                rollout_result.get("mutual_information_token_reward_soft_cap", 0.0)
            ),
            alfworld_mi_token_reward_clip=float(
                rollout_result.get("mutual_information_token_reward_clip", 0.0) or 0.0
            ),
            alfworld_mi_token_final_clipped_fraction=float(
                rollout_result.get(
                    "mutual_information_token_final_clipped_fraction", 0.0
                )
            ),
            alfworld_mi_token_unclipped_reward_mean=float(
                rollout_result.get(
                    "mutual_information_token_unclipped_reward_mean", 0.0
                )
            ),
            alfworld_mi_token_absolute_mode=float(
                bool(
                    rollout_result.get("mutual_information_token_absolute_mode", False)
                )
            ),
            alfworld_mi_abs_token_margin=float(
                rollout_result.get("mutual_information_abs_token_margin", 0.0)
            ),
            alfworld_mi_positive_token_fraction=float(
                rollout_result.get("mutual_information_positive_token_fraction", 0.0)
            ),
            alfworld_mi_clipped_token_fraction=float(
                rollout_result.get("mutual_information_clipped_token_fraction", 0.0)
            ),
            alfworld_mi_nll_margin=float(
                rollout_result.get("mutual_information_nll_margin", 0.0)
            ),
            alfworld_output_entropy_proxy=output_entropy_proxy,
            alfworld_schema_reward=float(rollout_result["schema_reward"]),
            alfworld_sr=float(rollout_result["sr"]),
            alfworld_wins=int(rollout_result["wins"]),
            alfworld_rollouts=int(rollout_result["n_rollouts"]),
            alfworld_schema_valid=float(bool(rollout_result["schema_valid"])),
            alfworld_generation_attempts=int(rollout_result["attempt_count"]),
            **{
                "online_skillbank/no_skill_sr": float(
                    rollout_result.get("outcome", {}).get("no_skill_sr", 0.0)
                ),
                "online_skillbank/singleton_sr": float(
                    rollout_result.get("outcome", {}).get("singleton_sr", 0.0)
                ),
                "online_skillbank/retrieved_sr": float(
                    rollout_result.get("outcome", {}).get("retrieved_skill_sr", 0.0)
                ),
                "online_skillbank/augmented_sr": float(
                    rollout_result.get("outcome", {}).get("augmented_sr", 0.0)
                ),
                "online_skillbank/bank_delta": float(
                    rollout_result.get("outcome", {}).get("bank_delta", 0.0)
                ),
                "online_skillbank/standalone_delta": float(
                    rollout_result.get("outcome", {}).get("standalone_delta", 0.0)
                ),
                "online_skillbank/weighted_marginal": float(
                    rollout_result.get("outcome", {}).get(
                        "weighted_marginal_score", 0.0
                    )
                ),
                "online_skillbank/positive_marginal": float(
                    rollout_result.get("outcome", {}).get(
                        "positive_marginal_indicator", 0.0
                    )
                ),
                "online_skillbank/schema_eligible": float(
                    rollout_result.get("outcome", {}).get("marginal_eligible", 0.0)
                ),
                "online_skillbank/reward": float(
                    rollout_result.get("outcome_reward", 0.0)
                ),
                "online_skillbank/bank_size": float(
                    rollout_result.get("outcome", {}).get("bank_before_size", 0)
                ),
                "online_skillbank/planned_compute_multiplier": float(
                    rollout_result.get("outcome", {}).get(
                        "planned_compute_multiplier", 0.0
                    )
                ),
                "online_skillbank/actual_compute_multiplier": float(
                    rollout_result.get("outcome", {}).get(
                        "actual_environment_compute_multiplier", 0.0
                    )
                ),
            },
        )
        return self._tensor_result(
            resp,
            float(rollout_result.get("terminal_reward", reward)),
            token_rewards=list(rollout_result.get("token_rewards", [])),
            sequence_reward=float(rollout_result.get("sequence_reward", reward)),
        )

    def _tensor_result(
        self,
        resp: ModelResponse,
        reward: float,
        *,
        token_rewards: Sequence[float] | None = None,
        sequence_reward: float | None = None,
    ) -> dict[str, torch.Tensor]:
        seq = list(resp.input_tokens) + list(resp.output_tokens)
        logprobs = [0.0] * resp.input_len + list(resp.output_logprobs)
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + list(resp.output_versions)
        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        if token_rewards is not None:
            values = [float(value) for value in token_rewards]
            if len(values) > resp.output_len:
                raise ValueError(
                    "token reward count exceeds generated output token count"
                )
            aligned = (
                [0.0] * resp.input_len
                + values
                + [0.0] * (resp.output_len - len(values))
            )
            res["token_rewards"] = torch.tensor(aligned, dtype=torch.float32)
            res["sequence_rewards"] = torch.tensor(
                reward if sequence_reward is None else sequence_reward,
                dtype=torch.float32,
            )
        return {key: value.unsqueeze(0) for key, value in res.items()}

    def _max_skill_generation_retries(self) -> int:
        return max(0, int(getattr(self, "skill_generation_max_retries", 0)))

    @staticmethod
    def _group_index_from_sample_key(sample_key: str) -> int:
        match = re.search(r"group_(\d+)", sample_key)
        return int(match.group(1)) if match else -1

    async def _run_skill_eval_blocking(
        self, fn: Callable[..., dict[str, Any]], *args: Any
    ) -> dict[str, Any]:
        executor = _get_skill_eval_executor(self.skill_eval_workers)
        if executor is None:
            return await asyncio.to_thread(fn, *args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, partial(fn, *args))

    def _collect_rollout_futures(
        self,
        *,
        futures: dict[Any, int],
        games: Sequence[dict[str, Any]],
        skill_name: str,
        episodes: list[dict[str, Any] | None],
        current_episodes: dict[int, dict[str, Any]],
        progress_lock: threading.Lock,
        progress_closed: threading.Event,
        write_progress_unlocked: Callable[..., None],
        context: str,
    ) -> int:
        pending = set(futures)
        deadline = (
            time.monotonic() + self.rollout_timeout_s
            if self.rollout_timeout_s is not None
            else None
        )
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
                rollout_index = futures[future]
                try:
                    episode = future.result()
                except EnvironmentProcessError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    episode = _invalid_skill_episode(
                        game=games[rollout_index],
                        skill_name=skill_name,
                        error="rollout_exception: " + repr(exc),
                    )
                    episode["status"] = "rollout_error"
                    episode["traceback"] = traceback.format_exc()
                with progress_lock:
                    episodes[rollout_index] = episode
                    current_episodes.pop(rollout_index, None)
                    write_progress_unlocked(rollout_index=rollout_index)

        if not pending:
            return 0

        timeout_text = (
            f"{self.rollout_timeout_s:.1f}s"
            if self.rollout_timeout_s is not None
            else "the configured deadline"
        )
        logger.warning(
            "%s timed out %d/%d ALFWorld rollouts after %s; marking them failed",
            context,
            len(pending),
            len(futures),
            timeout_text,
        )
        progress_closed.set()
        for future in pending:
            rollout_index = futures[future]
            future.cancel()
            with progress_lock:
                partial_episode = current_episodes.pop(rollout_index, None)
                episodes[rollout_index] = _rollout_timeout_episode(
                    game=games[rollout_index],
                    skill_name=skill_name,
                    error=f"rollout timed out after {timeout_text}",
                    timeout_s=self.rollout_timeout_s,
                    partial_episode=partial_episode,
                )
                write_progress_unlocked(rollout_index=rollout_index)
        return len(pending)

    def _round_barrier_deadline(self) -> float | None:
        timeout_s = float(getattr(self, "round_barrier_timeout_s", 0.0) or 0.0)
        return time.monotonic() + timeout_s if timeout_s > 0.0 else None

    def _missing_barrier_sample_names(
        self,
        paths: Sequence[Path],
        *,
        expected: int,
        canonical_names: set[str] | None = None,
    ) -> list[str]:
        if canonical_names is None:
            canonical_names = {
                f"group_{group_index:02d}_sample_{sample_index:02d}"
                for group_index in range(
                    math.ceil(expected / max(1, self.samples_per_round))
                )
                for sample_index in range(self.samples_per_round)
            }
        observed = {path.parent.name for path in paths}
        return sorted(canonical_names - observed)[:32]

    async def _wait_for_previous_round_async(
        self, round_dir: Path, *, expected_count: int | None = None
    ) -> list[Path]:
        expected = expected_count or self.samples_per_round
        deadline = self._round_barrier_deadline()
        last_log = 0.0
        while True:
            self._raise_if_workflow_failed()
            complete_paths = _complete_rollout_paths(round_dir)
            if len(complete_paths) >= expected:
                return complete_paths
            failures = _partial_rollout_failure_messages(
                round_dir,
                actor_error_threshold=self.fail_fast_actor_error_rate,
                infra_error_threshold=self.fail_fast_infra_error_rate,
            )
            failures.extend(
                _round_prompt_prepared_stall_messages(
                    round_dir,
                    expected=expected,
                    complete_count=len(complete_paths),
                )
            )
            if failures:
                raise RuntimeError(
                    "ALFWorld rollout infrastructure failure while waiting for round "
                    f"{round_dir}: {failures[:5]}"
                )
            if deadline is not None and time.monotonic() >= deadline:
                missing = self._missing_barrier_sample_names(
                    complete_paths, expected=expected
                )
                raise TimeoutError(
                    "timed out waiting for previous ALFWorld skill round: "
                    f"round={round_dir}, complete={len(complete_paths)}/{expected}, "
                    f"missing_samples={missing}"
                )
            now = time.time()
            if now - last_log >= _ROUND_WAIT_LOG_INTERVAL_S:
                logger.info(
                    "waiting for previous ALFWorld skill round %s: %s/%s rollouts complete",
                    round_dir,
                    len(complete_paths),
                    expected,
                )
                last_log = now
            await asyncio.sleep(5.0)

    async def _wait_for_round_generation_async(
        self,
        round_dir: Path,
        *,
        expected_count: int,
        canonical_group_count: int | None = None,
    ) -> list[Path]:
        expected = expected_count or self.samples_per_round
        canonical_names = (
            {
                f"group_{group_index:02d}_sample_{sample_index:02d}"
                for group_index in range(canonical_group_count)
                for sample_index in range(self.samples_per_round)
            }
            if canonical_group_count is not None
            else None
        )
        deadline = self._round_barrier_deadline()
        last_log = 0.0
        while True:
            self._raise_if_workflow_failed()
            ready_paths = _generation_ready_checkpoint_paths(round_dir)
            if canonical_names is not None:
                ready_paths = [
                    path for path in ready_paths if path.parent.name in canonical_names
                ]
            failures = _generation_failure_messages(round_dir)
            if failures:
                raise RuntimeError(
                    "ALFWorld skill generation failed while waiting for round "
                    f"{round_dir}: {failures[:5]}"
                )
            if len(ready_paths) >= expected:
                return ready_paths
            if deadline is not None and time.monotonic() >= deadline:
                missing = self._missing_barrier_sample_names(
                    ready_paths,
                    expected=expected,
                    canonical_names=canonical_names,
                )
                raise TimeoutError(
                    "timed out waiting for ALFWorld skill generation: "
                    f"round={round_dir}, ready={len(ready_paths)}/{expected}, "
                    f"missing_samples={missing}"
                )
            now = time.time()
            if now - last_log >= _ROUND_WAIT_LOG_INTERVAL_S:
                logger.info(
                    "waiting for ALFWorld skill generation phase %s: %s/%s skills generated",
                    round_dir,
                    len(ready_paths),
                    expected,
                )
                last_log = now
            await asyncio.sleep(2.0)

    async def _wait_for_group_generation_async(
        self,
        round_dir: Path,
        *,
        group_index: int,
        expected_count: int,
    ) -> list[Path]:
        """Wait only for the samples needed by one outcome-cycle coordinator."""

        expected = expected_count or self.samples_per_round
        canonical_names = {
            f"group_{group_index:02d}_sample_{sample_index:02d}"
            for sample_index in range(expected)
        }
        deadline = self._round_barrier_deadline()
        last_log = 0.0
        while True:
            self._raise_if_workflow_failed()
            ready_paths = [
                path
                for path in _generation_ready_checkpoint_paths(round_dir)
                if path.parent.name in canonical_names
            ]
            failures = _generation_failure_messages(round_dir)
            if failures:
                raise RuntimeError(
                    "ALFWorld skill generation failed while waiting for group "
                    f"{group_index} in {round_dir}: {failures[:5]}"
                )
            if len(ready_paths) >= expected:
                return ready_paths
            if deadline is not None and time.monotonic() >= deadline:
                missing = self._missing_barrier_sample_names(
                    ready_paths,
                    expected=expected,
                    canonical_names=canonical_names,
                )
                raise TimeoutError(
                    "timed out waiting for ALFWorld group generation: "
                    f"round={round_dir}, group={group_index}, "
                    f"ready={len(ready_paths)}/{expected}, "
                    f"missing_samples={missing}"
                )
            now = time.time()
            if now - last_log >= _ROUND_WAIT_LOG_INTERVAL_S:
                logger.info(
                    "waiting for ALFWorld group generation %s/group_%02d: %s/%s skills generated",
                    round_dir,
                    group_index,
                    len(ready_paths),
                    expected,
                )
                last_log = now
            await asyncio.sleep(0.5)

    def _training_step_index(self, round_in_category: int, group_index: int) -> int:
        global_group_index = round_in_category * self.groups_per_round + group_index
        return global_group_index // self.train_batch_size

    def _training_step_generation_paths(
        self, task_type: str, training_global_step: int
    ) -> tuple[list[Path], list[int]]:
        start_group = training_global_step * self.train_batch_size
        stop_group = start_group + self.train_batch_size
        generation_paths: list[Path] = []
        source_rounds: set[int] = set()
        for global_group_index in range(start_group, stop_group):
            round_index, group_index = divmod(global_group_index, self.groups_per_round)
            source_rounds.add(round_index)
            round_dir = self._round_dir(task_type, round_index)
            generation_paths.extend(
                sorted(
                    (round_dir / "skills").glob(
                        f"group_{group_index:02d}_*/generation.json"
                    )
                )
            )
        return generation_paths, sorted(source_rounds)

    def _write_training_step_skill_manifest(
        self, task_type: str, round_in_category: int, group_index: int
    ) -> dict[str, Any]:
        training_global_step = self._training_step_index(round_in_category, group_index)
        generation_paths, source_rounds = self._training_step_generation_paths(
            task_type, training_global_step
        )
        manifest_path = (
            self.artifact_dir
            / "train"
            / f"globalstep_{training_global_step:06d}"
            / "skill.json"
        )
        return _write_skill_manifest(
            manifest_path=manifest_path,
            generation_paths=generation_paths,
            mode="train",
            global_step=training_global_step,
            expected_skill_count=self.train_batch_size * self.samples_per_round,
            metadata={
                "task_type": task_type,
                "train_batch_size": self.train_batch_size,
                "groups_per_round": self.groups_per_round,
                "samples_per_input": self.samples_per_round,
                "source_rounds": source_rounds,
            },
        )

    def _write_generation_attempts(
        self, prepared: dict[str, Any], attempts: list[dict[str, Any]]
    ) -> None:
        sample_dir = Path(prepared["sample_dir"])
        latest = attempts[-1] if attempts else {}
        schema_valid = bool(latest.get("schema_valid"))
        max_retries = self._max_skill_generation_retries()
        exhausted = len(attempts) > max_retries
        status = (
            "generation_valid"
            if schema_valid
            else "generation_exhausted"
            if exhausted
            else "generation_retrying"
        )
        payload = {
            "task_type": prepared["task_type"],
            "prompt_category": prepared["prompt_category"],
            "round_in_category": prepared["round_in_category"],
            "group_index": prepared.get("group_index", 0),
            "sample_key": prepared["sample_key"],
            "attempt_count": len(attempts),
            "max_retries": max_retries,
            "schema_valid": schema_valid,
            "schema_error": latest.get("schema_error", ""),
            "attempts": attempts,
            "updated_at": time.time(),
        }
        _atomic_write_json(sample_dir / "generation_attempts.json", payload)
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": status,
                "task_type": prepared["task_type"],
                "round_in_category": prepared["round_in_category"],
                "group_index": prepared.get("group_index", 0),
                "sample_key": prepared["sample_key"],
                "schema_valid": schema_valid,
                "attempt_count": len(attempts),
                "max_retries": max_retries,
                "updated_at": time.time(),
            },
        )

    def _write_generation_failure(
        self, prepared: dict[str, Any], exc: Exception
    ) -> None:
        sample_dir = Path(prepared["sample_dir"])
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "generation_failed",
                "task_type": prepared["task_type"],
                "round_in_category": prepared["round_in_category"],
                "group_index": prepared.get("group_index", 0),
                "sample_key": prepared["sample_key"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": time.time(),
            },
        )

    def _write_preparation_failure(self, data: dict[str, Any], exc: Exception) -> None:
        """Publish a failure visible to siblings even before prompt preparation."""

        task_type = str(data.get("task_type") or MIXED_TASK_TYPE)
        round_in_category = int(data["round_in_category"])
        group_index = int(data.get("group_index", 0))
        failure_dir = self._round_dir(task_type, round_in_category) / "failures"
        failure_path = (
            failure_dir / f"group_{group_index:02d}_{uuid.uuid4().hex[:12]}.json"
        )
        _atomic_write_json(
            failure_path,
            {
                "status": "preparation_failed",
                "task_type": task_type,
                "round_in_category": round_in_category,
                "group_index": group_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": time.time(),
            },
        )

    def _write_training_generation(
        self,
        prepared: dict[str, Any],
        raw_generation: str,
        input_tokens: list[int],
        output_tokens: list[int],
        generation_attempts: list[dict[str, Any]],
    ) -> None:
        max_retries = self._max_skill_generation_retries()
        sample_dir = Path(prepared["sample_dir"])
        task_type = str(prepared["task_type"])
        input_task_type = str(prepared.get("input_task_type") or task_type)
        sample_key = str(prepared["sample_key"])
        round_in_category = int(prepared["round_in_category"])
        group_index = int(prepared.get("group_index", 0))
        skill_name = (
            f"skill/{task_type}/round_{round_in_category:04d}/"
            f"group_{group_index:02d}/{sample_key}"
        )
        final_attempt = generation_attempts[-1] if generation_attempts else {}
        parsed = final_attempt.get("parsed")
        parsed = parsed if isinstance(parsed, dict) else None
        valid = bool(final_attempt.get("schema_valid"))
        parse_error = str(final_attempt.get("schema_error") or "")
        repair_notes = list(final_attempt.get("repair_notes") or [])
        games = list(
            prepared.get("selected_games")
            or self._select_reward_games(
                input_task_type, round_in_category, group_index
            )
        )
        mi_target_tokens = self._mi_target_tokens(output_tokens)
        mi_token_reward_mask = (
            _skill_xml_token_reward_mask(
                self.tokenizer, raw_generation, mi_target_tokens
            )
            if valid
            else [0] * len(mi_target_tokens)
        )
        _atomic_write_json(
            sample_dir / "generation.json",
            {
                "skill_name": skill_name,
                "input_task_type": input_task_type,
                "raw": raw_generation,
                "parsed": parsed,
                "schema_valid": valid,
                "schema_error": parse_error,
                "repair_notes": repair_notes,
                "attempt_count": len(generation_attempts),
                "max_retries": max_retries,
                "attempts": generation_attempts,
                "input_token_count": len(input_tokens),
                "input_tokens": [int(token_id) for token_id in input_tokens],
                "output_token_count": len(output_tokens),
                "output_tokens": [int(token_id) for token_id in output_tokens],
                "mi_target_tokens": mi_target_tokens,
                "mi_token_reward_mask": mi_token_reward_mask,
                "group_index": group_index,
                "selected_games": games,
                "updated_at": time.time(),
            },
        )
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "generation_complete",
                "task_type": task_type,
                "round_in_category": round_in_category,
                "group_index": group_index,
                "sample_key": sample_key,
                "schema_valid": valid,
                "attempt_count": len(generation_attempts),
                "max_retries": max_retries,
                "updated_at": time.time(),
            },
        )

    def _round_dir(self, task_type: str, round_in_category: int) -> Path:
        return (
            self.artifact_dir
            / "categories"
            / task_type
            / f"round_{max(0, round_in_category):04d}"
        )

    def _claim_sample_slot(
        self, task_type: str, round_in_category: int, group_index: int = 0
    ) -> tuple[str, Path]:
        round_dir = self._round_dir(task_type, round_in_category)
        slots_dir = round_dir / "slots"
        slots_dir.mkdir(parents=True, exist_ok=True)
        for index in range(self.samples_per_round):
            sample_key = f"group_{group_index:02d}_sample_{index:02d}"
            sample_dir = round_dir / "skills" / sample_key
            checkpoint_path = sample_dir / "checkpoint.json"
            if checkpoint_path.exists():
                try:
                    checkpoint = _read_json(checkpoint_path)
                    if checkpoint.get("status") == "complete":
                        continue
                except Exception:
                    pass
            lock_path = slots_dir / f"{sample_key}.lock"
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not _remove_stale_lock(lock_path, foreign_host_stale_after_s=120.0):
                    continue
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
            with os.fdopen(fd, "w") as handle:
                handle.write(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "claimed_at": time.time(),
                        }
                    )
                )
            sample_dir.mkdir(parents=True, exist_ok=True)
            return sample_key, sample_dir

        # The group coordinator compares exactly samples_per_round canonical slots.
        raise RuntimeError(
            "online_skillbank could not claim a canonical sample slot in "
            f"{round_dir} for group {group_index}"
        )

    def _games_for_task(self, task_type: str) -> list[dict[str, str]]:
        with self._get_games_cache_lock():
            cached = self._games_cache.get(task_type)
            if cached is not None:
                return cached
            games = _collect_games(self.data_root, self.train_split, task_type)
            self._games_cache[task_type] = games
            return games

    def _get_games_cache_lock(self) -> threading.RLock:
        lock = getattr(self, "_games_cache_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._games_cache_lock = lock
        return lock

    def _select_reward_games(
        self,
        input_task_type: str,
        round_in_category: int,
        group_index: int = 0,
    ) -> list[dict[str, str]]:
        """Score every sample in a GRPO group on the same games of its input type."""

        if input_task_type == MIXED_TASK_TYPE:
            raise ValueError("reward game sampling requires a concrete input task type")
        games = self._games_for_task(input_task_type)
        if len(games) <= self.rollouts_per_skill:
            return list(games)
        rng = random.Random(700000 + round_in_category * 100000 + group_index * 1000)
        return rng.sample(games, k=self.rollouts_per_skill)


class ALFWorldSkillEvalWorkflow(ALFWorldEvaluationMixin, ALFWorldSkillGRPOWorkflow):
    pass
