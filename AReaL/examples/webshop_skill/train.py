# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from examples.skill_training.actor_server import frozen_actor_server
from examples.skill_training.checkpoint_retention import EvalCheckpointRetention
from examples.skill_training.gpu_layout import validate_skill_gpu_layout
from examples.skill_training.recovery import reconcile_artifacts_from_recover
from examples.skill_training.train_utils import (
    ensure_logp_mb_capacity,
    eval_baseline_workflow_kwargs,
    resolve_service_benchmark_layout,
    reward_workflow_kwargs,
)
from examples.webshop_skill.configs import WebShopSkillGRPOConfig
from examples.webshop_skill.task_split import (
    task_selection_contract,
    validate_official_manifest,
)

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config, save_config
from areal.dataset import get_custom_dataset
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("WebShopSkillTrain")


def _preflight_persisted_training_protocol(config: WebShopSkillGRPOConfig) -> None:
    """Reject a changed task pool before recovery archives existing artifacts."""

    runtime = config.webshop
    artifact_dir = Path(runtime.artifact_dir).expanduser()
    contract_path = artifact_dir / "webshop_training_contract.json"
    if contract_path.exists():
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("invalid WebShop training contract; use a new trial")
        expected_selection = (
            task_selection_contract(
                task_split=runtime.task_split,
                train_count=runtime.train_task_count,
                eval_count=runtime.eval_task_count,
                seed=runtime.task_seed,
                unique_asin=runtime.unique_asin_split,
            )
            if runtime.task_split != "training_holdout"
            else None
        )
        if payload.get("task_selection") != expected_selection:
            raise RuntimeError(
                "persisted WebShop task selection differs; use a new trial before recovery"
            )
        manifest_path = artifact_dir / "task_manifest.json"
        if runtime.task_split == "official_train" and manifest_path.exists():
            validate_official_manifest(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                eval_count=runtime.eval_task_count,
                seed=runtime.task_seed,
            )


def _resolve_layout(config: WebShopSkillGRPOConfig) -> tuple[int, int, int]:
    rounds, groups, samples = resolve_service_benchmark_layout(
        config,
        runtime=config.webshop,
        eval_config=config.webshop_eval,
        benchmark_key="webshop",
        benchmark_name="WebShop",
        logger=logger,
    )
    runtime = config.webshop
    if runtime.parallelize_same_step_rounds:
        batch_size = int(config.train_dataset.batch_size or 0)
        if bool(config.train_dataset.shuffle):
            raise ValueError(
                "parallelize_same_step_rounds requires train_dataset.shuffle=false"
            )
        if bool(config.dynamic_bs):
            raise ValueError("parallelize_same_step_rounds requires dynamic_bs=false")
        if int(config.rollout.max_head_offpolicyness) != 0:
            raise ValueError(
                "parallelize_same_step_rounds requires rollout.max_head_offpolicyness=0"
            )
        if batch_size <= 0 or batch_size % groups != 0:
            raise ValueError(
                "parallelize_same_step_rounds requires train batch_size to contain "
                "an integer number of complete rounds"
            )
        if rounds * groups % batch_size != 0:
            raise ValueError(
                "parallelize_same_step_rounds requires the dataset to contain an "
                "integer number of optimizer batches"
            )
    return rounds, groups, samples


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, WebShopSkillGRPOConfig)
    validate_skill_gpu_layout(config)
    if config.ref is not None and config.ref.path == config.actor.path:
        config.ref.attn_impl = config.actor.attn_impl
    ensure_logp_mb_capacity(config, logger)
    rounds, groups, samples = _resolve_layout(config)
    _preflight_persisted_training_protocol(config)
    reconcile_artifacts_from_recover(
        artifact_dir=Path(config.webshop.artifact_dir),
        recover_config=config.recover,
        benchmark="WebShop",
        groups_per_round=groups,
        train_batch_size=int(config.train_dataset.batch_size),
    )
    if os.getenv("RANK", "0") == "0":
        save_config(config, StatsLogger.get_log_path(config.stats_logger))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split=config.train_dataset.split,
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = None
    if config.webshop_eval.enabled:
        valid_dataset = get_custom_dataset(
            split=config.valid_dataset.split,
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )

    runtime = config.webshop
    workflow_kwargs = {
        "gconfig": config.gconfig,
        "tokenizer": tokenizer,
        "artifact_dir": runtime.artifact_dir,
        "env_service_url": runtime.env_service_url,
        "actor_model": config.actor_server.model_path,
        "actor_api_key": config.actor_server.api_key,
        "actor_timeout_s": runtime.actor_timeout_s,
        "actor_temperature": runtime.actor_temperature,
        "train_task_count": runtime.train_task_count,
        "eval_task_count": runtime.eval_task_count,
        "task_seed": runtime.task_seed,
        "unique_asin_split": runtime.unique_asin_split,
        "rollouts_per_skill": runtime.rollouts_per_skill,
        "max_rollout_steps": runtime.max_rollout_steps,
        "memory_window": runtime.memory_window,
        "observation_char_limit": runtime.observation_char_limit,
        "max_clickables": runtime.max_clickables,
        "invalid_action_retries": runtime.invalid_action_retries,
        "success_threshold": runtime.success_threshold,
        "actor_max_tokens": runtime.actor_max_tokens,
        "rollout_timeout_s": runtime.rollout_timeout_s,
        "episode_rollout_workers": runtime.episode_rollout_workers,
        "source_trajectories_per_prompt": runtime.source_trajectories_per_prompt,
        "prompt_observation_char_limit": runtime.prompt_observation_char_limit,
        "samples_per_round": samples,
        "rounds": rounds,
        "groups_per_round": groups,
        "train_batch_size": int(config.train_dataset.batch_size),
        "skill_generation_enable_thinking": runtime.skill_generation_enable_thinking,
        "skill_generation_max_retries": runtime.skill_generation_max_retries,
        "skill_description_max_words": runtime.skill_description_max_words,
        **reward_workflow_kwargs(runtime.reward),
        "task_split": runtime.task_split,
        "parallelize_same_step_rounds": runtime.parallelize_same_step_rounds,
        "outcome_condition_workers": runtime.outcome_condition_workers,
        "skillbank_top_k": runtime.skillbank.top_k,
        "skillbank_embedding_model": runtime.skillbank.embedding_model,
        "skillbank_embedding_batch_size": runtime.skillbank.embedding_batch_size,
        "skillbank_embedding_max_length": runtime.skillbank.embedding_max_length,
        "skillbank_embedding_device": runtime.skillbank.embedding_device,
        "skillbank_standalone_weight": runtime.skillbank.standalone_weight,
        "skillbank_retrieval_weight": runtime.skillbank.retrieval_weight,
        "skillbank_reward_weight": runtime.skillbank.reward_weight,
        "skillbank_update_min_marginal": runtime.skillbank.update_min_marginal,
        "skillbank_allow_noop_update": runtime.skillbank.allow_noop_update,
        "skillbank_max_size": runtime.skillbank.max_size,
    }
    eval_kwargs = {
        **workflow_kwargs,
        "gconfig": config.eval_gconfig,
        "eval_k": config.webshop_eval.k,
        "eval_task_count": config.webshop_eval.task_count,
        "eval_seed": config.webshop_eval.seed,
        "eval_max_parallel_rollouts_per_skill": (
            config.webshop_eval.max_parallel_rollouts_per_skill
        ),
        **eval_baseline_workflow_kwargs(config.webshop_eval),
    }
    checkpoint_retention = EvalCheckpointRetention.from_training_config(
        config,
        artifact_dir=config.webshop.artifact_dir,
        eval_config=config.webshop_eval,
        logger=logger,
    )
    checkpoint_retention.initialize()

    with frozen_actor_server(config.actor_server) as actor_base_url:
        workflow_kwargs["actor_base_url"] = actor_base_url
        eval_kwargs["actor_base_url"] = actor_base_url
        with PPOTrainer(
            config, train_dataset=train_dataset, valid_dataset=valid_dataset
        ) as trainer:
            trainer.train(
                workflow=config.workflow,
                workflow_kwargs=workflow_kwargs,
                eval_workflow=(
                    config.webshop_eval.workflow
                    if config.webshop_eval.enabled
                    else None
                ),
                eval_workflow_kwargs=eval_kwargs,
                early_stop_fn=checkpoint_retention.prune_after_eval,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
