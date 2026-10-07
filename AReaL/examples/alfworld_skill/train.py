# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).parent))
from examples.alfworld_skill.configs import ALFWorldSkillGRPOConfig
from examples.skill_training.actor_server import frozen_actor_server
from examples.skill_training.checkpoint_retention import EvalCheckpointRetention
from examples.skill_training.gpu_layout import validate_skill_gpu_layout
from examples.skill_training.recovery import reconcile_artifacts_from_recover
from examples.skill_training.train_utils import (
    ensure_logp_mb_capacity,
    eval_baseline_workflow_kwargs,
    resolve_training_rounds,
    reward_workflow_kwargs,
)

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config, save_config
from areal.dataset import get_custom_dataset
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("ALFWorldSkillTrain")


def _resolve_training_round_layout(
    config: ALFWorldSkillGRPOConfig,
) -> tuple[dict, int, int]:
    dataset_kwargs = dict(getattr(config.train_dataset, "dataset_kwargs", None) or {})

    runtime_groups = int(config.alfworld.groups_per_round or 0)
    configured_groups = dataset_kwargs.get("groups_per_round")
    if (
        configured_groups is not None
        and runtime_groups > 0
        and int(configured_groups) != runtime_groups
    ):
        raise ValueError(
            "groups_per_round is configured inconsistently: "
            f"alfworld={runtime_groups}, train_dataset={configured_groups}"
        )
    groups_per_round = int(
        configured_groups if configured_groups is not None else runtime_groups
    )
    if groups_per_round <= 0:
        raise ValueError("groups_per_round must be positive")

    runtime_rounds = int(config.alfworld.rounds_per_category or 0)
    configured_rounds = dataset_kwargs.get("rounds_per_category")
    if (
        configured_rounds is not None
        and runtime_rounds > 0
        and int(configured_rounds) > 0
        and int(configured_rounds) != runtime_rounds
    ):
        raise ValueError(
            "rounds_per_category is configured inconsistently: "
            f"alfworld={runtime_rounds}, train_dataset={configured_rounds}"
        )
    rounds_per_category = int(
        configured_rounds if configured_rounds is not None else runtime_rounds
    )

    generation_samples = int(config.gconfig.n_samples or 0)
    samples_per_round = int(config.alfworld.samples_per_round or 0)
    if generation_samples <= 0:
        raise ValueError("gconfig.n_samples must be positive")
    if samples_per_round <= 0:
        samples_per_round = generation_samples
    elif samples_per_round != generation_samples:
        raise ValueError(
            "alfworld.samples_per_round must equal gconfig.n_samples; "
            f"got {samples_per_round} and {generation_samples}"
        )
    config.alfworld.samples_per_round = samples_per_round

    runtime_task_types = list(config.alfworld.task_types or [])
    configured_task_types = dataset_kwargs.get("task_types")
    if configured_task_types is not None:
        configured_task_types = list(configured_task_types)
        if runtime_task_types and configured_task_types != runtime_task_types:
            raise ValueError(
                "task_types is configured inconsistently between alfworld and "
                "train_dataset.dataset_kwargs"
            )
        task_types = configured_task_types
    else:
        task_types = runtime_task_types

    total_train_steps = int(config.total_train_steps or 0)
    batch_size = int(config.train_dataset.batch_size or 0)
    total_train_epochs = int(config.total_train_epochs or 0)
    rounds_per_category, _ = resolve_training_rounds(
        configured_rounds=rounds_per_category,
        total_train_steps=total_train_steps,
        batch_size=batch_size,
        groups_per_round=groups_per_round,
        total_train_epochs=total_train_epochs,
        rounds_config_name="alfworld.rounds_per_category",
        benchmark_name="ALFWorld",
        logger=logger,
    )

    dataset_kwargs["rounds_per_category"] = rounds_per_category
    dataset_kwargs["groups_per_round"] = groups_per_round
    dataset_kwargs["task_types"] = task_types
    config.train_dataset.dataset_kwargs = dataset_kwargs
    config.alfworld.rounds_per_category = rounds_per_category
    config.alfworld.groups_per_round = groups_per_round
    config.alfworld.task_types = task_types
    if config.alfworld.parallelize_same_step_rounds:
        if not config.alfworld.trajectory_pool_enabled:
            raise ValueError(
                "parallelize_same_step_rounds requires trajectory_pool_enabled=true"
            )
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
        if batch_size <= 0 or batch_size % groups_per_round != 0:
            raise ValueError(
                "parallelize_same_step_rounds requires train batch_size to contain "
                "an integer number of complete rounds"
            )
        total_groups = rounds_per_category * groups_per_round
        if total_groups % batch_size != 0:
            raise ValueError(
                "parallelize_same_step_rounds requires the dataset to contain an "
                "integer number of optimizer batches"
            )
    return dataset_kwargs, rounds_per_category, groups_per_round


def _synchronize_eval_dataset_layout(config: ALFWorldSkillGRPOConfig) -> None:
    if not config.alfworld_eval.enabled or config.valid_dataset is None:
        return
    eval_k = int(config.alfworld_eval.k or 0)
    if eval_k <= 0:
        raise ValueError("alfworld_eval.k must be positive")

    dataset_kwargs = dict(config.valid_dataset.dataset_kwargs or {})
    mode = str(dataset_kwargs.get("mode", "eval"))
    if mode != "eval":
        raise ValueError(
            "alfworld_eval.enabled=true requires valid_dataset.dataset_kwargs.mode=eval"
        )

    configured_eval_k = dataset_kwargs.get("eval_k")
    if configured_eval_k is not None and int(configured_eval_k) != eval_k:
        raise ValueError(
            "eval_k is configured inconsistently: "
            f"alfworld_eval.k={eval_k}, valid_dataset={configured_eval_k}"
        )

    train_task_types = list(
        (config.train_dataset.dataset_kwargs or {}).get("task_types")
        or config.alfworld.task_types
        or []
    )
    valid_task_types = dataset_kwargs.get("task_types")
    if valid_task_types is not None:
        valid_task_types = list(valid_task_types)
        if train_task_types and valid_task_types != train_task_types:
            raise ValueError(
                "valid_dataset task_types must match the training task_types"
            )
    else:
        valid_task_types = train_task_types

    if eval_k != len(valid_task_types):
        raise ValueError(
            "eval requires exactly one skill per input task type: "
            f"eval_k={eval_k}, task_types={len(valid_task_types)}"
        )

    dataset_kwargs["mode"] = "eval"
    dataset_kwargs["eval_k"] = eval_k
    dataset_kwargs["task_types"] = valid_task_types
    config.valid_dataset.dataset_kwargs = dataset_kwargs


def _read_json(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _make_alfworld_post_eval_fn(
    config: ALFWorldSkillGRPOConfig,
    checkpoint_retention: EvalCheckpointRetention,
):
    early_stop_fn = _make_alfworld_early_stop_fn(config)

    def _post_eval(global_step: int) -> str | None:
        checkpoint_retention.prune_after_eval(global_step)
        if early_stop_fn is None:
            return None
        return early_stop_fn(global_step)

    return _post_eval


def _make_alfworld_early_stop_fn(config: ALFWorldSkillGRPOConfig):
    eval_config = config.alfworld_eval
    if not eval_config.enabled or not eval_config.early_stop_enabled:
        return None
    patience = int(eval_config.early_stop_patience)
    if patience <= 0:
        return None
    min_schema_valid_rate = float(eval_config.early_stop_min_schema_valid_rate)
    eval_root = pathlib.Path(config.alfworld.artifact_dir).expanduser() / "eval"

    def _complete_eval_rows() -> list[dict]:
        rows: list[dict] = []
        for summary_path in sorted(eval_root.glob("globalstep_*/summary.json")):
            try:
                summary = _read_json(summary_path)
            except Exception:
                continue
            if summary.get("status") != "complete":
                continue
            rows.append(summary)
        return sorted(
            rows,
            key=lambda item: int(item.get("checkpoint_global_step", -1)),
        )

    def _early_stop_reason(global_step: int) -> str | None:
        rows = [
            row
            for row in _complete_eval_rows()
            if int(row.get("checkpoint_global_step", -1)) <= global_step
        ]
        if len(rows) < patience:
            return None
        recent = rows[-patience:]
        bad_rows = [
            row
            for row in recent
            if float(row.get("schema_valid_rate", 0.0)) < min_schema_valid_rate
        ]
        if len(bad_rows) != patience:
            return None
        steps = [int(row.get("checkpoint_global_step", -1)) for row in recent]
        rates = [float(row.get("schema_valid_rate", 0.0)) for row in recent]
        reason = (
            "ALFWorld early stop: schema_valid_rate below "
            f"{min_schema_valid_rate} for {patience} consecutive eval steps; "
            f"steps={steps}, rates={rates}"
        )
        stop_payload = {
            "status": "early_stopped",
            "reason": reason,
            "global_step": global_step,
            "patience": patience,
            "min_schema_valid_rate": min_schema_valid_rate,
            "recent_steps": steps,
            "recent_schema_valid_rates": rates,
            "updated_at": time.time(),
        }
        stop_path = (
            pathlib.Path(config.alfworld.artifact_dir).expanduser() / "early_stop.json"
        )
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        with stop_path.open("w", encoding="utf-8") as handle:
            json.dump(stop_payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return reason

    return _early_stop_reason


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, ALFWorldSkillGRPOConfig)
    validate_skill_gpu_layout(config)
    if config.ref is not None and config.ref.path == config.actor.path:
        config.ref.attn_impl = config.actor.attn_impl
    ensure_logp_mb_capacity(config, logger)
    rollout_request_timeout = float(
        getattr(config.rollout, "request_timeout", 0.0) or 0.0
    )
    alfworld_rollout_timeout = float(config.alfworld.rollout_timeout_s or 0.0)
    if alfworld_rollout_timeout <= 0.0 and rollout_request_timeout > 0.0:
        timeout_margin = max(60.0, min(600.0, rollout_request_timeout * 0.1))
        alfworld_rollout_timeout = max(1.0, rollout_request_timeout - timeout_margin)
    dataset_kwargs, rounds_per_category, groups_per_round = (
        _resolve_training_round_layout(config)
    )
    _synchronize_eval_dataset_layout(config)
    reconcile_artifacts_from_recover(
        artifact_dir=pathlib.Path(config.alfworld.artifact_dir),
        recover_config=config.recover,
        benchmark="ALFWorld",
        groups_per_round=groups_per_round,
        train_batch_size=int(config.train_dataset.batch_size),
    )
    if os.getenv("RANK", "0") == "0":
        save_config(config, StatsLogger.get_log_path(config.stats_logger))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    skill_eval_workers = int(config.alfworld.skill_eval_workers or 0)
    if skill_eval_workers <= 0:
        skill_eval_workers = max(
            1, int(config.alfworld.samples_per_round) * groups_per_round
        )
    task_types = list(
        dataset_kwargs.get("task_types") or config.alfworld.task_types or []
    )

    train_dataset = get_custom_dataset(
        split=config.train_dataset.split,
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = None
    if config.alfworld_eval.enabled:
        if config.valid_dataset is None:
            raise ValueError(
                "alfworld_eval.enabled=true requires a valid_dataset config"
            )
        valid_dataset = get_custom_dataset(
            split=config.valid_dataset.split,
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )

    workflow_kwargs = {
        "gconfig": config.gconfig,
        "tokenizer": tokenizer,
        "artifact_dir": config.alfworld.artifact_dir,
        "actor_model": config.actor_server.model_path,
        "actor_api_key": config.actor_server.api_key,
        "actor_timeout_s": config.alfworld.actor_timeout_s,
        "actor_temperature": config.alfworld.actor_temperature,
        "repo_root": config.alfworld.repo_root,
        "data_root": config.alfworld.data_root,
        "train_split": config.alfworld.train_split,
        "rollouts_per_skill": config.alfworld.rollouts_per_skill,
        "max_rollout_steps": config.alfworld.max_rollout_steps,
        "memory_window": config.alfworld.memory_window,
        "max_commands": config.alfworld.max_commands,
        "prompt_observation_char_limit": (
            config.alfworld.prompt_observation_char_limit
        ),
        "prompt_result_char_limit": config.alfworld.prompt_result_char_limit,
        "rollout_timeout_s": alfworld_rollout_timeout,
        "skill_eval_workers": skill_eval_workers,
        "episode_rollout_workers": int(config.alfworld.episode_rollout_workers or 0),
        "progress_summary_interval_s": config.alfworld.progress_summary_interval_s,
        "samples_per_round": config.alfworld.samples_per_round,
        "rounds_per_category": rounds_per_category,
        "groups_per_round": groups_per_round,
        "train_batch_size": int(config.train_dataset.batch_size),
        "trajectory_pool_enabled": config.alfworld.trajectory_pool_enabled,
        "trajectory_pool_initial_size": config.alfworld.trajectory_pool_initial_size,
        "trajectory_pool_prompt_episodes": config.alfworld.trajectory_pool_prompt_episodes,
        "trajectory_pool_initial_workers": config.alfworld.trajectory_pool_initial_workers,
        "trajectory_pool_use_previous_step": config.alfworld.trajectory_pool_use_previous_step,
        "trajectory_pool_delete_consumed_step": config.alfworld.trajectory_pool_delete_consumed_step,
        "parallelize_same_step_rounds": config.alfworld.parallelize_same_step_rounds,
        "round_barrier_timeout_s": config.alfworld.round_barrier_timeout_s,
        "trajectory_pool_eval_prompts_per_step": (
            int(config.alfworld_eval.k) if config.alfworld_eval.enabled else 0
        ),
        "skill_generation_enable_thinking": (
            config.alfworld.skill_generation_enable_thinking
        ),
        "skill_prompt_version": config.alfworld.skill_prompt_version,
        "skill_generation_max_retries": config.alfworld.skill_generation_max_retries,
        "outcome_flattened_scheduler": config.alfworld.outcome_flattened_scheduler,
        "outcome_condition_workers": config.alfworld.outcome_condition_workers,
        "skill_output_format": config.alfworld.skill_output_format,
        "skill_description_max_words": config.alfworld.skill_description_max_words,
        "fail_fast_actor_error_rate": config.alfworld.fail_fast_actor_error_rate,
        "fail_fast_infra_error_rate": config.alfworld.fail_fast_infra_error_rate,
        "task_types": task_types,
        **reward_workflow_kwargs(config.alfworld.reward),
        "reward_online_skillbank_weight": config.alfworld.online_skillbank.weight,
        "online_skillbank_bank_weight": (config.alfworld.online_skillbank.bank_weight),
        "online_skillbank_standalone_weight": (
            config.alfworld.online_skillbank.standalone_weight
        ),
        "online_skillbank_top_k": config.alfworld.online_skillbank.top_k,
        "online_skillbank_seed": config.alfworld.online_skillbank.seed,
        "online_skillbank_warmup_steps": (
            config.alfworld.online_skillbank.warmup_steps
        ),
        "online_skillbank_warmup_update_type_count": (
            config.alfworld.online_skillbank.warmup_update_type_count
        ),
        "online_skillbank_later_update_type_count": (
            config.alfworld.online_skillbank.later_update_type_count
        ),
        "online_skillbank_zero_epsilon": (
            config.alfworld.online_skillbank.zero_epsilon
        ),
        "online_skillbank_embedding_model": (
            config.alfworld.online_skillbank.embedding_model
        ),
        "online_skillbank_embedding_batch_size": (
            config.alfworld.online_skillbank.embedding_batch_size
        ),
        "online_skillbank_embedding_max_length": (
            config.alfworld.online_skillbank.embedding_max_length
        ),
        "online_skillbank_max_compute_multiplier": (
            config.alfworld.online_skillbank.max_compute_multiplier
        ),
    }
    checkpoint_retention = EvalCheckpointRetention.from_training_config(
        config,
        artifact_dir=config.alfworld.artifact_dir,
        eval_config=config.alfworld_eval,
        logger=logger,
    )
    checkpoint_model_root = checkpoint_retention.checkpoint_model_root
    eval_workflow_kwargs = {
        **workflow_kwargs,
        "gconfig": config.eval_gconfig,
        "eval_data_split": config.alfworld_eval.split,
        "eval_k": config.alfworld_eval.k,
        "eval_tasks_per_type": config.alfworld_eval.tasks_per_type,
        "eval_seed": config.alfworld_eval.seed,
        **eval_baseline_workflow_kwargs(config.alfworld_eval),
        "checkpoint_model_root": str(checkpoint_model_root),
    }
    eval_workflow = (
        config.alfworld_eval.workflow if config.alfworld_eval.enabled else None
    )
    checkpoint_retention.initialize()
    early_stop_fn = _make_alfworld_post_eval_fn(config, checkpoint_retention)

    with frozen_actor_server(config.actor_server) as actor_base_url:
        workflow_kwargs["actor_base_url"] = actor_base_url
        eval_workflow_kwargs["actor_base_url"] = actor_base_url
        with PPOTrainer(
            config, train_dataset=train_dataset, valid_dataset=valid_dataset
        ) as trainer:
            trainer.train(
                workflow=config.workflow,
                workflow_kwargs=workflow_kwargs,
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
                early_stop_fn=early_stop_fn,
            )
            print("[training] optimizer loop completed successfully", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
