# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from typing import Any

from examples.skill_training.configs import (
    RepeatedNoSkillBaselineConfig,
    SkillRewardConfig,
)


def ensure_logp_mb_capacity(config: Any, logger: Any) -> None:
    """Keep actor/ref token packing compatible with the generation limit."""

    required = int(config.gconfig.max_tokens or 0)
    if required <= 0:
        return
    for role in ("actor", "ref"):
        engine = getattr(config, role, None)
        spec = getattr(engine, "mb_spec", None) if engine is not None else None
        current = getattr(spec, "max_tokens_per_mb", None) if spec is not None else None
        if current is not None and int(current) < required:
            logger.warning(
                "Increasing %s.mb_spec.max_tokens_per_mb from %s to %s to match "
                "gconfig.max_tokens; lower values cannot pack generated sequences.",
                role,
                current,
                required,
            )
            spec.max_tokens_per_mb = required


def reward_workflow_kwargs(reward: SkillRewardConfig) -> dict[str, object]:
    """Translate the shared reward config into workflow constructor arguments."""
    return {
        "reward_sr_weight": reward.sr_weight,
        "reward_mutual_information_weight": reward.mutual_information_weight,
        "reward_mutual_information_scale": reward.mutual_information_scale,
        "reward_mutual_information_token_margin_clip": reward.mutual_information_token_margin_clip,
        "reward_mutual_information_token_reward_soft_cap": reward.mutual_information_token_reward_soft_cap,
        "reward_mutual_information_token_reward_clip": reward.mutual_information_token_reward_clip,
        "reward_schema_valid_bonus": reward.schema_valid_bonus,
        "reward_schema_invalid_penalty": reward.schema_invalid_penalty,
    }


def eval_baseline_workflow_kwargs(
    baseline: RepeatedNoSkillBaselineConfig,
) -> dict[str, int]:
    """Validate and translate the shared repeated eval-baseline config."""

    repeats = int(baseline.no_skill_baseline_repeats)
    if repeats <= 0:
        raise ValueError("no_skill_baseline_repeats must be positive")
    return {
        "eval_no_skill_baseline_repeats": repeats,
        "eval_no_skill_baseline_seed": int(baseline.no_skill_baseline_seed),
    }


def resolve_training_rounds(
    *,
    configured_rounds: int,
    total_train_steps: int,
    batch_size: int,
    groups_per_round: int,
    total_train_epochs: int,
    rounds_config_name: str,
    benchmark_name: str,
    logger: Any,
) -> tuple[int, int]:
    """Resolve aligned rounds and the resulting optimizer-step capacity."""

    if groups_per_round <= 0:
        raise ValueError("groups_per_round must be positive")
    if batch_size <= 0 or total_train_epochs <= 0:
        raise ValueError("train batch_size and total_train_epochs must be positive")

    rounds = int(configured_rounds)
    if rounds <= 0:
        if total_train_steps <= 0:
            raise ValueError(f"{rounds_config_name}=0 requires total_train_steps > 0")
        required_groups = total_train_steps * batch_size
        rounds = math.ceil(required_groups / groups_per_round)
        alignment = batch_size // math.gcd(batch_size, groups_per_round)
        rounds = math.ceil(rounds / alignment) * alignment
        logger.info(
            "Auto-derived %s rounds=%d for steps=%d batch=%d groups=%d.",
            benchmark_name,
            rounds,
            total_train_steps,
            batch_size,
            groups_per_round,
        )

    available_steps = (
        math.ceil(rounds * groups_per_round / batch_size) * total_train_epochs
    )
    if total_train_steps > 0 and available_steps < total_train_steps:
        raise ValueError(
            f"{benchmark_name} dataset supplies only {available_steps} steps, "
            f"requested {total_train_steps}"
        )
    return rounds, available_steps


def resolve_service_benchmark_layout(
    config: Any,
    *,
    runtime: Any,
    eval_config: Any,
    benchmark_key: str,
    benchmark_name: str,
    logger: Any,
) -> tuple[int, int, int]:
    """Resolve the benchmark's task pool and common training batch layout."""

    if benchmark_key == "webshop":
        from examples.webshop_skill.task_split import validate_training_settings

        validate_training_settings(vars(runtime))
    full_webshop_train = (
        benchmark_key == "webshop"
        and getattr(runtime, "task_split", "training_holdout") == "official_train"
    )
    if not full_webshop_train and int(runtime.train_task_count) != 300:
        raise ValueError(
            f"{benchmark_name} training requires the fixed train_task_count=300"
        )
    if int(runtime.eval_task_count) != 100:
        raise ValueError(
            f"{benchmark_name} training requires the fixed eval_task_count=100"
        )
    if int(eval_config.task_count) != 100:
        raise ValueError(f"{benchmark_key}_eval.task_count must equal 100")
    if int(runtime.max_rollout_steps) <= 0:
        raise ValueError(f"{benchmark_key}.max_rollout_steps must be positive")

    dataset_kwargs = dict(config.train_dataset.dataset_kwargs or {})
    groups = int(runtime.groups_per_round or 0)
    configured_groups = dataset_kwargs.get("groups_per_round")
    if configured_groups is not None and int(configured_groups) != groups:
        raise ValueError(
            f"groups_per_round differs between {benchmark_key} and train_dataset"
        )
    if groups <= 0:
        raise ValueError(f"{benchmark_key}.groups_per_round must be positive")

    samples = int(runtime.samples_per_round or 0)
    generation_samples = int(config.gconfig.n_samples or 0)
    if generation_samples <= 0:
        raise ValueError("gconfig.n_samples must be positive")
    if samples <= 0:
        samples = generation_samples
    if samples != generation_samples:
        raise ValueError(
            f"{benchmark_key}.samples_per_round must equal gconfig.n_samples"
        )

    rounds = int(runtime.rounds or 0)
    configured_rounds = dataset_kwargs.get("rounds")
    if configured_rounds is not None and int(configured_rounds) > 0:
        if rounds > 0 and int(configured_rounds) != rounds:
            raise ValueError(
                f"rounds differs between {benchmark_key} and train_dataset"
            )
        rounds = int(configured_rounds)
    batch_size = int(config.train_dataset.batch_size or 0)
    total_steps = int(config.total_train_steps or 0)
    epochs = int(config.total_train_epochs or 0)
    rounds, _ = resolve_training_rounds(
        configured_rounds=rounds,
        total_train_steps=total_steps,
        batch_size=batch_size,
        groups_per_round=groups,
        total_train_epochs=epochs,
        rounds_config_name=f"{benchmark_key}.rounds",
        benchmark_name=benchmark_name,
        logger=logger,
    )

    runtime.samples_per_round = samples
    runtime.rounds = rounds
    config.train_dataset.dataset_kwargs = {
        **dataset_kwargs,
        "rounds": rounds,
        "groups_per_round": groups,
        "seed": int(config.seed),
    }
    if eval_config.enabled:
        if config.valid_dataset is None:
            raise ValueError(
                f"{benchmark_key}_eval.enabled=true requires valid_dataset"
            )
        valid_kwargs = dict(config.valid_dataset.dataset_kwargs or {})
        valid_kwargs.update(
            {
                "mode": "eval",
                "eval_k": int(eval_config.k),
                "seed": int(eval_config.seed),
            }
        )
        config.valid_dataset.dataset_kwargs = valid_kwargs
    return rounds, groups, samples
