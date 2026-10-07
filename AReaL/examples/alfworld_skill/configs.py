# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from examples.skill_training.configs import (
    FrozenActorServerConfig,
    RepeatedNoSkillBaselineConfig,
    SkillRewardConfig,
)

from areal.api.cli_args import GRPOConfig

_ALFWORLD_TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
]


@dataclass
class ALFWorldOnlineSkillbankConfig:
    """Online semantic skillbank reward and update settings."""

    weight: float = field(default=0.5)
    bank_weight: float = field(default=1.0)
    standalone_weight: float = field(default=0.0)
    top_k: int = field(default=3)
    seed: int = field(default=23_000)
    warmup_steps: int = field(default=20)
    warmup_update_type_count: int = field(default=2)
    later_update_type_count: int = field(default=1)
    zero_epsilon: float = field(default=0.0)
    embedding_model: str = field(default="sentence-transformers/all-mpnet-base-v2")
    embedding_batch_size: int = field(default=64)
    embedding_max_length: int = field(default=256)
    max_compute_multiplier: float = field(default=2.25)


@dataclass
class ALFWorldSkillRuntimeConfig:
    """Runtime settings for ALFWorld skill induction and rollout persistence."""

    artifact_dir: str = field(default="outputs/alfworld/trial0/alfworld_skill")
    repo_root: str = field(default=".")
    data_root: str = field(default="data/alfworld/json_2.1.1")
    train_split: str = field(default="train")
    rollouts_per_skill: int = field(default=8)
    max_rollout_steps: int = field(default=50)
    memory_window: int = field(default=5)
    max_commands: int = field(default=140)
    prompt_observation_char_limit: int = field(default=120)
    prompt_result_char_limit: int = field(default=120)
    actor_temperature: float = field(default=0.0)
    actor_timeout_s: float = field(default=120.0)
    rollout_timeout_s: float = field(default=0.0)
    skill_eval_workers: int = field(default=0)
    episode_rollout_workers: int = field(default=128)
    progress_summary_interval_s: float = field(default=10.0)
    samples_per_round: int = field(
        default=0,
        metadata={"help": "Skills per prompt group; 0 inherits gconfig.n_samples"},
    )
    rounds_per_category: int = field(
        default=0,
        metadata={
            "help": "Number of mixed training rounds; 0 derives it from total_train_steps, batch_size, and groups_per_round"
        },
    )
    groups_per_round: int = field(default=8)
    trajectory_pool_enabled: bool = field(default=True)
    trajectory_pool_initial_size: int = field(default=300)
    trajectory_pool_prompt_episodes: int = field(default=4)
    trajectory_pool_initial_workers: int = field(default=64)
    trajectory_pool_use_previous_step: bool = field(default=True)
    trajectory_pool_delete_consumed_step: bool = field(default=True)
    parallelize_same_step_rounds: bool = field(
        default=True,
        metadata={
            "help": "Allow complete rounds from one strict on-policy optimizer batch to run concurrently when trajectory_pool_enabled=true"
        },
    )
    round_barrier_timeout_s: float = field(
        default=21600.0,
        metadata={
            "help": "Fail a stalled generation, rollout, or trajectory-pool barrier with missing-slot diagnostics after this many seconds; 0 disables the deadline"
        },
    )
    skill_generation_enable_thinking: bool = field(default=False)
    skill_prompt_version: str = field(
        default="evidence_discovery",
        metadata={
            "help": "Skill induction prompt; only evidence_discovery is supported"
        },
    )
    skill_generation_max_retries: int = field(default=0)
    outcome_flattened_scheduler: bool = field(
        default=True,
        metadata={
            "help": "Submit no-skill, singleton, and pair episodes through one bounded outcome queue"
        },
    )
    outcome_condition_workers: int = field(default=32)
    # Skills must be exactly ``<skill>...</skill>``; only the body is rewarded.
    skill_output_format: str = field(default="skill_xml")
    skill_description_max_words: int = field(default=100)
    fail_fast_actor_error_rate: float = field(default=0.5)
    fail_fast_infra_error_rate: float = field(default=0.5)
    task_types: list[str] = field(default_factory=lambda: list(_ALFWORLD_TASK_TYPES))
    reward: SkillRewardConfig = field(default_factory=SkillRewardConfig)
    online_skillbank: ALFWorldOnlineSkillbankConfig = field(
        default_factory=ALFWorldOnlineSkillbankConfig
    )


@dataclass
class ALFWorldSkillEvalConfig(RepeatedNoSkillBaselineConfig):
    """Held-out evaluation settings for ALFWorld skill-generator checkpoints."""

    enabled: bool = field(default=True)
    workflow: str = field(
        default="areal.workflow.alfworld_skill.ALFWorldSkillEvalWorkflow"
    )
    split: str = field(default="valid_unseen")
    k: int = field(
        default=6, metadata={"help": "Eval skills per step; one per input task type"}
    )
    tasks_per_type: int = field(
        default=0,
        metadata={"help": "Eval games per task type from the split; 0 uses all games"},
    )
    seed: int = field(default=1001)
    no_skill_baseline_repeats: int = field(default=3)
    no_skill_baseline_seed: int = field(default=8_100_000)
    early_stop_enabled: bool = field(default=True)
    early_stop_patience: int = field(default=3)
    early_stop_min_schema_valid_rate: float = field(default=0.25)
    checkpoint_retention_enabled: bool = field(default=True)
    checkpoint_retention_keep_latest: int = field(default=1)
    checkpoint_retention_metric: str = field(default="alfworld_eval_all_sr")
    checkpoint_retention_protect_existing: bool = field(default=True)


@dataclass
class ALFWorldSkillGRPOConfig(GRPOConfig):
    """GRPO config for ALFWorld prompt skill-generator training."""

    workflow: str = field(
        default="areal.workflow.alfworld_skill.ALFWorldSkillGRPOWorkflow"
    )
    actor_server: FrozenActorServerConfig = field(
        default_factory=FrozenActorServerConfig
    )
    alfworld: ALFWorldSkillRuntimeConfig = field(
        default_factory=ALFWorldSkillRuntimeConfig
    )
    alfworld_eval: ALFWorldSkillEvalConfig = field(
        default_factory=ALFWorldSkillEvalConfig
    )
