# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from examples.skill_training.configs import (
    FrozenActorServerConfig,
    RepeatedNoSkillBaselineConfig,
    SkillRewardConfig,
)

from areal.api.cli_args import GRPOConfig


@dataclass
class WebShopSkillBankConfig:
    """Training-time semantic retrieval and counterfactual admission contract."""

    top_k: int = field(default=3)
    embedding_model: str = field(default="sentence-transformers/all-mpnet-base-v2")
    embedding_batch_size: int = field(default=64)
    embedding_max_length: int = field(default=256)
    embedding_device: str = field(default="cpu")
    reward_weight: float = field(default=0.5)
    standalone_weight: float = field(default=0.0)
    retrieval_weight: float = field(default=1.0)
    update_min_marginal: float = field(default=0.0)
    allow_noop_update: bool = field(default=True)
    max_size: int = field(default=0)


@dataclass
class WebShopSkillRuntimeConfig:
    artifact_dir: str = field(default="outputs/webshop/trial0/webshop_skill")
    env_service_url: str = field(default="http://127.0.0.1:31080")
    repo_root: str = field(default="webshop")
    products_file: str = field(default="data/webshop/items_shuffle.json")
    attributes_file: str = field(default="data/webshop/items_ins_v2.json")
    human_attributes_file: str = field(default="data/webshop/items_human_ins.json")
    search_index: str = field(default="data/webshop/search_engine/indexes")
    human_goals: bool = field(default=True)
    num_products: int | None = field(default=None)
    observation_mode: str = field(default="text_rich")
    task_split: str = field(default="official_train")
    environment_seed: int | None = field(default=233)
    train_task_count: int = field(default=-1)
    eval_task_count: int = field(default=100)
    task_seed: int = field(default=1)
    unique_asin_split: bool = field(default=False)
    rollouts_per_skill: int = field(default=8)
    max_rollout_steps: int = field(default=15)
    memory_window: int = field(default=5)
    observation_char_limit: int = field(default=5000)
    max_clickables: int = field(default=60)
    invalid_action_retries: int = field(default=1)
    success_threshold: float = field(default=0.999999)
    actor_temperature: float = field(default=0.0)
    actor_max_tokens: int = field(default=128)
    actor_timeout_s: float = field(default=120.0)
    rollout_timeout_s: float = field(default=7200.0)
    episode_rollout_workers: int = field(default=128)
    source_trajectories_per_prompt: int = field(default=4)
    prompt_observation_char_limit: int = field(default=400)
    skill_description_max_words: int = field(default=100)
    samples_per_round: int = field(default=0)
    rounds: int = field(default=0)
    groups_per_round: int = field(default=8)
    parallelize_same_step_rounds: bool = field(default=True)
    skill_generation_enable_thinking: bool = field(default=False)
    skill_generation_max_retries: int = field(default=0)
    outcome_condition_workers: int = field(default=32)
    reward: SkillRewardConfig = field(default_factory=SkillRewardConfig)
    skillbank: WebShopSkillBankConfig = field(default_factory=WebShopSkillBankConfig)


@dataclass
class WebShopSkillEvalConfig(RepeatedNoSkillBaselineConfig):
    enabled: bool = field(default=True)
    workflow: str = field(
        default="examples.webshop_skill.workflow.WebShopSkillEvalWorkflow"
    )
    k: int = field(default=4)
    task_count: int = field(default=100)
    seed: int = field(default=1001)
    max_parallel_rollouts_per_skill: int = field(default=64)
    checkpoint_retention_metric: str = field(
        default="webshop_eval_skillbank_policy_sr"
    )


@dataclass
class WebShopSkillGRPOConfig(GRPOConfig):
    workflow: str = field(
        default="examples.webshop_skill.workflow.WebShopSkillGRPOWorkflow"
    )
    actor_server: FrozenActorServerConfig = field(
        default_factory=FrozenActorServerConfig
    )
    webshop: WebShopSkillRuntimeConfig = field(
        default_factory=WebShopSkillRuntimeConfig
    )
    webshop_eval: WebShopSkillEvalConfig = field(default_factory=WebShopSkillEvalConfig)
