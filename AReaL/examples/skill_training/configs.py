# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrozenActorServerConfig:
    """OpenAI-compatible SGLang server used as a frozen environment actor."""

    enabled: bool = field(default=True)
    model_path: str = field(default="Qwen/Qwen3.5-4B")
    host: str = field(default="127.0.0.1")
    port: int = field(default=30080)
    cuda_visible_devices: str = field(default="0,1,2,3,4,5")
    tensor_parallel_size: int = field(default=1)
    data_parallel_size: int = field(default=6)
    dtype: str = field(default="bfloat16")
    mem_fraction_static: float = field(default=0.86)
    context_length: int = field(default=16384)
    startup_timeout_s: float = field(default=1800.0)
    api_key: str = field(default="")
    extra_args: list[str] = field(
        default_factory=lambda: [
            "--max-running-requests",
            "256",
            "--disable-cuda-graph",
        ]
    )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class SkillRewardConfig:
    """Task-independent reward shaping used by skill-generator training."""

    sr_weight: float = field(default=0.5)
    mutual_information_weight: float = field(default=0.1)
    mutual_information_scale: float = field(default=1.0)
    mutual_information_token_margin_clip: float | None = field(default=None)
    mutual_information_token_reward_soft_cap: float | None = field(
        default=None,
        metadata={"help": "Optional tanh soft cap for per-token MI reward values"},
    )
    mutual_information_token_reward_clip: float | None = field(
        default=0.15,
        metadata={"help": "Optional final hard clip for weighted per-token MI reward"},
    )
    schema_valid_bonus: float = field(default=0.02)
    schema_invalid_penalty: float = field(default=-0.1)


@dataclass
class RepeatedNoSkillBaselineConfig:
    """Shared fixed-baseline contract used by every skill benchmark eval."""

    no_skill_baseline_repeats: int = field(default=3)
    no_skill_baseline_seed: int = field(default=900_000)
    checkpoint_retention_enabled: bool = field(default=True)
    checkpoint_retention_keep_latest: int = field(default=1)
    checkpoint_retention_metric: str = field(default="mean_sr")
    checkpoint_retention_protect_existing: bool = field(default=False)
