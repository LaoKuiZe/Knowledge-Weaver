# SPDX-License-Identifier: MIT

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from examples.webshop_skill.core import category_metrics, episode_metrics
from examples.webshop_skill.runtime import _SKILLBANK_MODE, _finite

from areal.api import InferenceEngine
from areal.workflow.alfworld_skill import (
    _atomic_write_json,
    _clip_weighted_tokenwise_mi_rewards,
)


class WebShopRolloutMixin:
    def _evaluate_and_save(
        self,
        prepared: dict[str, Any],
        raw: str,
        input_tokens: list[int],
        output_tokens: list[int],
        attempts: list[dict[str, Any]],
        engine: InferenceEngine,
        group_reward_index: dict[str, Any],
    ) -> dict[str, Any]:
        sample_dir = Path(prepared["sample_dir"])
        sample_key = str(prepared["sample_key"])
        round_index = int(prepared["round_in_category"])
        group_index = int(prepared["group_index"])
        final = attempts[-1]
        parsed = final.get("parsed") if isinstance(final.get("parsed"), dict) else None
        valid = bool(final.get("schema_valid")) and parsed is not None
        task_indices = list(map(int, prepared["task_indices"]))
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "rollout_in_progress",
                "task_type": self.TASK_TYPE,
                "round_in_category": round_index,
                "group_index": group_index,
                "sample_key": sample_key,
                "schema_valid": valid,
                "n_rollouts_expected": len(task_indices),
                "updated_at": time.time(),
            },
        )
        if (
            group_reward_index.get("status") != "complete"
            or group_reward_index.get("mode") != _SKILLBANK_MODE
        ):
            raise RuntimeError(f"missing complete {_SKILLBANK_MODE} group reward index")
        group_sample_metrics = dict(
            group_reward_index.get("sample_metrics", {}).get(sample_key, {})
        )
        episodes = list(
            group_reward_index.get("singleton_episodes", {}).get(sample_key, [])
        )
        if len(episodes) != len(task_indices):
            raise RuntimeError(
                f"{_SKILLBANK_MODE} singleton episodes are incomplete for {sample_key}"
            )
        metrics = episode_metrics(episodes)
        sr = float(metrics["success_rate"])
        status_counts = Counter((str(item.get("status")) for item in episodes))
        baseline_episodes = list(group_reward_index.get("no_skill_episodes", []))
        baseline_metrics = episode_metrics(baseline_episodes)
        baseline = {
            "status": "complete",
            "sr": float(baseline_metrics["success_rate"]),
            "metrics": baseline_metrics,
            "episodes": baseline_episodes,
        }
        baseline_sr = baseline["sr"]
        baseline_delta_sr = sr - baseline_sr
        mi_index: dict[str, Any] = {"status": "disabled", "sample_scores": {}}
        if self.reward_mutual_information_weight > 0.0:
            mi_index = self._load_or_create_mi_index(
                engine, task_type=self.TASK_TYPE, round_in_category=round_index
            )
        mi_sample = (
            mi_index.get("sample_scores", {}).get(sample_key, {})
            if isinstance(mi_index.get("sample_scores"), dict)
            else {}
        )
        mi_score = _finite(
            mi_sample.get("reward_score", mi_sample.get("mi_score", 0.0))
        )
        mi_raw_score = _finite(
            mi_sample.get("raw_mi_score", mi_sample.get("mi_score", 0.0))
        )
        mi_token_scores = list(map(float, mi_sample.get("token_reward_scores", [])))
        mi_target_tokens = list(map(int, mi_sample.get("target_tokens", [])))
        if self.reward_mutual_information_weight > 0 and valid:
            expected = self._mi_target_tokens(output_tokens)
            if mi_sample.get("status") != "complete" or mi_target_tokens != expected:
                raise RuntimeError(f"tokenwise MI is incomplete for {sample_key}")
            if len(mi_token_scores) != len(mi_target_tokens):
                raise RuntimeError(
                    f"tokenwise MI reward length mismatch for {sample_key}"
                )
        reward, schema_reward, sr_reward, mi_reward = self._compute_training_reward(
            sr=sr, mutual_information_score=mi_score, schema_valid=valid
        )
        skillbank_reward = self.skillbank_reward_weight * float(
            group_sample_metrics.get("combined_sr_reward", 0.0)
        )
        reward += skillbank_reward
        token_clip = _clip_weighted_tokenwise_mi_rewards(
            mi_token_scores,
            weight=self.reward_mutual_information_weight,
            reward_clip=self.reward_mutual_information_token_reward_clip,
            absolute_mode=True,
        )
        token_rewards = list(map(float, token_clip["clipped"]))
        body_mask = mi_sample.get("token_reward_mask", [])
        body_positions = [i for i, include in enumerate(body_mask) if include]
        body_rewards = [token_rewards[i] for i in body_positions]
        body_clipped = [
            token_clip["unclipped"][i] != token_rewards[i] for i in body_positions
        ]
        result = {
            "skill_name": f"skill/{self.TASK_TYPE}/round_{round_index:04d}/group_{group_index:02d}/{sample_key}",
            "task_type": self.TASK_TYPE,
            "prompt_category": self.PROMPT_CATEGORY,
            "round_in_category": round_index,
            "group_index": group_index,
            "sample_key": sample_key,
            "schema_valid": valid,
            "schema_error": str(final.get("schema_error") or ""),
            "attempt_count": len(attempts),
            "wins": sum((bool(item.get("success")) for item in episodes)),
            "n_rollouts": len(episodes),
            "sr": sr,
            "mean_env_reward": float(metrics["mean_reward"]),
            "reward": reward,
            "terminal_reward": reward,
            "sequence_reward": reward,
            "token_rewards": token_rewards,
            "token_reward_target_tokens": mi_target_tokens,
            "schema_reward": schema_reward,
            "sr_reward": sr_reward,
            "skillbank_reward": skillbank_reward,
            "baseline_sr": baseline_sr,
            "baseline_delta_sr": baseline_delta_sr,
            "mutual_information_score": mi_score,
            "mutual_information_raw_score": mi_raw_score,
            "mutual_information_reward": mi_reward,
            "mi_token_reward_body_mean": mean(body_rewards) if body_rewards else 0.0,
            "mi_token_reward_body_count": len(body_rewards),
            "mi_token_reward_body_clipped_fraction": mean(body_clipped)
            if body_clipped
            else 0.0,
            "mutual_information_token_reward_mean": mean(token_rewards)
            if token_rewards
            else 0.0,
            "mutual_information_token_reward_min": min(token_rewards)
            if token_rewards
            else 0.0,
            "mutual_information_token_reward_max": max(token_rewards)
            if token_rewards
            else 0.0,
            "mutual_information_token_reward_count": len(token_rewards),
            "mutual_information_token_reward_clipped_fraction": float(
                token_clip["clipped_fraction"]
            ),
            "status_counts": dict(status_counts),
            "metrics": metrics,
            "per_category": category_metrics(episodes),
            "baseline": {
                key: value for key, value in baseline.items() if key != "episodes"
            },
            "mutual_information": mi_sample,
            "skillbank_counterfactual": group_sample_metrics,
            "skillbank": {
                "bank_size": int(group_reward_index.get("bank_size", 0)),
                "bank_digest": str(group_reward_index.get("bank_digest") or ""),
                "bank_top_k": int(group_reward_index.get("bank_top_k", 0)),
                "retrieval": group_reward_index.get("retrieval", []),
            },
            "episodes": episodes,
            "updated_at": time.time(),
        }
        denominator = (
            abs(sr_reward) + abs(mi_reward) + abs(skillbank_reward) + abs(schema_reward)
        )
        result["mutual_information_reward_abs_fraction"] = (
            abs(mi_reward) / denominator if denominator > 0 else 0.0
        )
        _atomic_write_json(sample_dir / "rollouts.json", result)
        _atomic_write_json(
            sample_dir / "metrics.json",
            {
                key: value
                for key, value in result.items()
                if key not in {"episodes", "token_rewards"}
            },
        )
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "complete",
                "task_type": self.TASK_TYPE,
                "round_in_category": round_index,
                "group_index": group_index,
                "sample_key": sample_key,
                "schema_valid": valid,
                "wins": result["wins"],
                "n_rollouts": len(episodes),
                "sr": sr,
                "reward": reward,
                "updated_at": time.time(),
            },
        )
        self._maybe_update_skillbank(int(prepared["training_global_step"]))
        return result
