# SPDX-License-Identifier: MIT

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from areal.api import InferenceEngine
from areal.workflow.alfworld_runtime import (
    _append_jsonl,
    _atomic_write_json,
    _baseline_entries_by_index,
    _per_task_type_metrics,
    _read_json,
    _remove_completed_episode_snapshots,
    _rollout_progress_event,
    _unlink_if_exists,
    logger,
)
from areal.workflow.skill_prompts import (
    _parse_skill_generation,
    _skill_xml_token_reward_mask,
)
from areal.workflow.skill_token_feedback import (
    _clip_weighted_tokenwise_mi_rewards,
)

# Placeholder mode recorded when token-wise MI is disabled or could not be scored.
_DISABLED_MI_MODE = "input_conditioned_token_mi"


class ALFWorldRolloutMixin:
    def _evaluate_and_save(
        self,
        prepared: dict[str, Any],
        raw_generation: str,
        input_tokens: list[int],
        output_tokens: list[int],
        generation_attempts: list[dict[str, Any]] | None = None,
        engine: InferenceEngine | None = None,
        precomputed_online_skillbank_index: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        max_retries = self._max_skill_generation_retries()
        sample_dir = Path(prepared["sample_dir"])
        task_type = str(prepared["task_type"])
        input_task_type = str(prepared.get("input_task_type") or task_type)
        prompt_category = str(prepared["prompt_category"])
        sample_key = str(prepared["sample_key"])
        round_in_category = int(prepared["round_in_category"])
        group_index = int(prepared.get("group_index", 0))
        skill_name = f"skill/{task_type}/round_{round_in_category:04d}/group_{group_index:02d}/{sample_key}"
        attempts = generation_attempts or []
        if attempts:
            final_attempt = attempts[-1]
            parsed = final_attempt.get("parsed")
            parsed = parsed if isinstance(parsed, dict) else None
            valid = bool(final_attempt.get("schema_valid"))
            parse_error = str(final_attempt.get("schema_error") or "")
            repair_notes = list(final_attempt.get("repair_notes") or [])
        else:
            parsed, valid, parse_error, repair_notes = _parse_skill_generation(
                raw_generation,
                prompt_category,
                skill_output_format=self.skill_output_format,
                max_words=int(getattr(self, "skill_description_max_words", 0)),
            )
            attempts = [
                {
                    "attempt": 1,
                    "raw": raw_generation,
                    "parsed": parsed,
                    "schema_valid": valid,
                    "schema_error": parse_error,
                    "repair_notes": repair_notes,
                    "input_token_count": len(input_tokens),
                    "output_token_count": len(output_tokens),
                    "updated_at": time.time(),
                }
            ]
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
                "attempt_count": len(attempts),
                "max_retries": max_retries,
                "attempts": attempts,
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
                "attempt_count": len(attempts),
                "max_retries": max_retries,
                "updated_at": time.time(),
            },
        )
        episodes: list[dict[str, Any] | None] = [None] * len(games)

        def write_episode_progress(rollout_index: int) -> None:
            now = time.time()
            complete_episode = episodes[rollout_index]
            if complete_episode is not None:
                _atomic_write_json(
                    sample_dir / f"episode_{rollout_index:02d}.json",
                    {
                        "skill_name": skill_name,
                        "task_type": task_type,
                        "prompt_category": prompt_category,
                        "round_in_category": round_in_category,
                        "group_index": group_index,
                        "sample_key": sample_key,
                        "rollout_index": rollout_index,
                        "episode": complete_episode,
                        "updated_at": now,
                    },
                )
                _unlink_if_exists(
                    sample_dir / f"current_episode_{rollout_index:02d}.json"
                )
                _append_jsonl(
                    sample_dir / "rollout_progress.jsonl",
                    _rollout_progress_event(
                        rollout_index=rollout_index,
                        episode=complete_episode,
                        final=True,
                        extra={
                            "skill_name": skill_name,
                            "task_type": task_type,
                            "round_in_category": round_in_category,
                            "group_index": group_index,
                            "sample_key": sample_key,
                        },
                    ),
                )

            partial_episodes = [episode for episode in episodes if episode is not None]
            partial_wins = sum(1 for episode in partial_episodes if episode.get("won"))
            partial_sr = partial_wins / max(1, len(partial_episodes))
            partial_payload = {
                "skill_name": skill_name,
                "task_type": task_type,
                "prompt_category": prompt_category,
                "round_in_category": round_in_category,
                "group_index": group_index,
                "sample_key": sample_key,
                "schema_valid": valid,
                "schema_error": parse_error,
                "attempt_count": len(attempts),
                "max_retries": max_retries,
                "wins": partial_wins,
                "n_rollouts": len(partial_episodes),
                "n_rollouts_expected": len(games),
                "sr": partial_sr,
                "status_counts": dict(
                    Counter(str(episode.get("status")) for episode in partial_episodes)
                ),
                "completed_episode_files": [
                    f"episode_{index:02d}.json"
                    for index, episode in enumerate(episodes)
                    if episode is not None
                ],
                "updated_at": now,
            }
            _atomic_write_json(sample_dir / "rollouts_partial.json", partial_payload)
            _atomic_write_json(
                sample_dir / "checkpoint.json",
                {
                    "status": "rollout_in_progress",
                    "task_type": task_type,
                    "round_in_category": round_in_category,
                    "group_index": group_index,
                    "sample_key": sample_key,
                    "schema_valid": valid,
                    "attempt_count": len(attempts),
                    "max_retries": max_retries,
                    "wins": partial_wins,
                    "n_rollouts": len(partial_episodes),
                    "n_rollouts_expected": len(games),
                    "sr": partial_sr,
                    "updated_at": now,
                },
            )

        outcome_metrics: dict[str, Any] = {
            "status": "disabled",
            "singleton_sr": 0.0,
            "no_skill_sr": 0.0,
            "standalone_delta": 0.0,
            "positive_marginal_indicator": 0.0,
            "negative_marginal_indicator": 0.0,
            "zero_marginal_indicator": 1.0,
            "outcome_reward": 0.0,
            "retrieved_skill_sr": 0.0,
            "augmented_sr": 0.0,
            "bank_delta": 0.0,
            "weighted_marginal_score": 0.0,
        }
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "online_skillbank_waiting",
                "task_type": task_type,
                "round_in_category": round_in_category,
                "group_index": group_index,
                "sample_key": sample_key,
                "schema_valid": valid,
                "attempt_count": len(attempts),
                "max_retries": max_retries,
                "updated_at": time.time(),
            },
        )
        if precomputed_online_skillbank_index is not None:
            online_skillbank_index = precomputed_online_skillbank_index
        else:
            online_skillbank_index = self._load_or_create_online_skillbank_group_index(
                task_type=task_type,
                round_in_category=round_in_category,
                group_index=group_index,
                games=games,
            )
        sample_metrics = online_skillbank_index.get("sample_metrics", {})
        metrics_key = (
            f"group_{int(prepared['training_step_group_index']):02d}/{sample_key}"
        )
        if not isinstance(sample_metrics, dict) or metrics_key not in sample_metrics:
            raise RuntimeError(
                f"online_skillbank index is missing metrics for {metrics_key}"
            )
        raw_outcome_metrics = sample_metrics[metrics_key]
        if not isinstance(raw_outcome_metrics, dict):
            raise RuntimeError(
                f"online_skillbank sample metrics are malformed for {metrics_key}"
            )
        outcome_metrics = {**outcome_metrics, **raw_outcome_metrics}
        # The singleton condition doubles as this sample's own skill rollouts.
        singleton_dir = Path(str(outcome_metrics.get("singleton_condition_dir", "")))
        singleton_index = _read_json(singleton_dir / "condition_index.json")
        singleton_episodes = singleton_index.get("episodes", [])
        if not isinstance(singleton_episodes, list) or len(singleton_episodes) != len(
            games
        ):
            raise RuntimeError(
                f"online_skillbank singleton condition is incomplete for {sample_key}"
            )
        episodes = [
            dict(episode) if isinstance(episode, dict) else None
            for episode in singleton_episodes
        ]
        for rollout_index in range(len(episodes)):
            write_episode_progress(rollout_index)
        episodes = [episode for episode in episodes if episode is not None]
        wins = sum((1 for episode in episodes if episode.get("won")))
        sr = wins / max(1, len(episodes))
        status_counts = Counter((str(episode.get("status")) for episode in episodes))
        n_episodes = max(1, len(episodes))
        actor_error_rate = status_counts.get("actor_error", 0) / n_episodes
        infra_error_count = sum(
            (
                status_counts.get(status, 0)
                for status in ("actor_error", "rollout_error")
            )
        )
        infra_error_rate = infra_error_count / n_episodes
        if (
            self.fail_fast_actor_error_rate > 0
            and actor_error_rate >= self.fail_fast_actor_error_rate
            or (
                self.fail_fast_infra_error_rate > 0
                and infra_error_rate >= self.fail_fast_infra_error_rate
            )
        ):
            causes = [
                self._diagnostic_text(
                    f"{episode.get('status')}: {episode.get('error', '')}\n{episode.get('traceback', '')}"
                )
                for episode in episodes
                if episode.get("status") in {"actor_error", "rollout_error"}
            ][:3]
            raise RuntimeError(
                f"ALFWorld rollout infrastructure failure: sample={sample_key} round={round_in_category} actor_error_rate={actor_error_rate:.3f} infra_error_rate={infra_error_rate:.3f} status_counts={dict(status_counts)}. The frozen actor server or rollout worker is unhealthy; aborting instead of training on failed rollouts. Causes: {causes}"
            )
        baseline_index = online_skillbank_index.get("baseline", {})
        if not isinstance(baseline_index, dict):
            raise RuntimeError("online_skillbank baseline index is malformed")
        baseline_entries = _baseline_entries_by_index(baseline_index)
        baseline_delta_sum = 0.0
        baseline_wins = 0
        baseline_compared = 0
        for rollout_index, episode in enumerate(episodes):
            baseline_entry = baseline_entries.get(rollout_index, {})
            baseline_won = bool(baseline_entry.get("won", False))
            skill_won = bool(episode.get("won"))
            baseline_wins += int(baseline_won)
            baseline_delta_sum += float(skill_won) - float(baseline_won)
            baseline_compared += 1
        baseline_sr = baseline_wins / max(1, baseline_compared)
        baseline_delta_sr = baseline_delta_sum / max(1, baseline_compared)
        mutual_information_index: dict[str, Any] = {
            "status": "disabled",
            "mode": _DISABLED_MI_MODE,
            "sample_scores": {},
        }
        mutual_information_score = 0.0
        mutual_information_raw_score = 0.0
        mutual_information_nll_margin = 0.0
        mutual_information_abs_token_margin = 0.0
        mutual_information_positive_token_fraction = 0.0
        mutual_information_clipped_token_fraction = 0.0
        mutual_information_token_unsmoothed_reward_mean = 0.0
        mutual_information_token_reward_compression_ratio = 1.0
        mutual_information_token_reward_soft_cap = 0.0
        mutual_information_token_absolute_mode = False
        mutual_information_token_reward_scores: list[float] = []
        mutual_information_target_tokens: list[int] = []
        mutual_information_error = ""
        if self.reward_mutual_information_weight > 0.0:
            if engine is None:
                mutual_information_error = "inference engine unavailable"
            else:
                try:
                    mutual_information_index = self._load_or_create_mi_index(
                        engine, task_type=task_type, round_in_category=round_in_category
                    )
                except Exception as exc:
                    mutual_information_error = repr(exc)
                    logger.warning(
                        "failed to load/create MI index for %s: %s",
                        sample_key,
                        mutual_information_error,
                    )
                    mutual_information_index = {
                        "status": "error",
                        "mode": _DISABLED_MI_MODE,
                        "sample_scores": {},
                        "error": mutual_information_error,
                    }
            sample_mi = (
                mutual_information_index.get("sample_scores", {}).get(sample_key, {})
                if isinstance(mutual_information_index.get("sample_scores"), dict)
                else {}
            )
            mutual_information_score = float(
                sample_mi.get("reward_score", sample_mi.get("mi_score", 0.0)) or 0.0
            )
            mutual_information_raw_score = float(
                sample_mi.get("raw_mi_score", sample_mi.get("mi_score", 0.0)) or 0.0
            )
            mutual_information_nll_margin = float(
                sample_mi.get("nll_margin", 0.0) or 0.0
            )
            mutual_information_abs_token_margin = float(
                sample_mi.get("mean_abs_token_margin", 0.0) or 0.0
            )
            mutual_information_positive_token_fraction = float(
                sample_mi.get("positive_token_fraction", 0.0) or 0.0
            )
            mutual_information_clipped_token_fraction = float(
                sample_mi.get("clipped_token_fraction", 0.0) or 0.0
            )
            mutual_information_token_unsmoothed_reward_mean = (
                self.reward_mutual_information_weight
                * self.reward_mutual_information_scale
                * float(sample_mi.get("mean_token_reward_input", 0.0) or 0.0)
            )
            mutual_information_token_reward_compression_ratio = float(
                sample_mi.get("token_reward_compression_ratio", 1.0) or 0.0
            )
            mutual_information_token_reward_soft_cap = float(
                sample_mi.get("token_reward_soft_cap", 0.0) or 0.0
            )
            mutual_information_token_absolute_mode = bool(
                sample_mi.get("absolute_margin_used_for_reward", False)
            )
            if valid:
                if sample_mi.get("status") != "complete":
                    raise RuntimeError(
                        f"token-wise MI scoring is incomplete for {sample_key}: {sample_mi.get('status', 'missing')} {mutual_information_index.get('score_error', '')}"
                    )
                mutual_information_target_tokens = [
                    int(value) for value in sample_mi.get("target_tokens", [])
                ]
                expected_target_tokens = self._mi_target_tokens(output_tokens)
                if mutual_information_target_tokens != expected_target_tokens:
                    raise RuntimeError(
                        f"token-wise MI target tokens do not match the sampled output for {sample_key}"
                    )
                mutual_information_token_reward_scores = [
                    float(value) for value in sample_mi.get("token_reward_scores", [])
                ]
                if len(mutual_information_token_reward_scores) != len(
                    mutual_information_target_tokens
                ):
                    raise RuntimeError(
                        f"token-wise MI reward count does not match target token count for {sample_key}"
                    )
        reward, schema_reward, sr_reward, mutual_information_reward = (
            self._compute_training_reward(
                sr=sr,
                mutual_information_score=mutual_information_score,
                schema_valid=valid,
            )
        )
        outcome_reward = self.reward_online_skillbank_weight * float(
            outcome_metrics.get("weighted_marginal_score", 0.0) or 0.0
        )
        reward += outcome_reward
        terminal_reward = float(reward)
        token_reward_clip_result = _clip_weighted_tokenwise_mi_rewards(
            mutual_information_token_reward_scores,
            weight=self.reward_mutual_information_weight,
            reward_clip=getattr(
                self, "reward_mutual_information_token_reward_clip", None
            ),
            absolute_mode=True,
        )
        token_rewards = list(token_reward_clip_result["clipped"])
        unclipped_token_rewards = list(token_reward_clip_result["unclipped"])
        mutual_information_token_final_clipped_fraction = float(
            token_reward_clip_result["clipped_fraction"]
        )
        mutual_information_reward = mean(token_rewards) if token_rewards else 0.0
        reward_abs_denominator = (
            abs(float(sr_reward))
            + abs(float(outcome_reward))
            + abs(float(schema_reward))
            + abs(float(mutual_information_reward))
        )
        mutual_information_reward_abs_fraction = (
            abs(float(mutual_information_reward)) / reward_abs_denominator
            if reward_abs_denominator > 0.0
            else 0.0
        )
        per_task_type = _per_task_type_metrics(episodes)
        baseline_summary = {
            "status": baseline_index.get("status", ""),
            "cache_dir": baseline_index.get("baseline_dir", ""),
            "signature": baseline_index.get("signature", ""),
            "wins": baseline_wins,
            "n_rollouts": baseline_compared,
            "sr": baseline_sr,
            "delta_sr": baseline_delta_sr,
            "delta_sum": baseline_delta_sum,
            "error": baseline_index.get("error", ""),
        }
        mi_sample = (
            mutual_information_index.get("sample_scores", {}).get(sample_key, {})
            if isinstance(mutual_information_index.get("sample_scores"), dict)
            else {}
        )
        mutual_information_summary = {
            "status": mi_sample.get(
                "status", mutual_information_index.get("status", "disabled")
            ),
            "mode": mutual_information_index.get("mode", _DISABLED_MI_MODE),
            "signature": mutual_information_index.get("signature", ""),
            "score": mutual_information_score,
            "raw_score": mutual_information_raw_score,
            "nll_margin": mutual_information_nll_margin,
            "mean_abs_token_margin": mutual_information_abs_token_margin,
            "positive_token_fraction": mutual_information_positive_token_fraction,
            "clipped_token_fraction": mutual_information_clipped_token_fraction,
            "token_margin_clip": float(
                mi_sample.get(
                    "token_margin_clip",
                    self.reward_mutual_information_token_margin_clip,
                )
                or 0.0
            ),
            "token_reward_count": len(token_rewards),
            "token_reward_mean": mean(token_rewards) if token_rewards else 0.0,
            "token_reward_min": min(token_rewards) if token_rewards else 0.0,
            "token_reward_max": max(token_rewards) if token_rewards else 0.0,
            "token_unsmoothed_reward_mean": mutual_information_token_unsmoothed_reward_mean,
            "token_reward_compression_ratio": mutual_information_token_reward_compression_ratio,
            "token_reward_soft_cap": mutual_information_token_reward_soft_cap,
            "token_reward_clip": getattr(
                self, "reward_mutual_information_token_reward_clip", None
            ),
            "token_final_clipped_fraction": mutual_information_token_final_clipped_fraction,
            "token_unclipped_reward_mean": mean(unclipped_token_rewards)
            if unclipped_token_rewards
            else 0.0,
            "token_unclipped_reward_min": min(unclipped_token_rewards)
            if unclipped_token_rewards
            else 0.0,
            "token_unclipped_reward_max": max(unclipped_token_rewards)
            if unclipped_token_rewards
            else 0.0,
            "absolute_margin_used_for_reward": mutual_information_token_absolute_mode,
            "token_credit_assignment": mutual_information_index.get(
                "token_credit_assignment", "terminal_sequence_reward"
            ),
            "reward_scale": float(
                mi_sample.get("reward_scale", self.reward_mutual_information_scale)
                or 0.0
            ),
            "matched_avg_nll": float(mi_sample.get("matched_avg_nll", 0.0) or 0.0),
            "marginal_avg_nll": float(mi_sample.get("marginal_avg_nll", 0.0) or 0.0),
            "target_token_count": int(mi_sample.get("target_token_count", 0) or 0),
            "n_inputs": int(mutual_information_index.get("n_inputs", 0) or 0),
            "mean_mi_score": float(
                mutual_information_index.get("mean_mi_score", 0.0) or 0.0
            ),
            "error": mutual_information_error
            or str(mutual_information_index.get("score_error", ""))
            or str(mutual_information_index.get("error", "")),
        }
        outcome_summary = {
            "status": str(outcome_metrics.get("status", "disabled")),
            "mode": str(online_skillbank_index.get("mode", "online_skillbank")),
            "singleton_sr": float(outcome_metrics.get("singleton_sr", 0.0) or 0.0),
            "no_skill_sr": float(outcome_metrics.get("no_skill_sr", 0.0) or 0.0),
            "standalone_delta": float(
                outcome_metrics.get("standalone_delta", 0.0) or 0.0
            ),
            "retrieved_skill_sr": float(
                outcome_metrics.get("retrieved_skill_sr", 0.0) or 0.0
            ),
            "augmented_sr": float(outcome_metrics.get("augmented_sr", 0.0) or 0.0),
            "bank_delta": float(outcome_metrics.get("bank_delta", 0.0) or 0.0),
            "weighted_marginal_score": float(
                outcome_metrics.get("weighted_marginal_score", 0.0) or 0.0
            ),
            "positive_marginal_indicator": float(
                outcome_metrics.get("positive_marginal_indicator", 0.0) or 0.0
            ),
            "positive_marginal_fraction": float(
                outcome_metrics.get("positive_marginal_fraction", 0.0) or 0.0
            ),
            "negative_marginal_fraction": float(
                outcome_metrics.get("negative_marginal_fraction", 0.0) or 0.0
            ),
            "zero_marginal_fraction": float(
                outcome_metrics.get("zero_marginal_fraction", 0.0) or 0.0
            ),
            "marginal_eligible": float(
                outcome_metrics.get("marginal_eligible", 0.0) or 0.0
            ),
            "reward": outcome_reward,
            "weight": float(self.reward_online_skillbank_weight),
            "bank_weight": float(getattr(self, "online_skillbank_bank_weight", 0.0)),
            "standalone_weight": float(
                getattr(self, "online_skillbank_standalone_weight", 0.0)
            ),
            "singleton_condition_dir": str(
                outcome_metrics.get("singleton_condition_dir", "")
            ),
            "signature": str(online_skillbank_index.get("signature", "")),
            "bank_before_signature": str(
                online_skillbank_index.get("bank_before_signature", "")
            ),
            "bank_before_size": int(
                online_skillbank_index.get("bank_before_size", 0) or 0
            ),
            "retrieval": online_skillbank_index.get("retrieval", {}),
            "planned_episode_budget": int(
                online_skillbank_index.get("planned_episode_budget", 0) or 0
            ),
            "planned_compute_multiplier": float(
                online_skillbank_index.get("planned_compute_multiplier", 0.0) or 0.0
            ),
            "actual_environment_episode_budget": int(
                online_skillbank_index.get("actual_environment_episode_budget", 0) or 0
            ),
            "actual_environment_compute_multiplier": float(
                online_skillbank_index.get("actual_environment_compute_multiplier", 0.0)
                or 0.0
            ),
        }
        rollouts_payload = {
            "skill_name": skill_name,
            "task_type": task_type,
            "input_task_type": input_task_type,
            "prompt_category": prompt_category,
            "round_in_category": round_in_category,
            "group_index": group_index,
            "sample_key": sample_key,
            "schema_valid": valid,
            "schema_error": parse_error,
            "attempt_count": len(attempts),
            "max_retries": max_retries,
            "wins": wins,
            "n_rollouts": len(episodes),
            "sr": sr,
            "reward": reward,
            "terminal_reward": terminal_reward,
            "sequence_reward": reward,
            "token_rewards": token_rewards,
            "token_reward_target_tokens": mutual_information_target_tokens,
            "task_reward": sr,
            "schema_reward": schema_reward,
            "sr_reward": sr_reward,
            "outcome_reward": outcome_reward,
            "mutual_information_reward": mutual_information_reward,
            "mutual_information_reward_abs_fraction": mutual_information_reward_abs_fraction,
            "mutual_information_score": mutual_information_score,
            "mutual_information_raw_score": mutual_information_raw_score,
            "mutual_information_nll_margin": mutual_information_nll_margin,
            "mutual_information_abs_token_margin": mutual_information_abs_token_margin,
            "mutual_information_positive_token_fraction": mutual_information_positive_token_fraction,
            "mutual_information_clipped_token_fraction": mutual_information_clipped_token_fraction,
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
            "mutual_information_token_unsmoothed_reward_mean": mutual_information_token_unsmoothed_reward_mean,
            "mutual_information_token_reward_compression_ratio": mutual_information_token_reward_compression_ratio,
            "mutual_information_token_reward_soft_cap": mutual_information_token_reward_soft_cap,
            "mutual_information_token_reward_clip": getattr(
                self, "reward_mutual_information_token_reward_clip", None
            ),
            "mutual_information_token_final_clipped_fraction": mutual_information_token_final_clipped_fraction,
            "mutual_information_token_unclipped_reward_mean": mean(
                unclipped_token_rewards
            )
            if unclipped_token_rewards
            else 0.0,
            "mutual_information_token_absolute_mode": mutual_information_token_absolute_mode,
            "baseline_sr": baseline_sr,
            "baseline_delta_sr": baseline_delta_sr,
            "baseline": baseline_summary,
            "outcome": outcome_summary,
            "mutual_information": mutual_information_summary,
            "per_task_type": per_task_type,
            "status_counts": dict(status_counts),
            "episodes": episodes,
            "updated_at": time.time(),
        }
        _atomic_write_json(sample_dir / "rollouts.json", rollouts_payload)
        self._record_training_step_trajectories(prepared=prepared, episodes=episodes)
        _atomic_write_json(
            sample_dir / "metrics.json",
            {
                "skill_name": skill_name,
                "task_type": task_type,
                "prompt_category": prompt_category,
                "round_in_category": round_in_category,
                "group_index": group_index,
                "sample_key": sample_key,
                "schema_valid": valid,
                "attempt_count": len(attempts),
                "max_retries": max_retries,
                "wins": wins,
                "n_rollouts": len(episodes),
                "sr": sr,
                "reward": reward,
                "task_reward": sr,
                "schema_reward": schema_reward,
                "sr_reward": sr_reward,
                "outcome_reward": outcome_reward,
                "mutual_information_reward": mutual_information_reward,
                "mutual_information_reward_abs_fraction": mutual_information_reward_abs_fraction,
                "mutual_information_score": mutual_information_score,
                "mutual_information_nll_margin": mutual_information_nll_margin,
                "mutual_information_abs_token_margin": mutual_information_abs_token_margin,
                "mutual_information_positive_token_fraction": mutual_information_positive_token_fraction,
                "mutual_information_clipped_token_fraction": mutual_information_clipped_token_fraction,
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
                "mutual_information_token_unsmoothed_reward_mean": mutual_information_token_unsmoothed_reward_mean,
                "mutual_information_token_reward_compression_ratio": mutual_information_token_reward_compression_ratio,
                "mutual_information_token_reward_soft_cap": mutual_information_token_reward_soft_cap,
                "mutual_information_token_absolute_mode": mutual_information_token_absolute_mode,
                "terminal_reward": terminal_reward,
                "sequence_reward": reward,
                "baseline_sr": baseline_sr,
                "baseline_delta_sr": baseline_delta_sr,
                "baseline": baseline_summary,
                "outcome": outcome_summary,
                "mutual_information": mutual_information_summary,
                "per_task_type": per_task_type,
                "updated_at": time.time(),
            },
        )
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "complete",
                "task_type": task_type,
                "round_in_category": round_in_category,
                "group_index": group_index,
                "sample_key": sample_key,
                "schema_valid": valid,
                "attempt_count": len(attempts),
                "max_retries": max_retries,
                "wins": wins,
                "n_rollouts": len(episodes),
                "sr": sr,
                "baseline_sr": baseline_sr,
                "baseline_delta_sr": baseline_delta_sr,
                "outcome": outcome_summary,
                "outcome_reward": outcome_reward,
                "mutual_information_score": mutual_information_score,
                "mutual_information_raw_score": mutual_information_raw_score,
                "mutual_information_nll_margin": mutual_information_nll_margin,
                "reward": reward,
                "terminal_reward": terminal_reward,
                "sequence_reward": reward,
                "mutual_information_token_reward_mean": mean(token_rewards)
                if token_rewards
                else 0.0,
                "mutual_information_token_reward_count": len(token_rewards),
                "mutual_information_token_unsmoothed_reward_mean": mutual_information_token_unsmoothed_reward_mean,
                "mutual_information_token_reward_compression_ratio": mutual_information_token_reward_compression_ratio,
                "mutual_information_token_reward_soft_cap": mutual_information_token_reward_soft_cap,
                "mutual_information_token_absolute_mode": mutual_information_token_absolute_mode,
                "updated_at": time.time(),
            },
        )
        _remove_completed_episode_snapshots(sample_dir)
        return rollouts_payload
