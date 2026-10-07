# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from areal.api import InferenceEngine
from areal.workflow.alfworld_runtime import (
    _MI_COMPUTE_LOCK,
    _acquire_json_lock,
    _atomic_write_json,
    _read_json,
    _release_json_lock,
    logger,
)
from areal.workflow.skill_prompts import (
    _resolve_skill_xml_token_reward_mask,
    skill_payload_to_prompt_text,
)
from areal.workflow.skill_token_feedback import (
    _logmeanexp,
    _mi_round_signature,
    _strip_trailing_token_ids,
    _tokenwise_mi_statistics,
)

# Only the XML-wrapped skill body receives token-wise MI credit.
_TOKEN_REWARD_MASK_MODE = "skill_xml_body_only_v1"


class SkillFeedbackMixin:
    def _mi_target_tokens(self, output_tokens: Sequence[int]) -> list[int]:
        gconfig = getattr(self, "gconfig", None)
        return _strip_trailing_token_ids(
            output_tokens, list(getattr(gconfig, "stop_token_ids", []) or [])
        )

    def _compute_training_reward(
        self, *, sr: float, mutual_information_score: float, schema_valid: bool
    ) -> tuple[float, float, float, float]:
        """Scalar SR/schema reward; MI is applied separately to knowledge tokens."""
        schema_reward = (
            self.reward_schema_valid_bonus
            if schema_valid
            else self.reward_schema_invalid_penalty
        )
        sr_reward = self.reward_sr_weight * float(sr)
        mi_summary = self.reward_mutual_information_weight * float(
            mutual_information_score
        )
        return (
            float(sr_reward + schema_reward),
            float(schema_reward),
            float(sr_reward),
            float(mi_summary),
        )

    def _mi_index_matches_current_token_config(self, payload: dict[str, Any]) -> bool:
        if payload.get("status") != "complete":
            return False
        if payload.get("reward_mode") != "tokenwise_abs_aligned":
            return False
        if bool(payload.get("absolute_margin_used_for_reward", False)) != True:
            return False
        if payload.get("token_reward_mask_mode") != _TOKEN_REWARD_MASK_MODE:
            return False

        def same_optional_float(left: Any, right: float | None) -> bool:
            if left is None or right is None:
                return left is None and right is None
            try:
                return math.isclose(
                    float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12
                )
            except (TypeError, ValueError):
                return False

        return (
            same_optional_float(
                payload.get("token_margin_clip"),
                self.reward_mutual_information_token_margin_clip,
            )
            and same_optional_float(
                payload.get("token_reward_soft_cap"),
                self.reward_mutual_information_token_reward_soft_cap,
            )
            and same_optional_float(
                payload.get("reward_scale"), self.reward_mutual_information_scale
            )
        )

    def _load_or_create_mi_index(
        self, engine: InferenceEngine, *, task_type: str, round_in_category: int
    ) -> dict[str, Any]:
        round_dir = self._round_dir(task_type, round_in_category)
        mi_dir = round_dir / "mi"
        mi_dir.mkdir(parents=True, exist_ok=True)
        index_name = "mi_tokenwise_abs_aligned_index.json"
        index_path = mi_dir / index_name
        # A cached index is reused only for the exact generations now in the round.
        inputs = self._mi_round_inputs(round_dir)

        def read_reusable() -> dict[str, Any] | None:
            if not index_path.exists():
                return None
            try:
                payload = _read_json(index_path)
            except Exception:
                return None
            if (
                self._mi_index_matches_current_token_config(payload)
                and payload.get("signature") == inputs["signature"]
            ):
                return payload
            return None

        payload = read_reusable()
        if payload is not None:
            return payload
        lock_path = mi_dir / f"{index_path.stem}.lock"
        fd = _acquire_json_lock(lock_path, timeout_s=1800.0)
        try:
            payload = read_reusable()
            if payload is not None:
                return payload
            return self._compute_and_write_mi_index(
                engine=engine,
                task_type=task_type,
                round_in_category=round_in_category,
                index_path=index_path,
                inputs=inputs,
            )
        finally:
            _release_json_lock(fd, lock_path)

    def _cached_token_reward_mask(
        self,
        generation: dict[str, Any],
        target_tokens: Sequence[int],
        *,
        schema_valid: bool,
    ) -> list[int]:
        # Resolving a mask decodes every token prefix, and each sample of a round
        # re-derives the round signature before reusing the shared index.
        if not schema_valid:
            return [0] * len(target_tokens)
        key = (str(generation.get("raw", "")), tuple(int(t) for t in target_tokens))
        cache = self.__dict__.setdefault("_mi_token_reward_mask_cache", {})
        mask = cache.get(key)
        if mask is None:
            mask = _resolve_skill_xml_token_reward_mask(
                self.tokenizer, generation, target_tokens, schema_valid=True
            )
            if len(cache) >= 4096:
                cache.clear()
            cache[key] = mask
        return list(mask)

    def _mi_round_inputs(self, round_dir: Path) -> dict[str, Any]:
        abs_mode_name = "input_skill_tokenwise_abs_aligned_mi_v1"
        records: list[dict[str, Any]] = []
        prompts_by_group: dict[int, dict[str, Any]] = {}
        for generation_path in sorted(round_dir.glob("skills/*/generation.json")):
            try:
                generation = _read_json(generation_path)
                _read_json(generation_path.parent / "prompt.json")
            except Exception:
                continue
            sample_key = generation_path.parent.name
            group_index = int(
                generation.get(
                    "group_index", self._group_index_from_sample_key(sample_key)
                )
            )
            parsed = generation.get("parsed")
            schema_valid = bool(generation.get("schema_valid")) and isinstance(
                parsed, dict
            )
            target_text = skill_payload_to_prompt_text(parsed) if schema_valid else ""
            if schema_valid:
                saved_target_tokens = generation.get("mi_target_tokens")
                if not isinstance(saved_target_tokens, list):
                    saved_target_tokens = generation.get("output_tokens")
                if isinstance(saved_target_tokens, list):
                    target_tokens = self._mi_target_tokens(saved_target_tokens)
                else:
                    target_tokens = self.tokenizer.encode(
                        target_text, add_special_tokens=False
                    )
            else:
                target_tokens = (
                    self.tokenizer.encode(target_text, add_special_tokens=False)
                    if target_text
                    else []
                )
            try:
                token_reward_mask = self._cached_token_reward_mask(
                    generation,
                    target_tokens,
                    schema_valid=schema_valid,
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"skill_xml token-wise MI mask is missing or invalid for {sample_key}"
                ) from exc
            saved_input_ids = generation.get("input_tokens")
            if not isinstance(saved_input_ids, list) or not saved_input_ids:
                raise RuntimeError(
                    f"aligned tokenwise MI requires saved input tokens: {sample_key}"
                )
            prompt_input_ids = list(map(int, saved_input_ids))
            existing_prompt = prompts_by_group.get(group_index)
            if existing_prompt is None:
                prompts_by_group[group_index] = {
                    "group_index": group_index,
                    "sample_key": sample_key,
                    "prompt_path": str(generation_path.parent / "prompt.json"),
                    "input_token_count": len(prompt_input_ids),
                    "input_ids": prompt_input_ids,
                }
            elif existing_prompt["input_ids"] != prompt_input_ids:
                raise RuntimeError(
                    f"tokenwise MI requires every sampled skill in one group to share the exact same input prompt; mismatch in group_{group_index:02d} at {sample_key}"
                )
            records.append(
                {
                    "sample_key": sample_key,
                    "group_index": group_index,
                    "generation_path": str(generation_path),
                    "schema_valid": schema_valid,
                    "target_text": target_text,
                    "target_token_count": len(target_tokens),
                    "target_tokens": [int(token_id) for token_id in target_tokens],
                    "token_reward_mask": token_reward_mask,
                }
            )
        prompt_items = [
            item for _, item in sorted(prompts_by_group.items(), key=lambda kv: kv[0])
        ]
        if len(prompt_items) != self.groups_per_round:
            raise RuntimeError(
                f"tokenwise MI expected {self.groups_per_round} input groups, found {len(prompt_items)}"
            )
        signature = _mi_round_signature(
            prompt_keys=[str(item["sample_key"]) for item in prompt_items],
            sample_keys=[str(record["sample_key"]) for record in records],
            target_texts=[str(record["target_text"]) for record in records],
            tokenizer_path=self.tokenizer_path,
            mode=abs_mode_name,
            target_token_ids=[list(record["target_tokens"]) for record in records],
            token_reward_masks=[
                list(record["token_reward_mask"]) for record in records
            ],
            token_margin_clip=self.reward_mutual_information_token_margin_clip,
            token_reward_soft_cap=self.reward_mutual_information_token_reward_soft_cap,
        )
        return {
            "mode": abs_mode_name,
            "records": records,
            "prompt_items": prompt_items,
            "signature": signature,
        }

    def _compute_and_write_mi_index(
        self,
        *,
        engine: InferenceEngine,
        task_type: str,
        round_in_category: int,
        index_path: Path,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        round_dir = self._round_dir(task_type, round_in_category)
        if inputs is None:
            inputs = self._mi_round_inputs(round_dir)
        abs_mode_name = str(inputs["mode"])
        records: list[dict[str, Any]] = list(inputs["records"])
        prompt_items: list[dict[str, Any]] = list(inputs["prompt_items"])
        prompt_group_indices = [int(item["group_index"]) for item in prompt_items]
        signature = str(inputs["signature"])
        if not prompt_items or not records:
            payload = {
                "status": "error",
                "mode": abs_mode_name,
                "signature": signature,
                "error": "missing prompts or generation records",
                "sample_scores": {},
                "updated_at": time.time(),
            }
            _atomic_write_json(index_path, payload)
            return payload
        score_jobs: list[tuple[int, int, str, int]] = []
        score_data: list[dict[str, torch.Tensor]] = []
        for record_index, record in enumerate(records):
            target_tokens = list(record.get("target_tokens") or [])
            if not target_tokens:
                continue
            for prompt_index, prompt_item in enumerate(prompt_items):
                prompt_ids = list(prompt_item["input_ids"])
                token_ids = prompt_ids + target_tokens
                loss_mask = [0] * len(prompt_ids) + [1] * len(target_tokens)
                score_jobs.append(
                    (
                        record_index,
                        prompt_index,
                        str(record["sample_key"]),
                        int(prompt_item["group_index"]),
                    )
                )
                score_data.append(
                    {
                        "input_ids": torch.tensor([token_ids], dtype=torch.int32),
                        "loss_mask": torch.tensor([loss_mask], dtype=torch.int32),
                        "attention_mask": torch.ones(
                            (1, len(token_ids)), dtype=torch.bool
                        ),
                    }
                )
        diagnostic_model_version = (
            int(engine.get_version()) if hasattr(engine, "get_version") else None
        )
        score_errors: list[str] = []
        score_results: list[torch.Tensor] = []
        if score_data:
            try:
                with _MI_COMPUTE_LOCK:
                    score_results = engine.compute_logp(score_data)
            except Exception as exc:
                score_errors.append(repr(exc))
                logger.warning(
                    "failed to compute MI logprobs for %s: %s", round_dir, exc
                )
        logp_by_sample: dict[str, dict[int, float]] = {
            str(record["sample_key"]): {} for record in records
        }
        diagnostic_logps_by_sample: dict[str, dict[int, list[float]]] = {}
        token_logps_by_sample: dict[str, dict[int, list[float]]] = {
            str(record["sample_key"]): {} for record in records
        }
        if score_results and len(score_results) == len(score_jobs):
            for score_index, (job, score_tensor) in enumerate(
                zip(score_jobs, score_results, strict=False)
            ):
                _, _, sample_key, prompt_group_index = job
                token_mask = score_data[score_index]["loss_mask"].reshape(-1).bool()
                score_mask = token_mask
                values = score_tensor.detach().float().reshape(-1)
                if values.numel() != score_mask.numel():
                    score_errors.append(
                        f"{sample_key}/group_{prompt_group_index}: score shape {values.numel()} != mask shape {score_mask.numel()}"
                    )
                    continue
                diagnostic_logps_by_sample.setdefault(sample_key, {})[
                    prompt_group_index
                ] = values[token_mask].detach().cpu().tolist()
                target_values = [
                    float(value) for value in values[score_mask].detach().cpu().tolist()
                ]
                expected_count = int(records[job[0]].get("target_token_count", 0) or 0)
                if len(target_values) != expected_count:
                    score_errors.append(
                        f"{sample_key}/group_{prompt_group_index}: expected {expected_count} target token scores, got {len(target_values)}"
                    )
                    continue
                logp_by_sample.setdefault(sample_key, {})[prompt_group_index] = sum(
                    target_values
                )
                token_logps_by_sample.setdefault(sample_key, {})[prompt_group_index] = (
                    target_values
                )
        elif score_results:
            score_errors.append(
                f"expected {len(score_jobs)} score results, got {len(score_results)}"
            )
        sample_scores: dict[str, dict[str, Any]] = {}
        mi_scores: list[float] = []
        for record in records:
            sample_key = str(record["sample_key"])
            group_index = int(record["group_index"])
            target_tokens = [int(value) for value in record.get("target_tokens", [])]
            target_token_count = len(target_tokens)
            token_reward_mask = [
                int(bool(value)) for value in record.get("token_reward_mask", [])
            ]
            if len(token_reward_mask) != target_token_count:
                token_reward_mask = [1] * target_token_count
            cross_logps = logp_by_sample.get(sample_key, {})
            matched_logp = cross_logps.get(group_index)
            marginal_logp = _logmeanexp(
                [
                    value
                    for prompt_group_index, value in cross_logps.items()
                    if prompt_group_index != group_index
                ]
            )
            common: dict[str, Any] = {
                "sample_key": sample_key,
                "group_index": group_index,
                "schema_valid": bool(record.get("schema_valid")),
                "target_token_count": target_token_count,
                "target_tokens": target_tokens,
                "matched_logp": matched_logp,
                "marginal_logp": marginal_logp
                if math.isfinite(marginal_logp)
                else None,
                "marginal_excludes_matched_group": True,
                "cross_logps": {
                    str(key): value for key, value in sorted(cross_logps.items())
                },
            }
            cross_token_logps = token_logps_by_sample.get(sample_key, {})
            common["diagnostic_cross_token_logps"] = {
                str(key): row
                for key, row in diagnostic_logps_by_sample.get(sample_key, {}).items()
            }
            matched_tokens = cross_token_logps.get(group_index)
            unmatched_tokens = [
                values
                for prompt_group_index, values in sorted(cross_token_logps.items())
                if prompt_group_index != group_index
            ]
            if matched_tokens is None or target_token_count <= 0:
                status = "unscored"
                token_stats: dict[str, Any] = {}
            else:
                try:
                    token_stats = _tokenwise_mi_statistics(
                        matched_tokens,
                        unmatched_tokens,
                        margin_clip=self.reward_mutual_information_token_margin_clip,
                        absolute_margin=True,
                        reward_soft_cap=self.reward_mutual_information_token_reward_soft_cap,
                    )
                    semantic_indices = [
                        index
                        for index, include in enumerate(token_reward_mask)
                        if include
                    ]
                    if not semantic_indices:
                        raise ValueError(
                            "token-wise MI has no rewardable semantic tokens"
                        )
                    semantic_stats = _tokenwise_mi_statistics(
                        [matched_tokens[index] for index in semantic_indices],
                        [
                            [values[index] for index in semantic_indices]
                            for values in unmatched_tokens
                        ],
                        margin_clip=self.reward_mutual_information_token_margin_clip,
                        absolute_margin=True,
                        reward_soft_cap=self.reward_mutual_information_token_reward_soft_cap,
                    )
                    aggregate_keys = (
                        "mean_token_margin",
                        "mean_abs_token_margin",
                        "mean_clipped_token_margin",
                        "mean_token_reward_input",
                        "mean_token_reward_value",
                        "min_token_reward_value",
                        "max_token_reward_value",
                        "token_reward_compression_ratio",
                        "positive_token_fraction",
                        "clipped_token_fraction",
                        "matched_avg_nll",
                        "marginal_avg_nll",
                        "nll_margin",
                    )
                    for key in aggregate_keys:
                        token_stats[f"all_token_{key}"] = token_stats[key]
                        token_stats[key] = semantic_stats[key]
                    status = "complete"
                except ValueError as exc:
                    token_stats = {}
                    status = "unscored"
                    score_errors.append(f"{sample_key}: {exc}")
            raw_score = float(token_stats.get("mean_token_margin", 0.0))
            clipped_score = float(token_stats.get("mean_clipped_token_margin", 0.0))
            reward_score = (
                float(token_stats.get("mean_token_reward_value", 0.0))
                * self.reward_mutual_information_scale
            )
            token_reward_values = [
                float(value) for value in token_stats.get("token_reward_values", [])
            ]
            if len(token_reward_values) != len(token_reward_mask):
                if status == "complete":
                    raise RuntimeError(
                        f"{sample_key}: token reward mask length does not match token-wise MI values"
                    )
                token_reward_values = [0.0] * len(token_reward_mask)
            token_reward_scores = [
                value * self.reward_mutual_information_scale * mask
                for value, mask in zip(
                    token_reward_values, token_reward_mask, strict=True
                )
            ]
            semantic_token_count = sum(token_reward_mask)
            masked_reward_score = (
                sum(token_reward_scores) / semantic_token_count
                if semantic_token_count > 0
                else 0.0
            )
            try:
                token_pieces = self.tokenizer.convert_ids_to_tokens(target_tokens)
            except Exception:
                token_pieces = []
            common.update(
                {
                    "status": status,
                    "mi_score": raw_score,
                    "raw_mi_score": raw_score,
                    "clipped_mi_score": clipped_score,
                    "reward_score": masked_reward_score
                    if status == "complete"
                    else 0.0,
                    "reward_mode": "tokenwise_abs_aligned",
                    "reward_scale": self.reward_mutual_information_scale,
                    "token_margin_clip": self.reward_mutual_information_token_margin_clip,
                    "token_reward_soft_cap": self.reward_mutual_information_token_reward_soft_cap,
                    "target_token_pieces": token_pieces,
                    "token_reward_mask": token_reward_mask,
                    "semantic_token_count": semantic_token_count,
                    **token_stats,
                    "token_reward_scores": token_reward_scores
                    if status == "complete"
                    else [],
                }
            )
            if status == "complete":
                mi_scores.append(raw_score)
            sample_scores[sample_key] = common
        reward_scores: list[float] = []
        for sample in sample_scores.values():
            raw_score = float(sample.get("mi_score", 0.0) or 0.0)
            reward_score = float(sample.get("reward_score", 0.0) or 0.0)
            sample["reward_score"] = reward_score
            sample["reward_mode"] = "tokenwise_abs_aligned"
            sample["reward_scale"] = self.reward_mutual_information_scale
            if sample.get("status") == "complete":
                reward_scores.append(float(reward_score))
        mode_name = abs_mode_name
        score_error = "; ".join(score_errors)
        payload = {
            "status": "complete" if not score_error else "partial",
            "mode": mode_name,
            "reward_mode": "tokenwise_abs_aligned",
            "reward_score_alignment": "remote_loss_mask_target_positions_v1",
            "signature": signature,
            "task_type": task_type,
            "round_in_category": round_in_category,
            "n_inputs": len(prompt_items),
            "n_samples": len(records),
            "prompt_group_indices": prompt_group_indices,
            "marginal_excludes_matched_group": True,
            "mean_mi_score": mean(mi_scores) if mi_scores else 0.0,
            "min_mi_score": min(mi_scores) if mi_scores else 0.0,
            "max_mi_score": max(mi_scores) if mi_scores else 0.0,
            "mean_reward_score": mean(reward_scores) if reward_scores else 0.0,
            "min_reward_score": min(reward_scores) if reward_scores else 0.0,
            "max_reward_score": max(reward_scores) if reward_scores else 0.0,
            "reward_scale": self.reward_mutual_information_scale,
            "token_margin_clip": self.reward_mutual_information_token_margin_clip,
            "token_reward_soft_cap": self.reward_mutual_information_token_reward_soft_cap,
            "token_credit_assignment": "direct_token_advantage",
            "token_reward_mask_mode": _TOKEN_REWARD_MASK_MODE,
            "absolute_margin_used_for_reward": True,
            "diagnostic_model_version": diagnostic_model_version
            if diagnostic_model_version is not None
            and int(engine.get_version()) == diagnostic_model_version
            else None,
            "diagnostic_score_alignment": "remote_loss_mask_target_positions_v1",
            "diagnostic_prompt_tokens_by_group": {
                str(item["group_index"]): item["input_ids"] for item in prompt_items
            },
            "score_error": score_error,
            "sample_scores": sample_scores,
            "updated_at": time.time(),
        }
        _atomic_write_json(index_path, payload)
        return payload
