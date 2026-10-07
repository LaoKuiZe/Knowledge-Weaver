# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from statistics import mean
from typing import Any


def _logmeanexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    max_value = max((float(value) for value in values))
    if not math.isfinite(max_value):
        return max_value
    return max_value + math.log(
        sum((math.exp(float(value) - max_value) for value in values)) / len(values)
    )


def _output_token_nll_stats(logprobs: Sequence[Any]) -> tuple[float, int]:
    """Return sampled-output mean token NLL and the number of scored tokens."""
    finite_logprobs: list[float] = []
    for logprob in logprobs:
        try:
            value = float(logprob)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite_logprobs.append(value)
    if not finite_logprobs:
        return 0.0, 0
    return -sum(finite_logprobs) / len(finite_logprobs), len(finite_logprobs)


def _strip_trailing_token_ids(
    token_ids: Sequence[int], stop_token_ids: Sequence[int]
) -> list[int]:
    result = [int(token_id) for token_id in token_ids]
    stops = {int(token_id) for token_id in stop_token_ids if token_id is not None}
    while result and result[-1] in stops:
        result.pop()
    return result


def _tokenwise_mi_statistics(
    matched_logps: Sequence[float],
    unmatched_logps: Sequence[Sequence[float]],
    *,
    margin_clip: float | None,
    absolute_margin: bool = False,
    reward_soft_cap: float | None = None,
) -> dict[str, Any]:
    """Compute per-token matched-vs-unmatched margins and reward values."""
    matched = [float(value) for value in matched_logps]
    unmatched = [[float(value) for value in row] for row in unmatched_logps]
    if margin_clip is not None and margin_clip <= 0.0:
        raise ValueError("margin_clip must be positive when enabled")
    if reward_soft_cap is not None and reward_soft_cap <= 0.0:
        raise ValueError("reward_soft_cap must be positive when enabled")
    if not matched:
        raise ValueError("matched_logps must not be empty")
    if not unmatched:
        raise ValueError("at least one unmatched input is required")
    if any(len(row) != len(matched) for row in unmatched):
        raise ValueError("all matched and unmatched token log-prob lists must align")
    if any(not math.isfinite(value) for row in [matched, *unmatched] for value in row):
        raise ValueError("token log-probabilities must be finite")

    marginal = [
        _logmeanexp([row[token_index] for row in unmatched])
        for token_index in range(len(matched))
    ]
    margins = [
        matched_value - marginal_value
        for matched_value, marginal_value in zip(matched, marginal, strict=True)
    ]
    clipped = (
        list(margins)
        if margin_clip is None
        else [
            min(float(margin_clip), max(-float(margin_clip), margin))
            for margin in margins
        ]
    )
    reward_inputs = [abs(value) if absolute_margin else value for value in clipped]
    reward_values = (
        list(reward_inputs)
        if reward_soft_cap is None
        else [
            float(reward_soft_cap) * math.tanh(value / float(reward_soft_cap))
            for value in reward_inputs
        ]
    )
    mean_abs_reward_input = mean(abs(value) for value in reward_inputs)
    mean_abs_reward_value = mean(abs(value) for value in reward_values)
    compression_ratio = (
        mean_abs_reward_value / mean_abs_reward_input
        if mean_abs_reward_input > 0.0
        else 1.0
    )
    return {
        "matched_token_logps": matched,
        "marginal_token_logps": marginal,
        "token_margins": margins,
        "clipped_token_margins": clipped,
        "token_reward_inputs": reward_inputs,
        "token_reward_values": reward_values,
        "mean_token_margin": mean(margins),
        "mean_abs_token_margin": mean(abs(value) for value in margins),
        "mean_clipped_token_margin": mean(clipped),
        "mean_token_reward_input": mean(reward_inputs),
        "mean_token_reward_value": mean(reward_values),
        "min_token_reward_value": min(reward_values),
        "max_token_reward_value": max(reward_values),
        "token_reward_compression_ratio": compression_ratio,
        "absolute_margin_used_for_reward": bool(absolute_margin),
        "token_reward_soft_cap": reward_soft_cap,
        "positive_token_fraction": (
            sum(value > 0.0 for value in margins) / len(margins)
        ),
        "clipped_token_fraction": (
            0.0
            if margin_clip is None
            else sum(abs(value) > float(margin_clip) for value in margins)
            / len(margins)
        ),
        "matched_avg_nll": -mean(matched),
        "marginal_avg_nll": -mean(marginal),
        "nll_margin": -mean(margins),
        "unmatched_input_count": len(unmatched),
    }


def _mi_round_signature(
    *,
    prompt_keys: Sequence[str],
    sample_keys: Sequence[str],
    target_texts: Sequence[str],
    tokenizer_path: str,
    mode: str,
    target_token_ids: Sequence[Sequence[int]] | None = None,
    token_reward_masks: Sequence[Sequence[int]] | None = None,
    token_margin_clip: float | None = None,
    token_reward_soft_cap: float | None = None,
) -> str:
    payload: dict[str, Any] = {
        "mode": mode,
        "prompt_keys": list(prompt_keys),
        "sample_keys": list(sample_keys),
        "target_text_hashes": [
            hashlib.sha1(text.encode("utf-8")).hexdigest() for text in target_texts
        ],
        "tokenizer_path": tokenizer_path,
    }
    if target_token_ids is not None:
        payload["target_token_hashes"] = [
            hashlib.sha1(
                json.dumps([int(token_id) for token_id in tokens]).encode("utf-8")
            ).hexdigest()
            for tokens in target_token_ids
        ]
    if token_reward_masks is not None:
        payload["token_reward_mask_hashes"] = [
            hashlib.sha1(
                json.dumps([int(bool(value)) for value in mask]).encode("utf-8")
            ).hexdigest()
            for mask in token_reward_masks
        ]
    if token_margin_clip is not None:
        payload["token_margin_clip"] = float(token_margin_clip)
    if token_reward_soft_cap is not None:
        payload["token_reward_soft_cap"] = float(token_reward_soft_cap)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _clip_weighted_tokenwise_mi_rewards(
    scores: Sequence[float],
    *,
    weight: float,
    reward_clip: float | None,
    absolute_mode: bool,
) -> dict[str, Any]:
    unbounded = [float(weight) * float(score) for score in scores]
    if reward_clip is None:
        clipped = list(unbounded)
    else:
        bound = float(reward_clip)
        if absolute_mode:
            clipped = [min(bound, max(0.0, value)) for value in unbounded]
        else:
            clipped = [min(bound, max(-bound, value)) for value in unbounded]
    changed = sum(
        not math.isclose(before, after, rel_tol=0.0, abs_tol=1.0e-12)
        for before, after in zip(unbounded, clipped, strict=True)
    )
    return {
        "unclipped": unbounded,
        "clipped": clipped,
        "clipped_fraction": changed / max(1, len(unbounded)),
        "clip": reward_clip,
    }
