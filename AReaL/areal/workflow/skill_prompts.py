# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.utils.hf_utils import apply_chat_template
from areal.workflow.alfworld_runtime import (
    _COMPACT_TRAJECTORY_PROFILES,
    _PROMPT_OBSERVATION_CHAR_LIMIT,
    _PROMPT_RESULT_CHAR_LIMIT,
    _PROMPT_TRACE_MAX_STEPS,
    _SKILL_PROMPT_TOTAL_TOKEN_BUDGET,
    _SKILL_PROMPT_VERSIONS,
    _TRAJECTORY_RENDER_VERSION,
    _UNUSABLE_PROMPT_STEP_STATUSES,
    ALFWORLD_ACTION_SPACE,
    SKILL_XML_CLOSE_TAG,
    SKILL_XML_OPEN_TAG,
    SKILL_XML_OUTPUT_REQUIREMENTS,
    ZERO_SHOT_TRAJECTORIES,
    _acquire_json_lock,
    _atomic_write_json,
    _manifest_trajectory_refs,
    _read_json,
    _release_json_lock,
    _shorten,
    _skill_manifest_input_id,
)
from areal.workflow.alfworld_skill_evidence_prompt import (
    EVIDENCE_DISCOVERY_SYSTEM_PROMPT,
    EVIDENCE_DISCOVERY_USER_CONTENT_TEMPLATE,
)
from areal.workflow.skill_prompt import skill_prompt_budgets

# Shown next to every input's category; lists all six ALFWorld task types.
_CATEGORY_NOTE = (
    "one of: pick_and_place, look_at_obj_in_light, pick_heat_then_place, "
    "pick_clean_then_place, pick_cool_then_place, pick_two_obj_and_place"
)


def _split_skill_generation_sections(raw_generation: str) -> tuple[str, str, list[str]]:
    text = str(raw_generation or "").strip().lstrip("\ufeff")
    notes: list[str] = []
    thinking = ""
    answer = ""

    thinking_match = re.search(
        r"<(?:think|thinking)>\s*(.*?)\s*</(?:think|thinking)>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        notes.append("thinking_extracted")

    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        answer = answer_match.group(1).strip()
        notes.append("answer_extracted")
        return thinking, answer, notes

    slash_answer_match = re.search(
        r"/answer\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if slash_answer_match:
        answer = slash_answer_match.group(1).strip()
        notes.append("slash_answer_extracted")
        return thinking, answer, notes

    if thinking_match:
        answer = text[thinking_match.end() :].strip()
        if answer:
            notes.append("answer_after_thinking_extracted")
            return thinking, answer, notes

    return thinking, text, notes


def _parse_skill_generation(
    raw_generation: str,
    expected_category: str,
    *,
    skill_output_format: str = "skill_xml",
    max_words: int = 0,
) -> tuple[dict[str, Any] | None, bool, str, list[str]]:
    """Parse exactly one ``<skill>...</skill>`` body into a description payload."""

    if str(skill_output_format or "").strip().lower() != "skill_xml":
        raise ValueError("skill_output_format must be 'skill_xml'")
    tagged_text = str(raw_generation or "").strip().lstrip("\ufeff")
    tagged_text = re.sub(
        r"(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>)+\s*$",
        "",
        tagged_text,
        flags=re.IGNORECASE,
    ).strip()
    match = re.fullmatch(
        r"<skill>\s*(.*?)\s*</skill>",
        tagged_text,
        flags=re.DOTALL,
    )
    if match is None:
        return (
            {"description": ""},
            False,
            "skill_xml output must be exactly <skill>...</skill>",
            [],
        )
    description = match.group(1).strip()
    if not description:
        return (
            {"description": ""},
            False,
            "description must be non-empty",
            ["skill_xml_removed"],
        )
    # Literal UI labels (e.g. WebShop's "< Prev") and comparisons are
    # evidence, not markup. Reject actual tags, not individual angle brackets.
    if re.search(
        r"</?[A-Za-z_][\w:.-]*(?:\s+[^<>]*?)?\s*/?>"
        r"|<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<![A-Za-z][^<>]*>",
        description,
        flags=re.DOTALL,
    ):
        return (
            {"description": description},
            False,
            "skill_xml guidance must not contain nested tags",
            ["skill_xml_removed", "strict_format_rejected"],
        )
    parsed = {"description": description}
    if max_words > 0 and len(re.findall(r"\S+", description)) > max_words:
        return (
            parsed,
            False,
            f"description must contain at most {max_words} words",
            ["skill_xml_removed", "strict_format_rejected"],
        )
    valid, error = validate_skill_payload(parsed, expected_category)
    return parsed, valid, error, ["skill_xml_removed"]


def validate_skill_payload(
    payload: dict[str, Any], expected_category: str
) -> tuple[bool, str]:
    _ = expected_category
    if "description" not in payload:
        return False, "missing required field: description"
    if (
        not isinstance(payload.get("description"), str)
        or not payload["description"].strip()
    ):
        return False, "description must be a non-empty string"
    normalized = re.sub(r"\s+", " ", payload["description"].strip().lower())
    if normalized in {"...", "…", "n/a", "none", "null", "todo", "tbd"}:
        return False, "description must not be a placeholder"
    if normalized.startswith("<") and normalized.endswith(">"):
        return False, "description must not be a placeholder"
    return True, ""


def skill_payload_to_prompt_text(payload: dict[str, Any]) -> str:
    return str(payload["description"]).strip()


def _skill_xml_token_reward_mask(
    tokenizer: PreTrainedTokenizerFast,
    raw_generation: str,
    target_tokens: Sequence[int],
) -> list[int]:
    """Mask fixed XML wrapper tokens out of token-wise MI credit."""

    tokens = [int(token_id) for token_id in target_tokens]
    if not tokens:
        return []
    try:
        # Use the exact teacher-forcing text, including whitespace, to align offsets.
        tagged_text = tokenizer.decode(tokens, skip_special_tokens=False)
    except Exception:  # noqa: BLE001
        tagged_text = str(raw_generation or "")
    match_text = tagged_text.lstrip("\ufeff")
    removed_prefix_chars = len(tagged_text) - len(match_text)
    match = re.fullmatch(
        r"\s*<skill>\s*(.*?)\s*</skill>\s*",
        match_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("cannot align MI mask: target is not exact skill_xml output")
    # Derive boundaries from sampled token prefixes; re-encoding can change BPE segmentation.
    offsets: list[tuple[int, int]] = []
    previous_end = 0
    try:
        for token_end in range(1, len(tokens) + 1):
            prefix_text = tokenizer.decode(
                tokens[:token_end], skip_special_tokens=False
            )
            common_prefix_length = 0
            for left, right in zip(prefix_text, tagged_text, strict=False):
                if left != right:
                    break
                common_prefix_length += 1
            current_end = max(previous_end, common_prefix_length)
            current_end = min(current_end, len(tagged_text))
            offsets.append((previous_end, current_end))
            previous_end = current_end
    except Exception as exc:  # noqa: BLE001
        raise ValueError("cannot align MI mask: tokenizer decode unavailable") from exc
    if not offsets or offsets[-1][1] != len(tagged_text):
        raise ValueError("cannot align MI mask: sampled token decode mismatch")
    body_start, body_end = match.span(1)
    body_start += removed_prefix_chars
    body_end += removed_prefix_chars
    mask = [int(end > body_start and start < body_end) for start, end in offsets]
    if not any(mask):
        raise ValueError("cannot align MI mask: body has no rewardable tokens")
    if all(mask):
        raise ValueError("cannot align MI mask: wrapper shares all body tokens")
    return mask


def _resolve_skill_xml_token_reward_mask(
    tokenizer: PreTrainedTokenizerFast,
    generation: dict[str, Any],
    target_tokens: Sequence[int],
    *,
    schema_valid: bool,
) -> list[int]:
    """Load a safe XML-body mask or rebuild it during resume."""

    tokens = [int(token_id) for token_id in target_tokens]
    if not schema_valid:
        return [0] * len(tokens)
    expected = _skill_xml_token_reward_mask(
        tokenizer,
        str(generation.get("raw", "")),
        tokens,
    )
    saved = generation.get("mi_token_reward_mask")
    if isinstance(saved, list) and len(saved) == len(tokens):
        mask = [int(bool(value)) for value in saved]
        if mask == expected:
            return mask
    # Rebuild missing masks from exact teacher-forcing tokens, never from length alone.
    return expected


def _render_episode_first_steps(
    episode: dict[str, Any],
    *,
    max_steps: int,
    task_limit: int = 220,
    obs_limit: int = _PROMPT_OBSERVATION_CHAR_LIMIT,
    action_limit: int = 100,
    result_limit: int = _PROMPT_RESULT_CHAR_LIMIT,
) -> str:
    steps = episode.get("steps") or []
    step_limit = max(0, min(max_steps, _PROMPT_TRACE_MAX_STEPS))
    rendered_steps = steps[:step_limit]
    lines = [
        f"Task type: {episode.get('task_type', '')}",
        f"Task: {_shorten(episode.get('task_desc'), limit=task_limit)}",
        (
            "Outcome: "
            + ("SUCCESS" if episode.get("won") else "FAILURE")
            + f"; status={episode.get('status')}; steps={episode.get('steps_taken', len(steps))}"
        ),
    ]
    for step in rendered_steps:
        lines.append(
            f"S{int(step.get('t', 0)) + 1:02d} | "
            f"Obs: {_shorten(step.get('observation'), limit=obs_limit)} | "
            f"Action: {_shorten(step.get('action'), limit=action_limit)} | "
            f"Result: {_shorten(step.get('result') or step.get('error'), limit=result_limit)}"
        )
    if len(steps) > step_limit:
        lines.append(
            "=== TRACE TRUNCATED: "
            f"showing first {step_limit} of {len(steps)} steps; "
            f"{len(steps) - step_limit} later steps omitted ==="
        )
    return "\n".join(lines)


def _render_trajectory_bundle_first_steps(
    episodes: list[dict[str, Any]],
    *,
    max_steps: int,
    obs_limit: int,
    result_limit: int,
) -> str:
    if not episodes:
        return ZERO_SHOT_TRAJECTORIES
    chunks = []
    for index, episode in enumerate(episodes, start=1):
        outcome = "success" if episode.get("won") else "failure"
        if "label" in episode:
            heading = (
                f"=== Trajectory {index} "
                f"(label={int(bool(episode.get('label')))}, {outcome}) ===\n"
            )
        else:
            heading = f"=== Trajectory {index} ({outcome}) ===\n"
        chunks.append(
            heading
            + _render_episode_first_steps(
                episode,
                max_steps=max_steps,
                obs_limit=obs_limit,
                result_limit=result_limit,
            )
        )
    return "\n\n".join(chunks)


def _trajectory_step_needs_detail(
    step: dict[str, Any], *, previous_score: float | None
) -> bool:
    status = str(step.get("status") or "ok").strip().lower()
    result = str(step.get("result") or step.get("error") or "").lower()
    error = str(step.get("error") or "").strip()
    try:
        score = float(step.get("score", 0.0) or 0.0)
        reward = float(step.get("reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
        reward = 0.0
    return bool(
        status not in {"", "ok"}
        or step.get("in_admissible") is False
        or bool(error)
        or reward != 0.0
        or (previous_score is not None and score != previous_score)
        or step.get("done")
        or step.get("won")
        or "nothing happens" in result
    )


def _compact_step_result(result: str, *, limit: int) -> str:
    """Keep state evidence while removing common ALFWorld boilerplate."""

    if limit <= 0:
        return ""
    text = " ".join(str(result or "").split())
    if not text:
        return ""
    lowered = text.lower()
    if "available commands:" in lowered:
        return _shorten("no progress", limit=limit)

    # Prefer new state evidence over repeated action/location text before truncation.
    seen_at = lowered.find("you see ")
    if seen_at >= 0:
        text = text[seen_at:]
    elif lowered.startswith("you arrive at "):
        _, separator, remainder = text.partition(".")
        text = (
            remainder.strip()
            if separator and remainder.strip()
            else "Arrival confirmed."
        )
    return _shorten(text, limit=limit)


def _compact_observation(observation: str, *, limit: int) -> str:
    """Remove action-menu text and retain the visible environment state."""

    text = " ".join(str(observation or "").split())
    if not text or limit <= 0:
        return ""
    lowered = text.lower()
    command_at = lowered.find("available commands:")
    if command_at >= 0:
        text = text[:command_at].strip()
        lowered = text.lower()
    if not text:
        return ""
    seen_at = lowered.find("you see ")
    if seen_at > 0:
        text = text[seen_at:]
    return _shorten(text, limit=limit)


def _render_episode_for_prompt(
    episode: dict[str, Any],
    *,
    detail_tail_steps: int,
    summary_result_limit: int,
    include_state_metadata: bool,
    task_limit: int = 220,
    obs_limit: int = _PROMPT_OBSERVATION_CHAR_LIMIT,
    result_limit: int = _PROMPT_RESULT_CHAR_LIMIT,
) -> str:
    """Render every action while avoiding repeated adjacent environment states."""

    steps = [step for step in (episode.get("steps") or []) if isinstance(step, dict)]
    lines = [
        f"Task type: {episode.get('task_type', '')}",
        f"Task: {_shorten(episode.get('task_desc'), limit=task_limit)}",
        (
            "Outcome: "
            + ("SUCCESS" if episode.get("won") else "FAILURE")
            + f"; status={episode.get('status')}; steps={episode.get('steps_taken', len(steps))}"
        ),
    ]
    detailed_from = max(0, len(steps) - max(0, detail_tail_steps))
    previous_result = ""
    previous_score: float | None = None
    for index, step in enumerate(steps):
        action = " ".join(str(step.get("action") or "").split()) or "<no action>"
        result = str(step.get("result") or step.get("error") or "")
        observation = " ".join(str(step.get("observation") or "").split())
        detail = index >= detailed_from or _trajectory_step_needs_detail(
            step, previous_score=previous_score
        )
        parts = [f"S{int(step.get('t', index)) + 1:02d}"]
        if index == 0 and observation:
            initial_observation = _compact_observation(observation, limit=obs_limit)
            if initial_observation:
                parts.append(f"Obs: {initial_observation}")
        parts.append(f"Action: {action}")
        normalized_previous = " ".join(previous_result.split())
        if observation and index > 0 and observation != normalized_previous:
            observation_limit = obs_limit if detail else max(24, summary_result_limit)
            compact_observation = _compact_observation(
                observation, limit=observation_limit
            )
            if compact_observation:
                parts.append(f"Obs: {compact_observation}")
        if detail:
            detailed_result = _compact_step_result(result, limit=result_limit)
            if detailed_result:
                parts.append(f"Result: {detailed_result}")
            state_bits: list[str] = []
            status = str(step.get("status") or "").strip()
            if status and status.lower() != "ok":
                state_bits.append(f"status={status}")
            try:
                reward = float(step.get("reward", 0.0) or 0.0)
                score = float(step.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                reward = 0.0
                score = 0.0
            if reward != 0.0:
                state_bits.append(f"reward={reward:g}")
            if previous_score is not None and score != previous_score:
                state_bits.append(f"score={score:g}")
            if step.get("done"):
                state_bits.append("done=true")
            if step.get("won"):
                state_bits.append("won=true")
            if state_bits and include_state_metadata:
                parts.append("State: " + ", ".join(state_bits))
            previous_score = score
        else:
            compact_result = _compact_step_result(
                result,
                limit=summary_result_limit,
            )
            if compact_result:
                parts.append(f"Result: {compact_result}")
            try:
                previous_score = float(step.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                previous_score = 0.0
        lines.append(" | ".join(parts))
        previous_result = result

    return "\n".join(lines)


def _render_trajectory_bundle_compact(
    episodes: list[dict[str, Any]],
    *,
    detail_tail_steps: int,
    summary_result_limit: int,
    include_state_metadata: bool,
    obs_limit: int,
    result_limit: int,
) -> str:
    if not episodes:
        return ZERO_SHOT_TRAJECTORIES
    chunks = []
    for index, episode in enumerate(episodes, start=1):
        outcome = "success" if episode.get("won") else "failure"
        if "label" in episode:
            heading = (
                f"=== Trajectory {index} "
                f"(label={int(bool(episode.get('label')))}, {outcome}) ===\n"
            )
        else:
            heading = f"=== Trajectory {index} ({outcome}) ===\n"
        chunks.append(
            heading
            + _render_episode_for_prompt(
                episode,
                detail_tail_steps=detail_tail_steps,
                summary_result_limit=summary_result_limit,
                include_state_metadata=include_state_metadata,
                obs_limit=obs_limit,
                result_limit=result_limit,
            )
        )
    return "\n\n".join(chunks)


def _render_trajectory_bundle(
    episodes: list[dict[str, Any]],
    *,
    max_steps: int,
    obs_limit: int,
    result_limit: int,
) -> str:
    """Render every action within the length of a first-``max_steps`` rendering.

    External bank generation has no tokenizer, so it bounds the complete compact
    rendering by characters; training bounds prompts by tokens instead.
    """

    if not episodes:
        return ZERO_SHOT_TRAJECTORIES
    reference = _render_trajectory_bundle_first_steps(
        episodes,
        max_steps=max_steps,
        obs_limit=obs_limit,
        result_limit=result_limit,
    )
    for (
        _,
        detail_tail_steps,
        summary_result_limit,
        include_state_metadata,
    ) in _COMPACT_TRAJECTORY_PROFILES:
        trajectories = _render_trajectory_bundle_compact(
            episodes,
            detail_tail_steps=min(detail_tail_steps, max(0, max_steps)),
            summary_result_limit=min(summary_result_limit, result_limit),
            include_state_metadata=include_state_metadata,
            obs_limit=obs_limit,
            result_limit=result_limit,
        )
        if len(trajectories) <= len(reference):
            return trajectories
    raise ValueError(
        "complete compact trajectories exceed the first-step reference character "
        "budget; refusing to drop or truncate actions"
    )


def _normalize_skill_prompt_version(value: str | None) -> str:
    version = str(value or "evidence_discovery").strip().lower()
    if version not in _SKILL_PROMPT_VERSIONS:
        choices = ", ".join(sorted(_SKILL_PROMPT_VERSIONS))
        raise ValueError(f"skill_prompt_version must be one of: {choices}")
    return version


def _skill_generation_messages(
    *,
    prompt_category: str,
    trajectories: str,
    skill_output_format: str = "skill_xml",
    skill_prompt_version: str = "evidence_discovery",
) -> list[dict[str, str]]:
    _normalize_skill_prompt_version(skill_prompt_version)
    if str(skill_output_format or "").strip().lower() != "skill_xml":
        raise ValueError("skill_output_format must be 'skill_xml'")
    system_prompt = (
        EVIDENCE_DISCOVERY_SYSTEM_PROMPT + "\nReturn exactly one skill wrapped by "
        f"{SKILL_XML_OPEN_TAG} and {SKILL_XML_CLOSE_TAG}, with no text outside "
        "the tags."
    )
    user_template = (
        EVIDENCE_DISCOVERY_USER_CONTENT_TEMPLATE + SKILL_XML_OUTPUT_REQUIREMENTS
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_template.format(
                category=prompt_category,
                category_note=_CATEGORY_NOTE,
                action_space=ALFWORLD_ACTION_SPACE,
                trajectories=trajectories,
            ),
        },
    ]


def _skill_prompt_token_count(
    tokenizer: PreTrainedTokenizerFast,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> int:
    return len(
        apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    )


def _skill_prompt_budget(gconfig: GenerationHyperparameters) -> int:
    _, prompt_budget = skill_prompt_budgets(
        gconfig,
        total_token_budget=_SKILL_PROMPT_TOTAL_TOKEN_BUDGET,
    )
    return prompt_budget


def _fit_skill_prompt_messages(
    *,
    tokenizer: PreTrainedTokenizerFast,
    prompt_category: str,
    sampled_episodes: list[dict[str, Any]],
    max_prompt_tokens: int,
    prompt_observation_char_limit: int,
    prompt_result_char_limit: int,
    enable_thinking: bool = False,
    skill_output_format: str = "skill_xml",
    skill_prompt_version: str = "evidence_discovery",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Render every sampled action, compacting step details until the prompt fits."""

    prompt_version = _normalize_skill_prompt_version(skill_prompt_version)
    budget_metadata = {
        "max_prompt_tokens": max_prompt_tokens,
        "total_token_budget": _SKILL_PROMPT_TOTAL_TOKEN_BUDGET,
        "dropped_history": False,
        "full_history_preserved": True,
        "skill_prompt_version": prompt_version,
        "trajectory_render_version": _TRAJECTORY_RENDER_VERSION,
        "prompt_observation_char_limit": prompt_observation_char_limit,
        "prompt_result_char_limit": prompt_result_char_limit,
    }
    if not sampled_episodes:
        messages = _skill_generation_messages(
            prompt_category=prompt_category,
            trajectories=ZERO_SHOT_TRAJECTORIES,
            skill_output_format=skill_output_format,
            skill_prompt_version=prompt_version,
        )
        prompt_token_count = _skill_prompt_token_count(
            tokenizer, messages, enable_thinking=enable_thinking
        )
        if prompt_token_count > max_prompt_tokens:
            raise ValueError(
                "zero-shot skill prompt exceeds the prompt token budget: "
                f"{prompt_token_count} > {max_prompt_tokens}"
            )
        return messages, {
            **budget_metadata,
            "prompt_token_count": prompt_token_count,
            "was_compacted": False,
            "source_step_count": 0,
            "rendered_action_count": 0,
            "detail_profile": "zero_shot",
            "detailed_tail_steps": 0,
        }

    source_step_count = sum(
        len([step for step in (episode.get("steps") or []) if isinstance(step, dict)])
        for episode in sampled_episodes
    )
    max_source_steps = max(
        (
            len(
                [
                    step
                    for step in (episode.get("steps") or [])
                    if isinstance(step, dict)
                ]
            )
            for episode in sampled_episodes
        ),
        default=0,
    )
    smallest_count: int | None = None
    for profile_index, (
        profile_name,
        detail_tail_steps,
        summary_result_limit,
        include_state_metadata,
    ) in enumerate(_COMPACT_TRAJECTORY_PROFILES):
        trajectories = _render_trajectory_bundle_compact(
            sampled_episodes,
            detail_tail_steps=detail_tail_steps,
            summary_result_limit=min(summary_result_limit, prompt_result_char_limit),
            include_state_metadata=include_state_metadata,
            obs_limit=prompt_observation_char_limit,
            result_limit=prompt_result_char_limit,
        )
        messages = _skill_generation_messages(
            prompt_category=prompt_category,
            trajectories=trajectories,
            skill_output_format=skill_output_format,
            skill_prompt_version=prompt_version,
        )
        prompt_token_count = _skill_prompt_token_count(
            tokenizer, messages, enable_thinking=enable_thinking
        )
        smallest_count = (
            prompt_token_count
            if smallest_count is None
            else min(smallest_count, prompt_token_count)
        )
        if prompt_token_count > max_prompt_tokens:
            continue
        return messages, {
            **budget_metadata,
            "prompt_token_count": prompt_token_count,
            "was_compacted": True,
            "rendered_max_steps": max_source_steps,
            "source_step_count": source_step_count,
            "rendered_action_count": source_step_count,
            "detail_profile_index": profile_index,
            "detail_profile": profile_name,
            "detailed_tail_steps": detail_tail_steps,
            "included_state_metadata": include_state_metadata,
            "trajectory_char_count": len(trajectories),
        }

    raise ValueError(
        "complete compact trajectories cannot fit within the model prompt budget "
        "without dropping history: "
        f"smallest={smallest_count}, ceiling={max_prompt_tokens}, "
        f"source_steps={source_step_count}"
    )


def _prompt_episode_ref(
    episode: dict[str, Any],
    *,
    source_rollouts_path: Path | None,
    source_episode_path: Path | None = None,
    rollout_index: int | None = None,
) -> dict[str, Any]:
    steps = episode.get("steps")
    steps_taken = episode.get("steps_taken")
    if steps_taken is None and isinstance(steps, list):
        steps_taken = len(steps)
    valid_action_count = None
    if isinstance(steps, list):
        valid_action_count = sum(
            bool(str(step.get("action") or "").strip())
            and str(step.get("status") or "ok").strip().lower()
            not in _UNUSABLE_PROMPT_STEP_STATUSES
            for step in steps
            if isinstance(step, dict)
        )
    return {
        "source_rollouts_path": str(source_rollouts_path)
        if source_rollouts_path
        else "",
        "source_episode_path": str(source_episode_path) if source_episode_path else "",
        "rollout_index": -1 if rollout_index is None else int(rollout_index),
        "won": bool(episode.get("won")),
        "status": episode.get("status"),
        "steps_taken": steps_taken,
        "valid_action_count": valid_action_count,
        "task_type": episode.get("task_type"),
        "gamefile": episode.get("gamefile"),
    }


def _trajectory_pool_episode_ref(
    episode: dict[str, Any],
    *,
    source_path: Path,
    rollout_index: int,
    pool_source: str,
    source_training_global_step: int | None = None,
) -> dict[str, Any]:
    ref = _prompt_episode_ref(
        episode,
        source_rollouts_path=source_path,
        rollout_index=rollout_index,
    )
    ref["pool_source"] = pool_source
    ref["source_training_global_step"] = source_training_global_step
    return ref


def _build_skill_manifest(
    *,
    generation_paths: Sequence[Path],
    mode: str,
    global_step: int,
    expected_skill_count: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for generation_path in sorted(set(generation_paths)):
        prompt_path = generation_path.with_name("prompt.json")
        try:
            generation = _read_json(generation_path)
            prompt = _read_json(prompt_path)
        except Exception:
            continue
        parsed = generation.get("parsed")
        parsed = parsed if isinstance(parsed, dict) else None
        trajectory_refs = _manifest_trajectory_refs(prompt)
        input_fingerprint = _skill_manifest_input_id(prompt, trajectory_refs)
        sample_key = generation_path.parent.name
        source_round = str(prompt.get("round_in_category", ""))
        if source_round != "":
            source_round = f"round_{int(source_round):04d}"
        elif prompt.get("source_round_index") is not None:
            source_round = f"round_{int(prompt['source_round_index']):04d}"
        skill_id = f"{source_round}/{sample_key}" if source_round else sample_key
        records.append(
            {
                "id": skill_id,
                "text": skill_payload_to_prompt_text(parsed) if parsed else "",
                "raw": str(generation.get("raw") or ""),
                "valid": bool(generation.get("schema_valid")),
                "input_fingerprint": input_fingerprint,
                "trajectory_ids": [
                    str(ref["trajectory_id"]) for ref in trajectory_refs
                ],
            }
        )

    records.sort(key=lambda item: str(item["id"]))
    records_by_input: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_input.setdefault(str(record["input_fingerprint"]), []).append(record)

    input_groups: list[dict[str, Any]] = []
    for input_index, input_records in enumerate(records_by_input.values()):
        input_id = f"input_{input_index:02d}"
        skills = [
            {
                "id": str(record["id"]),
                "skill": str(record["text"]),
                "raw": str(record["raw"]),
                "valid": bool(record["valid"]),
            }
            for record in input_records
        ]
        input_groups.append(
            {
                "input_id": input_id,
                "input_type": (
                    "trajectory_bundle"
                    if input_records[0]["trajectory_ids"]
                    else "zero_shot"
                ),
                "relation": "same_input" if len(skills) > 1 else "distinct_input",
                "trajectory_ids": list(input_records[0]["trajectory_ids"]),
                "skills": skills,
            }
        )

    generation_signature = hashlib.sha1(
        json.dumps(
            [
                {
                    "id": record["id"],
                    "input_fingerprint": record["input_fingerprint"],
                    "text": record["text"],
                    "raw": record["raw"],
                }
                for record in records
            ],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    complete = len(records) >= expected_skill_count
    compact_metadata = {
        key: value
        for key, value in (metadata or {}).items()
        if key
        in {
            "task_type",
            "source_rounds",
            "checkpoint_global_step",
            "checkpoint_model_version",
            "source_round_index",
        }
    }
    return {
        "schema_version": 2,
        "status": "complete" if complete else "partial",
        "mode": mode,
        "global_step": global_step,
        "skill_count": len(records),
        "input_count": len(input_groups),
        **compact_metadata,
        "input_groups": input_groups,
        "relation_note": (
            "Skills inside one input_group share the same input trajectories; "
            "different input_id values denote different inputs."
        ),
        "generation_signature": generation_signature,
        "updated_at": time.time(),
    }


def _write_skill_manifest(
    *,
    manifest_path: Path,
    generation_paths: Sequence[Path],
    mode: str,
    global_step: int,
    expected_skill_count: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    fd = _acquire_json_lock(lock_path)
    try:
        payload = _build_skill_manifest(
            generation_paths=generation_paths,
            mode=mode,
            global_step=global_step,
            expected_skill_count=expected_skill_count,
            metadata=metadata,
        )
        if manifest_path.exists():
            try:
                existing = _read_json(manifest_path)
            except Exception:
                existing = {}
            # Never overwrite a complete manifest with a stale partial path list.
            if (
                existing.get("status") == "complete"
                and payload.get("status") != "complete"
            ):
                return existing
            if (
                existing.get("schema_version") == payload["schema_version"]
                and existing.get("generation_signature")
                == payload["generation_signature"]
                and existing.get("status") == payload["status"]
            ):
                return existing
        _atomic_write_json(manifest_path, payload)
        return payload
    finally:
        _release_json_lock(fd, lock_path)
