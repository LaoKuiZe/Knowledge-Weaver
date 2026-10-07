# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from examples.webshop_skill.evidence_prompt import (
    EVIDENCE_DISCOVERY_SYSTEM_PROMPT,
    EVIDENCE_DISCOVERY_USER_TEMPLATE,
)

from areal.utils.http_session import get_thread_session

NO_SKILL_TEXT = "No additional guidance."
DEFAULT_ACTOR_MODEL = "Qwen/Qwen3.5-4B"
ZERO_SHOT_TRAJECTORIES = (
    "No prior rollout trajectories are available yet. Generate the best compact, "
    "transferable WebShop skill from the task definition and action space alone."
)

WEBSHOP_SKILL_XML_OUTPUT_REQUIREMENTS = """

# Strict output requirements
- Return exactly: <skill><one natural-language guidance paragraph></skill>
- The opening tag must be the first non-whitespace text and the closing tag the last.
- Use exactly one opening tag and one closing tag. Do not add analysis, markdown, labels,
  nested tags, or text outside the wrapper.
"""


class ChatModel(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> str: ...


@dataclass(frozen=True)
class OpenAIChatClient:
    base_url: str
    model: str
    api_key: str = ""
    timeout_s: float = 120.0
    max_retries: int = 3
    enable_thinking: bool = False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "seed": int(seed),
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        last_error = ""
        for attempt in range(max(1, self.max_retries)):
            try:
                response = get_thread_session().post(
                    self.base_url.rstrip("/") + "/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )
                if (
                    getattr(response, "status_code", None) == 429
                    and attempt + 1 < max(1, self.max_retries)
                ):
                    try:
                        delay = float(response.headers.get("Retry-After", 30 * (attempt + 1)))
                    except (TypeError, ValueError):
                        delay = 30 * (attempt + 1)
                    time.sleep(min(120.0, max(1.0, delay)))
                    continue
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise RuntimeError(f"response has no choices: {data!r}")
                message = choices[0].get("message")
                if not isinstance(message, dict):
                    raise RuntimeError(
                        f"response choice has no message: {choices[0]!r}"
                    )
                content = message.get("content")
                if not isinstance(content, str):
                    raise RuntimeError(f"response content is not text: {message!r}")
                return content
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                if attempt + 1 < max(1, self.max_retries):
                    time.sleep(attempt + 1)
        raise RuntimeError(f"chat request failed after retries: {last_error}")


@dataclass(frozen=True)
class ParsedAction:
    action: str
    valid: bool
    reason: str
    candidate: str


def _collapse_whitespace(text: Any) -> str:
    return " ".join(str(text or "").split())


def _truncate(text: Any, limit: int) -> str:
    value = _collapse_whitespace(text)
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


_USABLE_SOURCE_STATUSES = frozenset({"complete", "max_steps"})


def is_usable_webshop_source_episode(episode: dict[str, Any]) -> bool:
    if str(episode.get("status") or "").strip().lower() not in _USABLE_SOURCE_STATUSES:
        return False
    trace = episode.get("trace")
    if not isinstance(trace, list):
        return False
    return any(
        str(row.get("action") or "").strip() and row.get("valid_action") is not False
        for row in trace
        if isinstance(row, dict)
    )


def compact_webshop_source_episode(
    episode: dict[str, Any],
    *,
    observation_char_limit: int,
    detail_tail_steps: int = 0,
    summary_observation_char_limit: int = 0,
    include_state_metadata: bool = True,
    include_intermediate_results: bool = True,
    actions_only: bool = False,
) -> dict[str, Any]:
    """Keep every action, abbreviating page text outside the tail and key steps."""
    trace = [
        row
        for row in list(episode.get("trace") or [])
        if isinstance(row, dict) and str(row.get("action") or "").strip()
    ]
    compact_trace: list[dict[str, Any]] = []
    previous_result = ""
    detailed_from = max(0, len(trace) - max(0, int(detail_tail_steps)))
    summary_limit = max(
        1, int(summary_observation_char_limit or observation_char_limit)
    )
    for index, row in enumerate(trace):
        valid_action = row.get("valid_action") is not False
        reward = float(row.get("reward", 0.0) or 0.0)
        done = bool(row.get("done"))
        detail = index >= detailed_from or not valid_action or reward != 0.0 or done
        observation = _collapse_whitespace(row.get("observation"))
        result = _collapse_whitespace(row.get("next_observation"))
        # Avoid repeating a step result as the next observation unless the timeline is discontinuous.
        rendered_observation = ""
        if index == 0 or (observation and observation != previous_result):
            rendered_observation = _truncate(
                observation,
                observation_char_limit if detail else summary_limit,
            )
        rendered_result = (
            _truncate(
                result,
                observation_char_limit if detail else summary_limit,
            )
            if detail or include_intermediate_results
            else ""
        )
        compact_row: dict[str, Any] = {
            "step": int(row.get("step", index)),
            "observation": rendered_observation,
            "action": _collapse_whitespace(row.get("action")),
            "next_observation": rendered_result,
            "done": done,
            "valid_action": valid_action,
            "invalid_reason": _truncate(row.get("invalid_reason"), 160),
        }
        if include_state_metadata:
            compact_row["reward"] = reward
        compact_trace.append(compact_row)
        previous_result = result
    if actions_only:
        # Preserve every action and its outcome flags, not repeated page/error prose.
        for compact_row in compact_trace:
            compact_row["observation"] = ""
            compact_row["next_observation"] = ""
            compact_row["invalid_reason"] = ""
    # The guidance the episode ran with is omitted; the curator sees only its trace.
    return {
        "task_index": int(episode.get("task_index", -1)),
        "instruction": _truncate(episode.get("instruction"), 360),
        "status": str(episode.get("status") or ""),
        "reward": float(episode.get("reward", 0.0)),
        "success": bool(episode.get("success")),
        "done": bool(episode.get("done")),
        "trace": compact_trace,
    }


def _extract_action_candidate(raw: str) -> str:
    match = re.search(r"Action\s*:\s*([^\n]+)", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    direct = raw.strip().splitlines()
    return direct[-1].strip() if direct else ""


def parse_webshop_action(raw: str, available_actions: dict[str, Any]) -> ParsedAction:
    candidate = _extract_action_candidate(raw)
    match = re.fullmatch(r"\s*(search|click)\s*\[(.*)]\s*", candidate, re.IGNORECASE)
    if not match:
        return ParsedAction("", False, "expected search[...] or click[...]", candidate)

    kind = match.group(1).lower()
    argument = _collapse_whitespace(match.group(2))
    if not argument:
        return ParsedAction("", False, "action argument is empty", candidate)

    if kind == "search":
        if not bool(available_actions.get("has_search_bar")):
            return ParsedAction(
                "", False, "search is unavailable on this page", candidate
            )
        return ParsedAction(f"search[{argument}]", True, "", candidate)

    clickables = [str(item) for item in available_actions.get("clickables", [])]
    exact = {item.casefold(): item for item in clickables}
    selected = exact.get(argument.casefold())
    if selected is None:
        return ParsedAction(
            "", False, "click target is not currently visible", candidate
        )
    return ParsedAction(f"click[{selected}]", True, "", candidate)


def format_webshop_memory(
    trace: Sequence[dict[str, Any]],
    *,
    memory_window: int,
    observation_char_limit: int,
) -> str:
    if memory_window <= 0:
        return ""
    rows = list(trace[-memory_window:])
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"Memory step {index}:",
                f"Observation: {_truncate(row.get('observation'), observation_char_limit)}",
                f"Action: {_truncate(row.get('action'), 240)}",
                f"Result: {_truncate(row.get('next_observation'), observation_char_limit)}",
                f"Reward: {float(row.get('reward', 0.0)):.4f}; done: {bool(row.get('done'))}",
            ]
        )
    return "\n".join(lines)


def webshop_actor_messages(
    *,
    instruction: str,
    observation: str,
    available_actions: dict[str, Any],
    skill: str,
    trace: Sequence[dict[str, Any]],
    memory_window: int,
    observation_char_limit: int,
    max_clickables: int,
    correction: str = "",
) -> list[dict[str, str]]:
    clickables = sorted(str(item) for item in available_actions.get("clickables", []))
    if max_clickables > 0:
        clickables = clickables[:max_clickables]
    clickable_text = "\n".join(f"- {item}" for item in clickables) or "- (none)"
    search_text = (
        "available" if available_actions.get("has_search_bar") else "unavailable"
    )
    memory = format_webshop_memory(
        trace,
        memory_window=memory_window,
        observation_char_limit=min(observation_char_limit, 900),
    )
    system = (
        "You are a WebShop text agent. Select the next action to satisfy the shopping "
        "instruction and maximize the final WebShop reward.\n"
        "Respond using exactly two lines:\n"
        "Thought: <concise reasoning, at most 35 words>\n"
        "Action: <one valid action>\n"
        "Use search[concise keywords] only when search is available. Use click[exact target] "
        "by copying one currently visible clickable target. Do not add any other text.\n"
        "Guidance skill:\n"
        f"{skill or NO_SKILL_TEXT}"
    )
    memory_block = f"Recent online memory:\n{memory}\n" if memory else ""
    correction_block = (
        f"Previous response was invalid: {correction}\n" if correction else ""
    )
    user = (
        f"Shopping instruction: {instruction}\n"
        f"Current page observation:\n{_truncate(observation, observation_char_limit)}\n"
        f"{memory_block}"
        f"Search action: {search_text}\n"
        f"Visible click targets:\n{clickable_text}\n"
        f"{correction_block}"
        "Return Thought and Action now:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render_webshop_trajectory(compact: dict[str, Any]) -> str:
    """Render one compacted episode as an outcome line plus one line per action."""
    lines = [
        f"Task: {compact['instruction']}",
        (
            f"End: {'SUCCESS' if compact['success'] else 'FAILURE'};"
            f"{compact['status']};r={compact['reward']:.3f}"
        ),
    ]
    for row in compact["trace"]:
        parts: list[str] = []
        if row.get("observation"):
            parts.append(f"S={row['observation']}")
        parts.append(str(row["action"]))
        if row.get("next_observation"):
            parts.append(f"R={row['next_observation']}")
        if row.get("valid_action") is False:
            parts.append(
                "INVALID"
                + (f"({row['invalid_reason']})" if row.get("invalid_reason") else "")
            )
        if float(row.get("reward", 0.0) or 0.0) != 0.0:
            parts.append(f"r={float(row['reward']):g}")
        if row.get("done"):
            parts.append("DONE")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def render_webshop_trajectories(episodes: Sequence[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"## Source trajectory {index}\n" + render_webshop_trajectory(episode)
        for index, episode in enumerate(episodes, start=1)
    )


def webshop_skill_messages(
    episodes: Sequence[dict[str, Any]], *, max_words: int
) -> list[dict[str, str]]:
    """Build the curator prompt from compacted source episodes."""
    trajectories = render_webshop_trajectories(episodes) or ZERO_SHOT_TRAJECTORIES
    system = (
        EVIDENCE_DISCOVERY_SYSTEM_PROMPT
        + "\n\nReturn exactly one <skill>...</skill> wrapper and no text outside it."
    )
    user = (
        EVIDENCE_DISCOVERY_USER_TEMPLATE.format(
            trajectories=trajectories,
            max_words=max_words,
        )
        + WEBSHOP_SKILL_XML_OUTPUT_REQUIREMENTS
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def select_trajectory_bundles(
    episodes: Sequence[dict[str, Any]],
    *,
    num_skills: int,
    trajectories_per_skill: int,
    seed: int,
    success_threshold: float,
) -> list[list[dict[str, Any]]]:
    if num_skills <= 0:
        raise ValueError("num_skills must be positive")
    if trajectories_per_skill <= 0:
        raise ValueError("trajectories_per_skill must be positive")
    usable = [item for item in episodes if is_usable_webshop_source_episode(item)]
    if not usable:
        raise ValueError("no usable source trajectories are available")

    success_pool = [
        item
        for item in usable
        if bool(item.get("success"))
        or float(item.get("reward", 0.0)) >= success_threshold
    ]
    failure_pool = [
        item
        for item in usable
        if not bool(item.get("success"))
        and float(item.get("reward", 0.0)) < success_threshold
    ]
    rng = random.Random(seed)
    for items in (success_pool, failure_pool):
        items.sort(key=lambda item: int(item.get("task_index", 0)))
        rng.shuffle(items)
    all_items = sorted(usable, key=lambda item: int(item.get("task_index", 0)))
    rng.shuffle(all_items)

    bundles: list[list[dict[str, Any]]] = []
    for bundle_index in range(num_skills):
        selected: list[dict[str, Any]] = []
        seen: set[int] = set()

        def extend_from(pool: list[dict[str, Any]], count: int, offset: int) -> None:
            for index in range(len(pool) * 2):
                if count <= 0:
                    break
                candidate = pool[(offset + index) % len(pool)]
                key = int(candidate.get("task_index", id(candidate)))
                if key in seen:
                    continue
                selected.append(candidate)
                seen.add(key)
                count -= 1

        success_target = trajectories_per_skill // 2
        failure_target = trajectories_per_skill - success_target
        extend_from(success_pool, success_target, bundle_index * max(1, success_target))
        extend_from(failure_pool, failure_target, bundle_index * max(1, failure_target))
        for offset in range(len(all_items) * 2):
            if len(selected) >= trajectories_per_skill:
                break
            candidate = all_items[
                (bundle_index * trajectories_per_skill + offset) % len(all_items)
            ]
            key = int(candidate.get("task_index", id(candidate)))
            if key not in seen:
                selected.append(candidate)
                seen.add(key)
        if len(selected) < trajectories_per_skill:
            for offset in range(trajectories_per_skill - len(selected)):
                selected.append(all_items[(bundle_index + offset) % len(all_items)])
        bundles.append(selected)
    return bundles


def episode_metrics(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(episodes)
    if not rows:
        return {
            "n": 0,
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "purchase_rate": 0.0,
            "error_rate": 0.0,
            "mean_steps": 0.0,
            "invalid_action_rate": 0.0,
        }
    rewards = [float(item.get("reward", 0.0)) for item in rows]
    steps = [len(item.get("trace") or []) for item in rows]
    invalid = sum(
        int(step.get("valid_action") is False)
        for item in rows
        for step in (item.get("trace") or [])
    )
    total_steps = sum(steps)
    return {
        "n": len(rows),
        "mean_reward": mean(rewards),
        "success_rate": mean(float(bool(item.get("success"))) for item in rows),
        "purchase_rate": mean(float(bool(item.get("done"))) for item in rows),
        "error_rate": mean(float(item.get("status") == "error") for item in rows),
        "mean_steps": mean(steps),
        "invalid_action_rate": invalid / max(1, total_steps),
    }


def category_metrics(episodes: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        category = str(episode.get("category") or "unknown")
        buckets.setdefault(category, []).append(episode)
    return {key: episode_metrics(value) for key, value in sorted(buckets.items())}


def paired_condition_metrics(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    base_by_task = {int(item["task_index"]): item for item in baseline}
    candidate_by_task = {int(item["task_index"]): item for item in candidate}
    task_ids = sorted(base_by_task.keys() & candidate_by_task.keys())
    deltas = [
        float(candidate_by_task[index].get("reward", 0.0))
        - float(base_by_task[index].get("reward", 0.0))
        for index in task_ids
    ]
    tolerance = 1.0e-9
    baseline_success = [
        float(bool(base_by_task[index].get("success"))) for index in task_ids
    ]
    candidate_success = [
        float(bool(candidate_by_task[index].get("success"))) for index in task_ids
    ]
    return {
        "n_paired": len(task_ids),
        "mean_reward_delta": mean(deltas) if deltas else 0.0,
        "success_rate_delta": (
            mean(candidate_success) - mean(baseline_success) if task_ids else 0.0
        ),
        "improved_fraction": (
            sum(delta > tolerance for delta in deltas) / len(deltas) if deltas else 0.0
        ),
        "tied_fraction": (
            sum(abs(delta) <= tolerance for delta in deltas) / len(deltas)
            if deltas
            else 0.0
        ),
        "worse_fraction": (
            sum(delta < -tolerance for delta in deltas) / len(deltas) if deltas else 0.0
        ),
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def fixed_task_split(
    goals: Sequence[dict[str, Any]],
    *,
    train_count: int,
    eval_count: int,
    seed: int,
    unique_asin: bool = True,
) -> dict[str, Any]:
    """Select disjoint train/eval goals, deduplicating by ASIN before seeded shuffling by default."""

    if train_count <= 0:
        raise ValueError("train_count must be positive")
    if eval_count <= 0:
        raise ValueError("eval_count must be positive")

    candidates: list[int] = []
    seen_asins: set[str] = set()
    for index, goal in enumerate(goals):
        asin = str(goal.get("asin") or "").strip()
        if unique_asin and asin:
            if asin in seen_asins:
                continue
            seen_asins.add(asin)
        candidates.append(index)

    required = train_count + eval_count
    if len(candidates) < required:
        qualifier = "unique-ASIN " if unique_asin else ""
        raise ValueError(
            f"requested {required} tasks but only {len(candidates)} {qualifier}goals "
            "are available"
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    train_indices = candidates[:train_count]
    eval_indices = candidates[train_count:required]

    def summarize(indices: Sequence[int]) -> dict[str, Any]:
        categories: dict[str, int] = {}
        asins: list[str] = []
        for index in indices:
            goal = goals[index]
            category = str(goal.get("category") or "unknown")
            categories[category] = categories.get(category, 0) + 1
            asins.append(str(goal.get("asin") or ""))
        return {
            "count": len(indices),
            "unique_asins": len(set(asins)),
            "category_counts": dict(sorted(categories.items())),
        }

    def task_records(indices: Sequence[int]) -> list[dict[str, Any]]:
        return [
            {
                "index": int(index),
                "asin": str(goals[index].get("asin") or ""),
                "category": str(goals[index].get("category") or "unknown"),
                "instruction": str(goals[index].get("instruction_text") or ""),
            }
            for index in indices
        ]

    train_tasks = task_records(train_indices)
    eval_tasks = task_records(eval_indices)
    signature = hashlib.sha256(
        json.dumps(
            {
                "seed": int(seed),
                "unique_asin": bool(unique_asin),
                "train_tasks": train_tasks,
                "eval_tasks": eval_tasks,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "version": 1,
        "manifest_signature": signature,
        "seed": int(seed),
        "unique_asin": bool(unique_asin),
        "goal_count": len(goals),
        "candidate_count": len(candidates),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
        "train_summary": summarize(train_indices),
        "eval_summary": summarize(eval_indices),
    }


class WebShopRuntime:
    """One shared WebShop simulator with lightweight per-episode environments."""

    def __init__(
        self,
        *,
        repo_root: Path,
        products_file: Path,
        attributes_file: Path,
        human_attributes_file: Path,
        search_index: Path,
        human_goals: bool,
        num_products: int | None,
        observation_mode: str,
        environment_seed: int | None = None,
    ) -> None:
        for path in (
            repo_root,
            products_file,
            attributes_file,
            human_attributes_file,
            search_index,
        ):
            if not path.exists():
                raise FileNotFoundError(f"required WebShop path does not exist: {path}")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        print("[webshop-env] importing simulator utilities", flush=True)
        from web_agent_site import utils as webshop_utils

        webshop_utils.DEFAULT_FILE_PATH = str(products_file)
        webshop_utils.DEFAULT_ATTR_PATH = str(attributes_file)
        webshop_utils.HUMAN_ATTR_PATH = str(human_attributes_file)

        print("[webshop-env] importing Lucene/JVM", flush=True)
        from examples.webshop_skill.native_runtime import lucene_searcher_class

        LuceneSearcher = lucene_searcher_class()

        print("[webshop-env] Lucene imported; importing simulator engine", flush=True)
        from web_agent_site.engine import engine as webshop_engine
        from web_agent_site.envs import web_agent_text_env

        webshop_engine.DEFAULT_FILE_PATH = str(products_file)
        webshop_engine.DEFAULT_ATTR_PATH = str(attributes_file)
        webshop_engine.HUMAN_ATTR_PATH = str(human_attributes_file)

        def load_searcher(num_products: int | None = None):
            del num_products
            return LuceneSearcher(str(search_index))

        webshop_engine.init_search_engine = load_searcher
        web_agent_text_env.init_search_engine = load_searcher
        self.env_class = web_agent_text_env.WebAgentTextEnv
        self.human_goals = human_goals
        self.num_products = num_products
        self.environment_seed = environment_seed
        # Prices are sampled before SimServer applies its fixed goal shuffle.
        state = random.getstate()
        if environment_seed is not None:
            random.seed(environment_seed)
        try:
            print("[webshop-env] building catalog and search index", flush=True)
            self.server = web_agent_text_env.SimServer(
                "http://127.0.0.1:3000",
                str(products_file),
                num_products=num_products,
                human_goals=human_goals,
            )
        finally:
            if environment_seed is not None:
                random.setstate(state)
        self.observation_mode = observation_mode
        self._session_lock = threading.Lock()

    @property
    def goal_count(self) -> int:
        return len(self.server.goals)

    def task_context(self, task_index: int) -> dict[str, Any]:
        """Reset one task and return the context used for frozen retrieval."""

        env = self.env_class(
            observation_mode=self.observation_mode,
            server=self.server,
            session_prefix=f"context_{int(task_index)}_{uuid.uuid4().hex[:12]}_",
        )
        bootstrap_session = str(env.session)
        try:
            observation, _ = env.reset(session=int(task_index))
            active_session = str(env.session)
            with self._session_lock:
                if bootstrap_session != active_session:
                    self.server.user_sessions.pop(bootstrap_session, None)
                goal = dict(self.server.user_sessions[active_session]["goal"])
            return {
                "task_index": int(task_index),
                "category": str(goal.get("category") or "unknown"),
                "instruction": str(goal.get("instruction_text") or ""),
                "initial_observation": str(observation or ""),
            }
        finally:
            try:
                with self._session_lock:
                    self.server.user_sessions.pop(str(env.session), None)
                    self.server.user_sessions.pop(bootstrap_session, None)
                env.close()
            except Exception:  # noqa: BLE001
                pass

    def rollout(
        self,
        *,
        task_index: int,
        actor: ChatModel,
        skill: str,
        condition: str,
        max_steps: int,
        memory_window: int,
        observation_char_limit: int,
        max_clickables: int,
        actor_temperature: float,
        actor_max_tokens: int,
        actor_seed: int,
        invalid_action_retries: int,
        success_threshold: float,
        session_key: str = "",
    ) -> dict[str, Any]:
        started_at = time.time()
        episode: dict[str, Any] = {
            "task_index": int(task_index),
            "condition": condition,
            "skill": skill,
            "trace": [],
            "reward": 0.0,
            "success": False,
            "done": False,
            "status": "running",
            "error": "",
            "started_at": started_at,
        }
        env = None
        try:
            session_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", session_key).strip("_")
            if not session_prefix:
                session_prefix = uuid.uuid4().hex
            session_prefix = session_prefix[:96] + "_"
            env = self.env_class(
                observation_mode=self.observation_mode,
                server=self.server,
                session_prefix=session_prefix,
            )
            bootstrap_session = str(env.session)
            observation, _ = env.reset(session=int(task_index))
            active_session = str(env.session)
            with self._session_lock:
                if bootstrap_session != active_session:
                    self.server.user_sessions.pop(bootstrap_session, None)
                goal = dict(self.server.user_sessions[active_session]["goal"])
            episode.update(
                {
                    "instruction": str(goal.get("instruction_text", "")),
                    "category": str(goal.get("category") or "unknown"),
                    "query": str(goal.get("query") or ""),
                    "goal_attribute_count": len(goal.get("attributes") or []),
                    "goal_option_count": len(goal.get("goal_options") or []),
                }
            )

            final_reward = 0.0
            done = False
            for step_index in range(max_steps):
                available_actions = env.get_available_actions()
                raw_response = ""
                parsed = ParsedAction("", False, "no response", "")
                for retry in range(max(0, invalid_action_retries) + 1):
                    correction = parsed.reason if retry else ""
                    messages = webshop_actor_messages(
                        instruction=episode["instruction"],
                        observation=observation,
                        available_actions=available_actions,
                        skill=skill,
                        trace=episode["trace"],
                        memory_window=memory_window,
                        observation_char_limit=observation_char_limit,
                        max_clickables=max_clickables,
                        correction=correction,
                    )
                    raw_response = actor.chat(
                        messages,
                        temperature=actor_temperature,
                        seed=actor_seed + task_index * 1000 + step_index,
                        max_tokens=actor_max_tokens,
                    )
                    parsed = parse_webshop_action(raw_response, available_actions)
                    if parsed.valid:
                        break

                if parsed.valid:
                    next_observation, reward, done, _ = env.step(parsed.action)
                    final_reward = float(reward)
                else:
                    next_observation = observation
                    reward = 0.0
                    done = False
                episode["trace"].append(
                    {
                        "step": step_index,
                        "observation": observation,
                        "available_actions": available_actions,
                        "raw_response": raw_response,
                        "action": parsed.action or parsed.candidate,
                        "valid_action": parsed.valid,
                        "invalid_reason": parsed.reason,
                        "next_observation": next_observation,
                        "reward": float(reward),
                        "done": bool(done),
                    }
                )
                observation = next_observation
                if done:
                    break

            episode.update(
                {
                    "reward": final_reward,
                    "success": final_reward >= success_threshold,
                    "done": bool(done),
                    "status": "complete" if done else "max_steps",
                }
            )
        except Exception as exc:  # noqa: BLE001
            episode.update({"status": "error", "error": repr(exc)})
        finally:
            if env is not None:
                try:
                    with self._session_lock:
                        self.server.user_sessions.pop(str(env.session), None)
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
            episode["elapsed_s"] = time.time() - started_at
        return episode
