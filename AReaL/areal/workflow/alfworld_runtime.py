# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Sequence
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer
from areal.workflow.skill_eval_baseline import (
    BaselineRecordSchema,
    aggregate_repeated_baseline_runs,
)
from areal.workflow.skill_prompt import (
    DEFAULT_SKILL_PROMPT_TOTAL_TOKEN_BUDGET,
)

if TYPE_CHECKING:
    from areal.workflow.alfworld_environment import _OutcomeEpisodeDispatcher

logger = logging.getLogger("ALFWorldSkillWorkflow")
_TEXTWORLD_GYM_LOCK = threading.Lock()
_LOCK_STALE_AFTER_S = 3600.0
_ROUND_WAIT_LOG_INTERVAL_S = 30.0
_DEFAULT_SKILL_GENERATION_MAX_RETRIES = 0
_SKILL_EVAL_EXECUTOR_LOCK = threading.Lock()
_SKILL_EVAL_EXECUTOR: ThreadPoolExecutor | None = None
_SKILL_EVAL_EXECUTOR_WORKERS = 0
_EPISODE_ROLLOUT_EXECUTOR_LOCK = threading.Lock()
_EPISODE_ROLLOUT_EXECUTOR: ProcessPoolExecutor | None = None
_EPISODE_ROLLOUT_EXECUTOR_WORKERS = 0
_OUTCOME_CONDITION_EXECUTOR_LOCK = threading.Lock()
_OUTCOME_CONDITION_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_OUTCOME_EPISODE_DISPATCHER_LOCK = threading.Lock()
_OUTCOME_EPISODE_DISPATCHERS: dict[int, _OutcomeEpisodeDispatcher] = {}
_ROLLOUT_WORKER_TOKENIZERS: dict[str, PreTrainedTokenizerFast] = {}
_MI_COMPUTE_LOCK = threading.Lock()
_GROUP_SIMILARITY_EMBEDDER_LOCK = threading.RLock()
_GROUP_SIMILARITY_EMBEDDERS: dict[str, tuple[Any, Any]] = {}
_GROUP_SIMILARITY_CHUNKS: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_GROUP_SIMILARITY_CACHE_ROWS = 0
_GROUP_SIMILARITY_CACHE_MAX_ROWS = 8192
_DEFAULT_GROUP_SIMILARITY_MODEL = "sentence-transformers/all-mpnet-base-v2"
_OUTCOME_RETRYABLE_INFRA_STATUSES = frozenset(
    {
        "actor_error",
        "env_error",
        "no_admissible_commands",
        "rollout_error",
        "outcome_rollout_error",
        "rollout_timeout",
    }
)
_ONLINE_SKILLBANK_MODE = "online_semantic_top3_marginal_v1"
_ONLINE_SKILLBANK_QUERY_VERSION = "alfworld_episode_initial_state_v1"
_PROMPT_TRACE_MAX_STEPS = 30
_PROMPT_OBSERVATION_CHAR_LIMIT = 120
_PROMPT_RESULT_CHAR_LIMIT = 120
_SKILL_PROMPT_TOTAL_TOKEN_BUDGET = DEFAULT_SKILL_PROMPT_TOTAL_TOKEN_BUDGET
_SKILL_PROMPT_VERSIONS = frozenset({"evidence_discovery"})
_TRAJECTORY_RENDER_VERSION = "full_compact_v1"
_PROMPT_CONTRACT_VERSION = 1
_COMPACT_TRAJECTORY_PROFILES = (
    ("tail8_result64", 8, 64, True),
    ("tail6_result48", 6, 48, True),
    ("tail6_result32", 6, 32, True),
    ("tail4_result24", 4, 24, True),
    ("tail2_result12", 2, 12, True),
    ("tail2_result12_plain", 2, 12, False),
)
_NO_SKILL_BASELINE_TEXT = "No additional guidance."
_GENERATION_READY_STATUSES = frozenset(
    {
        "generation_complete",
        "online_skillbank_waiting",
        "rollout_in_progress",
        "complete",
    }
)
_GENERATION_FAILURE_STATUSES = frozenset({"generation_failed"})
ALFWORLD_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
)
# Artifact namespace for training rounds that mix all configured input task types.
MIXED_TASK_TYPE = "mixed_id_tasks"
PROMPT_CATEGORY_BY_TASK_TYPE = {
    "pick_and_place_simple": "pick_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_heat_then_place_in_recep": "pick_heat_then_place",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep": "pick_clean_then_place",
    "pick_cool_then_place_in_recep": "pick_cool_then_place",
}
ALFWORLD_ACTION_SPACE = "go to {recep}, open/close {recep}, take {obj} from {recep}, put {obj} in/on {recep}, use {obj}, heat/cool/clean {obj} with {appliance}, examine {obj}, look"
SKILL_XML_OPEN_TAG = "<skill>"
SKILL_XML_CLOSE_TAG = "</skill>"
SKILL_XML_OUTPUT_REQUIREMENTS = f"\n\n# Strict output requirements\n- Return exactly: {SKILL_XML_OPEN_TAG}<one natural-language guidance paragraph>{SKILL_XML_CLOSE_TAG}\n- The opening tag must be the first non-whitespace text and the closing tag the last.\n- Use exactly one opening tag and one closing tag; do not add analysis, markdown, labels,\n  nested tags, or text outside the wrapper.\n"
ZERO_SHOT_TRAJECTORIES = "No prior rollout trajectories are available yet. Generate the best compact, transferable skill for this task category from the task definition and action space alone."


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_data_root(repo_root: Path) -> Path:
    env_root = os.getenv("ALFWORLD_DATA")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        return root / "json_2.1.1" if (root / "json_2.1.1").exists() else root
    return repo_root / "alfworld_full_data" / "json_2.1.1"


def _ensure_repo_importable(repo_root: Path) -> None:
    root = str(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _shorten(text: Any, limit: int = 700) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3].rstrip() + "..."


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("failed to remove completed rollout snapshot %s: %s", path, exc)


def _remove_completed_episode_snapshots(sample_dir: Path) -> None:
    """Remove progress snapshots only after durable final artifacts exist."""
    _unlink_if_exists(sample_dir / "current_episode.json")
    for snapshot_path in sample_dir.glob("current_episode_*.json"):
        _unlink_if_exists(snapshot_path)


def _get_skill_eval_executor(max_workers: int) -> ThreadPoolExecutor | None:
    if max_workers <= 0:
        return None
    global _SKILL_EVAL_EXECUTOR, _SKILL_EVAL_EXECUTOR_WORKERS
    with _SKILL_EVAL_EXECUTOR_LOCK:
        if (
            _SKILL_EVAL_EXECUTOR is not None
            and _SKILL_EVAL_EXECUTOR_WORKERS >= max_workers
        ):
            return _SKILL_EVAL_EXECUTOR
        old_executor = _SKILL_EVAL_EXECUTOR
        _SKILL_EVAL_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="alfworld-skill-eval",
        )
        _SKILL_EVAL_EXECUTOR_WORKERS = max_workers
    if old_executor is not None:
        old_executor.shutdown(wait=False, cancel_futures=False)
    return _SKILL_EVAL_EXECUTOR


def _get_episode_rollout_executor(max_workers: int) -> ProcessPoolExecutor | None:
    if max_workers <= 0:
        return None
    global _EPISODE_ROLLOUT_EXECUTOR, _EPISODE_ROLLOUT_EXECUTOR_WORKERS
    with _EPISODE_ROLLOUT_EXECUTOR_LOCK:
        old_executor = _EPISODE_ROLLOUT_EXECUTOR
        # One dead worker (native crash, OOM kill) breaks the pool permanently;
        # replace it so later episodes do not all fail with BrokenProcessPool.
        broken = old_executor is not None and bool(getattr(old_executor, "_broken", False))
        if (
            old_executor is not None
            and not broken
            and _EPISODE_ROLLOUT_EXECUTOR_WORKERS >= max_workers
        ):
            return old_executor
        if broken:
            logger.warning("Rebuilding the ALFWorld rollout process pool after a worker died")
            max_workers = max(max_workers, _EPISODE_ROLLOUT_EXECUTOR_WORKERS)
        # Use processes for rollout parallelism: TextWorld's global parser state is not thread-safe.
        _EPISODE_ROLLOUT_EXECUTOR = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=get_context("spawn"),
        )
        _EPISODE_ROLLOUT_EXECUTOR_WORKERS = max_workers
    if old_executor is not None:
        old_executor.shutdown(wait=False, cancel_futures=False)
    return _EPISODE_ROLLOUT_EXECUTOR


def _get_outcome_condition_executor(max_workers: int) -> ThreadPoolExecutor:
    """Return a process-wide coordinator pool for independent outcome conditions."""

    workers = max(1, int(max_workers))
    with _OUTCOME_CONDITION_EXECUTOR_LOCK:
        executor = _OUTCOME_CONDITION_EXECUTORS.get(workers)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="alfworld-outcome-condition",
            )
            _OUTCOME_CONDITION_EXECUTORS[workers] = executor
        return executor


def _rollout_worker_tokenizer(
    tokenizer_path: str | None,
) -> PreTrainedTokenizerFast | None:
    if not tokenizer_path:
        return None
    tokenizer = _ROLLOUT_WORKER_TOKENIZERS.get(tokenizer_path)
    if tokenizer is None:
        tokenizer = load_hf_tokenizer(tokenizer_path)
        _ROLLOUT_WORKER_TOKENIZERS[tokenizer_path] = tokenizer
    return tokenizer


def _rollout_progress_event(
    *,
    rollout_index: int,
    episode: dict[str, Any],
    final: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rollout_index": rollout_index,
        "final": final,
        "task_type": episode.get("task_type"),
        "gamefile": episode.get("gamefile"),
        "status": episode.get("status"),
        "steps_taken": episode.get("steps_taken", 0),
        "final_score": episode.get("final_score", 0.0),
        "done": bool(episode.get("done")),
        "won": bool(episode.get("won")),
        "outcome_reward": episode.get("outcome_reward", 0.0),
        "updated_at": time.time(),
    }
    if extra:
        payload.update(extra)
    return payload


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    tmp.write_text(payload)
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_stale_lock(
    lock_path: Path,
    *,
    foreign_host_stale_after_s: float | None = None,
) -> bool:
    try:
        payload = _read_json(lock_path)
    except Exception:
        payload = None

    if not isinstance(payload, dict):
        return _remove_old_unreadable_lock(lock_path)

    lock_host = payload.get("host")
    if lock_host and lock_host != socket.gethostname():
        if foreign_host_stale_after_s is not None:
            return _remove_old_unreadable_lock(
                lock_path,
                payload.get("claimed_at"),
                stale_after_s=foreign_host_stale_after_s,
            )
        return False

    try:
        pid = int(payload.get("pid", -1))
    except (TypeError, ValueError):
        pid = -1

    if pid <= 0:
        return _remove_old_unreadable_lock(lock_path, payload.get("claimed_at"))

    if _process_is_alive(pid):
        return False

    try:
        lock_path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _remove_old_unreadable_lock(
    lock_path: Path,
    claimed_at: Any = None,
    *,
    stale_after_s: float = _LOCK_STALE_AFTER_S,
) -> bool:
    try:
        age_s = time.time() - float(claimed_at)
    except (TypeError, ValueError):
        try:
            age_s = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return True
        except OSError:
            return False

    if age_s < stale_after_s:
        return False

    try:
        lock_path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _task_desc_from_json(path: Path) -> str:
    data = json.loads(path.read_text())
    anns = data.get("turk_annotations", {}).get("anns", [])
    if anns:
        return str(anns[0].get("task_desc", ""))
    params = data.get("pddl_params", {})
    return f"{data.get('task_type', '')}: {params}"


def _collect_games(data_root: Path, split: str, task_type: str) -> list[dict[str, str]]:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"ALFWorld split does not exist: {split_root}")

    games: list[dict[str, str]] = []
    task_dirs = [
        path for path in sorted(split_root.glob(f"{task_type}-*")) if path.is_dir()
    ]
    trajs = [
        traj
        for task_dir in task_dirs
        for traj in sorted(task_dir.glob("*/traj_data.json"))
    ]
    if not trajs:
        trajs = sorted(split_root.rglob("traj_data.json"))
    for traj in trajs:
        root = traj.parent
        if "movable" in str(root) or "Sliced" in str(root):
            continue
        game = root / "game.tw-pddl"
        if not game.exists():
            continue
        try:
            game_data = json.loads(game.read_text())
            traj_data = json.loads(traj.read_text())
        except Exception:
            continue
        if not game_data.get("solvable", False):
            continue
        if traj_data.get("task_type") != task_type:
            continue
        games.append(
            {
                "task_type": task_type,
                "gamefile": str(game.resolve()),
                "traj_json": str(traj.resolve()),
                "task_desc": _task_desc_from_json(traj),
            }
        )
    if not games:
        raise RuntimeError(
            f"No games found for task_type={task_type!r} in split={split!r}"
        )
    return games


def _baseline_games_signature(
    games: Sequence[dict[str, Any]],
    *,
    actor_model: str,
    max_rollout_steps: int,
    memory_window: int,
    max_commands: int,
    actor_temperature: float,
) -> str:
    payload = {
        "actor_model": actor_model,
        "max_rollout_steps": max_rollout_steps,
        "memory_window": memory_window,
        "max_commands": max_commands,
        "actor_temperature": actor_temperature,
        "skill_text": _NO_SKILL_BASELINE_TEXT,
        "games": [
            {
                "task_type": game.get("task_type", ""),
                "gamefile": game.get("gamefile", ""),
                "traj_json": game.get("traj_json", ""),
            }
            for game in games
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _outcome_task_id(game: dict[str, Any], rollout_index: int) -> str:
    """Stable task key shared by all online-bank reward conditions."""

    return f"{rollout_index:04d}:{game.get('gamefile', '')}"


def _episodes_as_outcomes(
    episodes: Sequence[dict[str, Any]],
    games: Sequence[dict[str, Any]],
    task_indices: Sequence[int],
) -> dict[str, bool]:
    if len(episodes) != len(task_indices):
        raise ValueError("episode count does not match outcome task indices")
    outcomes: dict[str, bool] = {}
    for episode, task_index in zip(episodes, task_indices, strict=True):
        game = games[int(task_index)]
        if str(episode.get("gamefile", "")) != str(game.get("gamefile", "")):
            raise ValueError("outcome episode gamefile does not match paired task")
        outcomes[_outcome_task_id(game, int(task_index))] = bool(episode.get("won"))
    return outcomes


def _online_skillbank_retrieval_query(
    *, task_type: str, task_description: str, initial_observation: str
) -> str:
    """Match the frozen, once-per-episode semantic retrieval query."""

    def clean(value: str) -> str:
        return " ".join(str(value or "").split())

    return "\n".join(
        (
            f"Task type: {clean(task_type)}",
            f"Task description: {clean(task_description)}",
            f"Current observation: {clean(initial_observation)}",
            "Previous 8 commands: none (retrieval occurs before the first action)",
            "Recent 5 online-memory action/result pairs: none "
            "(retrieval occurs before the first action)",
        )
    )


def _online_skillbank_initial_observation(episode: dict[str, Any]) -> str:
    steps = episode.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[0], dict):
        return str(steps[0].get("observation") or "")
    return ""


def _online_skillbank_prompt_text(
    entries: Sequence[dict[str, Any]], *, candidate_text: str = ""
) -> str:
    """Render retrieved entries and an optional candidate without extra policy."""

    bodies = [str(entry.get("text") or "").strip() for entry in entries]
    bodies = [body for body in bodies if body]
    candidate = str(candidate_text or "").strip()
    if candidate:
        bodies.append(candidate)
    if not bodies:
        return _NO_SKILL_BASELINE_TEXT
    if len(bodies) == 1:
        return bodies[0]
    return "Retrieved guidance skills:\n" + "\n\n".join(
        f"{index}. {body}" for index, body in enumerate(bodies, start=1)
    )


def _online_skillbank_skills_signature(skills: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps(
        list(skills),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_path_component(text: Any, *, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text or "")).strip("._-")
    return cleaned[:80] if cleaned else fallback


def _baseline_index_from_episodes(
    episodes: Sequence[dict[str, Any]],
    games: Sequence[dict[str, Any]],
    *,
    baseline_dir: Path,
    signature: str,
    skill_name: str,
) -> dict[str, Any]:
    entries = []
    wins = 0
    for rollout_index, game in enumerate(games):
        episode = episodes[rollout_index] if rollout_index < len(episodes) else None
        episode = episode if isinstance(episode, dict) else {}
        won = bool(episode.get("won"))
        wins += int(won)
        entries.append(
            {
                "rollout_index": rollout_index,
                "task_type": game.get("task_type", ""),
                "gamefile": game.get("gamefile", ""),
                "won": won,
                "status": episode.get("status", "missing"),
                "steps_taken": episode.get("steps_taken", 0),
                "episode_file": f"episode_{rollout_index:02d}.json",
            }
        )
    n_rollouts = len(entries)
    return {
        "status": "complete",
        "mode": "no_skill_baseline",
        "signature": signature,
        "skill_name": skill_name,
        "skill_text": _NO_SKILL_BASELINE_TEXT,
        "baseline_dir": str(baseline_dir),
        "wins": wins,
        "n_rollouts": n_rollouts,
        "sr": wins / max(1, n_rollouts),
        "entries": entries,
        "updated_at": time.time(),
    }


def _baseline_entries_by_index(
    index_payload: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    entries = index_payload.get("entries", [])
    if not isinstance(entries, list):
        return {}
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            rollout_index = int(entry.get("rollout_index", -1))
        except (TypeError, ValueError):
            continue
        if rollout_index >= 0:
            result[rollout_index] = entry
    return result


def _empty_baseline_delta_bucket() -> dict[str, Any]:
    return {
        "n": 0,
        "skill_wins": 0,
        "baseline_wins": 0,
        "improved_0_to_1": 0,
        "regressed_1_to_0": 0,
        "unchanged_0": 0,
        "unchanged_1": 0,
        "delta_sum": 0.0,
        "skill_sr": 0.0,
        "baseline_sr": 0.0,
        "delta_sr": 0.0,
    }


def _update_baseline_delta_bucket(
    bucket: dict[str, Any], *, skill_won: bool, baseline_win_rate: float
) -> None:
    baseline_win_rate = min(1.0, max(0.0, float(baseline_win_rate)))
    skill_value = float(bool(skill_won))
    bucket["n"] += 1
    bucket["skill_wins"] += int(skill_won)
    bucket["baseline_wins"] += baseline_win_rate
    bucket["improved_0_to_1"] += skill_value * (1.0 - baseline_win_rate)
    bucket["regressed_1_to_0"] += (1.0 - skill_value) * baseline_win_rate
    bucket["unchanged_1"] += skill_value * baseline_win_rate
    bucket["unchanged_0"] += (1.0 - skill_value) * (1.0 - baseline_win_rate)
    bucket["delta_sum"] += skill_value - baseline_win_rate


def _finalize_baseline_delta_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    n = max(1, int(bucket.get("n", 0)))
    bucket["skill_sr"] = float(bucket.get("skill_wins", 0)) / n
    bucket["baseline_sr"] = float(bucket.get("baseline_wins", 0)) / n
    bucket["delta_sr"] = float(bucket.get("delta_sum", 0.0)) / n
    return bucket


def _baseline_delta_metrics(
    episodes: Sequence[dict[str, Any]],
    baseline_index: dict[str, Any],
) -> dict[str, Any]:
    entries = _baseline_entries_by_index(baseline_index)
    overall = _empty_baseline_delta_bucket()
    by_task_type: dict[str, dict[str, Any]] = {}
    missing_baseline = 0
    for rollout_index, episode in enumerate(episodes):
        baseline_entry = entries.get(rollout_index)
        if baseline_entry is None:
            missing_baseline += 1
            continue
        skill_won = bool(episode.get("won"))
        baseline_win_rate = float(
            baseline_entry.get("win_rate", float(bool(baseline_entry.get("won"))))
        )
        _update_baseline_delta_bucket(
            overall,
            skill_won=skill_won,
            baseline_win_rate=baseline_win_rate,
        )
        task_type = str(
            episode.get("task_type") or baseline_entry.get("task_type") or "unknown"
        )
        task_bucket = by_task_type.setdefault(
            task_type,
            _empty_baseline_delta_bucket(),
        )
        _update_baseline_delta_bucket(
            task_bucket,
            skill_won=skill_won,
            baseline_win_rate=baseline_win_rate,
        )

    return {
        "status": baseline_index.get("status", ""),
        "baseline_signature": baseline_index.get("signature", ""),
        "baseline_cache_dir": baseline_index.get("baseline_dir", ""),
        "missing_baseline_entries": missing_baseline,
        "overall": _finalize_baseline_delta_bucket(overall),
        "per_task_type": {
            task_type: _finalize_baseline_delta_bucket(bucket)
            for task_type, bucket in sorted(by_task_type.items())
        },
    }


def _load_prompt_episode_ref(ref: dict[str, Any]) -> dict[str, Any] | None:
    episode_path_text = str(ref.get("source_episode_path") or "")
    if episode_path_text:
        try:
            payload = _read_json(Path(episode_path_text))
        except Exception:
            return None
        episode = payload.get("episode")
        if not isinstance(episode, dict):
            return None
        copied = dict(episode)
    else:
        rollout_path_text = str(ref.get("source_rollouts_path") or "")
        if not rollout_path_text:
            return None
        try:
            payload = _read_json(Path(rollout_path_text))
        except Exception:
            return None
        episodes = payload.get("episodes", [])
        rollout_index = int(ref.get("rollout_index", -1))
        if rollout_index < 0 or rollout_index >= len(episodes):
            return None
        episode = episodes[rollout_index]
        if not isinstance(episode, dict):
            return None
        copied = dict(episode)

    copied["_source_rollouts_path"] = str(ref.get("source_rollouts_path") or "")
    if episode_path_text:
        copied["_source_episode_path"] = episode_path_text
    copied["_source_rollout_index"] = int(ref.get("rollout_index", -1))
    return copied


def _complete_rollout_paths(round_dir: Path) -> list[Path]:
    rollout_paths: list[Path] = []
    for checkpoint_path in sorted(round_dir.glob("skills/*/checkpoint.json")):
        try:
            checkpoint = _read_json(checkpoint_path)
        except Exception:
            continue
        rollout_path = checkpoint_path.with_name("rollouts.json")
        if checkpoint.get("status") == "complete" and rollout_path.exists():
            rollout_paths.append(rollout_path)
    return rollout_paths


def _partial_rollout_failure_messages(
    round_dir: Path,
    *,
    actor_error_threshold: float,
    infra_error_threshold: float,
) -> list[str]:
    messages: list[str] = []
    if actor_error_threshold <= 0 and infra_error_threshold <= 0:
        return messages
    for partial_path in sorted(round_dir.glob("skills/*/rollouts_partial.json")):
        if partial_path.with_name("rollouts.json").exists():
            continue
        try:
            payload = _read_json(partial_path)
        except Exception:
            continue
        status_counts_raw = payload.get("status_counts", {})
        if not isinstance(status_counts_raw, dict):
            continue
        status_counts = Counter({str(k): int(v) for k, v in status_counts_raw.items()})
        expected = int(
            payload.get("n_rollouts_expected") or payload.get("n_rollouts") or 0
        )
        observed = sum(status_counts.values())
        if expected <= 0 or observed < expected:
            continue
        actor_errors = status_counts.get("actor_error", 0)
        infra_errors = actor_errors + status_counts.get("rollout_error", 0)
        actor_error_rate = actor_errors / max(1, expected)
        infra_error_rate = infra_errors / max(1, expected)
        if (
            actor_error_threshold > 0 and actor_error_rate >= actor_error_threshold
        ) or (infra_error_threshold > 0 and infra_error_rate >= infra_error_threshold):
            messages.append(
                "sample={sample} actor_error_rate={actor:.3f} "
                "infra_error_rate={infra:.3f} status_counts={counts}".format(
                    sample=payload.get("sample_key") or partial_path.parent.name,
                    actor=actor_error_rate,
                    infra=infra_error_rate,
                    counts=dict(status_counts),
                )
            )
    return messages


def _round_prompt_prepared_stall_messages(
    round_dir: Path,
    *,
    expected: int,
    complete_count: int,
    stall_timeout_s: float = 900.0,
) -> list[str]:
    if complete_count > 0 or expected <= 0:
        return []
    checkpoint_paths = sorted(round_dir.glob("skills/*/checkpoint.json"))
    if len(checkpoint_paths) < expected:
        return []
    statuses: Counter[str] = Counter()
    mtimes: list[float] = []
    for checkpoint_path in checkpoint_paths:
        try:
            payload = _read_json(checkpoint_path)
        except Exception:
            continue
        statuses[str(payload.get("status"))] += 1
        try:
            mtimes.append(checkpoint_path.stat().st_mtime)
        except OSError:
            pass
    if statuses.get("prompt_prepared", 0) < expected or not mtimes:
        return []
    age_s = time.time() - max(mtimes)
    if age_s < stall_timeout_s:
        return []
    return [
        "round has {prepared}/{expected} prompt_prepared checkpoints, "
        "0 completed rollouts, and no checkpoint update for {age:.0f}s".format(
            prepared=statuses.get("prompt_prepared", 0),
            expected=expected,
            age=age_s,
        )
    ]


def _generation_ready_checkpoint_paths(round_dir: Path) -> list[Path]:
    checkpoint_paths: list[Path] = []
    for checkpoint_path in sorted(round_dir.glob("skills/*/checkpoint.json")):
        try:
            checkpoint = _read_json(checkpoint_path)
        except Exception:
            continue
        if checkpoint.get("status") in _GENERATION_READY_STATUSES:
            checkpoint_paths.append(checkpoint_path)
    return checkpoint_paths


def _generation_failure_messages(round_dir: Path) -> list[str]:
    failures: list[str] = []
    for checkpoint_path in sorted(round_dir.glob("skills/*/checkpoint.json")):
        try:
            checkpoint = _read_json(checkpoint_path)
        except Exception:
            continue
        status = str(checkpoint.get("status", ""))
        if status not in _GENERATION_FAILURE_STATUSES:
            continue
        sample_key = str(checkpoint.get("sample_key") or checkpoint_path.parent.name)
        error_type = str(checkpoint.get("error_type") or "generation_error")
        error = str(checkpoint.get("error") or "unknown error")
        failures.append(f"{sample_key}: {error_type}: {error}")
    for failure_path in sorted(round_dir.glob("failures/*.json")):
        try:
            failure = _read_json(failure_path)
        except Exception:
            continue
        if failure.get("status") != "preparation_failed":
            continue
        group_index = int(failure.get("group_index", -1))
        error_type = str(failure.get("error_type") or "preparation_error")
        error = str(failure.get("error") or "unknown error")
        failures.append(f"group_{group_index:02d} preparation: {error_type}: {error}")
    return failures


_USABLE_PROMPT_EPISODE_STATUSES = frozenset({"won", "done", "max_steps", "complete"})
_UNUSABLE_PROMPT_STEP_STATUSES = frozenset(
    {"actor_error", "env_error", "parse_error", "rollout_error", "rollout_timeout"}
)


def _is_usable_prompt_episode(episode: dict[str, Any]) -> bool:
    status = str(episode.get("status") or "").strip().lower()
    if status not in _USABLE_PROMPT_EPISODE_STATUSES:
        return False

    valid_action_count = episode.get("valid_action_count")
    if valid_action_count is not None:
        try:
            return int(valid_action_count) > 0
        except (TypeError, ValueError):
            return False

    steps = episode.get("steps")
    if isinstance(steps, list):
        return any(
            str(step.get("action") or "").strip()
            and str(step.get("status") or "ok").strip().lower()
            not in _UNUSABLE_PROMPT_STEP_STATUSES
            for step in steps
            if isinstance(step, dict)
        )

    try:
        return int(episode.get("steps_taken") or 0) > 0
    except (TypeError, ValueError):
        return False


def _sanitize_trajectory_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Keep only environment evidence and the binary outcome for future prompts."""
    clean_steps: list[dict[str, Any]] = []
    for raw_step in episode.get("steps") or []:
        if not isinstance(raw_step, dict):
            continue
        clean_steps.append(
            {
                "t": int(raw_step.get("t", len(clean_steps)) or 0),
                "observation": str(raw_step.get("observation") or ""),
                "action": str(raw_step.get("action") or ""),
                "result": str(raw_step.get("result") or ""),
                "error": str(raw_step.get("error") or ""),
                "status": str(raw_step.get("status") or ""),
                "reward": float(raw_step.get("reward", 0.0) or 0.0),
                "score": float(raw_step.get("score", 0.0) or 0.0),
                "done": bool(raw_step.get("done")),
                "won": bool(raw_step.get("won")),
                "in_admissible": bool(raw_step.get("in_admissible")),
            }
        )
    won = bool(episode.get("won"))
    return {
        "schema_version": 1,
        "task_type": str(episode.get("task_type") or ""),
        "gamefile": str(episode.get("gamefile") or ""),
        "traj_json": str(episode.get("traj_json") or ""),
        "task_desc": str(episode.get("task_desc") or ""),
        "status": str(episode.get("status") or ""),
        "steps": clean_steps,
        "steps_taken": len(clean_steps),
        "done": bool(episode.get("done")),
        "won": won,
        "label": int(won),
        "outcome_reward": float(won),
        "final_score": float(episode.get("final_score", 0.0) or 0.0),
    }


def _sample_trajectory_pool_episodes(
    refs: Sequence[dict[str, Any]],
    *,
    seed: int,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Uniformly sample clean trajectories from the static+previous-step union."""
    shuffled = [dict(ref) for ref in refs if _is_usable_prompt_episode(ref)]
    random.Random(seed).shuffle(shuffled)
    selected_episodes: list[dict[str, Any]] = []
    selected_refs: list[dict[str, Any]] = []
    for ref in shuffled:
        episode = _load_prompt_episode_ref(ref)
        if episode is None or not _is_usable_prompt_episode(episode):
            continue
        selected_episodes.append(episode)
        selected_refs.append(ref)
        if len(selected_episodes) >= count:
            break
    if len(selected_episodes) < count:
        raise RuntimeError(
            "trajectory pool does not contain enough usable episodes: "
            f"requested={count}, available={len(selected_episodes)}"
        )
    source_counts = Counter(
        str(ref.get("pool_source") or "unknown") for ref in selected_refs
    )
    return selected_episodes, {
        "sampling_mode": "uniform_static_plus_previous_step",
        "candidate_count": len(shuffled),
        "sampled_total": len(selected_episodes),
        "sampled_successes": sum(bool(item.get("won")) for item in selected_episodes),
        "sampled_failures": sum(
            not bool(item.get("won")) for item in selected_episodes
        ),
        "source_counts": dict(source_counts),
        "selected_refs": selected_refs,
    }


def _per_task_type_metrics(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        task_type = str(episode.get("task_type", "unknown"))
        item = metrics.setdefault(task_type, {"wins": 0, "n": 0, "status_counts": {}})
        item["n"] += 1
        if episode.get("won"):
            item["wins"] += 1
        status = str(episode.get("status"))
        item["status_counts"][status] = item["status_counts"].get(status, 0) + 1
    for item in metrics.values():
        item["sr"] = item["wins"] / max(1, item["n"])
    return metrics


def _stable_seed(seed: int, label: str) -> int:
    return int(seed) + sum((index + 1) * ord(ch) for index, ch in enumerate(label))


def _sample_fixed_games(
    games: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    copied = [dict(game) for game in games]
    if count <= 0 or len(copied) <= count:
        return copied
    rng = random.Random(seed)
    return rng.sample(copied, k=count)


def _task_sampling_seed(seed: int, task_type: str) -> int:
    # Frozen salts: changing them would change which tasks a fixed seed draws.
    group = "id" if task_type in ALFWORLD_TASK_TYPES[:4] else "ood"
    return _stable_seed(seed, f"{group}:{task_type}")


def _select_games_by_type(
    *,
    data_root: Path,
    split: str,
    task_types: Sequence[str],
    tasks_per_type: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select games for each task type; tasks_per_type <= 0 keeps every game."""

    selected: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "split": split,
        "seed": seed,
        "tasks_per_type": tasks_per_type,
        "task_types": {},
    }
    for task_type in task_types:
        games = _collect_games(data_root, split, task_type)
        sampled = _sample_fixed_games(
            games, count=tasks_per_type, seed=_task_sampling_seed(seed, task_type)
        )
        for index, game in enumerate(sampled):
            game["eval_task_index"] = index
        selected.extend(sampled)
        metadata["task_types"][task_type] = {
            "available": len(games),
            "selected": len(sampled),
        }
    return selected, metadata


def _episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for episode in episodes if episode.get("won"))
    return {
        "wins": wins,
        "n": len(episodes),
        "sr": wins / max(1, len(episodes)),
        "status_counts": dict(
            Counter(str(episode.get("status")) for episode in episodes)
        ),
    }


def _aggregate_eval_no_skill_baseline_runs(
    runs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_repeated_baseline_runs(
        runs,
        schema=BaselineRecordSchema(
            records_key="entries",
            task_id_key="rollout_index",
            success_key="won",
            category_key="task_type",
        ),
    )
    aggregated_entries = []
    for summary in aggregate["task_summaries"]:
        representative = dict(summary["representative"])
        outcomes = list(summary["repeat_successes"])
        win_rate = float(summary["success_rate"])
        aggregated_entries.append(
            {
                **representative,
                "won": win_rate >= 0.5,
                "win_rate": win_rate,
                "repeat_wins": sum(outcomes),
                "repeat_count": int(summary["repeat_count"]),
                "repeat_outcomes": outcomes,
            }
        )

    def episode_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        task_count = int(metrics.get("task_count", 0))
        success_rate = float(metrics.get("success_rate", 0.0))
        return {
            "wins": success_rate * task_count,
            "n": task_count,
            "sr": success_rate,
            "sr_std": float(metrics.get("success_rate_std", 0.0)),
            "repeat_srs": list(metrics.get("repeat_success_rates", [])),
        }

    overall_sr = float(aggregate["metrics"].get("success_rate", 0.0))
    return {
        "wins": overall_sr * len(aggregated_entries),
        "n_rollouts": len(aggregated_entries),
        "sr": overall_sr,
        "sr_std": float(aggregate["metrics_std"].get("success_rate", 0.0)),
        "repeat_srs": [float(run.get("sr", 0.0)) for run in runs],
        "per_task_type": {
            task_type: episode_metrics(metrics)
            for task_type, metrics in aggregate["categories"].items()
        },
        "entries": aggregated_entries,
    }


def _acquire_json_lock(
    lock_path: Path,
    *,
    timeout_s: float = 120.0,
    foreign_host_stale_after_s: float | None = None,
) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            _remove_stale_lock(
                lock_path,
                foreign_host_stale_after_s=foreign_host_stale_after_s,
            )
            if time.monotonic() - start >= timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(0.2)
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "claimed_at": time.time(),
                    }
                )
            )
        return os.open(str(lock_path), os.O_RDONLY)


def _release_json_lock(fd: int, lock_path: Path) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _trajectory_ref_id(ref: dict[str, Any]) -> str:
    source_path_text = str(
        ref.get("source_rollouts_path") or ref.get("source_episode_path") or ""
    )
    source_path = Path(source_path_text) if source_path_text else None
    round_name = ""
    source_sample_key = ""
    if source_path is not None:
        source_sample_key = source_path.parent.name
        round_name = next(
            (part for part in reversed(source_path.parts) if part.startswith("round_")),
            "",
        )
    rollout_index_value = ref.get("rollout_index", -1)
    rollout_index = int(rollout_index_value if rollout_index_value is not None else -1)
    if round_name and source_sample_key and rollout_index >= 0:
        return f"{round_name}/{source_sample_key}/trajectory_{rollout_index:03d}"

    gamefile = str(ref.get("gamefile") or "")
    game_path = Path(gamefile) if gamefile else None
    if game_path is not None and len(game_path.parts) >= 3:
        game_id = "/".join(game_path.parts[-3:-1])
        if rollout_index >= 0:
            return f"{game_id}/trajectory_{rollout_index:03d}"
        return game_id

    encoded = json.dumps(ref, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "trajectory_" + hashlib.sha1(encoded).hexdigest()[:16]


def _manifest_trajectory_refs(prompt: dict[str, Any]) -> list[dict[str, Any]]:
    sampling = prompt.get("sampling") or prompt.get("trajectory_sampling") or {}
    selected_refs = (
        sampling.get("selected_refs", []) if isinstance(sampling, dict) else []
    )
    refs: list[dict[str, Any]] = []
    for raw_ref in selected_refs:
        if not isinstance(raw_ref, dict):
            continue
        source_path_text = str(
            raw_ref.get("source_rollouts_path")
            or raw_ref.get("source_episode_path")
            or ""
        )
        source_path = Path(source_path_text) if source_path_text else None
        gamefile = str(raw_ref.get("gamefile") or "")
        game_path = Path(gamefile) if gamefile else None
        refs.append(
            {
                "trajectory_id": _trajectory_ref_id(raw_ref),
                "source_round": (
                    next(
                        (
                            part
                            for part in reversed(source_path.parts)
                            if part.startswith("round_")
                        ),
                        "",
                    )
                    if source_path is not None
                    else ""
                ),
                "source_sample_key": (
                    source_path.parent.name if source_path is not None else ""
                ),
                "rollout_index": int(
                    raw_ref.get("rollout_index", -1)
                    if raw_ref.get("rollout_index", -1) is not None
                    else -1
                ),
                "task_type": str(raw_ref.get("task_type") or ""),
                "game_id": (
                    "/".join(game_path.parts[-3:-1])
                    if game_path is not None and len(game_path.parts) >= 3
                    else ""
                ),
                "gamefile": gamefile,
                "won": bool(raw_ref.get("won")),
                "status": str(raw_ref.get("status") or ""),
                "steps_taken": raw_ref.get("steps_taken"),
                "source_rollouts_path": str(raw_ref.get("source_rollouts_path") or ""),
                "source_episode_path": str(raw_ref.get("source_episode_path") or ""),
            }
        )
    return refs


def _skill_manifest_input_id(
    prompt: dict[str, Any], trajectory_refs: Sequence[dict[str, Any]]
) -> str:
    if trajectory_refs:
        identity: Any = [
            {
                "trajectory_id": ref["trajectory_id"],
                "source_rollouts_path": ref["source_rollouts_path"],
                "source_episode_path": ref["source_episode_path"],
                "rollout_index": ref["rollout_index"],
            }
            for ref in trajectory_refs
        ]
    else:
        identity = {
            "messages": prompt.get("messages", []),
            "prompt_render": prompt.get("prompt_render", {}),
        }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "input_" + hashlib.sha1(encoded).hexdigest()[:16]


def _embed_group_similarity_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Cache exact input chunks without changing padding or batch boundaries."""
    global _GROUP_SIMILARITY_CACHE_ROWS
    if not texts:
        return torch.empty((0, 0), dtype=torch.float32)
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("embedding batch_size and max_length must be positive")
    with _GROUP_SIMILARITY_EMBEDDER_LOCK:
        cached = _GROUP_SIMILARITY_EMBEDDERS.get(model_name)
        if cached is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            model.to("cpu")
            cached = (tokenizer, model)
            _GROUP_SIMILARITY_EMBEDDERS[model_name] = cached
        tokenizer, model = cached

        chunks: list[torch.Tensor] = []
        use_cache = os.environ.get("SKILL_EMBEDDING_CACHE", "1") != "0"
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                chunk_texts = tuple(texts[start : start + batch_size])
                cache_key = (
                    model_name,
                    id(tokenizer),
                    id(model),
                    max_length,
                    batch_size,
                    chunk_texts,
                )
                cached_chunk = (
                    _GROUP_SIMILARITY_CHUNKS.get(cache_key) if use_cache else None
                )
                if cached_chunk is not None:
                    _GROUP_SIMILARITY_CHUNKS.move_to_end(cache_key)
                    chunks.append(cached_chunk)
                    continue
                encoded = tokenizer(
                    list(chunk_texts),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to("cpu") for key, value in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                chunk = torch.nn.functional.normalize(pooled, p=2, dim=1).float().cpu()
                chunks.append(chunk)
                if use_cache and len(chunk_texts) <= _GROUP_SIMILARITY_CACHE_MAX_ROWS:
                    _GROUP_SIMILARITY_CHUNKS[cache_key] = chunk
                    _GROUP_SIMILARITY_CACHE_ROWS += len(chunk_texts)
                    while (
                        _GROUP_SIMILARITY_CACHE_ROWS > _GROUP_SIMILARITY_CACHE_MAX_ROWS
                    ):
                        _, evicted = _GROUP_SIMILARITY_CHUNKS.popitem(last=False)
                        _GROUP_SIMILARITY_CACHE_ROWS -= len(evicted)
        # cat allocates independent storage, so callers cannot corrupt the cache.
        return torch.cat(chunks, dim=0)
