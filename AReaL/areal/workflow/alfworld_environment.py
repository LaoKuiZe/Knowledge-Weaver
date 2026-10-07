# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import itertools
import os
import queue
import re
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import PreTrainedTokenizerFast

from areal.utils.hf_utils import apply_chat_template
from areal.utils.http_session import get_thread_session
from areal.workflow.alfworld_env_process import EnvironmentProcessError
from areal.workflow.alfworld_runtime import (
    _NO_SKILL_BASELINE_TEXT,
    _OUTCOME_EPISODE_DISPATCHER_LOCK,
    _OUTCOME_EPISODE_DISPATCHERS,
    _TEXTWORLD_GYM_LOCK,
    _atomic_write_json,
    _ensure_repo_importable,
    _get_episode_rollout_executor,
    _rollout_worker_tokenizer,
    _shorten,
    _unlink_if_exists,
    logger,
)


_BROKEN_POOL_RERUNS = 2


class _OutcomeEpisodeDispatcher:
    """Bound ProcessPool admission with a shared queue that prioritizes retries."""

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, int(max_workers))
        self._sequence = itertools.count()
        self._queue: queue.PriorityQueue[
            tuple[int, int, Future, dict[str, Any], float | None]
        ] = queue.PriorityQueue()
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"alfworld-outcome-episode-{index:03d}",
                daemon=True,
            )
            for index in range(self.max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float | None,
        priority: int,
    ) -> Future:
        result: Future = Future()
        self._queue.put(
            (
                int(priority),
                next(self._sequence),
                result,
                dict(payload),
                timeout_s,
            )
        )
        return result

    def _worker(self) -> None:
        while True:
            _, _, result, payload, timeout_s = self._queue.get()
            try:
                if result.set_running_or_notify_cancel():
                    self._run_episode(result, payload, timeout_s)
            except BaseException as exc:  # noqa: BLE001
                result.set_exception(exc)
            finally:
                self._queue.task_done()

    def _run_episode(
        self, result: Future, payload: dict[str, Any], timeout_s: float | None
    ) -> None:
        for attempt in range(_BROKEN_POOL_RERUNS + 1):
            executor = _get_episode_rollout_executor(self.max_workers)
            if executor is None:
                raise RuntimeError(
                    "outcome episode dispatcher requires a shared process pool"
                )
            try:
                process_future = executor.submit(_run_rollout_episode_process, payload)
                episode = process_future.result(timeout=timeout_s)
            except BrokenProcessPool:
                # Any dead worker fails every in-flight episode; rerun this one on
                # the rebuilt pool before reporting an infrastructure failure.
                if attempt == _BROKEN_POOL_RERUNS:
                    raise
                logger.warning(
                    "Rerunning ALFWorld episode %s on a rebuilt rollout pool",
                    payload.get("rollout_index"),
                )
                continue
            except FutureTimeoutError:
                cancelled = process_future.cancel()
                timeout_text = (
                    f"{timeout_s:.1f}s"
                    if timeout_s is not None
                    else "the configured deadline"
                )
                result.set_result(
                    _rollout_timeout_episode(
                        game=dict(payload["game"]),
                        skill_name=str(payload["skill_name"]),
                        error="outcome rollout timed out after " + timeout_text,
                        timeout_s=timeout_s,
                    )
                )
                if not cancelled:
                    # Retain the admission slot until the uncancellable process exits to preserve the global bound.
                    try:
                        process_future.result()
                    except BaseException:  # noqa: BLE001
                        pass
                return
            result.set_result(episode)
            return


def _get_outcome_episode_dispatcher(max_workers: int) -> _OutcomeEpisodeDispatcher:
    workers = max(1, int(max_workers))
    with _OUTCOME_EPISODE_DISPATCHER_LOCK:
        dispatcher = _OUTCOME_EPISODE_DISPATCHERS.get(workers)
        if dispatcher is None:
            dispatcher = _OutcomeEpisodeDispatcher(workers)
            _OUTCOME_EPISODE_DISPATCHERS[workers] = dispatcher
        return dispatcher


def _normalize_command(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^['\"`*\-\d\.\):\s]+", "", text)
    text = text.splitlines()[0] if text else ""
    text = text.strip(" '\"`")
    return re.sub(r"\s+", " ", text)


def _parse_react_action_response(raw: str) -> str:
    for line in raw.splitlines():
        match = re.match(r"\s*Action\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if match:
            return _normalize_command(match.group(1))

    match = re.search(
        r"Action\s*:\s*(.+?)(?:\n|$)", raw, flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return _normalize_command(match.group(1))
    return ""


def _match_admissible_command(pred: str, admissible: list[str]) -> tuple[str, bool]:
    normalized_to_cmd = {_normalize_command(cmd): cmd for cmd in admissible}
    if pred in normalized_to_cmd:
        return normalized_to_cmd[pred], True
    return pred, False


def format_online_memory(
    online_memory: list[dict[str, Any]], memory_window: int
) -> str:
    rows = online_memory[-memory_window:] if memory_window > 0 else []
    if not rows:
        return ""
    lines: list[str] = []
    for index, item in enumerate(rows, start=1):
        lines.extend(
            [
                f"Memory step {index}:",
                f"Observation: {_shorten(item.get('observation'), limit=420)}",
                f"Action: {_shorten(item.get('action'), limit=120)}",
                f"Result: {_shorten(item.get('result'), limit=420)}",
                (
                    f"Reward: {item.get('reward', 0.0)}; score: {item.get('score', 0.0)}; "
                    f"done: {item.get('done', False)}; won: {item.get('won', False)}"
                ),
            ]
        )
    return "\n".join(lines)


def actor_command_choice_messages(
    task_desc: str,
    obs: str,
    admissible: list[str],
    skill: str,
    online_memory: str,
    *,
    max_commands: int | None = 140,
) -> list[dict[str, str]]:
    visible_commands = sorted(admissible)
    if max_commands is not None and max_commands > 0:
        visible_commands = visible_commands[:max_commands]
    commands_text = "\n".join(f"- {cmd}" for cmd in visible_commands)
    system = (
        "You are an ALFWorld TextWorld agent. Choose the next action using this exact format:\n"
        "Thought: <concise reasoning, max 30 words>\n"
        "Action: <copy exactly one command from the admissible command list>\n"
        "Write the Action line immediately after the Thought line. Do not add any other text.\n"
        "Guidance skill:\n"
        f"{skill}"
    )
    memory_block = (
        f"Recent online episodic memory:\n{online_memory}\n" if online_memory else ""
    )
    user = (
        f"Task: {task_desc}\n"
        f"Observation: {obs[-1800:]}\n"
        f"{memory_block}"
        f"Admissible commands:\n{commands_text}\n\n"
        "Respond with Thought and Action:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _drain_outcome_condition_futures(
    futures: Sequence[tuple[str, Future]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    errors: list[tuple[str, BaseException]] = []
    # Drain every submitted coordinator before releasing the group lock.
    # Workers never write canonical episode files, but coordinators do.
    for name, future in futures:
        try:
            results[name] = future.result()
        except BaseException as exc:  # noqa: BLE001
            errors.append((name, exc))
    if errors:
        name, error = errors[0]
        raise RuntimeError(
            f"flattened outcome condition failed: {name}: {error!r}"
        ) from error
    return results


class ActorChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 120.0,
        tokenizer: PreTrainedTokenizerFast | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        if self.tokenizer is None:
            endpoint = "/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_tokens,
            }
            if self.enable_thinking is not None:
                payload["enable_thinking"] = bool(self.enable_thinking)
        else:
            endpoint = "/v1/completions"
            prompt = apply_chat_template(
                self.tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_tokens,
                "stop": ["<|im_end|>"],
            }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        last_error = ""
        for attempt in range(3):
            try:
                resp = get_thread_session().post(
                    self.base_url + endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )
                if getattr(resp, "status_code", None) == 429 and attempt < 2:
                    try:
                        delay = float(
                            resp.headers.get("Retry-After", 30 * (attempt + 1))
                        )
                    except (TypeError, ValueError):
                        delay = 30 * (attempt + 1)
                    time.sleep(min(120.0, max(1.0, delay)))
                    continue
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Actor response is not a JSON object: {data!r}")
                if self.tokenizer is not None:
                    return self._completion_response_as_chat(data)
                return data
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Actor chat failed: {last_error}")

    @staticmethod
    def _completion_response_as_chat(data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Actor completion response has no choices: {data!r}")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError(f"Actor completion choice is not an object: {first!r}")
        text = first.get("text", "")
        if not isinstance(text, str):
            raise RuntimeError(f"Actor completion text is not a string: {first!r}")
        return {
            **data,
            "choices": [
                {
                    **first,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                }
            ],
        }


def _make_env(repo_root: Path, gamefile: str, max_steps: int):
    if os.environ.get("ALFWORLD_ISOLATE_ENV_PROCESS", "0") == "1":
        from areal.workflow.alfworld_env_process import IsolatedTextWorldEnv

        return IsolatedTextWorldEnv(repo_root, gamefile, max_steps)
    _ensure_repo_importable(repo_root)
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
        AlfredInfos,
    )

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        facts=False,
        extras=["gamefile", "expert_plan"],
    )
    wrappers = [
        AlfredDemangler(shuffle=False),
        AlfredInfos,
        AlfredExpert(AlfredExpertType.HANDCODED),
    ]
    # Lock TextWorld's global state; keep actor calls outside the lock to overlap serving.
    with _TEXTWORLD_GYM_LOCK:
        env_id = textworld.gym.register_games(
            [gamefile],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=max_steps,
            wrappers=wrappers,
        )
        return textworld.gym.make(env_id)


def _batch_item(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _as_float(value: Any) -> float:
    try:
        return float(_batch_item(value, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _choose_actor_command(
    client: ActorChatClient,
    *,
    task_desc: str,
    obs: str,
    admissible: list[str],
    skill: str,
    online_memory: str,
    temperature: float,
    seed: int,
    max_commands: int,
) -> tuple[str, bool, str]:
    resp = client.chat(
        actor_command_choice_messages(
            task_desc,
            obs,
            admissible,
            skill,
            online_memory,
            max_commands=max_commands,
        ),
        temperature=temperature,
        seed=seed,
        max_tokens=192,
    )
    raw = str(resp["choices"][0]["message"]["content"])
    pred = _parse_react_action_response(raw)
    if not pred:
        pred = (
            "look"
            if any(_normalize_command(cmd) == "look" for cmd in admissible)
            else ""
        )
    action, exact_valid = _match_admissible_command(pred, admissible)
    return action, exact_valid, raw


def _rollout_episode(
    *,
    repo_root: Path,
    actor_client: ActorChatClient,
    game: dict[str, str],
    skill_name: str,
    skill_text: str,
    max_rollout_steps: int,
    memory_window: int,
    max_commands: int,
    actor_temperature: float,
    seed_base: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    env = _make_env(repo_root, game["gamefile"], max_steps=max_rollout_steps)
    env_lock = (
        nullcontext()
        if getattr(env, "process_isolated", False)
        else _TEXTWORLD_GYM_LOCK
    )
    online_memory: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    previous_score = 0.0
    final_score = 0.0
    final_done = False
    final_won = False
    status = "max_steps"

    def episode_snapshot(current_status: str) -> dict[str, Any]:
        valid_steps = [step for step in steps if step.get("status") != "parse_error"]
        episode = {
            "skill": skill_name,
            "task_type": game["task_type"],
            "gamefile": game["gamefile"],
            "traj_json": game["traj_json"],
            "task_desc": game["task_desc"],
            "status": current_status,
            "steps": list(steps),
            "steps_taken": len(steps),
            "final_score": final_score,
            "outcome_reward": 1.0 if final_won else 0.0,
            "done": final_done,
            "won": final_won,
            "parse_error_count": sum(
                1 for step in steps if step.get("status") == "parse_error"
            ),
            "env_error_count": sum(
                1 for step in steps if step.get("status") == "env_error"
            ),
            "in_admissible_rate": (
                mean(1.0 if step.get("in_admissible") else 0.0 for step in valid_steps)
                if valid_steps
                else 0.0
            ),
        }
        if "eval_task_index" in game:
            episode["eval_task_index"] = game["eval_task_index"]
        return episode

    def emit_progress(current_status: str) -> None:
        if progress_callback is not None:
            progress_callback(episode_snapshot(current_status))

    try:
        with env_lock:
            obs, infos = env.reset()
        for step_i in range(max_rollout_steps):
            current_obs = str(_batch_item(obs, ""))
            admissible = [
                str(cmd)
                for cmd in list(_batch_item(infos.get("admissible_commands"), []))
            ]
            memory_text = format_online_memory(online_memory, memory_window)
            if not admissible:
                steps.append(
                    {
                        "t": step_i,
                        "raw": "",
                        "action": "",
                        "observation": _shorten(current_obs, limit=1200),
                        "admissible_count": 0,
                        "exact_valid": False,
                        "status": "no_admissible_commands",
                        "reward": 0.0,
                        "score": previous_score,
                        "done": False,
                        "won": False,
                        "in_admissible": False,
                        "result": "",
                        "error": "environment returned no admissible commands",
                    }
                )
                status = "no_admissible_commands"
                emit_progress(status)
                break

            try:
                action, exact_valid, raw = _choose_actor_command(
                    actor_client,
                    task_desc=game["task_desc"],
                    obs=current_obs,
                    admissible=admissible,
                    skill=skill_text,
                    online_memory=memory_text,
                    temperature=actor_temperature,
                    seed=seed_base + step_i,
                    max_commands=max_commands,
                )
            except Exception as exc:  # noqa: BLE001
                steps.append(
                    {
                        "t": step_i,
                        "raw": "",
                        "action": "",
                        "observation": _shorten(current_obs, limit=1200),
                        "admissible_count": len(admissible),
                        "exact_valid": False,
                        "status": "actor_error",
                        "reward": 0.0,
                        "score": previous_score,
                        "done": False,
                        "won": False,
                        "in_admissible": False,
                        "result": "",
                        "error": repr(exc),
                    }
                )
                status = "actor_error"
                emit_progress(status)
                break

            step_record: dict[str, Any] = {
                "t": step_i,
                "raw": raw,
                "action": action,
                "observation": _shorten(current_obs, limit=1200),
                "admissible_count": len(admissible),
                "exact_valid": exact_valid,
                "status": "ok",
                "reward": 0.0,
                "score": previous_score,
                "done": False,
                "won": False,
                "in_admissible": action.lower() in {cmd.lower() for cmd in admissible},
                "result": "",
                "error": None,
            }

            try:
                with env_lock:
                    next_obs, scores, dones, next_infos = env.step([action])
            except Exception as exc:  # noqa: BLE001
                if getattr(env, "process_isolated", False):
                    # A dead/invalid environment transport is an infrastructure
                    # failure, never a valid unsuccessful reward observation.
                    raise
                step_record.update({"status": "env_error", "error": repr(exc)})
                steps.append(step_record)
                status = "env_error"
                emit_progress(status)
                break

            score = _as_float(scores)
            reward = score - previous_score
            done = bool(_batch_item(dones, False))
            won = bool(_batch_item(next_infos.get("won"), False))
            result_text = str(_batch_item(next_obs, ""))
            step_record.update(
                {
                    "reward": reward,
                    "score": score,
                    "done": done,
                    "won": won,
                    "result": _shorten(result_text),
                }
            )
            steps.append(step_record)
            online_memory.append(
                {
                    "observation": current_obs,
                    "action": action,
                    "result": result_text,
                    "reward": reward,
                    "score": score,
                    "done": done,
                    "won": won,
                }
            )
            obs, infos = next_obs, next_infos
            previous_score = score
            final_score = score
            final_done = done
            final_won = won
            emit_progress("won" if won else "done" if done else "in_progress")
            if done:
                status = "won" if won else "done"
                break
    finally:
        with env_lock:
            env.close()

    return episode_snapshot(status)


def _write_rollout_episode_file(
    *,
    sample_dir: Path,
    rollout_index: int,
    episode: dict[str, Any],
    final: bool,
    metadata: dict[str, Any],
    index_width: int = 2,
) -> None:
    now = time.time()
    payload = {
        **metadata,
        "rollout_index": rollout_index,
        "episode": episode,
        "updated_at": now,
    }
    suffix = f"{rollout_index:0{index_width}d}"
    if final:
        _atomic_write_json(sample_dir / f"episode_{suffix}.json", payload)
        _unlink_if_exists(sample_dir / f"current_episode_{suffix}.json")
        return
    _atomic_write_json(sample_dir / "current_episode.json", payload)
    _atomic_write_json(
        sample_dir / f"current_episode_{suffix}.json",
        payload,
    )


def _run_rollout_episode_process(payload: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(str(payload["repo_root"]))
    sample_dir = Path(str(payload["sample_dir"]))
    game = dict(payload["game"])
    skill_name = str(payload["skill_name"])
    rollout_index = int(payload["rollout_index"])
    index_width = int(payload.get("index_width", 2))
    progress_metadata = dict(payload["progress_metadata"])
    persist_progress = bool(payload.get("persist_progress", True))
    tokenizer = _rollout_worker_tokenizer(payload.get("tokenizer_path"))
    actor_client = ActorChatClient(
        base_url=str(payload["actor_base_url"]),
        model=str(payload["actor_model"]),
        api_key=str(payload.get("actor_api_key") or ""),
        timeout_s=float(payload["actor_timeout_s"]),
        tokenizer=tokenizer,
        enable_thinking=payload.get("actor_enable_thinking"),
    )
    progress_interval = max(
        0.0, float(os.environ.get("SKILL_PROGRESS_INTERVAL_SECONDS", "5"))
    )
    last_progress_write = float("-inf")

    def progress_callback(episode: dict[str, Any]) -> None:
        nonlocal last_progress_write
        now = time.monotonic()
        if (
            episode.get("status") == "in_progress"
            and now - last_progress_write < progress_interval
        ):
            return
        _write_rollout_episode_file(
            sample_dir=sample_dir,
            rollout_index=rollout_index,
            episode=episode,
            final=False,
            metadata=progress_metadata,
            index_width=index_width,
        )
        last_progress_write = now

    try:
        episode = _rollout_episode(
            repo_root=repo_root,
            actor_client=actor_client,
            game=game,
            skill_name=skill_name,
            skill_text=str(payload["skill_text"]),
            max_rollout_steps=int(payload["max_rollout_steps"]),
            memory_window=int(payload["memory_window"]),
            max_commands=int(payload["max_commands"]),
            actor_temperature=float(payload["actor_temperature"]),
            seed_base=int(payload["seed_base"]),
            progress_callback=progress_callback if persist_progress else None,
        )
    except EnvironmentProcessError:
        logger.error(
            "Isolated ALFWorld environment failed during skill rollout", exc_info=True
        )
        raise
    except Exception as exc:
        episode = _invalid_skill_episode(
            game=game, skill_name=skill_name, error="rollout_exception: " + repr(exc)
        )
        episode["status"] = "rollout_error"
        episode["traceback"] = traceback.format_exc()
    if persist_progress:
        _write_rollout_episode_file(
            sample_dir=sample_dir,
            rollout_index=rollout_index,
            episode=episode,
            final=True,
            metadata=progress_metadata,
            index_width=index_width,
        )
    return episode


def _run_no_skill_result_episode_process(payload: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(str(payload["repo_root"]))
    game = dict(payload["game"])
    rollout_index = int(payload["rollout_index"])
    tokenizer = _rollout_worker_tokenizer(payload.get("tokenizer_path"))
    actor_client = ActorChatClient(
        base_url=str(payload["actor_base_url"]),
        model=str(payload["actor_model"]),
        api_key=str(payload.get("actor_api_key") or ""),
        timeout_s=float(payload["actor_timeout_s"]),
        tokenizer=tokenizer,
        enable_thinking=payload.get("actor_enable_thinking"),
    )
    skill_name = str(payload.get("skill_name") or "baseline/no_skill")
    try:
        episode = _rollout_episode(
            repo_root=repo_root,
            actor_client=actor_client,
            game=game,
            skill_name=skill_name,
            skill_text=str(payload.get("skill_text", _NO_SKILL_BASELINE_TEXT)),
            max_rollout_steps=int(payload["max_rollout_steps"]),
            memory_window=int(payload["memory_window"]),
            max_commands=int(payload["max_commands"]),
            actor_temperature=float(payload["actor_temperature"]),
            seed_base=int(payload["seed_base"]),
            progress_callback=None,
        )
    except EnvironmentProcessError:
        logger.error(
            "Isolated ALFWorld environment failed during baseline rollout",
            exc_info=True,
        )
        raise
    except Exception as exc:  # noqa: BLE001
        episode = _invalid_skill_episode(
            game=game,
            skill_name=skill_name,
            error="no_skill_baseline_exception: " + repr(exc),
        )
        episode["status"] = "no_skill_baseline_error"
        episode["traceback"] = traceback.format_exc()

    if payload.get("return_episode"):
        return episode

    entry = {
        "rollout_index": rollout_index,
        "task_type": game.get("task_type", ""),
        "gamefile": game.get("gamefile", ""),
        "traj_json": game.get("traj_json", ""),
        "won": bool(episode.get("won")),
    }
    if "eval_task_index" in game:
        entry["eval_task_index"] = game["eval_task_index"]
    return entry


def _invalid_skill_episode(
    *,
    game: dict[str, str],
    skill_name: str,
    error: str,
) -> dict[str, Any]:
    episode = {
        "skill": skill_name,
        "task_type": game["task_type"],
        "gamefile": game["gamefile"],
        "traj_json": game["traj_json"],
        "task_desc": game["task_desc"],
        "status": "invalid_skill_json",
        "steps": [],
        "steps_taken": 0,
        "final_score": 0.0,
        "outcome_reward": 0.0,
        "done": False,
        "won": False,
        "parse_error_count": 0,
        "env_error_count": 0,
        "in_admissible_rate": 0.0,
        "error": error,
    }
    if "eval_task_index" in game:
        episode["eval_task_index"] = game["eval_task_index"]
    return episode


def _rollout_timeout_episode(
    *,
    game: dict[str, Any],
    skill_name: str,
    error: str,
    timeout_s: float | None,
    partial_episode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if partial_episode:
        episode = copy.deepcopy(partial_episode)
        episode.setdefault("skill", skill_name)
        episode.setdefault("task_type", game["task_type"])
        episode.setdefault("gamefile", game["gamefile"])
        episode.setdefault("traj_json", game["traj_json"])
        episode.setdefault("task_desc", game["task_desc"])
    else:
        episode = _invalid_skill_episode(
            game=game,
            skill_name=skill_name,
            error=error,
        )
    episode["status"] = "rollout_timeout"
    episode["done"] = False
    episode["won"] = False
    episode["outcome_reward"] = 0.0
    episode["error"] = error
    episode["timeout_s"] = timeout_s
    if "eval_task_index" in game:
        episode["eval_task_index"] = game["eval_task_index"]
    return episode
