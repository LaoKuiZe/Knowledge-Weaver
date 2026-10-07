# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import wait as wait_futures
from pathlib import Path
from typing import Any

import torch

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, ModelResponse
from areal.utils import stats_tracker
from areal.utils.hf_utils import apply_chat_template
from areal.workflow.alfworld_environment import (
    _invalid_skill_episode,
    _run_no_skill_result_episode_process,
    _run_rollout_episode_process,
)
from areal.workflow.alfworld_runtime import (
    _NO_SKILL_BASELINE_TEXT,
    _TRAJECTORY_RENDER_VERSION,
    MIXED_TASK_TYPE,
    PROMPT_CATEGORY_BY_TASK_TYPE,
    _aggregate_eval_no_skill_baseline_runs,
    _append_jsonl,
    _atomic_write_json,
    _baseline_delta_metrics,
    _baseline_games_signature,
    _episode_metrics,
    _get_episode_rollout_executor,
    _per_task_type_metrics,
    _read_json,
    _remove_completed_episode_snapshots,
    _rollout_progress_event,
    _select_games_by_type,
    _unlink_if_exists,
    logger,
)
from areal.workflow.skill_eval_baseline import RepeatedNoSkillBaselinePipeline
from areal.workflow.skill_prompts import (
    _fit_skill_prompt_messages,
    _parse_skill_generation,
    _skill_prompt_budget,
    _split_skill_generation_sections,
    _write_skill_manifest,
    skill_payload_to_prompt_text,
)
from areal.workflow.skill_reports import _write_eval_summary
from areal.workflow.skill_token_feedback import _output_token_nll_stats


class ALFWorldEvaluationMixin:
    """Generate one held-out eval skill per input task type and score its unseen-task SR."""

    def __init__(
        self,
        *args: Any,
        eval_data_split: str = "valid_unseen",
        eval_k: int = 6,
        eval_tasks_per_type: int = 0,
        eval_seed: int = 1001,
        eval_no_skill_baseline_repeats: int = 3,
        eval_no_skill_baseline_seed: int = 8_100_000,
        checkpoint_model_root: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if eval_k != len(self.task_types):
            raise ValueError(
                "eval requires exactly one skill per input task type: "
                f"eval_k={eval_k}, task_types={len(self.task_types)}"
            )
        if eval_tasks_per_type < 0:
            raise ValueError("eval_tasks_per_type must be non-negative")
        if eval_no_skill_baseline_repeats <= 0:
            raise ValueError("eval_no_skill_baseline_repeats must be positive")
        self.eval_data_split = eval_data_split
        self.eval_k = int(eval_k)
        self.eval_tasks_per_type = int(eval_tasks_per_type)
        self.eval_seed = int(eval_seed)
        self.eval_no_skill_baseline_repeats = int(eval_no_skill_baseline_repeats)
        self.eval_no_skill_baseline_seed = int(eval_no_skill_baseline_seed)
        self.checkpoint_model_root = (
            Path(checkpoint_model_root).expanduser().resolve()
            if checkpoint_model_root
            else None
        )

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor] | None:
        model_version = int(engine.get_version())
        prepared = await asyncio.to_thread(
            self._prepare_eval_sample, data, model_version
        )
        input_ids = apply_chat_template(
            self.tokenizer,
            prepared["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.skill_generation_enable_thinking,
        )
        generation_attempts: list[dict[str, Any]] = []
        resp: ModelResponse | None = None
        raw = ""
        max_retries = self._max_skill_generation_retries()
        for attempt_index in range(max_retries + 1):
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)
            raw = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=False)
            parsed, schema_valid, schema_error, repair_notes = _parse_skill_generation(
                raw,
                str(prepared["prompt_category"]),
                skill_output_format=self.skill_output_format,
                max_words=int(getattr(self, "skill_description_max_words", 0)),
            )
            thinking_text, answer_text, section_notes = (
                _split_skill_generation_sections(raw)
            )
            output_entropy_proxy, output_entropy_token_count = _output_token_nll_stats(
                resp.output_logprobs
            )
            generation_attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "raw": raw,
                    "thinking": thinking_text,
                    "answer": answer_text,
                    "parsed": parsed,
                    "schema_valid": schema_valid,
                    "schema_error": schema_error,
                    "repair_notes": section_notes + repair_notes,
                    "input_token_count": len(resp.input_tokens),
                    "output_token_count": len(resp.output_tokens),
                    "output_entropy_proxy": output_entropy_proxy,
                    "output_entropy_token_count": output_entropy_token_count,
                    "updated_at": time.time(),
                }
            )
            await asyncio.to_thread(
                self._write_eval_generation_attempts, prepared, generation_attempts
            )
            if schema_valid:
                break
        if resp is None:
            raise RuntimeError("skill generator did not return a response")

        result = await self._run_skill_eval_blocking(
            self._evaluate_eval_and_save,
            prepared,
            raw,
            list(resp.input_tokens),
            list(resp.output_tokens),
            generation_attempts,
        )
        all_sr = float(result.get("overall", {}).get("sr", 0.0))
        baseline_delta = result.get("baseline_delta", {})
        baseline_delta = baseline_delta if isinstance(baseline_delta, dict) else {}
        no_skill_baseline = result.get("no_skill_baseline", {})
        no_skill_baseline = (
            no_skill_baseline if isinstance(no_skill_baseline, dict) else {}
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            alfworld_eval_all_sr=all_sr,
            alfworld_eval_baseline_all_sr=float(no_skill_baseline.get("sr", 0.0)),
            alfworld_eval_baseline_all_sr_std=float(
                no_skill_baseline.get("sr_std", 0.0)
            ),
            alfworld_eval_baseline_repeats=int(
                no_skill_baseline.get("repeat_count", 0)
            ),
            alfworld_eval_baseline_total_rollouts=int(
                no_skill_baseline.get("total_rollouts", 0)
            ),
            alfworld_eval_all_delta_sr=float(
                baseline_delta.get("overall", {}).get("delta_sr", 0.0)
            ),
            alfworld_eval_schema_valid=float(bool(result.get("schema_valid"))),
            alfworld_eval_generation_attempts=int(result.get("attempt_count", 0)),
            alfworld_eval_output_entropy_proxy=output_entropy_proxy,
        )
        input_task_type = str(
            result.get("input_task_type") or result.get("matched_task_type") or ""
        )
        if input_task_type:
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                **{
                    f"alfworld_eval_matched_{input_task_type}_sr": float(
                        result.get("matched_task_type_sr", 0.0) or 0.0
                    )
                }
            )
        eval_summary = result.get("_eval_summary")
        if isinstance(eval_summary, dict) and eval_summary.get("status") == "complete":
            matched = eval_summary.get("matched_task_type_srs") or {}
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                **{
                    f"alfworld_eval_matched_{task_name}_sr": float(
                        matched.get(task_name, 0.0) or 0.0
                    )
                    for task_name in self.task_types
                }
            )
        return self._tensor_result(resp, all_sr)

    @staticmethod
    def _eval_step_name(checkpoint_global_step: int) -> str:
        if checkpoint_global_step < 0:
            return "globalstep_pretrain"
        return f"globalstep_{checkpoint_global_step:06d}"

    def _eval_games(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _select_games_by_type(
            data_root=self.data_root,
            split=self.eval_data_split,
            task_types=self.task_types,
            tasks_per_type=self.eval_tasks_per_type,
            seed=self.eval_seed,
        )

    def _prepare_eval_sample(
        self, data: dict[str, Any], model_version: int
    ) -> dict[str, Any]:
        self._ensure_prompt_contract()
        eval_index = int(data.get("eval_index", 0))
        trajectory_pool_enabled = bool(getattr(self, "trajectory_pool_enabled", False))
        eval_k = int(data.get("eval_k", self.eval_k))
        checkpoint_global_step = model_version - 1
        step_name = self._eval_step_name(checkpoint_global_step)
        eval_dir = self.artifact_dir / "eval" / step_name
        sample_key = f"eval_skill_{eval_index:02d}"
        sample_dir = eval_dir / "skills" / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = sample_dir / "prompt.json"
        if trajectory_pool_enabled and prompt_path.exists():
            try:
                existing = _read_json(prompt_path)
                if existing.get(
                    "trajectory_pool_enabled"
                ) is True and self._prepared_prompt_matches_current(existing):
                    self._maybe_delete_consumed_previous_step_pool(
                        max(0, checkpoint_global_step + 1)
                    )
                    return {
                        "mode": "eval",
                        "eval_index": eval_index,
                        "eval_k": eval_k,
                        "sample_key": sample_key,
                        "sample_dir": str(sample_dir),
                        "eval_dir": str(eval_dir),
                        "task_type": MIXED_TASK_TYPE,
                        "input_task_type": str(existing["input_task_type"]),
                        "prompt_category": str(existing["prompt_category"]),
                        "checkpoint_model_version": model_version,
                        "checkpoint_global_step": checkpoint_global_step,
                        "messages": list(existing.get("messages") or []),
                        "selected_games": list(existing.get("selected_games") or []),
                        "game_sampling": dict(existing.get("game_sampling") or {}),
                        "skill_prompt_version": str(
                            existing.get("skill_prompt_version") or ""
                        ),
                        "trajectory_render_version": str(
                            existing.get("trajectory_render_version") or ""
                        ),
                    }
            except Exception:
                logger.warning(
                    "failed to reuse prepared eval prompt %s",
                    prompt_path,
                    exc_info=True,
                )

        input_task_type = self.task_types[eval_index]
        prompt_category = PROMPT_CATEGORY_BY_TASK_TYPE[input_task_type]
        # Passing checkpoint+1 selects the just-completed checkpoint step as
        # the dynamic half of the static+latest trajectory union.
        pool_step = max(0, checkpoint_global_step + 1)
        sampled_episodes, sampling = self._select_trajectory_pool_prompt(
            input_task_type=input_task_type,
            training_global_step=pool_step,
            seed=(
                self.eval_seed * 100000
                + max(0, checkpoint_global_step) * 1000
                + eval_index
            ),
        )
        sampling["eval_checkpoint_global_step"] = checkpoint_global_step

        games, game_sampling = self._eval_games()
        messages, prompt_render = _fit_skill_prompt_messages(
            tokenizer=self.tokenizer,
            prompt_category=prompt_category,
            sampled_episodes=sampled_episodes,
            max_prompt_tokens=_skill_prompt_budget(self.gconfig),
            prompt_observation_char_limit=self.prompt_observation_char_limit,
            prompt_result_char_limit=self.prompt_result_char_limit,
            enable_thinking=self.skill_generation_enable_thinking,
            skill_output_format=self.skill_output_format,
            skill_prompt_version=self.skill_prompt_version,
        )
        prompt_payload = {
            "mode": "eval",
            "task_type": MIXED_TASK_TYPE,
            "input_task_type": input_task_type,
            "prompt_category": prompt_category,
            "eval_index": eval_index,
            "eval_k": eval_k,
            "checkpoint_model_version": model_version,
            "checkpoint_global_step": checkpoint_global_step,
            "trajectory_pool_enabled": trajectory_pool_enabled,
            "skill_prompt_version": self.skill_prompt_version,
            "trajectory_render_version": _TRAJECTORY_RENDER_VERSION,
            "messages": messages,
            "prompt_render": prompt_render,
            "sampled_episodes": [
                {
                    "task_type": episode.get("task_type"),
                    "gamefile": episode.get("gamefile"),
                    "won": episode.get("won"),
                    "label": int(bool(episode.get("won"))),
                    "status": episode.get("status"),
                    "steps_taken": episode.get("steps_taken"),
                    "source_rollouts_path": episode.get("_source_rollouts_path"),
                    "source_episode_path": episode.get("_source_episode_path"),
                }
                for episode in sampled_episodes
            ],
            "trajectory_sampling": sampling,
            "game_sampling": game_sampling,
            "selected_games": games,
            "created_at": time.time(),
        }
        _atomic_write_json(prompt_path, prompt_payload)
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "prompt_prepared",
                "mode": "eval",
                "eval_index": eval_index,
                "input_task_type": input_task_type,
                "checkpoint_model_version": model_version,
                "checkpoint_global_step": checkpoint_global_step,
                "updated_at": time.time(),
            },
        )
        _atomic_write_json(
            eval_dir / "eval_state.json",
            {
                "status": "running",
                "checkpoint_model_version": model_version,
                "checkpoint_global_step": checkpoint_global_step,
                "expected_skill_count": eval_k,
                "input_task_types": list(self.task_types),
                "tasks_per_type": self.eval_tasks_per_type,
                "updated_at": time.time(),
            },
        )
        if trajectory_pool_enabled:
            self._maybe_delete_consumed_previous_step_pool(
                max(0, checkpoint_global_step + 1)
            )
        return {
            "mode": "eval",
            "eval_index": eval_index,
            "eval_k": eval_k,
            "sample_key": sample_key,
            "sample_dir": str(sample_dir),
            "eval_dir": str(eval_dir),
            "task_type": MIXED_TASK_TYPE,
            "input_task_type": input_task_type,
            "prompt_category": prompt_category,
            "checkpoint_model_version": model_version,
            "checkpoint_global_step": checkpoint_global_step,
            "messages": messages,
            "selected_games": games,
            "game_sampling": game_sampling,
        }

    def _write_eval_generation_attempts(
        self, prepared: dict[str, Any], attempts: list[dict[str, Any]]
    ) -> None:
        sample_dir = Path(prepared["sample_dir"])
        latest = attempts[-1] if attempts else {}
        schema_valid = bool(latest.get("schema_valid"))
        max_retries = self._max_skill_generation_retries()
        exhausted = len(attempts) > max_retries
        status = (
            "generation_valid"
            if schema_valid
            else "generation_exhausted"
            if exhausted
            else "generation_retrying"
        )
        payload = {
            "mode": "eval",
            "eval_index": prepared["eval_index"],
            "eval_k": prepared["eval_k"],
            "checkpoint_model_version": prepared["checkpoint_model_version"],
            "checkpoint_global_step": prepared["checkpoint_global_step"],
            "attempt_count": len(attempts),
            "max_retries": max_retries,
            "schema_valid": schema_valid,
            "schema_error": latest.get("schema_error", ""),
            "attempts": attempts,
            "updated_at": time.time(),
        }
        _atomic_write_json(sample_dir / "generation_attempts.json", payload)
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": status,
                "mode": "eval",
                "eval_index": prepared["eval_index"],
                "checkpoint_model_version": prepared["checkpoint_model_version"],
                "checkpoint_global_step": prepared["checkpoint_global_step"],
                "schema_valid": schema_valid,
                "attempt_count": len(attempts),
                "max_retries": max_retries,
                "updated_at": time.time(),
            },
        )

    def _eval_no_skill_baseline_pipeline(
        self, games: Sequence[dict[str, Any]]
    ) -> RepeatedNoSkillBaselinePipeline:
        games_signature = _baseline_games_signature(
            games,
            actor_model=self.actor_model,
            max_rollout_steps=self.max_rollout_steps,
            memory_window=self.memory_window,
            max_commands=self.max_commands,
            actor_temperature=self.actor_temperature,
        )
        signature_payload = {
            "benchmark": "alfworld",
            "games_signature": games_signature,
        }
        return RepeatedNoSkillBaselinePipeline(
            cache_root=self.artifact_dir / "eval" / "no_skill_baselines",
            signature_payload=signature_payload,
            repeat_count=self.eval_no_skill_baseline_repeats,
            seed_base=self.eval_no_skill_baseline_seed,
            task_count=len(games),
            final_filename="baseline_index.json",
            metadata={
                "skill_name": "eval/baseline/no_skill",
                "skill_text": _NO_SKILL_BASELINE_TEXT,
                "actor_model": self.actor_model,
                "max_rollout_steps": self.max_rollout_steps,
                "memory_window": self.memory_window,
                "max_commands": self.max_commands,
                "actor_temperature": self.actor_temperature,
            },
            lock_timeout_s=max(7200.0, float(self.rollout_timeout_s or 0.0) + 600.0),
        )

    def _load_or_create_eval_no_skill_baseline(
        self, games: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        game_list = list(games)
        pipeline = self._eval_no_skill_baseline_pipeline(game_list)

        def run_repeat(
            baseline_dir: Path, signature: str, repeat_index: int
        ) -> dict[str, Any]:
            return self._run_eval_no_skill_baseline_repeat(
                baseline_dir=baseline_dir,
                signature=signature,
                games=game_list,
                repeat_index=repeat_index,
            )

        return pipeline.load_or_run(
            run_repeat=run_repeat,
            aggregate_runs=_aggregate_eval_no_skill_baseline_runs,
        )

    def _run_eval_no_skill_baseline_repeat(
        self,
        *,
        baseline_dir: Path,
        signature: str,
        games: list[dict[str, Any]],
        repeat_index: int,
    ) -> dict[str, Any]:
        skill_name = "eval/baseline/no_skill"
        repeat_seed = self.eval_no_skill_baseline_seed + repeat_index * 100_000
        metadata = {
            "mode": "eval_no_skill_baseline",
            "signature": signature,
            "skill_name": skill_name,
            "repeat_index": repeat_index,
            "seed_base": repeat_seed,
            "n_rollouts_expected": len(games),
        }

        entries: list[dict[str, Any] | None] = [None] * len(games)
        executor = _get_episode_rollout_executor(self.episode_rollout_workers)
        futures = {
            executor.submit(
                _run_no_skill_result_episode_process,
                {
                    "repo_root": str(self.repo_root),
                    "game": game,
                    "skill_name": skill_name,
                    "rollout_index": rollout_index,
                    "max_rollout_steps": self.max_rollout_steps,
                    "memory_window": self.memory_window,
                    "max_commands": self.max_commands,
                    "actor_base_url": self.actor_base_url,
                    "actor_model": self.actor_model,
                    "actor_api_key": self.actor_api_key,
                    "actor_timeout_s": self.actor_timeout_s,
                    "actor_temperature": self.actor_temperature,
                    "tokenizer_path": self.tokenizer_path,
                    "seed_base": repeat_seed + rollout_index * 100,
                },
            ): rollout_index
            for rollout_index, game in enumerate(games)
        }
        pending = set(futures)
        deadline = (
            time.monotonic() + self.rollout_timeout_s
            if self.rollout_timeout_s is not None
            else None
        )
        while pending:
            wait_timeout = None
            if deadline is not None:
                wait_timeout = max(0.0, deadline - time.monotonic())
                if wait_timeout <= 0.0:
                    break
            done, pending = wait_futures(
                pending,
                timeout=wait_timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                break
            for future in done:
                rollout_index = futures[future]
                try:
                    entries[rollout_index] = future.result()
                except Exception:  # noqa: BLE001
                    game = games[rollout_index]
                    entries[rollout_index] = {
                        "rollout_index": rollout_index,
                        "task_type": game.get("task_type", ""),
                        "gamefile": game.get("gamefile", ""),
                        "traj_json": game.get("traj_json", ""),
                        "eval_task_index": game.get("eval_task_index", -1),
                        "won": False,
                    }

        if pending:
            timeout_text = (
                f"{self.rollout_timeout_s:.1f}s"
                if self.rollout_timeout_s is not None
                else "the configured deadline"
            )
            logger.warning(
                "eval no-skill baseline %s repeat %d timed out %d/%d rollouts after %s",
                signature,
                repeat_index,
                len(pending),
                len(futures),
                timeout_text,
            )
            for future in pending:
                rollout_index = futures[future]
                future.cancel()
                game = games[rollout_index]
                entries[rollout_index] = {
                    "rollout_index": rollout_index,
                    "task_type": game.get("task_type", ""),
                    "gamefile": game.get("gamefile", ""),
                    "traj_json": game.get("traj_json", ""),
                    "eval_task_index": game.get("eval_task_index", -1),
                    "won": False,
                }

        complete_entries: list[dict[str, Any]] = []
        for rollout_index, entry in enumerate(entries):
            if entry is None:
                game = games[rollout_index]
                entry = {
                    "rollout_index": rollout_index,
                    "task_type": game.get("task_type", ""),
                    "gamefile": game.get("gamefile", ""),
                    "traj_json": game.get("traj_json", ""),
                    "eval_task_index": game.get("eval_task_index", -1),
                    "won": False,
                }
            complete_entries.append(entry)
        wins = sum(1 for entry in complete_entries if entry.get("won"))
        result = {
            **metadata,
            "status": "complete",
            "wins": wins,
            "n_rollouts": len(complete_entries),
            "sr": wins / max(1, len(complete_entries)),
            "per_task_type": _per_task_type_metrics(complete_entries),
            "entries": complete_entries,
            "updated_at": time.time(),
        }
        return result

    def _evaluate_eval_and_save(
        self,
        prepared: dict[str, Any],
        raw_generation: str,
        input_tokens: list[int],
        output_tokens: list[int],
        generation_attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sample_dir = Path(prepared["sample_dir"])
        eval_dir = Path(prepared["eval_dir"])
        eval_index = int(prepared["eval_index"])
        eval_k = int(prepared["eval_k"])
        checkpoint_model_version = int(prepared["checkpoint_model_version"])
        checkpoint_global_step = int(prepared["checkpoint_global_step"])
        sample_key = str(prepared["sample_key"])
        input_task_type = str(prepared.get("input_task_type") or MIXED_TASK_TYPE)
        skill_name = f"skill/eval/globalstep_{checkpoint_global_step}/{sample_key}"
        final_attempt = generation_attempts[-1] if generation_attempts else {}
        parsed = final_attempt.get("parsed")
        parsed = parsed if isinstance(parsed, dict) else None
        valid = bool(final_attempt.get("schema_valid"))
        parse_error = str(final_attempt.get("schema_error") or "")
        repair_notes = list(final_attempt.get("repair_notes") or [])
        games = list(prepared["selected_games"])
        max_retries = self._max_skill_generation_retries()

        _atomic_write_json(
            sample_dir / "generation.json",
            {
                "mode": "eval",
                "skill_name": skill_name,
                "input_task_type": input_task_type,
                "raw": raw_generation,
                "parsed": parsed,
                "schema_valid": valid,
                "schema_error": parse_error,
                "repair_notes": repair_notes,
                "attempt_count": len(generation_attempts),
                "max_retries": max_retries,
                "attempts": generation_attempts,
                "input_token_count": len(input_tokens),
                "output_token_count": len(output_tokens),
                "eval_index": eval_index,
                "eval_k": eval_k,
                "checkpoint_model_version": checkpoint_model_version,
                "checkpoint_global_step": checkpoint_global_step,
                "selected_games": games,
                "updated_at": time.time(),
            },
        )
        _write_skill_manifest(
            manifest_path=eval_dir / "skill.json",
            generation_paths=sorted((eval_dir / "skills").glob("*/generation.json")),
            mode="eval",
            global_step=checkpoint_global_step,
            expected_skill_count=eval_k,
            metadata={
                "checkpoint_model_version": checkpoint_model_version,
                "checkpoint_global_step": checkpoint_global_step,
            },
        )
        try:
            eval_baseline_index = self._load_or_create_eval_no_skill_baseline(games)
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to load/create eval no-skill baseline")
            eval_baseline_index = {
                "status": "error",
                "mode": "eval_no_skill_baseline",
                "signature": "",
                "baseline_dir": "",
                "entries": [],
                "error": repr(exc),
            }

        episodes: list[dict[str, Any] | None] = [None] * len(games)
        current_episodes: dict[int, dict[str, Any]] = {}
        progress_lock = threading.Lock()
        progress_closed = threading.Event()
        last_partial_write_at = 0.0

        def write_progress_unlocked(
            *, rollout_index: int, current_episode: dict[str, Any] | None = None
        ) -> None:
            nonlocal last_partial_write_at
            now = time.time()
            if current_episode is not None:
                current_episodes[rollout_index] = current_episode
                _atomic_write_json(
                    sample_dir / f"current_episode_{rollout_index:03d}.json",
                    {
                        "mode": "eval",
                        "skill_name": skill_name,
                        "eval_index": eval_index,
                        "rollout_index": rollout_index,
                        "episode": current_episode,
                        "updated_at": now,
                    },
                )
                _append_jsonl(
                    sample_dir / "rollout_progress.jsonl",
                    _rollout_progress_event(
                        rollout_index=rollout_index,
                        episode=current_episode,
                        final=False,
                        extra={
                            "mode": "eval",
                            "skill_name": skill_name,
                            "eval_index": eval_index,
                            "checkpoint_global_step": checkpoint_global_step,
                        },
                    ),
                )
            else:
                complete_episode = episodes[rollout_index]
                if complete_episode is not None:
                    _atomic_write_json(
                        sample_dir / f"episode_{rollout_index:03d}.json",
                        {
                            "mode": "eval",
                            "skill_name": skill_name,
                            "eval_index": eval_index,
                            "checkpoint_model_version": checkpoint_model_version,
                            "checkpoint_global_step": checkpoint_global_step,
                            "rollout_index": rollout_index,
                            "episode": complete_episode,
                            "updated_at": now,
                        },
                    )
                    _unlink_if_exists(
                        sample_dir / f"current_episode_{rollout_index:03d}.json"
                    )
                    _append_jsonl(
                        sample_dir / "rollout_progress.jsonl",
                        _rollout_progress_event(
                            rollout_index=rollout_index,
                            episode=complete_episode,
                            final=True,
                            extra={
                                "mode": "eval",
                                "skill_name": skill_name,
                                "eval_index": eval_index,
                                "checkpoint_global_step": checkpoint_global_step,
                            },
                        ),
                    )

            should_write_summary = (
                current_episode is None
                or self.progress_summary_interval_s <= 0.0
                or now - last_partial_write_at >= self.progress_summary_interval_s
            )
            if not should_write_summary:
                return
            last_partial_write_at = now

            partial = [episode for episode in episodes if episode is not None]
            partial.extend(
                current_episodes[index]
                for index in sorted(current_episodes)
                if episodes[index] is None
            )
            partial_payload = {
                "mode": "eval",
                "skill_name": skill_name,
                "eval_index": eval_index,
                "eval_k": eval_k,
                "checkpoint_model_version": checkpoint_model_version,
                "checkpoint_global_step": checkpoint_global_step,
                "schema_valid": valid,
                "schema_error": parse_error,
                "attempt_count": len(generation_attempts),
                "max_retries": max_retries,
                "overall": _episode_metrics(partial),
                "n_rollouts_expected": len(games),
                "episode_snapshot_files": [
                    f"current_episode_{index:03d}.json"
                    for index in sorted(current_episodes)
                    if episodes[index] is None
                ],
                "completed_episode_files": [
                    f"episode_{index:03d}.json"
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
                    "mode": "eval",
                    "eval_index": eval_index,
                    "checkpoint_model_version": checkpoint_model_version,
                    "checkpoint_global_step": checkpoint_global_step,
                    "schema_valid": valid,
                    "overall": partial_payload["overall"],
                    "updated_at": now,
                },
            )

        if not valid or parsed is None:
            episodes = [
                _invalid_skill_episode(
                    game=game,
                    skill_name=skill_name,
                    error=parse_error,
                )
                for game in games
            ]
        else:
            skill_text = skill_payload_to_prompt_text(parsed)
            progress_metadata = {
                "mode": "eval",
                "skill_name": skill_name,
                "eval_index": eval_index,
                "checkpoint_model_version": checkpoint_model_version,
                "checkpoint_global_step": checkpoint_global_step,
            }
            write_progress_unlocked(rollout_index=0)

            executor = _get_episode_rollout_executor(self.episode_rollout_workers)
            futures = {
                executor.submit(
                    _run_rollout_episode_process,
                    {
                        "repo_root": str(self.repo_root),
                        "sample_dir": str(sample_dir),
                        "game": game,
                        "skill_name": skill_name,
                        "skill_text": skill_text,
                        "rollout_index": rollout_index,
                        "index_width": 3,
                        "max_rollout_steps": self.max_rollout_steps,
                        "memory_window": self.memory_window,
                        "max_commands": self.max_commands,
                        "actor_base_url": self.actor_base_url,
                        "actor_model": self.actor_model,
                        "actor_api_key": self.actor_api_key,
                        "actor_timeout_s": self.actor_timeout_s,
                        "actor_temperature": self.actor_temperature,
                        "tokenizer_path": self.tokenizer_path,
                        "seed_base": (
                            900000
                            + max(0, checkpoint_global_step) * 100000
                            + eval_index * 1000
                            + rollout_index * 100
                        ),
                        "progress_metadata": progress_metadata,
                    },
                ): rollout_index
                for rollout_index, game in enumerate(games)
            }
            self._collect_rollout_futures(
                futures=futures,
                games=games,
                skill_name=skill_name,
                episodes=episodes,
                current_episodes=current_episodes,
                progress_lock=progress_lock,
                progress_closed=progress_closed,
                write_progress_unlocked=write_progress_unlocked,
                context=sample_key,
            )

        progress_closed.set()
        complete_episodes = [episode for episode in episodes if episode is not None]
        baseline_delta = _baseline_delta_metrics(
            complete_episodes,
            eval_baseline_index,
        )
        no_skill_baseline_summary = {
            key: value for key, value in eval_baseline_index.items() if key != "entries"
        }
        per_task_type = _per_task_type_metrics(complete_episodes)
        matched_task_metrics = dict(per_task_type.get(input_task_type) or {})
        payload = {
            "mode": "eval",
            "skill_name": skill_name,
            "input_task_type": input_task_type,
            "eval_index": eval_index,
            "eval_k": eval_k,
            "checkpoint_model_version": checkpoint_model_version,
            "checkpoint_global_step": checkpoint_global_step,
            "schema_valid": valid,
            "schema_error": parse_error,
            "attempt_count": len(generation_attempts),
            "max_retries": max_retries,
            "overall": _episode_metrics(complete_episodes),
            "per_task_type": per_task_type,
            "matched_task_type": input_task_type,
            "matched_task_type_metrics": matched_task_metrics,
            "matched_task_type_sr": float(matched_task_metrics.get("sr", 0.0) or 0.0),
            "no_skill_baseline": no_skill_baseline_summary,
            "baseline_delta": baseline_delta,
            "episodes": complete_episodes,
            "updated_at": time.time(),
        }
        _atomic_write_json(sample_dir / "rollouts.json", payload)
        metrics_payload = {
            key: value for key, value in payload.items() if key != "episodes"
        }
        _atomic_write_json(sample_dir / "metrics.json", metrics_payload)
        _atomic_write_json(
            sample_dir / "checkpoint.json",
            {
                "status": "complete",
                "mode": "eval",
                "eval_index": eval_index,
                "checkpoint_model_version": checkpoint_model_version,
                "checkpoint_global_step": checkpoint_global_step,
                "schema_valid": valid,
                "overall": payload["overall"],
                "baseline_delta": baseline_delta,
                "updated_at": time.time(),
            },
        )
        _remove_completed_episode_snapshots(sample_dir)
        summary = _write_eval_summary(
            eval_dir=eval_dir,
            artifact_dir=self.artifact_dir,
            expected_skill_count=eval_k,
            checkpoint_model_root=self.checkpoint_model_root,
            task_types=self.task_types,
        )
        metrics_payload["_eval_summary_status"] = summary["status"]
        metrics_payload["_eval_summary"] = summary
        metrics_payload["_eval_dir"] = str(eval_dir)
        _atomic_write_json(
            eval_dir / "eval_state.json",
            {
                "status": summary["status"],
                "checkpoint_model_version": checkpoint_model_version,
                "checkpoint_global_step": checkpoint_global_step,
                "expected_skill_count": eval_k,
                "completed_skill_count": summary["completed_skill_count"],
                "schema_valid_count": summary["schema_valid_count"],
                "schema_valid_rate": summary["schema_valid_rate"],
                "all_mean_sr": summary["all_mean_sr"],
                "matched_task_type_srs": summary.get("matched_task_type_srs", {}),
                "all_mean_baseline_delta_sr": summary["all_mean_baseline_delta_sr"],
                "all_mean_0_to_1": summary["all_mean_0_to_1"],
                "all_mean_1_to_0": summary["all_mean_1_to_0"],
                "updated_at": time.time(),
            },
        )
        return metrics_payload
