# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import requests
import yaml

AREAL_ROOT = Path(__file__).resolve().parents[2]
if str(AREAL_ROOT) not in sys.path:
    sys.path.insert(0, str(AREAL_ROOT))

from examples.alfworld_skill.sglang_server import (  # noqa: E402
    ServerSpec,
    managed_sglang_server,
)

from areal.utils import logging  # noqa: E402
from areal.workflow.alfworld_runtime import ALFWORLD_TASK_TYPES  # noqa: E402
from areal.workflow.alfworld_skill import (  # noqa: E402
    PROMPT_CATEGORY_BY_TASK_TYPE,
    _parse_skill_generation,
    _render_trajectory_bundle,
    _sample_trajectory_pool_episodes,
    _skill_generation_messages,
)

logger = logging.getLogger("ExternalSkillGenerator")

_SKILL_ID_PATTERN = re.compile(r"skill_(\d{6})")
_SUPPORTED_BACKENDS = {"remote_api", "existing_server", "local_sglang"}
_SUPPORTED_TASK_TYPE_MODES = {"all", "custom"}


@dataclass(frozen=True)
class LocalServerConfig:
    model_path: str
    host: str
    port: int
    gpus: str
    tp_size: int
    dp_size: int
    dtype: str
    mem_fraction_static: float
    context_length: int
    startup_timeout_seconds: float
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class ExternalSkillGenerationConfig:
    backend: str
    base_url: str
    api_key: str
    model: str
    request_model: str
    local_server: LocalServerConfig | None
    timeout_seconds: float
    transport_retries: int
    concurrency: int
    extra_body: dict[str, Any]
    trajectory_pool_indices: tuple[Path, ...]
    task_type_mode: str
    task_types: tuple[str, ...]
    num_skills: int
    seed: int
    trajectories_per_prompt: int
    temperature: float
    top_p: float
    max_output_tokens: int
    reasoning_effort: str | None
    max_attempts_per_skill: int
    prompt_trace_max_steps: int
    prompt_observation_char_limit: int
    prompt_result_char_limit: int
    max_skill_words: int
    truncate_overlong_skill: bool
    output_path: Path


@dataclass(frozen=True)
class PromptJob:
    skill_id: str
    input_task_type: str
    expected_category: str
    messages: list[dict[str, str]]


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {key!r} must be a mapping")
    return value


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"configuration field {key!r} must be non-empty")
    return value


def _resolve_path(value: Any, *, config_dir: Path, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"configuration field {field!r} must be non-empty")
    path = Path(text).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def _normalize_openai_base_url(value: str) -> str:
    base_url = str(value or "").strip().rstrip("/")
    if not base_url:
        return ""
    return base_url if base_url.endswith("/v1") else base_url + "/v1"


def _model_folder_name(value: str) -> str:
    candidate = Path(str(value).rstrip("/")).name or str(value)
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate.strip())
    normalized = normalized.strip("_.-")
    if not normalized:
        raise ValueError(f"cannot derive output model folder from {value!r}")
    return normalized


def _resolve_model_path(value: Any, *, config_dir: Path) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("api.model_path must be non-empty for local_sglang")
    expanded = Path(text).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())
    relative = (config_dir / expanded).resolve()
    if relative.exists() or text.startswith("."):
        return str(relative)
    return text


def _api_key(api: dict[str, Any], *, backend: str) -> str:
    env_name = str(api.get("api_key_env") or "").strip()
    configured = str(api.get("api_key") or "").strip()
    value = str(os.getenv(env_name, "")).strip() if env_name else configured
    if env_name and not value and backend == "remote_api":
        raise ValueError(f"environment variable {env_name!r} is not set")
    if not value and backend == "remote_api":
        raise ValueError(
            "remote_api requires api.api_key_env or a non-empty api.api_key"
        )
    return value


def _resolve_task_types(sampling: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    raw_mode = sampling.get("task_type_mode")
    raw_task_types = sampling.get("task_types")
    if raw_mode is None:
        task_type_mode = "custom" if raw_task_types is not None else "all"
    else:
        task_type_mode = str(raw_mode).strip().lower()
    if task_type_mode not in _SUPPORTED_TASK_TYPE_MODES:
        raise ValueError(
            "sampling.task_type_mode must be one of "
            f"{sorted(_SUPPORTED_TASK_TYPE_MODES)}, got {task_type_mode!r}"
        )

    if task_type_mode == "all":
        task_types = ALFWORLD_TASK_TYPES
    else:
        if not isinstance(raw_task_types, list) or not raw_task_types:
            raise ValueError(
                "sampling.task_types must be a non-empty list when "
                "sampling.task_type_mode=custom"
            )
        task_types = tuple(str(item).strip() for item in raw_task_types)

    if task_type_mode != "custom" and raw_task_types is not None:
        raise ValueError(
            "sampling.task_types cannot be combined with "
            f"sampling.task_type_mode={task_type_mode}; use custom mode instead"
        )
    if any(not task_type for task_type in task_types):
        raise ValueError("sampling.task_types cannot contain empty values")
    if len(set(task_types)) != len(task_types):
        raise ValueError("sampling.task_types cannot contain duplicates")
    unsupported = sorted(set(task_types) - set(PROMPT_CATEGORY_BY_TASK_TYPE))
    if unsupported:
        raise ValueError("unsupported task types: " + ", ".join(unsupported))
    return task_type_mode, task_types


def load_config(path: Path) -> ExternalSkillGenerationConfig:
    config_path = path.expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")

    api = _mapping(payload, "api")
    sampling = _mapping(payload, "sampling")
    generation = _mapping(payload, "generation")
    prompt = _mapping(payload, "prompt")
    output = _mapping(payload, "output")

    backend = str(api.get("backend") or "remote_api").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"api.backend must be one of {sorted(_SUPPORTED_BACKENDS)}, got {backend!r}"
        )
    model = _required_text(api, "model")
    local_server: LocalServerConfig | None = None
    if backend == "local_sglang":
        server = payload.get("local_server") or {}
        if not isinstance(server, dict):
            raise ValueError("configuration section 'local_server' must be a mapping")
        model_path = _resolve_model_path(
            api.get("model_path") or model,
            config_dir=config_path.parent,
        )
        raw_extra_args = server.get("extra_args") or []
        if not isinstance(raw_extra_args, list):
            raise ValueError("local_server.extra_args must be a list")
        local_server = LocalServerConfig(
            model_path=model_path,
            host=str(server.get("host") or "127.0.0.1").strip(),
            port=int(server.get("port", 34180)),
            gpus=str(server.get("gpus") or "0").strip(),
            tp_size=int(server.get("tp_size", 1)),
            dp_size=int(server.get("dp_size", 1)),
            dtype=str(server.get("dtype") or "bfloat16").strip(),
            mem_fraction_static=float(server.get("mem_fraction_static", 0.86)),
            context_length=int(server.get("context_length", 16384)),
            startup_timeout_seconds=float(
                server.get("startup_timeout_seconds", 1800.0)
            ),
            extra_args=tuple(str(item) for item in raw_extra_args),
        )
        base_url = f"http://{local_server.host}:{local_server.port}/v1"
        request_model = str(api.get("request_model") or model_path).strip()
    else:
        base_url = _normalize_openai_base_url(_required_text(api, "base_url"))
        request_model = str(api.get("request_model") or model).strip()
    if not request_model:
        raise ValueError("api.request_model must be non-empty")

    raw_indices = sampling.get("trajectory_pool_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("sampling.trajectory_pool_indices must be a non-empty list")
    trajectory_pool_indices = tuple(
        _resolve_path(
            item,
            config_dir=config_path.parent,
            field="sampling.trajectory_pool_indices",
        )
        for item in raw_indices
    )
    missing_indices = [
        str(item) for item in trajectory_pool_indices if not item.is_file()
    ]
    if missing_indices:
        raise FileNotFoundError(
            "trajectory-pool index files do not exist: " + ", ".join(missing_indices)
        )

    task_type_mode, task_types = _resolve_task_types(sampling)

    extra_body = api.get("extra_body") or {}
    if not isinstance(extra_body, dict):
        raise ValueError("api.extra_body must be a mapping")

    if str(output.get("path") or "").strip():
        output_path = _resolve_path(
            output.get("path"),
            config_dir=config_path.parent,
            field="output.path",
        )
    else:
        output_root = _resolve_path(
            output.get("root"),
            config_dir=config_path.parent,
            field="output.root",
        )
        output_model_name = str(output.get("model_name") or model).strip()
        output_path = output_root / _model_folder_name(output_model_name) / "all.jsonl"

    config = ExternalSkillGenerationConfig(
        backend=backend,
        base_url=base_url,
        api_key=_api_key(api, backend=backend),
        model=model,
        request_model=request_model,
        local_server=local_server,
        timeout_seconds=float(api.get("timeout_seconds", 180.0)),
        transport_retries=int(api.get("transport_retries", 2)),
        concurrency=int(api.get("concurrency", 8)),
        extra_body=dict(extra_body),
        trajectory_pool_indices=trajectory_pool_indices,
        task_type_mode=task_type_mode,
        task_types=task_types,
        num_skills=int(sampling.get("num_skills", 50)),
        seed=int(sampling.get("seed", 42)),
        trajectories_per_prompt=int(sampling.get("trajectories_per_prompt", 4)),
        temperature=float(generation.get("temperature", 1.0)),
        top_p=float(generation.get("top_p", 1.0)),
        max_output_tokens=int(generation.get("max_output_tokens", 256)),
        reasoning_effort=(
            str(generation.get("reasoning_effort") or "").strip() or None
        ),
        max_attempts_per_skill=int(generation.get("max_attempts_per_skill", 3)),
        prompt_trace_max_steps=int(prompt.get("trace_max_steps", 24)),
        prompt_observation_char_limit=int(prompt.get("observation_char_limit", 120)),
        prompt_result_char_limit=int(prompt.get("result_char_limit", 120)),
        max_skill_words=int(prompt.get("max_skill_words", 100)),
        truncate_overlong_skill=bool(prompt.get("truncate_overlong_skill", False)),
        output_path=output_path,
    )
    _validate_config(config)
    return config


def _validate_config(config: ExternalSkillGenerationConfig) -> None:
    positive_values = {
        "api.timeout_seconds": config.timeout_seconds,
        "api.concurrency": config.concurrency,
        "sampling.num_skills": config.num_skills,
        "sampling.trajectories_per_prompt": config.trajectories_per_prompt,
        "generation.max_output_tokens": config.max_output_tokens,
        "generation.max_attempts_per_skill": config.max_attempts_per_skill,
        "prompt.trace_max_steps": config.prompt_trace_max_steps,
        "prompt.observation_char_limit": config.prompt_observation_char_limit,
        "prompt.result_char_limit": config.prompt_result_char_limit,
        "prompt.max_skill_words": config.max_skill_words,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError("configuration fields must be positive: " + ", ".join(invalid))
    if config.transport_retries < 0:
        raise ValueError("api.transport_retries must be non-negative")
    if not 0.0 <= config.temperature:
        raise ValueError("generation.temperature must be non-negative")
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError("generation.top_p must be in (0, 1]")
    if config.local_server is not None:
        server_values = {
            "local_server.port": config.local_server.port,
            "local_server.tp_size": config.local_server.tp_size,
            "local_server.dp_size": config.local_server.dp_size,
            "local_server.mem_fraction_static": (
                config.local_server.mem_fraction_static
            ),
            "local_server.context_length": config.local_server.context_length,
            "local_server.startup_timeout_seconds": (
                config.local_server.startup_timeout_seconds
            ),
        }
        invalid_server = [
            name for name, value in server_values.items() if float(value) <= 0
        ]
        if invalid_server:
            raise ValueError(
                "local server configuration must be positive: "
                + ", ".join(invalid_server)
            )


def _load_trajectory_refs(
    index_paths: tuple[Path, ...],
) -> dict[str, list[dict[str, Any]]]:
    refs_by_task: dict[str, list[dict[str, Any]]] = {}
    for index_path in index_paths:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"trajectory-pool index is not complete: {index_path}")
        refs = payload.get("refs")
        if not isinstance(refs, list):
            raise ValueError(f"trajectory-pool index has no refs list: {index_path}")
        for raw_ref in refs:
            if not isinstance(raw_ref, dict):
                continue
            resolved_ref = dict(raw_ref)
            task_type = str(resolved_ref.get("task_type") or "").strip()
            for field in ("source_rollouts_path", "source_episode_path"):
                raw_path = str(resolved_ref.get(field) or "").strip()
                if not raw_path:
                    continue
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    resolved_ref[field] = str((index_path.parent / path).resolve())
                    continue
                if path.is_file():
                    continue
                relocated_candidates = (
                    index_path.parent / "samples" / path.name,
                    index_path.parent / "episodes" / task_type / path.name,
                )
                relocated = next(
                    (
                        candidate
                        for candidate in relocated_candidates
                        if candidate.is_file()
                    ),
                    None,
                )
                if relocated is not None:
                    resolved_ref[field] = str(relocated.resolve())
            if task_type:
                refs_by_task.setdefault(task_type, []).append(resolved_ref)
    return refs_by_task


def build_prompt_jobs(config: ExternalSkillGenerationConfig) -> list[PromptJob]:
    refs_by_task = _load_trajectory_refs(config.trajectory_pool_indices)
    for task_type in config.task_types:
        available = len(refs_by_task.get(task_type, []))
        if available < config.trajectories_per_prompt:
            raise ValueError(
                "trajectory pool has too few refs for "
                f"{task_type}: required={config.trajectories_per_prompt}, "
                f"available={available}"
            )

    jobs: list[PromptJob] = []
    for index in range(config.num_skills):
        sample_seed = config.seed * 100_000 + index * 100
        task_type = random.Random(sample_seed).choice(config.task_types)
        episodes, _ = _sample_trajectory_pool_episodes(
            refs_by_task[task_type],
            seed=sample_seed,
            count=config.trajectories_per_prompt,
        )
        if len(episodes) != config.trajectories_per_prompt:
            raise RuntimeError(
                f"failed to load {config.trajectories_per_prompt} trajectories for "
                f"skill_{index:06d}"
            )
        prompt_category = PROMPT_CATEGORY_BY_TASK_TYPE[task_type]
        trajectories = _render_trajectory_bundle(
            episodes,
            max_steps=config.prompt_trace_max_steps,
            obs_limit=config.prompt_observation_char_limit,
            result_limit=config.prompt_result_char_limit,
        )
        messages = _skill_generation_messages(
            prompt_category=prompt_category,
            trajectories=trajectories,
            skill_output_format="skill_xml",
        )
        jobs.append(
            PromptJob(
                skill_id=f"skill_{index:06d}",
                input_task_type=task_type,
                expected_category=prompt_category,
                messages=messages,
            )
        )
    return jobs


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("API response contains no choices")
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        if parts:
            return "".join(parts)
    raise ValueError("API response message contains no text content")


def _completion_diagnostics(response: Any, raw: str) -> str:
    choices = getattr(response, "choices", None) or []
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
    return (
        f"finish_reason={finish_reason or 'unknown'} content_chars={len(raw)} "
        f"completion_tokens={completion_tokens if completion_tokens is not None else 'unknown'} "
        f"reasoning_tokens={reasoning_tokens if reasoning_tokens is not None else 'unknown'}"
    )


def _validation_retry_messages(
    job: PromptJob,
    *,
    raw: str,
    error: str,
    max_skill_words: int,
) -> list[dict[str, str]]:
    """Ask the model to repair one invalid response without changing the base prompt."""
    return [
        *[dict(message) for message in job.messages],
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                f"The previous response failed validation: {error}. "
                "Rewrite the same guidance so it satisfies the original output "
                "contract. Return exactly one <skill>...</skill> block with no "
                "text outside it, and do not use nested tags. "
                f"The skill body must contain at most {max_skill_words} "
                "whitespace-separated words."
            ),
        },
    ]


def _truncate_skill_body(content: str, *, max_words: int) -> str:
    """Return a coherent text prefix containing at most ``max_words`` words."""
    matches = list(re.finditer(r"\S+", content))
    if max_words <= 0 or len(matches) <= max_words:
        return content.strip()
    prefix = content[: matches[max_words - 1].end()].strip()
    sentence_ends = list(re.finditer(r"[.!?](?:[\"')\]]*)", prefix))
    if sentence_ends:
        return prefix[: sentence_ends[-1].end()].strip()
    return prefix


async def generate_job(
    client: Any,
    config: ExternalSkillGenerationConfig,
    job: PromptJob,
) -> dict[str, str]:
    last_error = "generation did not run"
    messages = [dict(message) for message in job.messages]
    for attempt in range(1, config.max_attempts_per_skill + 1):
        diagnostics = "response_unavailable"
        try:
            request: dict[str, Any] = {
                "model": config.request_model,
                "messages": messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_completion_tokens": config.max_output_tokens,
                "n": 1,
            }
            if config.reasoning_effort:
                request["reasoning_effort"] = config.reasoning_effort
            if config.extra_body:
                request["extra_body"] = config.extra_body
            response = await client.chat.completions.create(
                **request,
            )
            raw = _response_text(response)
            diagnostics = _completion_diagnostics(response, raw)
            parsed, valid, error, _ = _parse_skill_generation(
                raw,
                job.expected_category,
                skill_output_format="skill_xml",
                max_words=config.max_skill_words,
            )
            overlong_error = (
                f"description must contain at most {config.max_skill_words} words"
            )
            if (
                not valid
                and config.truncate_overlong_skill
                and error == overlong_error
                and isinstance(parsed, dict)
            ):
                original_content = str(parsed.get("description") or "").strip()
                content = _truncate_skill_body(
                    original_content,
                    max_words=config.max_skill_words,
                )
                reparsed, revalid, _, _ = _parse_skill_generation(
                    f"<skill>{content}</skill>",
                    job.expected_category,
                    skill_output_format="skill_xml",
                    max_words=config.max_skill_words,
                )
                if revalid and isinstance(reparsed, dict):
                    logger.info(
                        "truncated overlong skill id=%s original_words=%s max_words=%s",
                        job.skill_id,
                        len(re.findall(r"\S+", original_content)),
                        config.max_skill_words,
                    )
                    return {
                        "id": job.skill_id,
                        "model": config.model,
                        "content": str(reparsed["description"]).strip(),
                    }
            if valid and isinstance(parsed, dict):
                content = str(parsed.get("description") or "").strip()
                if content:
                    return {
                        "id": job.skill_id,
                        "model": config.model,
                        "content": content,
                    }
            last_error = error or "parsed skill body is empty"
            messages = _validation_retry_messages(
                job,
                raw=raw,
                error=last_error,
                max_skill_words=config.max_skill_words,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "skill generation attempt failed id=%s attempt=%s/%s error=%s %s",
            job.skill_id,
            attempt,
            config.max_attempts_per_skill,
            last_error,
            diagnostics,
        )
    raise RuntimeError(
        f"failed to generate a valid skill for {job.skill_id}: {last_error}"
    )


def _skill_index(record: dict[str, str]) -> int:
    match = _SKILL_ID_PATTERN.fullmatch(str(record.get("id") or ""))
    if match is None:
        raise ValueError(f"invalid skill id in output: {record.get('id')!r}")
    return int(match.group(1))


def load_existing_records(path: Path, *, model: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(decoded, dict):
            decoded = decoded.get("skills")
        if not isinstance(decoded, list):
            raise ValueError("JSON output must be a list or an object with skills")
        raw_records = list(enumerate(decoded, start=1))
    else:
        raw_records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.strip():
                raw_records.append((line_number, json.loads(line)))

    records: dict[str, dict[str, str]] = {}
    for record_number, payload in raw_records:
        if not isinstance(payload, dict):
            raise ValueError(f"output record {record_number} must be an object")
        if set(payload) != {"id", "model", "content"}:
            raise ValueError(
                f"output record {record_number} must contain only id, model, content"
            )
        skill_id = str(payload["id"])
        _skill_index(payload)
        if str(payload["model"]) != model:
            raise ValueError(
                f"output record {record_number} belongs to model "
                f"{payload['model']!r}, "
                f"expected {model!r}"
            )
        if not str(payload["content"]).strip():
            raise ValueError(f"output record {record_number} has empty content")
        if skill_id in records:
            raise ValueError(f"duplicate skill id in output: {skill_id}")
        records[skill_id] = {
            "id": skill_id,
            "model": str(payload["model"]),
            "content": str(payload["content"]),
        }
    return records


def write_records(path: Path, records: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=_skill_index)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        if path.suffix.lower() == ".json":
            json.dump(ordered, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        else:
            for record in ordered:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _async_openai_class() -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "the external skill generator requires the AReaL openai dependency"
        ) from exc
    return AsyncOpenAI


def _verify_local_server_model(
    *, base_url: str, api_key: str, expected_model_ids: set[str]
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(
        base_url.rstrip("/") + "/models",
        headers=headers,
        timeout=10.0,
    )
    response.raise_for_status()
    payload = response.json()
    served_ids = {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }
    expected = {item for item in expected_model_ids if item}
    for item in list(expected):
        path = Path(item).expanduser()
        if path.exists():
            expected.add(str(path.resolve()))
    if not served_ids or served_ids.isdisjoint(expected):
        raise RuntimeError(
            f"local server model IDs {sorted(served_ids)} do not match expected "
            f"{sorted(expected)}"
        )


@contextmanager
def _generation_endpoint(config: ExternalSkillGenerationConfig):
    if config.backend == "local_sglang":
        if config.local_server is None:
            raise RuntimeError("local_sglang backend is missing local server config")
        server = config.local_server
        spec = ServerSpec(
            model_path=server.model_path,
            host=server.host,
            port=server.port,
            gpus=server.gpus,
            tp_size=server.tp_size,
            dp_size=server.dp_size,
            dtype=server.dtype,
            mem_fraction_static=server.mem_fraction_static,
            context_length=server.context_length,
            startup_timeout_s=server.startup_timeout_seconds,
            api_key=config.api_key,
            extra_args=server.extra_args,
            enabled=True,
        )
        with managed_sglang_server(spec) as server_root:
            base_url = _normalize_openai_base_url(server_root)
            _verify_local_server_model(
                base_url=base_url,
                api_key=config.api_key,
                expected_model_ids={config.request_model, server.model_path},
            )
            yield base_url
        return

    if config.backend == "existing_server":
        _verify_local_server_model(
            base_url=config.base_url,
            api_key=config.api_key,
            expected_model_ids={config.request_model},
        )
    yield config.base_url


async def run(config: ExternalSkillGenerationConfig) -> None:
    AsyncOpenAI = _async_openai_class()

    jobs = build_prompt_jobs(config)
    records = load_existing_records(config.output_path, model=config.model)
    expected_ids = {job.skill_id for job in jobs}
    unexpected = sorted(set(records) - expected_ids)
    if unexpected:
        raise ValueError(
            "output contains ids outside the configured num_skills range: "
            + ", ".join(unexpected)
        )
    pending = [job for job in jobs if job.skill_id not in records]
    if not pending:
        logger.info(
            "External skillbank already complete path=%s count=%s",
            config.output_path,
            len(records),
        )
        return

    semaphore = asyncio.Semaphore(config.concurrency)
    write_lock = asyncio.Lock()

    with _generation_endpoint(config) as base_url:
        async with AsyncOpenAI(
            base_url=base_url,
            api_key=config.api_key or "EMPTY",
            timeout=config.timeout_seconds,
            max_retries=config.transport_retries,
        ) as client:

            async def generate_and_persist(job: PromptJob) -> None:
                async with semaphore:
                    record = await generate_job(client, config, job)
                async with write_lock:
                    records[job.skill_id] = record
                    write_records(config.output_path, records)
                    logger.info(
                        "Generated external skill id=%s progress=%s/%s",
                        job.skill_id,
                        len(records),
                        config.num_skills,
                    )

            results = await asyncio.gather(
                *(generate_and_persist(job) for job in pending),
                return_exceptions=True,
            )

    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise RuntimeError(
            f"external skill generation failed for {len(errors)} slots; "
            f"completed={len(records)}/{config.num_skills}; first_error={errors[0]}"
        )
    if len(records) != config.num_skills:
        raise RuntimeError(
            f"external skillbank is incomplete: {len(records)}/{config.num_skills}"
        )
    logger.info(
        "External skillbank complete path=%s count=%s model=%s",
        config.output_path,
        len(records),
        config.model,
    )


async def probe_one(config: ExternalSkillGenerationConfig) -> None:
    """Generate one skill with one API attempt without touching the output file."""
    AsyncOpenAI = _async_openai_class()
    job = build_prompt_jobs(config)[0]
    probe_config = replace(config, max_attempts_per_skill=1)
    with _generation_endpoint(probe_config) as base_url:
        async with AsyncOpenAI(
            base_url=base_url,
            api_key=probe_config.api_key or "EMPTY",
            timeout=probe_config.timeout_seconds,
            max_retries=probe_config.transport_retries,
        ) as client:
            record = await generate_job(client, probe_config, job)
    print(json.dumps(record, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a small external-model ALFWorld skillbank from the same "
            "four-trajectory prompt sampling used by training."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--probe-one",
        action="store_true",
        help="send one API request for skill_000000 without writing the output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    asyncio.run(probe_one(config) if args.probe_one else run(config))


if __name__ == "__main__":
    main()
