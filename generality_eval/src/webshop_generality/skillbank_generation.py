"""Build a WebShop knowledge bank through the training prompt/input protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import requests
import yaml
from omegaconf import OmegaConf

from examples.webshop_skill.configs import WebShopSkillGRPOConfig
from examples.webshop_skill.core import (
    atomic_write_json,
    is_usable_webshop_source_episode,
    select_trajectory_bundles,
)
from examples.webshop_skill.sglang_server import ManagedSGLangServer, server_spec
from examples.webshop_skill.workflow import WebShopSkillGRPOWorkflow

from areal.api import ModelRequest
from areal.api.cli_args import parse_cli_args, to_structured_cfg
from areal.engine.sglang_remote import SGLangBackend
from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer

from webshop_generality.run_semantic_skillbank_eval import (
    check_gpus,
    digest,
    endpoint_root,
    output_lock,
    read_json,
    validate_episode,
    verify_actor,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "generality_eval/config/webshop_skillbank_generation.yaml"
VERSION = "webshop_training_skillbank_v1"


def path_from_root(value):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in value:
        raise ValueError("Set all environment variables used by the requested path")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_settings(path, overrides=()):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("config must be a YAML mapping")
    config_path = path_from_root(settings["training_config"])
    config, _ = parse_cli_args(
        [
            "--config",
            str(config_path),
            *settings.get("training_overrides", []),
            *overrides,
        ]
    )
    # Same Hydra composition and dataclass defaults as train.main, without creating
    # name-resolution services, training logs, GPU workers, or optimizer state.
    config = to_structured_cfg(config, WebShopSkillGRPOConfig)
    return settings, config


def workflow_adapter(config, tokenizer=None):
    """Use side-effect-free training methods without starting a training workflow."""
    workflow = WebShopSkillGRPOWorkflow.__new__(WebShopSkillGRPOWorkflow)
    for key, value in OmegaConf.to_container(config.webshop, resolve=True).items():
        setattr(workflow, key, value)
    workflow.train_batch_size = int(config.train_dataset.batch_size)
    if tokenizer is not None:
        workflow.tokenizer = tokenizer
        workflow.gconfig = OmegaConf.to_object(
            config.gconfig
        ).new_with_stop_and_pad_token_ids(tokenizer)
    return workflow


def pool_protocol(config):
    """Only task/actor semantics; endpoints and worker counts may change on resume."""
    keys = (
        "human_goals",
        "num_products",
        "observation_mode",
        "train_task_count",
        "eval_task_count",
        "task_seed",
        "unique_asin_split",
        "max_rollout_steps",
        "memory_window",
        "observation_char_limit",
        "max_clickables",
        "invalid_action_retries",
        "success_threshold",
        "actor_temperature",
        "actor_max_tokens",
    )
    return {
        key: OmegaConf.to_container(config.webshop, resolve=True)[key] for key in keys
    }


def read_pool(pool_dir, config):
    pool_dir = Path(pool_dir)
    pool = read_json(pool_dir / "pool.json")
    if not isinstance(pool, dict) or pool.get("status") != "complete":
        raise ValueError(
            "Prepare a complete real trace pool with the collect command first"
        )
    if pool.get("version") != VERSION or pool.get("protocol") != pool_protocol(config):
        raise ValueError(
            "Trace-pool protocol differs from this training config; collect a new pool"
        )
    if pool.get("actor_model") != "Qwen/Qwen3.5-4B":
        raise ValueError("The source actor must be Qwen/Qwen3.5-4B")
    tasks = pool["tasks"]
    indices = [task["task_index"] for task in tasks]
    if indices != pool["task_manifest"]["train_indices"]:
        raise ValueError("Pool does not cover the full training task split")
    if len(set(indices)) != len(indices) or set(indices) & set(
        pool["task_manifest"]["eval_indices"]
    ):
        raise ValueError("Pool has duplicate tasks or evaluation leakage")
    episodes = []
    for task in tasks:
        episode = read_json(
            pool_dir / "episodes" / f"task_{task['task_index']:06d}.json"
        )
        validate_episode(
            episode, task, success_threshold=float(config.webshop.success_threshold)
        )
        if (
            episode.get("pool_signature") != pool["pool_signature"]
            or episode.get("actor_model") != pool["actor_model"]
        ):
            raise ValueError("Episode belongs to a different trace-pool run")
        episodes.append(episode)
    if not any(is_usable_webshop_source_episode(ep) for ep in episodes):
        raise ValueError("Trace pool contains no usable executed actions")
    return pool, episodes


def build_jobs(config, tokenizer, episodes, *, count, seed):
    if count < 1:
        raise ValueError("skill_count must be positive")
    workflow = workflow_adapter(config, tokenizer)
    if workflow.groups_per_round < 1 or workflow.source_trajectories_per_prompt < 1:
        raise ValueError("Training groups and source trajectory count must be positive")
    first_group = (
        max(workflow.groups_per_round, workflow.train_batch_size)
        if workflow.parallelize_same_step_rounds
        else workflow.groups_per_round
    )
    jobs = []
    for index in range(count):
        # One candidate (sample_index=0) per logical group, not eight duplicate
        # inputs. Skip the initial zero-shot optimizer step (or round in serial mode).
        round_index, group_index = divmod(
            first_group + index, workflow.groups_per_round
        )
        source_seed = seed * 100_000 + round_index * 1_000 + group_index * 100
        selected = select_trajectory_bundles(
            episodes,
            num_skills=1,
            trajectories_per_skill=workflow.source_trajectories_per_prompt,
            seed=source_seed,
            success_threshold=workflow.success_threshold,
        )[0]
        if not selected:
            raise ValueError("Training selector found no usable source traces")
        messages, metadata = workflow._fit_skill_messages(selected)
        input_ids = apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=workflow.skill_generation_enable_thinking,
        )
        jobs.append(
            {
                "id": f"skill_{index:06d}",
                "round_in_category": round_index,
                "group_index": group_index,
                "sample_index": 0,
                "source_seed": source_seed,
                "source_task_indices": [ep["task_index"] for ep in selected],
                "source_successes": [bool(ep["success"]) for ep in selected],
                "messages": messages,
                "prompt_render": metadata,
                "input_ids": input_ids,
            }
        )
    return workflow, jobs


def api_key(section):
    # Credentials are deliberately never part of saved settings/contracts.
    return os.environ.get(str(section.get("api_key_env") or "OPENAI_API_KEY"), "")


def model_location(section):
    value = os.path.expandvars(str(section.get("model_path") or section["model"]))
    if "$" in value:
        raise ValueError("Set the environment variable used by model_path")
    if value.startswith(("/", ".", "~")) or (ROOT / value).exists():
        return str(path_from_root(value))
    return value


def request_model(section):
    return str(section.get("request_model") or model_location(section))


@contextmanager
def model_endpoint(section, output_dir, *, role):
    backend = section["backend"]
    key = api_key(section)
    if backend == "remote_api":
        url = os.path.expandvars(str(section["base_url"]))
        if "$" in url or not url.startswith(("http://", "https://")):
            raise ValueError(
                "Configure a valid generator.base_url or its environment variable"
            )
        yield endpoint_root(url), key
        return
    if backend not in {"existing_server", "local_sglang"}:
        raise ValueError("backend must be remote_api, existing_server, or local_sglang")
    model_path = model_location(section)
    if backend == "local_sglang" and Path(model_path).is_absolute():
        directory = Path(model_path)
        if not (directory / "config.json").is_file() or not (
            list(directory.glob("*.safetensors"))
            or list(directory.glob("pytorch_model*.bin"))
        ):
            raise ValueError(
                "model directory must contain config.json and model weights, not optimizer/FSDP shards"
            )
    section = {**section, "api_key": key}
    section["base_url"] = endpoint_root(
        os.path.expandvars(str(section.get("base_url") or ""))
    )
    if backend == "existing_server" and not section["base_url"]:
        raise ValueError("existing_server requires base_url")
    if backend == "local_sglang" and section["base_url"]:
        raise ValueError(
            "local_sglang requires empty base_url; use existing_server to attach"
        )
    spec = server_spec(
        role=role,
        section=section,
        model=model_path,
        output_dir=output_dir,
        force_no_start=backend == "existing_server",
    )
    if spec.start:
        check_gpus(spec)
    with ManagedSGLangServer(spec) as url:
        verify_actor(url, key, request_model(section))
        yield url, key


def generation_request(workflow, job, section):
    if section["backend"] != "remote_api":
        request = ModelRequest(
            input_ids=job["input_ids"],
            gconfig=workflow.gconfig.new(n_samples=1),
            tokenizer=workflow.tokenizer,
        )
        built = SGLangBackend().build_generation_request(
            request, with_lora=False, version=0
        )
        return built.endpoint, built.payload
    gconfig = workflow.gconfig
    # Fail rather than silently dropping a non-portable training decoding rule.
    configured_stops = set(gconfig.stop_token_ids) - {
        workflow.tokenizer.eos_token_id,
        workflow.tokenizer.pad_token_id,
    }
    if (
        gconfig.min_new_tokens
        or gconfig.ignore_eos
        or gconfig.use_beam_search
        or gconfig.top_k < int(1e8)
        or configured_stops
    ):
        raise ValueError(
            "This training decoding config cannot be represented faithfully by Chat Completions; use local_sglang"
        )
    body = {
        "model": section.get("request_model") or section["model"],
        "messages": job["messages"],
        "n": 1,
        "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
        "top_p": gconfig.top_p,
        "max_completion_tokens": int(
            section.get("max_completion_tokens") or gconfig.max_new_tokens
        ),
        "frequency_penalty": gconfig.frequency_penalty,
    }
    if gconfig.stop:
        body["stop"] = gconfig.stop
    if section.get("reasoning_effort"):
        body["reasoning_effort"] = section["reasoning_effort"]
    return "/v1/chat/completions", body


def skill_attempt(workflow, raw, response, attempt_index):
    """Keep strict training syntax, truncating only an overlong XML skill body."""
    limit = int(workflow.skill_description_max_words)
    if not 1 <= limit <= 100:
        raise ValueError("External skill word limit must be between 1 and 100")
    result = workflow._generation_attempt(raw, response, attempt_index)
    description = (result.get("parsed") or {}).get("description", "")
    words = list(re.finditer(r"\S+", description))
    truncated = False
    if (
        result["schema_error"] == f"description must contain at most {limit} words"
        and len(words) > limit
    ):
        content = description[: words[limit - 1].end()]
        checked = workflow._generation_attempt(
            f"<skill>{content}</skill>", response, attempt_index
        )
        # Preserve raw output, token statistics and section extraction for audit.
        for key in ("parsed", "schema_valid", "schema_error", "repair_notes"):
            result[key] = checked[key]
        result["repair_notes"] = [
            *result["repair_notes"],
            "external_word_limit_truncated",
        ]
        truncated = True
    result["word_truncation"] = {
        "policy": "truncate",
        "applied": truncated,
        "max_words": limit,
        "original_word_count": len(words),
        "final_word_count": len(
            re.findall(r"\S+", (result.get("parsed") or {}).get("description", ""))
        ),
    }
    return result


def response_attempt(workflow, job, payload, *, backend, attempt_index):
    if backend == "remote_api":
        choice = payload["choices"][0]
        raw = choice["message"].get("content") or ""
        if not isinstance(raw, str):
            raise ValueError("API returned non-text content")
        response = SimpleNamespace(
            input_tokens=job["input_ids"], output_tokens=[], output_logprobs=[]
        )
    else:
        parsed = SGLangBackend().parse_generation_response(payload)
        raw = workflow.tokenizer.decode(parsed.output_tokens, skip_special_tokens=False)
        response = SimpleNamespace(
            input_tokens=job["input_ids"],
            **{
                "output_tokens": parsed.output_tokens,
                "output_logprobs": parsed.output_logprobs,
            },
        )
        if parsed.stop_reason == "abort":
            raise RuntimeError("SGLang aborted generation")
    result = skill_attempt(workflow, raw, response, attempt_index)
    if backend == "remote_api":
        result.update(
            input_token_count=None,
            output_token_count=None,
            provider_usage=payload.get("usage"),
            finish_reason=choice.get("finish_reason"),
        )
    return result


def guarded_contract(output_dir, contract):
    path = output_dir / "run_manifest.json"
    previous = read_json(path)
    if path.exists() and previous != contract:
        raise ValueError(
            "Output directory has a different or corrupt run manifest; choose a new output directory"
        )
    atomic_write_json(path, contract)


def generate_bank(settings, config, *, prepare_only=False):
    sampling = settings["sampling"]
    section = dict(settings["generator"])
    section["model"] = os.path.expandvars(str(section["model"]))
    if "$" in section["model"]:
        raise ValueError("Set CURATOR_MODEL to a checkpoint directory saved by training")
    if not 1 <= int(config.webshop.skill_description_max_words) <= 100:
        raise ValueError("External skill word limit must be between 1 and 100")
    pool_dir = path_from_root(settings["pool_dir"])
    pool, episodes = read_pool(pool_dir, config)
    tokenizer = load_hf_tokenizer(str(config.tokenizer_path))
    workflow, jobs = build_jobs(
        config,
        tokenizer,
        episodes,
        count=int(sampling["skill_count"]),
        seed=int(sampling["seed"]),
    )
    bank_path = path_from_root(settings["output"]["path"])
    if bank_path.suffix != ".jsonl":
        raise ValueError("output.path must name a .jsonl file")
    output_dir = bank_path.parent
    max_attempts = int(
        sampling.get(
            "max_attempts_per_skill", 1 + workflow._max_skill_generation_retries()
        )
    )
    workers = int(sampling["workers"])
    if workers < 1 or max_attempts < 1:
        raise ValueError("workers and max_attempts_per_skill must be positive")
    # Validate API support before issuing requests or writing incomplete outputs.
    generation_request(workflow, jobs[0], section)
    contract = {
        "version": VERSION,
        "pool_signature": pool["pool_signature"],
        "episode_content_signature": digest(episodes),
        "seed": int(sampling["seed"]),
        "skill_count": len(jobs),
        "model": section["model"],
        "backend": section["backend"],
        "model_path": model_location(section),
        "request_model": section.get("request_model"),
        "provider_endpoint_signature": digest(
            os.path.expandvars(str(section.get("base_url") or ""))
        )
        if section["backend"] == "remote_api"
        else None,
        "reasoning_effort": section.get("reasoning_effort"),
        "tokenizer": str(config.tokenizer_path),
        "gconfig": asdict(workflow.gconfig),
        "prompt_signature": digest(jobs),
        "word_limit": workflow.skill_description_max_words,
        "overlong_skill_policy": "truncate",
        "max_completion_tokens": section.get("max_completion_tokens"),
    }
    with output_lock(output_dir):
        guarded_contract(output_dir, contract)
        for job in jobs:
            atomic_write_json(output_dir / "prompts" / f"{job['id']}.json", job)
        if prepare_only:
            print(
                f"Prepared {len(jobs)} exact training prompts: {output_dir}", flush=True
            )
            return
        records = {}
        for job in jobs:
            saved = read_json(output_dir / "generations" / f"{job['id']}.json")
            # Resume valid generations after re-checking them; retry the rest.
            if (
                saved
                and saved.get("schema_valid")
                and isinstance(saved.get("raw"), str)
            ):
                checked = skill_attempt(
                    workflow,
                    saved["raw"],
                    SimpleNamespace(
                        input_tokens=[], output_tokens=[], output_logprobs=[]
                    ),
                    0,
                )
                if not checked["schema_valid"] or checked["parsed"] != saved["parsed"]:
                    raise ValueError(
                        "Cached generation no longer passes the external skill policy"
                    )
                records[job["id"]] = {
                    "id": job["id"],
                    "model": section["model"],
                    "content": checked["parsed"]["description"],
                }

        def publish():
            bank = bank_path
            temporary = bank.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(
                    json.dumps(records[key], ensure_ascii=False) + "\n"
                    for key in sorted(records)
                ),
                encoding="utf-8",
            )
            temporary.replace(bank)
            atomic_write_json(
                output_dir / "summary.json",
                {
                    "status": "complete" if len(records) == len(jobs) else "incomplete",
                    "model": section["model"],
                    "requested": len(jobs),
                    "completed": len(records),
                    "seed": sampling["seed"],
                    "word_limit": workflow.skill_description_max_words,
                    "overlong_skill_policy": "truncate",
                    "max_attempts_per_skill": max_attempts,
                    "skillbank": bank_path.name,
                    "sampling": "training selector; one sample per logical group; static real trace pool",
                },
            )

        publish()
        pending = [job for job in jobs if job["id"] not in records]
        if pending:
            with model_endpoint(section, output_dir, role="generator") as (url, key):

                def run_one(job):
                    endpoint, body = generation_request(workflow, job, section)
                    attempt_dir = output_dir / "attempts" / job["id"]
                    offset = len(list(attempt_dir.glob("attempt_*.json")))
                    for attempt in range(max_attempts):
                        response = None
                        try:
                            response = requests.post(
                                url + endpoint,
                                json=body,
                                headers={"Authorization": f"Bearer {key}"}
                                if key
                                else {},
                                timeout=float(section.get("timeout_s", 180)),
                            )
                            response.raise_for_status()
                            result = response_attempt(
                                workflow,
                                job,
                                response.json(),
                                backend=section["backend"],
                                attempt_index=attempt,
                            )
                        except Exception as exc:
                            # Do not persist exception text: gateways may echo API credentials.
                            result = {
                                "schema_valid": False,
                                "error_type": type(exc).__name__,
                                "attempt": attempt + 1,
                                "http_status": response.status_code
                                if response is not None
                                else None,
                            }
                        result["lifetime_attempt"] = offset + attempt + 1
                        atomic_write_json(
                            attempt_dir / f"attempt_{offset + attempt + 1:03d}.json",
                            result,
                        )
                        atomic_write_json(
                            output_dir / "generations" / f"{job['id']}.json", result
                        )
                        if result["schema_valid"]:
                            return {
                                "id": job["id"],
                                "model": section["model"],
                                "content": result["parsed"]["description"],
                            }
                    return None

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for future in as_completed(
                        [executor.submit(run_one, job) for job in pending]
                    ):
                        record = future.result()
                        if record:
                            records[record["id"]] = record
                        publish()
                        print(
                            f"[skillbank] valid={len(records)}/{len(jobs)}", flush=True
                        )
        if len(records) != len(jobs):
            raise RuntimeError(
                f"Only {len(records)}/{len(jobs)} valid skills; inspect attempts/. Re-run to retry missing slots with unchanged prompts."
            )
        print(f"Skillbank: {bank_path}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "generate", "prepare"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Hydra override, identical to the training launcher",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    settings, config = load_settings(args.config, args.train_override)
    if args.command == "collect":
        from webshop_generality.collect_skill_traces import collect

        collect(settings, config)
    else:
        generate_bank(settings, config, prepare_only=args.command == "prepare")


if __name__ == "__main__":
    main()
