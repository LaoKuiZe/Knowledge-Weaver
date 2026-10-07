"""Build a knowledge bank with a trained curator checkpoint or an API model.

Phases: collect rolls out the frozen executor on training tasks; generate has
the curator write the bank from those trajectories; all runs both; prepare
renders the curator prompts without sending any request.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def alfworld_task_types():
    """Import lazily, so the WebShop path never loads the ALFWorld runtime."""
    if str(ROOT / "AReaL") not in sys.path:
        sys.path.insert(0, str(ROOT / "AReaL"))
    from areal.workflow.alfworld_runtime import ALFWORLD_TASK_TYPES

    return ALFWORLD_TASK_TYPES


def runtime_environment():
    env = os.environ.copy()
    defaults = {
        "PROJECT_ROOT": ROOT,
        "ALFWORLD_ROOT": ROOT,
        "WEBSHOP_ROOT": env.get(
            "WEBSHOP_REPO_ROOT", ROOT / ".benchmark-runtime/webshop"
        ),
        "ALFWORLD_DATA_ROOT": ROOT / "data/alfworld/json_2.1.1",
        "WEBSHOP_DATA_ROOT": ROOT / "data/webshop",
        "AREAL_OUTPUT_ROOT": ROOT / "outputs",
        "AREAL_CHECKPOINT_ROOT": ROOT / "checkpoints",
        "AREAL_NAME_RESOLVE_ROOT": ROOT / "outputs/name_resolve",
        "CACHE_ROOT": ROOT / ".cache",
        "ACTOR_MODEL_PATH": "Qwen/Qwen3.5-4B",
        "WANDB_MODE": "disabled",
    }
    for key, value in defaults.items():
        env.setdefault(key, str(value))
    env.setdefault("WEBSHOP_REPO_ROOT", env["WEBSHOP_ROOT"])
    env.setdefault("ALFWORLD_DATA", str(Path(env["ALFWORLD_DATA_ROOT"]).parent))
    paths = [
        ROOT / "generality_eval/src",
        ROOT / "AReaL",
        ROOT,
        Path(env["WEBSHOP_REPO_ROOT"]),
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(p) for p in paths), env.get("PYTHONPATH", "")]
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def repo_path(value):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in value:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def extract_alfworld_pool(result, pool, *, per_type, seed, trajectories_per_prompt=4):
    """Validate train-only evidence, manifest completeness, task identity, and usable actions."""
    task_types = alfworld_task_types()
    if result.get("run_config", {}).get("split") != "train":
        raise ValueError("Generation evidence must use train tasks")
    if result.get("status") != "complete":
        raise ValueError("Source evaluation must be complete")
    manifest = result.get("task_manifest")
    records = result.get("episodes_by_k", {}).get("0", [])
    expected = per_type * len(task_types)
    if (
        not isinstance(manifest, list)
        or len(manifest) != expected
        or len(records) != expected
    ):
        raise ValueError(f"Expected {expected} manifest tasks and source episodes")
    tasks = {row["rollout_index"]: row for row in manifest}
    if set(tasks) != set(range(expected)) or sorted(
        row["rollout_index"] for row in records
    ) != list(range(expected)):
        raise ValueError("Source rollout indices must uniquely cover all tasks")
    if Counter(row["task_type"] for row in manifest) != Counter(
        {task: per_type for task in task_types}
    ):
        raise ValueError("Source manifest must cover all six task types equally")
    usable = []
    counts = Counter()
    for record in records:
        episode = record["episode"]
        task = tasks[record["rollout_index"]]
        if record.get("status") != "complete":
            raise ValueError("Source episode record must be complete")
        if any(episode.get(key) != task.get(key) for key in ("gamefile", "task_type")):
            raise ValueError("Source episode does not match its manifest task")
        parts = Path(task["gamefile"]).parts
        if "train" not in parts or {"valid_seen", "valid_unseen"}.intersection(parts):
            raise ValueError("Source manifest contains a non-train task")
        steps = episode.get("steps") or []
        valid = sum(bool(str(s.get("action") or "").strip()) for s in steps)
        if episode.get("status") in {"won", "done", "max_steps"} and valid:
            usable.append((record, valid))
            counts[episode["task_type"]] += 1
    if set(counts) != set(task_types) or min(counts.values()) < trajectories_per_prompt:
        raise ValueError("Insufficient valid evidence for all six task types")
    refs = []
    for record, valid in usable:
        episode = record["episode"]
        index = record["rollout_index"]
        relative = Path("episodes") / episode["task_type"] / f"episode_{index:04d}.json"
        write_json(pool / relative, {"episode": episode, "rollout_index": index})
        refs.append(
            {
                "source_episode_path": str(relative),
                "source_rollouts_path": "",
                "rollout_index": index,
                "won": bool(episode.get("won")),
                "status": episode["status"],
                "steps_taken": len(episode.get("steps") or []),
                "valid_action_count": valid,
                "task_type": episode["task_type"],
                "gamefile": episode["gamefile"],
                "pool_source": "frozen_executor_train",
            }
        )
    write_json(
        pool / "index.json",
        {
            "status": "complete",
            "split": "train",
            "seed": seed,
            "per_task_type_count": dict(counts),
            "refs": refs,
            "source_run_config": result["run_config"],
        },
    )


def run(command, *, dry_run):
    print(shlex.join([str(x) for x in command]), flush=True)
    if not dry_run:
        subprocess.run([str(x) for x in command], check=True, cwd=ROOT)


def collect_alfworld(settings, *, dry_run):
    collection = settings["collection"]
    output = repo_path(collection["output_dir"])
    pool = repo_path(collection["pool_dir"])
    # The shared evaluator requires a nonempty bank; k=0 injects only its canonical
    # no-skill text, so this protocol anchor is never retrieved or shown to the actor.
    anchor = output / "no_skill_anchor.jsonl"
    if not dry_run:
        anchor.parent.mkdir(parents=True, exist_ok=True)
        anchor.write_text(
            json.dumps({"id": "unused", "model": "none", "content": "No skill."}) + "\n"
        )
    per_type = int(collection.get("tasks_per_type", 20))
    if per_type < int(settings["sampling"].get("trajectories_per_prompt", 4)):
        raise ValueError("tasks_per_type must cover trajectories_per_prompt")
    command = [
        sys.executable,
        "-m",
        "alfworld_generality.run_semantic_skillbank_eval",
        "--skillbank",
        anchor,
        "--expected-skill-count",
        "1",
        "--top-k-values",
        "0",
        "--output-dir",
        output,
        "--repo-root",
        ROOT,
        "--data-root",
        repo_path(collection["data_root"]),
        "--split",
        "train",
        "--tasks-per-type",
        str(per_type),
        "--expected-total-games",
        str(per_type * len(alfworld_task_types())),
    ]
    options = {
        "checkpoint_path": collection["model"],
        "seed": collection.get("seed", 13000000),
        "max_rollout_steps": 50,
        "memory_window": 5,
        "max_commands": 140,
        "rollout_workers": collection.get("workers", 16),
        "initial_observation_workers": 8,
        "embedding_model": collection.get(
            "embedding_model", "sentence-transformers/all-mpnet-base-v2"
        ),
        "gpus": collection.get("gpus", "0"),
        "tp_size": collection.get("tp_size", 1),
        "dp_size": collection.get("dp_size", 1),
        "port": collection.get("port", 34080),
    }
    for key in ("base_url", "request_model", "api_key_env"):
        if collection.get(key):
            options[key] = collection[key]
    for key, value in options.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    run(command, dry_run=dry_run)
    if not dry_run:
        extract_alfworld_pool(
            json.loads((output / "results.json").read_text()),
            pool,
            per_type=per_type,
            seed=options["seed"],
            trajectories_per_prompt=int(
                settings["sampling"].get("trajectories_per_prompt", 4)
            ),
        )


def alfworld_generation_config(settings):
    settings = json.loads(json.dumps(settings))
    settings.pop("collection", None)
    settings["sampling"]["trajectory_pool_indices"] = [
        str(repo_path(p)) for p in settings["sampling"]["trajectory_pool_indices"]
    ]
    settings["output"]["path"] = str(repo_path(settings["output"]["path"]))
    for key in ("model", "model_path", "base_url", "request_model"):
        if settings["api"].get(key):
            value = os.path.expandvars(settings["api"][key])
            if "$" in value:
                raise ValueError(
                    f"Set the environment variable for api.{key}, usually CURATOR_MODEL"
                )
            if key in {"model", "model_path"} and (
                value.startswith(("/", ".", "~")) or (ROOT / value).exists()
            ):
                value = str(repo_path(value))
            settings["api"][key] = value
    return settings


def check_curator(settings, task):
    """Exit before collection when the generate phase could not load its curator."""
    section = settings["api" if task == "alfworld" else "generator"]
    hint = (
        "Set CURATOR_MODEL to a checkpoint directory saved by training, or set "
        f"{task.upper()}_API_MODEL, {task.upper()}_API_BASE_URL and "
        f"{task.upper()}_API_KEY"
    )
    values = [
        os.path.expandvars(str(section.get(key) or ""))
        for key in ("model_path", "model")
    ]
    location = values[0] or values[1]
    if not location or any("$" in value for value in values):
        raise SystemExit(hint)
    if section.get("backend") != "local_sglang" or not (
        location.startswith(("/", ".", "~")) or (ROOT / location).exists()
    ):
        return
    path = repo_path(location)
    if not (path / "config.json").is_file() or not (
        any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    ):
        raise SystemExit(
            f"Curator model {path} must be a directory with config.json and model "
            f"weights. {hint}"
        )


def apply_api_environment(settings, task):
    """Select a task's remote curator without changing executor settings."""
    prefix = task.upper() + "_API_"
    names = {key: prefix + key for key in ("MODEL", "BASE_URL", "KEY")}
    values = {key: os.environ.get(name, "").strip() for key, name in names.items()}
    if not any(values.values()):
        return False
    missing = [names[key] for key, value in values.items() if not value]
    if missing:
        raise ValueError("API generation requires " + ", ".join(missing))
    if not values["BASE_URL"].startswith(("https://", "http://")):
        raise ValueError(f"{names['BASE_URL']} must start with http:// or https://")
    section = settings["api" if task == "alfworld" else "generator"]
    section.update(
        backend="remote_api",
        model=values["MODEL"],
        model_path="",
        request_model="",
        base_url=values["BASE_URL"],
        api_key_env=names["KEY"],
    )
    section.pop("api_key", None)
    if task == "alfworld":
        # Local SGLang template options are not portable API parameters.
        section["extra_body"] = {}
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("alfworld", "webshop"))
    parser.add_argument(
        "--phase", choices=("all", "collect", "generate", "prepare"), default="all"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    os.environ.update(runtime_environment())
    config = (
        args.config
        or ROOT / f"generality_eval/config/{args.task}_skillbank_generation.yaml"
    ).resolve()
    settings = yaml.safe_load(config.read_text())
    phases = ("collect", "generate") if args.phase == "all" else (args.phase,)
    api_override = (
        apply_api_environment(settings, args.task) if args.phase != "collect" else False
    )
    if "generate" in phases:
        check_curator(settings, args.task)
    output_override = os.environ.get("BANK_OUTPUT_PATH", "").strip()
    if output_override:
        output_path = repo_path(output_override)
        if output_path.suffix != ".jsonl":
            raise ValueError("BANK_OUTPUT_PATH must name a .jsonl file")
        settings["output"] = {"path": str(output_path)}
    if args.task == "webshop":
        with tempfile.TemporaryDirectory(prefix="knowledgeweaver-bank-") as temporary:
            if api_override or output_override:
                config = Path(temporary) / "generation.yaml"
                config.write_text(yaml.safe_dump(settings, sort_keys=False))
            for phase in phases:
                run(
                    [
                        sys.executable,
                        "-m",
                        "webshop_generality.skillbank_generation",
                        phase,
                        "--config",
                        config,
                    ],
                    dry_run=args.dry_run,
                )
        return
    for phase in phases:
        if phase == "collect":
            collect_alfworld(settings, dry_run=args.dry_run)
            continue
        if args.dry_run:
            print(f"{phase}: shared ALFWorld curator generator, config={config}")
            continue
        normalized = alfworld_generation_config(settings)
        with tempfile.TemporaryDirectory(prefix="knowledgeweaver-bank-") as temporary:
            normalized_path = Path(temporary) / "generation.yaml"
            normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=False))
            if phase == "prepare":
                # Validation and prompt rendering only. No curator request is sent.
                sys.path[:0] = [str(ROOT / "AReaL")]
                from examples.alfworld_skill.generate_external_skills import (
                    load_config,
                    build_prompt_jobs,
                )

                jobs = build_prompt_jobs(load_config(normalized_path))
                output = (
                    Path(normalized["output"]["path"]).parent / "prepared_prompts.json"
                )
                write_json(output, [job.__dict__ for job in jobs])
                print(f"Prepared {len(jobs)} prompts: {output}")
            else:
                run(
                    [
                        sys.executable,
                        ROOT
                        / "AReaL/examples/alfworld_skill/generate_external_skills.py",
                        "--config",
                        normalized_path,
                    ],
                    dry_run=False,
                )


if __name__ == "__main__":
    main()
