# SPDX-License-Identifier: MIT

"""Knowledge-bank I/O, semantic retrieval, and admission helpers for skill training."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from areal.workflow.alfworld_skill import _embed_group_similarity_texts

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


@dataclass(frozen=True)
class SkillEntry:
    """One body-only natural-language skill."""

    source_index: int
    skill_id: str
    model: str
    content: str

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def _skill_records(path: Path) -> list[tuple[int, Any]]:
    if path.suffix.lower() == ".json":
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(decoded, dict):
            decoded = decoded.get("skills")
        if not isinstance(decoded, list):
            raise ValueError("JSON skillbank must be a list or an object with skills")
        return list(enumerate(decoded, start=1))

    records: list[tuple[int, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append((line_number, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} at line {line_number}: {exc}"
            ) from exc
    return records


def load_skillbank(
    path: Path, *, expected_count: int = 0, allow_empty: bool = False
) -> list[SkillEntry]:
    """Load the canonical ``id/model/content`` JSON or JSONL skillbank schema."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Skillbank does not exist: {resolved}")

    skills: list[SkillEntry] = []
    seen_ids: set[str] = set()
    seen_bodies: set[str] = set()
    for record_number, payload in _skill_records(resolved):
        if not isinstance(payload, dict):
            raise ValueError(f"Skillbank record {record_number} must be a JSON object")
        if set(payload) != {"id", "model", "content"}:
            raise ValueError(
                f"Skillbank record {record_number} must contain only id, model, content"
            )
        skill_id = str(payload["id"]).strip()
        model = str(payload["model"]).strip()
        content = " ".join(str(payload["content"]).split())
        if not skill_id or not model or not content:
            raise ValueError(
                f"Skillbank record {record_number} has an empty id, model, or content"
            )
        if skill_id in seen_ids:
            raise ValueError(f"Duplicate skill id in {resolved}: {skill_id}")
        normalized_body = content.casefold()
        if normalized_body in seen_bodies:
            raise ValueError(
                f"Duplicate normalized skill content in {resolved}: {skill_id}"
            )
        if "<skill" in content.lower() or "</skill" in content.lower():
            raise ValueError(
                f"Skillbank record {record_number} must contain only the skill body"
            )
        seen_ids.add(skill_id)
        seen_bodies.add(normalized_body)
        skills.append(
            SkillEntry(
                source_index=len(skills),
                skill_id=skill_id,
                model=model,
                content=content,
            )
        )

    if expected_count > 0 and len(skills) != expected_count:
        raise ValueError(
            f"Expected {expected_count} skills in {resolved}, found {len(skills)}"
        )
    if not skills and not allow_empty:
        raise ValueError(f"Skillbank is empty: {resolved}")
    return skills


def write_skillbank(path: Path, skills: Sequence[SkillEntry]) -> None:
    """Atomically persist body-only records in deterministic JSONL order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(
        "".join(
            json.dumps(
                {
                    "id": skill.skill_id,
                    "model": skill.model,
                    "content": skill.content,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for skill in skills
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def skillbank_snapshot_digest(skills: Sequence[SkillEntry]) -> str:
    """Hash the canonical body-only skillbank records in their retrieval order."""

    payload = [
        {
            "id": skill.skill_id,
            "model": skill.model,
            "content": skill.content,
        }
        for skill in skills
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def compute_skillbank_counterfactual_reward(
    *,
    singleton_sr: float,
    no_skill_sr: float,
    bank_plus_candidate_sr: float,
    bank_only_sr: float,
    standalone_weight: float = 0.5,
    retrieval_weight: float = 0.5,
) -> dict[str, float]:
    """Compute the two paired outcome deltas and their weighted reward."""

    standalone_weight = float(standalone_weight)
    retrieval_weight = float(retrieval_weight)
    if min(standalone_weight, retrieval_weight) < 0.0:
        raise ValueError("skillbank counterfactual weights must be non-negative")
    if abs(standalone_weight + retrieval_weight - 1.0) > 1.0e-9:
        raise ValueError("skillbank counterfactual weights must sum to one")
    standalone_delta = float(singleton_sr) - float(no_skill_sr)
    bank_marginal_delta = float(bank_plus_candidate_sr) - float(bank_only_sr)
    reward = (
        standalone_weight * standalone_delta + retrieval_weight * bank_marginal_delta
    )
    return {
        "singleton_sr": float(singleton_sr),
        "no_skill_sr": float(no_skill_sr),
        "standalone_delta": standalone_delta,
        "bank_plus_candidate_sr": float(bank_plus_candidate_sr),
        "bank_only_sr": float(bank_only_sr),
        "bank_marginal_delta": bank_marginal_delta,
        "combined_sr_reward": reward,
        "combined_reward": reward,
    }


def select_online_skillbank_admission(
    *,
    candidates: Sequence[dict[str, Any]],
    current_skills: Sequence[SkillEntry],
    min_marginal: float = 0.0,
    allow_noop: bool = True,
    max_size: int = 0,
) -> dict[str, Any] | None:
    """Admit the highest positive combined-SR candidate that is valid and new."""

    existing_bodies = {
        " ".join(skill.content.split()).casefold() for skill in current_skills
    }
    eligible = []
    if max_size <= 0 or len(current_skills) < max_size:
        for candidate in candidates:
            content = " ".join(str(candidate.get("content") or "").split())
            if (
                bool(candidate.get("schema_valid"))
                and content
                and content.casefold() not in existing_bodies
                and float(candidate.get("combined_sr_reward", 0.0)) > min_marginal
            ):
                eligible.append({**candidate, "content": content})
    if not eligible:
        if allow_noop:
            return None
        raise RuntimeError("no eligible online skillbank admission candidate")
    return min(
        eligible,
        key=lambda item: (
            -float(item["combined_sr_reward"]),
            str(item.get("sample_key") or ""),
        ),
    )


def build_retrieval_query(
    *,
    task_type: str,
    task_description: str,
    initial_observation: str,
) -> str:
    """Build the once-per-episode retrieval query used during train and eval."""

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


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    max_length: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Encode text on CPU with the shared MPNet mean-pooling embedder."""

    if device != "cpu":
        raise ValueError("shared skillbank retrieval currently requires device='cpu'")
    return _embed_group_similarity_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )


def retrieve_top_skills(
    *,
    skills: Sequence[SkillEntry],
    skill_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    top_k: int,
) -> list[list[dict[str, Any]]]:
    """Return deterministic cosine-ranked skill metadata for every query."""

    if skill_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("skill_embeddings and query_embeddings must be rank-2")
    if skill_embeddings.shape[0] != len(skills):
        raise ValueError("skill embedding rows must match the skill count")
    if skill_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("skill and query embedding dimensions must match")
    if top_k < 0 or top_k > len(skills):
        raise ValueError(f"top_k must be between 0 and {len(skills)}")

    similarities = query_embeddings.float() @ skill_embeddings.float().T
    rows: list[list[dict[str, Any]]] = []
    for query_index in range(int(query_embeddings.shape[0])):
        scores = similarities[query_index].tolist()
        ordered = sorted(
            range(len(skills)),
            key=lambda index: (-float(scores[index]), skills[index].skill_id, index),
        )[:top_k]
        rows.append(
            [
                {
                    "rank": rank,
                    "skill_id": skills[index].skill_id,
                    "model": skills[index].model,
                    "source_index": skills[index].source_index,
                    "cosine_similarity": float(scores[index]),
                }
                for rank, index in enumerate(ordered, start=1)
            ]
        )
    return rows


def format_retrieved_skill_text(
    selected: Sequence[dict[str, Any]],
    skills_by_id: dict[str, SkillEntry],
    *,
    no_skill_text: str,
) -> str:
    """Build static actor guidance for one complete episode."""

    if not selected:
        return no_skill_text
    bodies: list[str] = []
    for item in selected:
        skill_id = str(item.get("skill_id") or "")
        if skill_id not in skills_by_id:
            raise KeyError(f"Retrieved unknown skill id: {skill_id}")
        bodies.append(skills_by_id[skill_id].content)
    if len(bodies) == 1:
        return bodies[0]
    return "Retrieved guidance skills:\n" + "\n\n".join(
        f"{index}. {body}" for index, body in enumerate(bodies, start=1)
    )


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "SkillEntry",
    "build_retrieval_query",
    "compute_skillbank_counterfactual_reward",
    "embed_texts",
    "format_retrieved_skill_text",
    "load_skillbank",
    "retrieve_top_skills",
    "select_online_skillbank_admission",
    "skillbank_snapshot_digest",
    "write_skillbank",
]
