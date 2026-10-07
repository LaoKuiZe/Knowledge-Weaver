"""Knowledge-bank loading and semantic retrieval for ALFWorld and WebShop evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


@dataclass(frozen=True)
class SkillEntry:
    """One natural-language entry from an id/model/content knowledge bank."""

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


def load_skillbank(path: Path, *, expected_count: int = 50) -> list[SkillEntry]:
    """Load a JSONL or JSON bank and fail fast on non-conforming records."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Skillbank does not exist: {resolved}")

    skills: list[SkillEntry] = []
    seen_ids: set[str] = set()
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
        if "<skill" in content.lower() or "</skill" in content.lower():
            raise ValueError(
                f"Skillbank record {record_number} must contain only the skill body"
            )
        seen_ids.add(skill_id)
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
    if not skills:
        raise ValueError(f"Skillbank is empty: {resolved}")
    return skills


def build_retrieval_query(
    *,
    task_type: str,
    task_description: str,
    initial_observation: str,
) -> str:
    """Build the once-per-episode retrieval query used by the evaluator."""

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
    cache_dir: Path | None = None,
) -> torch.Tensor:
    """Encode text with attention-mask mean pooling and L2 normalization."""

    if not texts:
        return torch.empty((0, 0), dtype=torch.float32)
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")

    from transformers import AutoModel, AutoTokenizer

    load_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir.expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model.eval()
    model.to(device)

    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunks.append(
                torch.nn.functional.normalize(pooled, p=2, dim=1).float().cpu()
            )
    return torch.cat(chunks, dim=0)


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
    """Build the static actor guidance injected for one complete episode."""

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
