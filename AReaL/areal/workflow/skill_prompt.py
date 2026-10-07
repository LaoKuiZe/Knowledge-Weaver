# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

DEFAULT_SKILL_PROMPT_TOTAL_TOKEN_BUDGET = 8192


def skill_prompt_budgets(
    gconfig: Any,
    *,
    total_token_budget: int = DEFAULT_SKILL_PROMPT_TOTAL_TOKEN_BUDGET,
) -> tuple[int, int]:
    """Return the effective total and prompt-only token budgets."""

    configured_total = int(
        getattr(gconfig, "max_tokens", total_token_budget) or total_token_budget
    )
    effective_total = min(max(1, configured_total), max(1, total_token_budget))
    generation_budget = max(
        1, int(getattr(gconfig, "max_new_tokens", 1) or 1)
    )
    return effective_total, max(1, effective_total - generation_budget)

