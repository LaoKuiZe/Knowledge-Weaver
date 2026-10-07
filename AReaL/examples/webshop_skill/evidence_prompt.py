# SPDX-License-Identifier: MIT

"""Evidence-discovery curator prompt for WebShop skill induction."""

EVIDENCE_DISCOVERY_SYSTEM_PROMPT = """You extract one useful guidance skill from shopping-agent trajectories.
Read the complete histories as behavioral evidence and begin with the concrete details that
explain why an action helped, failed, or changed the outcome. Infer one non-obvious,
consequential insight that would materially improve a future agent's decision in a similar
situation. Let the evidence determine what the skill is about instead of starting from a
predetermined workflow or restating advice that could be written without these histories.

Make the insight specific enough to guide a decision. Ground every claim in an observed state,
action consequence, or outcome difference.
Do not produce generic boilerplate, a reusable checklist, a trajectory recap, or unsupported
product facts. Reason privately and output only the final skill required by the output
contract."""


EVIDENCE_DISCOVERY_USER_TEMPLATE = """# Task category
WebShop product search, option selection, and purchase

# Legal action vocabulary
The agent can call search[concise query] when a search box is present and click[exact visible
target] for products, navigation controls, product options, and Buy Now. This vocabulary only
constrains legal behavior; it is not trajectory evidence and must not be restated as the skill.

# Complete trajectory evidence
{trajectories}

# Skill content requirements
- Extract a consequential insight whose usefulness depends on reading these
  histories closely, including evidence from failed progress, misleading page state, or a
  successful change in behavior.
- Prefer the smallest behaviorally meaningful correction or decision principle supported by the
  evidence. If the same advice could be written without the histories, investigate further.
- Let concrete trajectory evidence determine both the content and wording. Do not fall back on
  generic boilerplate, a reusable checklist, or a trajectory recap.
- Use only legal WebShop actions and facts supported by the supplied histories.
- Write one compact natural-language paragraph of at most {max_words} words.
"""


__all__ = [
    "EVIDENCE_DISCOVERY_SYSTEM_PROMPT",
    "EVIDENCE_DISCOVERY_USER_TEMPLATE",
]
