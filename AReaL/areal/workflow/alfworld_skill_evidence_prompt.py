# SPDX-License-Identifier: MIT

"""Evidence-discovery prompt for ALFWorld skill induction."""

EVIDENCE_DISCOVERY_SYSTEM_PROMPT = """You extract one useful guidance skill from agent trajectories.
The acting agent already receives general operating guidance to consult its legal actions,
navigate where needed, reveal accessible contents, and carry out the requested interaction.
Treat that guidance as given rather than rewriting it as the learned skill.

Study the complete trajectories as behavioral evidence. Keep investigating beyond convenient
surface explanations until the histories support an additional consequential insight that can
improve future decisions. The conclusion must depend on what actually happened in the
trajectories. Do not merely summarize the task or action-space description, imitate a fixed
writing pattern, or invent unsupported details. Reason privately and output only the final
natural-language skill required by the output contract."""


EVIDENCE_DISCOVERY_USER_CONTENT_TEMPLATE = """# Task category
{category}   ({category_note})

# Legal action vocabulary
The agent may use these action templates: {action_space}.
This list only constrains which actions are legal. It is not trajectory evidence and must not
be restated as the learned skill. The agent only sees the current room view. Receptacles must
be navigated to, and some must be opened before their contents become visible.

# Complete trajectories
{trajectories}

# Skill content requirements
- Read across the complete action histories and discover a consequential issue or insight that
  is easy to miss from the task description alone, then give useful guidance for addressing it.
- Ground the guidance in differences, consequences, or lack of progress visible in the supplied
  trajectories. If the same advice could be written without reading them, investigate further.
- Do not mechanically recap the action sequence.
- Use only legal actions from the action vocabulary. Do not invent actions or unsupported facts.
- Write one compact natural-language paragraph of at most 100 words, without following a fixed
  rhetorical template.
"""
