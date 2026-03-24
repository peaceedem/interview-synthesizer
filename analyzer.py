"""
Analysis engine — uses Claude to synthesize customer interview transcripts.
Requires ANTHROPIC_API_KEY environment variable.
"""

import os
from typing import Generator
import anthropic


def synthesize(transcripts: list[dict]) -> Generator[str, None, None]:
    """
    Yields text chunks from Claude's streaming analysis of all transcripts.
    Each chunk is a raw string fragment — the caller stitches them together.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it as an environment variable."
        )

    client = anthropic.Anthropic(api_key=api_key)

    n = len(transcripts)
    formatted = "\n\n".join(
        f"--- INTERVIEW {i + 1}: {t['label']} ---\n{t['text'].strip()}"
        for i, t in enumerate(transcripts)
    )

    prompt = f"""You are a senior UX researcher and product strategist. A founder has shared {n} customer interview transcript{"s" if n != 1 else ""} with you and needs a deep, actionable synthesis to guide their next product decisions.

Read every word carefully. Your analysis should surface what customers actually said, not generic observations — use their language, cite their words, and connect patterns across interviews.

{formatted}

---

Write a thorough synthesis in the following structure. Use markdown formatting throughout.

## Executive Summary
In 3-4 sentences: what is the single most important thing these interviews reveal, and what should the product team do about it? Be direct and specific — no hedging.

## Key Themes
Identify the 4–7 most significant patterns across interviews. For each theme:
- **Theme name** — a crisp label
- What it means: 2-3 sentences explaining the pattern and why it matters
- Evidence: 1-2 direct quotes with speaker attribution in italics (e.g. *— {transcripts[0]['label'] if transcripts else 'Interviewee'}*)
- Prevalence: note which interviews reflect this theme

## Pain Points
The real frustrations, frictions, and failures customers experience. Go beyond surface complaints — diagnose the underlying problem. For each:
- **Pain point title**
- What's actually happening and why it's painful
- Direct quote capturing the frustration
- Impact on the customer's workflow or decision-making

## Feature Requests
Both explicit requests and implicit needs you can infer from what customers struggled with or wished existed. For each:
- **Request title**
- What they want and the underlying job-to-be-done
- Quote or evidence from the transcript
- Who needs this most (segment, role, use case)

## Objections & Hesitations
Anything that would slow adoption, cause hesitation, or lead to churn. For each:
- **Objection**
- What's driving it (price, trust, complexity, timing, alternatives, etc.)
- Quote
- How a product or messaging change could address it

## Recommended Actions
5–7 concrete, prioritized actions for the product team. Each action should be specific enough that an engineer or designer could act on it tomorrow. For each:
- **Action title**
- What to build, change, or investigate
- Evidence base: which interviews and quotes support this
- Priority: 🔴 High / 🟡 Medium / 🟢 Low
- Expected impact: what customer outcome this drives

## Open Questions
2–4 important questions these interviews raise that need further research before the team can act with confidence. Be specific — not "learn more about X" but "we need to understand whether customers would pay for X given that they already use Y."

---

Be ruthlessly specific. If you see a pattern in only one interview, say so. If something is unclear from the transcripts, say so rather than speculating. The product team is depending on this to make real decisions."""

    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
