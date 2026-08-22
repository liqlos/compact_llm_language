"""Format cost micro-benchmark: identical working-memory content, six serializations.

Measures what the user-facing question actually asks: how much does the
CHOICE OF REPRESENTATION of machine reasoning state cost in tokens?
All formats carry exactly the same numbers/questions/actions; only the
surface form differs. Measured under both rcc-approx-1 and tiktoken o200k.

Evidence context:
- Structured Context Engineering (arXiv:2602.05447): YAML most token-efficient
  of common formats; exotic formats (TOON) suffer a "grep tax" at scale.
- Notation Matters (arXiv:2605.29676): TOON/TRON savings can reverse inside
  multi-turn loops via parsing cascades.
RIR/1 lines are deliberately near-natural-language symbols to stay parseable.
"""

from __future__ import annotations

import json

from bench.scenarios import _doc  # deterministic filler generator

FACTS = [
    ("PostgreSQL 16.3 released 2024-05-09", ("obs-0001",), 0.98),
    ("p95 latency 142ms under load test LT-77", ("obs-0001",), 0.97),
    ("timeout configured 30000ms", ("obs-0001",), 1.0),
    ("schema version v9.2.1", ("obs-0002",), 0.99),
    ("default port 5432", ("obs-0002",), 1.0),
]
QUESTIONS = ["writer lock ordering between replicas?", "is v9.2.1 backward compatible?"]
ACTIONS = ["verify lock ordering via simulation against obs-0003",
           "changelog diff v9.2.0..v9.2.1 from obs-0002"]
CONFLICT = "source A says 142ms p95, source B says 141ms p95 @obs-0001,@obs-0004"

FILLER_NOTE = _doc("supporting", paras=6)


def prose() -> str:
    parts = ["So far I have established the following facts about the deployment:"]
    for text, _, conf in FACTS:
        parts.append(f"- It is confirmed that {text} (confidence approximately {conf:.2f}).")
    parts.append(f"There is an open conflict to resolve: {CONFLICT}.")
    parts.append("Questions that remain open at this point:")
    for q in QUESTIONS:
        parts.append(f"- We still need to answer: {q}")
    parts.append("The next actions I intend to take are the following:")
    for a in ACTIONS:
        parts.append(f"- My next step will be to {a}")
    return "\n".join(parts)


def markdown() -> str:
    out = ["## Facts", ""]
    for text, src, conf in FACTS:
        s = ",".join(src)
        out.append(f"- **{text}** (conf={conf:.2f}, src=@{s})")
    out += ["", "## Conflict", "", f"- {CONFLICT}", "", "## Questions", ""]
    out += [f"- {q}" for q in QUESTIONS]
    out += ["", "## Next actions", ""]
    out += [f"- [ ] {a}" for a in ACTIONS]
    return "\n".join(out)


def json_fmt() -> str:
    return json.dumps({
        "facts": [{"text": t, "src": list(s), "conf": c} for t, s, c in FACTS],
        "conflict": CONFLICT,
        "questions": QUESTIONS,
        "next_actions": ACTIONS,
        "note": FILLER_NOTE,
    }, separators=(",", ":"))


def yaml_fmt() -> str:
    lines = ["facts:"]
    for text, src, conf in FACTS:
        lines.append(f"  - text: {text}")
        lines.append(f"    src: [{','.join(src)}]")
        lines.append(f"    conf: {conf:.2f}")
    lines.append(f"conflict: {CONFLICT}")
    lines.append("questions:")
    lines += [f"  - {q}" for q in QUESTIONS]
    lines.append("next_actions:")
    lines += [f"  - {a}" for a in ACTIONS]
    lines.append(f"note: {FILLER_NOTE.splitlines()[0]}")
    return "\n".join(lines)


def csv_rows() -> str:
    lines = ["kind,id,text,src,conf"]
    for i, (text, src, conf) in enumerate(FACTS, 1):
        lines.append(f'F,f{i:02d},"{text}","{",".join(src)}",{conf:.2f}')
    lines.append(f'C,c01,"{CONFLICT}",,')
    for i, q in enumerate(QUESTIONS, 1):
        lines.append(f'Q,q{i:02d},"{q}",,')
    for i, a in enumerate(ACTIONS, 1):
        lines.append(f'N,n{i:02d},"{a}",,')
    return "\n".join(lines)


def rir() -> str:
    """What RCC's Scratch layer actually emits (numbers verified verbatim)."""
    lines = ["<SCRATCH format=RIR/1>"]
    for i, (text, src, conf) in enumerate(FACTS, 1):
        lines.append(f"F f{i:02d} {text} conf={conf:.2f} @{','.join(src)}")
    lines.append(f"C c01 {CONFLICT}")
    for i, q in enumerate(QUESTIONS, 1):
        lines.append(f"Q q{i:02d} {q}")
    for i, a in enumerate(ACTIONS, 1):
        lines.append(f"N n{i:02d} {a}")
    return "\n".join(lines)


FORMATS = {
    "prose": prose,
    "markdown": markdown,
    "json": json_fmt,
    "yaml": yaml_fmt,
    "csv": csv_rows,
    "rir1": rir,
}


def measure(tokenizer) -> dict[str, int]:
    return {name: tokenizer(fn()) for name, fn in FORMATS.items()}
