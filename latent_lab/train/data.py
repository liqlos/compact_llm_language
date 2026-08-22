"""Supervised data for visible-to-latent curriculum (toy scale).

Each example carries: input token ids, a visible CoT span [lo, hi) that the
curriculum may replace with latent steps, and answer token ids. Deterministic
templates only — no dataset downloads at this stage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumExample:
    example_id: str
    input_ids: list[int]
    cot_span: tuple[int, int]      # [lo, hi) replaceable reasoning span
    answer_ids: list[int]


def make_chain_dataset(n: int = 32, vocab_base: int = 100) -> list[CurriculumExample]:
    """A→B→C chains as token sequences.

    layout: [q_a, q_b, <cot_start>, c1, c2, c3, <cot_end>, ans]
    The model must map (a,b) -> c through the CoT span.
    """
    out = []
    for i in range(n):
        a, b, c = i % 16 + vocab_base, (i * 7 + 3) % 16 + vocab_base, \
                  (i * 13 + 5) % 16 + vocab_base
        inp = [1, a, b, 2]          # BOS, question tokens, SEP
        cot_lo = len(inp)
        inp += [20, 21, 22]         # placeholder CoT tokens
        ans = [c]
        out.append(CurriculumExample(
            example_id=f"chain-{i:04d}",
            input_ids=inp,
            cot_span=(cot_lo, len(inp)),
            answer_ids=ans,
        ))
    return out
