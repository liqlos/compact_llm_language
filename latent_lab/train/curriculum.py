"""Curriculum: gradually replace visible CoT spans with latent steps.

`replace_fraction` in {0.0, 0.5, 1.0} per stage. Collapse detection: if the
model's answer quality is unchanged when latent state is zeroed, the latent
block is being IGNORED — logged as a collapse case, never counted as progress.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    replace_fraction: float
    k_steps: int


STAGES = (
    CurriculumStage("s0_visible", 0.0, 0),
    CurriculumStage("s1_half", 0.5, 2),
    CurriculumStage("s2_full_latent", 1.0, 4),
)


def apply_stage(example, stage: CurriculumStage):
    """Return (input_ids with span masked/replaced, effective k).

    Replacement uses dedicated boundary anchor tokens:
      <lat_open>=3, <lat_pad>=4, <lat_close>=5
    Boundary tokens are control events — not serialized reasoning.
    """
    lo, hi = example.cot_span
    ids = list(example.input_ids)
    n_span = hi - lo
    n_replace = round(n_span * stage.replace_fraction)
    if stage.replace_fraction > 0.0 and n_replace == 0:
        n_replace = 1
    new_ids = (
        ids[:lo]
        + [3] + [4] * max(0, n_replace - 2) + [5]
        + ids[hi:]
    ) if n_replace >= 2 else ids[:lo] + [3] + ids[hi:]
    return new_ids, (stage.k_steps if n_replace > 0 else 0)


def detect_collapse(acc_with_latent: float, acc_zero_ablation: float,
                    tolerance: float = 1e-6) -> bool:
    """True when zeroing latent state does NOT hurt -> latent path ignored."""
    return abs(acc_with_latent - acc_zero_ablation) <= tolerance
