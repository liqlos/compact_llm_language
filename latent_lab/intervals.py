"""Layer-interval selection for hybrid Qwen architectures.

Qwen3.5 hybrid layout repeats natural 4-layer groups:
    3 x Gated DeltaNet (+FFN) then 1 x gated Attention (+FFN)
Output-side intervals are NOT assumed optimal: candidates cover early,
middle, output-side and the full-decoder control.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerInterval:
    lo: int
    hi: int
    name: str

    def __post_init__(self) -> None:
        if not (0 <= self.lo < self.hi):
            raise ValueError(f"non-contiguous interval {self}")


def natural_groups(layer_types: tuple[str, ...], group: int = 4) -> list[tuple[int, int]]:
    """Contiguous groups of `group` layers: [(0,4), (4,8), ...]."""
    if group <= 0 or len(layer_types) % group != 0:
        # fall back to single-layer grouping when layout is not a multiple
        return [(i, i + 1) for i in range(len(layer_types))]
    return [(i, i + group) for i in range(0, len(layer_types), group)]


def candidate_intervals(layer_types: tuple[str, ...]) -> list[LayerInterval]:
    """Standard comparison set for a given layer-type layout.

    early / middle / output-side = one natural group each;
    full = the whole decoder (control; equivalent to full-decoder recurrence).
    """
    n = len(layer_types)
    groups = natural_groups(layer_types)
    if not groups:
        return []
    out: list[LayerInterval] = []
    seen: set[tuple[int, int]] = set()

    def add(lo: int, hi: int, name: str) -> None:
        if (lo, hi) not in seen and lo < hi <= n:
            seen.add((lo, hi))
            out.append(LayerInterval(lo, hi, name))

    g_first = groups[0]
    g_mid = groups[len(groups) // 2]
    g_last = groups[-1]
    add(g_first[0], g_first[1], "early")
    add(g_mid[0], g_mid[1], "middle")
    add(g_last[0], g_last[1], "output_side")
    add(0, n, "full_decoder_control")
    return out


def is_contiguous(lo: int, hi: int, n_layers: int) -> bool:
    return 0 <= lo < hi <= n_layers
