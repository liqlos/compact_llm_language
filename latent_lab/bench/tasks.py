"""Deterministic task families with exact scorers.

These exercise whether a configuration actually USES its latent state:
multi-hop chains cannot be answered from the problem text alone — the
intermediate hop must be carried through machine state. Scorers are exact,
not model-judged.

Anti-cheat hooks live in runner.py (zero-state, shuffled steps, swapped
states, truncated depth).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TaskInstance:
    task_id: str
    family: str
    question: str
    facts: tuple[tuple[str, str], ...]   # (key, value) evidence pairs
    expected: str

    def payload_seed(self) -> bytes:
        h = hashlib.sha256()
        h.update(self.task_id.encode())
        for k, v in self.facts:
            h.update(k.encode())
            h.update(v.encode())
        return h.digest()


def _vec_from_seed(seed: bytes, dim: int = 8) -> tuple[float, ...]:
    vals = [b / 255.0 for b in seed[:dim]]
    while len(vals) < dim:
        seed = hashlib.sha256(seed).digest()
        vals.extend(b / 255.0 for b in seed)
    return tuple(vals[:dim])


class Scorer(Protocol):
    def __call__(self, answer: str, task: TaskInstance) -> float: ...


def exact_match(answer: str, task: TaskInstance) -> float:
    return 1.0 if answer.strip() == task.expected else 0.0


class MultiHopChain:
    """Builds A->B->C lookup chains; expected answer is the final hop value."""

    FAMILY = "multi_hop_chain"

    @staticmethod
    def make(task_id: str, chain: tuple[str, ...], values: dict[str, str]) -> TaskInstance:
        if len(chain) < 2:
            raise ValueError("chain needs >=2 hops")
        facts = tuple((chain[i], values[chain[i]]) for i in range(len(chain) - 1))
        return TaskInstance(
            task_id=task_id,
            family=MultiHopChain.FAMILY,
            question=f"Follow {chain[0]} to its final value.",
            facts=facts,
            expected=values[chain[-1]],
        )

    @staticmethod
    def to_payload(task: TaskInstance) -> tuple:
        """Encode facts as vectors keyed by hash — mock backend input."""
        return tuple(
            _vec_from_seed(hashlib.sha256(f"{k}={v}".encode()).digest())
            for k, v in task.facts
        )


TASKS: list[TaskInstance] = [
    MultiHopChain.make("mh01", ("a", "b", "c"), {"a": "has b", "b": "has c", "c": "42"}),
    MultiHopChain.make("mh02", ("x", "y", "z"), {"x": "owns y", "y": "owns z", "z": "flag"}),
]
