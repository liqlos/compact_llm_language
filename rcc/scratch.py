"""RIR/1 — symbolic machine-state scratch layer.

Compact typed records of semantic working state, rendered at the END of
compiled context so appends never invalidate the prompt prefix:

    <SCRATCH format=RIR/1 atoms=3>
    F f01 p95 latency 142ms under LT-77 @obs-0001
    Q q01 writer lock ordering? @obs-0002
    N n01 verify q01 via simulation
    </SCRATCH>

Exactness guard (Telegraph English fine-facts lesson): numbers are what lossy
rewriting destroys, so an atom may PARAPHRASE prose but every numeric token it
contains must appear VERBATIM in referenced source observations. Violations
raise at add() time -- fabricated or drifted digits can never enter machine
state. Sources resolve through the hash-verified store, so validation works
even after the observation itself has been masked out of active context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUM_RE = re.compile(r"\d+(?:[.,:\-]\d+)*")
_KINDS = ("F", "Q", "N", "C")


class ScratchError(Exception):
    pass


@dataclass(frozen=True)
class Atom:
    aid: str                  # "f01", "q02", ...
    kind: str                 # F | Q | N | C
    text: str
    src: tuple[str, ...] = ()
    conf: float | None = None


class Scratch:
    """Append-only symbolic state bound to one session."""

    def __init__(self, session):
        self._session = session
        self._atoms: list[Atom] = []
        self._counter: dict[str, int] = {}

    # ---- ingestion ----------------------------------------------------

    def add(self, kind: str, text: str, src: tuple[str, ...] = (),
            conf: float | None = None) -> str:
        if kind not in _KINDS:
            raise ScratchError(f"unknown atom kind {kind!r}")
        self._validate_numbers(text, src)
        self._counter[kind] = self._counter.get(kind, 0) + 1
        aid = f"{kind.lower()}{self._counter[kind]:02d}"
        self._atoms.append(Atom(aid=aid, kind=kind, text=text, src=tuple(src), conf=conf))
        if getattr(self._session, "journal", None) is not None:
            self._session.journal.log_atom(aid, kind, text, list(src), conf)
        return aid

    def _validate_numbers(self, text: str, src: tuple[str, ...]) -> None:
        numbers = set(_NUM_RE.findall(text))
        if not numbers:
            if src == () and any(ch.isdigit() for ch in text):
                pass  # unreachable; kept for clarity
            return
        if not src:
            raise ScratchError(
                f"numeric atom without source provenance: {sorted(numbers)!r}"
            )
        corpus_parts = []
        for obs_id in src:
            item = self._session._by_id.get(obs_id)
            if item is None:
                raise ScratchError(f"unknown source reference {obs_id!r}")
            if item.ref.run_id != self._session.run_id:
                raise ScratchError(
                    f"cross-run source {obs_id!r} ({item.ref.run_id} != {self._session.run_id})"
                )
            corpus_parts.append(self._session.store.get_text(item.ref))  # fail-closed
        corpus = "\n".join(corpus_parts)
        missing = sorted(n for n in numbers if n not in corpus)
        if missing:
            raise ScratchError(
                f"numbers not verbatim in sources: {missing} "
                "(machine state may quote, never rewrite, numbers)"
            )

    # ---- rendering / introspection --------------------------------------

    def render(self) -> str:
        """Header once, then one line per atom. No closing tag: the block is
        always the LAST section of compiled context, so every append is a pure
        suffix addition and the prompt prefix is byte-stable across turns."""
        if not self._atoms:
            return ""
        lines = ["<SCRATCH format=RIR/1>"]
        for a in self._atoms:
            line = f"{a.kind} {a.aid} {a.text}"
            if a.conf is not None:
                line += f" conf={a.conf:.2f}"
            if a.src:
                line += " @" + ",".join(a.src)
            lines.append(line)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._atoms)

    def atoms(self) -> list[Atom]:
        return list(self._atoms)
