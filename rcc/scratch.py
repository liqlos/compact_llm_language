"""RIR/1 — symbolic machine-state scratch layer.

Compact typed records of semantic working state, rendered at the END of
compiled context so appends never invalidate the prompt prefix:

    <SCRATCH format=RIR/1 atoms=3>
    F f01 p95 latency 142ms under LT-77 @obs-0001
    Q q01 writer lock ordering? @obs-0002
    N n01 verify q01 via simulation
    </SCRATCH>

Exactness guard (Telegraph English fine-facts lesson): numbers are what lossy
rewriting destroys, so an atom may PARAPHRASE prose but every normalized
numeric token it contains must equal a complete signed decimal/exponent token
in referenced source observations. Substring evidence is forbidden. Sources
resolve through the hash-verified store, so validation still works after the
observation is masked out of active context.
"""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_NUM_RE = re.compile(
    r"(?<!\d)[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+-]?\d+)?(?!\d)"
)
_AID_RE = re.compile(r"^([fqnc])(\d{2,})$")
_UNSAFE_RENDER_RE = re.compile(r"[\x00-\x1f\x7f@<>]|conf=|b64json=")
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


def extract_numeric_tokens(text: str) -> tuple[str, ...]:
    """Extract signed decimal/exponent tokens at exact digit boundaries."""
    return tuple(match.group(0) for match in _NUM_RE.finditer(text))


def _normalized_number(token: str) -> Decimal:
    try:
        return Decimal(token.replace(",", "."))
    except InvalidOperation as exc:  # defensive: the regex should preclude this
        raise ScratchError(f"invalid numeric token {token!r}") from exc


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
        if not isinstance(text, str):
            raise ScratchError("atom text must be a string")
        self._validate_confidence(conf)
        self._validate_numbers(text, src)
        self._counter[kind] = self._counter.get(kind, 0) + 1
        aid = f"{kind.lower()}{self._counter[kind]:02d}"
        self._atoms.append(Atom(aid=aid, kind=kind, text=text, src=tuple(src), conf=conf))
        if getattr(self._session, "journal", None) is not None:
            self._session.journal.log_atom(aid, kind, text, list(src), conf)
        return aid

    @staticmethod
    def _validate_confidence(conf: float | None) -> None:
        if conf is None:
            return
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            raise ScratchError("confidence must be a finite number in [0, 1]")
        if not math.isfinite(float(conf)) or not 0.0 <= float(conf) <= 1.0:
            raise ScratchError("confidence must be a finite number in [0, 1]")

    def _validate_numbers(self, text: str, src: tuple[str, ...]) -> None:
        if not isinstance(src, tuple) or not all(isinstance(obs_id, str) for obs_id in src):
            raise ScratchError("source provenance must be a tuple of observation ids")
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
        numbers = extract_numeric_tokens(text)
        if not numbers:
            return
        if not src:
            raise ScratchError(
                f"numeric atom without source provenance: {sorted(numbers)!r}"
            )
        corpus = "\n".join(corpus_parts)
        corpus_numbers = {
            _normalized_number(token) for token in extract_numeric_tokens(corpus)
        }
        missing = sorted({
            token for token in numbers
            if _normalized_number(token) not in corpus_numbers
        })
        if missing:
            raise ScratchError(
                f"numbers not verbatim-equivalent in source numeric tokens: {missing} "
                "(substring matches are forbidden)"
            )

    def _restore_atom(self, atom: Atom) -> None:
        """Validate and append one atom from persisted state."""
        if atom.kind not in _KINDS:
            raise ScratchError(f"unknown atom kind {atom.kind!r}")
        match = _AID_RE.fullmatch(atom.aid)
        if match is None or match.group(1) != atom.kind.lower():
            raise ScratchError(f"invalid atom id {atom.aid!r} for kind {atom.kind!r}")
        if any(existing.aid == atom.aid for existing in self._atoms):
            raise ScratchError(f"duplicate atom id {atom.aid!r}")
        if not isinstance(atom.text, str):
            raise ScratchError("atom text must be a string")
        self._validate_confidence(atom.conf)
        self._validate_numbers(atom.text, atom.src)
        self._atoms.append(atom)
        self._counter[atom.kind] = max(
            self._counter.get(atom.kind, 0), int(match.group(2))
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
            if _UNSAFE_RENDER_RE.search(a.text):
                payload = json.dumps(
                    {
                        "text": a.text,
                        "conf": a.conf,
                        "src": list(a.src),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                encoded = base64.urlsafe_b64encode(payload).decode("ascii")
                line = f"{a.kind} {a.aid} b64json={encoded}"
            else:
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
