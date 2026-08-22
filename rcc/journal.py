"""Append-only session journal: immutable checkpoints + replayed deltas.

Design (per master plan Phase 4):
- Every ingestion event is appended as one JSONL line: say / observe / atom.
- `checkpoint()` writes the full exported state as a special line.
- `restore()` replays: latest checkpoint (optionally <= upto_seq) + subsequent
  events. Nothing is ever rewritten in place -> auditable, rollback-safe,
  prefix-friendly.
- `compact()` writes a fresh checkpoint and truncates the log to it
  (atomic tmp+replace), bounding replay time without losing history semantics.

Single-writer assumption; appends are flushed per event.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .session import Policy, ResearchSession
from .store import RawStore
from .tokens import count_tokens


class JournalError(Exception):
    pass


@dataclass(frozen=True)
class JournalEntry:
    seq: int
    etype: str            # say | observe | atom | checkpoint
    payload: dict


class Journal:
    def __init__(self, path: Path | str, session: ResearchSession):
        self.path = Path(path)
        self.session = session
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.entries():
            from dataclasses import asdict

            self._append("meta", {
                "run_id": session.run_id,
                "policy": asdict(session.policy),
            })

    # ---- writing -------------------------------------------------------

    def _append(self, etype: str, payload: dict) -> int:
        entry = JournalEntry(seq=self.session._seq, etype=etype, payload=payload)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"seq": entry.seq, "type": etype, "payload": payload}) + "\n")
        return entry.seq

    def log_say(self, role: str, content: str, protected: bool) -> None:
        self._append("say", {"role": role, "content": content, "protected": protected})

    def log_observe(self, label: str, content: str, obs_id: str) -> None:
        self._append("observe", {"label": label, "content": content, "obs_id": obs_id})

    def log_atom(self, aid: str, kind: str, text: str, src: list[str], conf: float | None) -> None:
        self._append("atom", {"aid": aid, "kind": kind, "text": text,
                              "src": src, "conf": conf})

    def checkpoint(self) -> int:
        """Persist full current state; returns the checkpoint line's seq."""
        return self._append("checkpoint", self.session._export_state())

    def compact(self) -> None:
        """Fold history into meta + one checkpoint line (atomic rewrite)."""
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            from dataclasses import asdict

            f.write(json.dumps({
                "seq": 0,
                "type": "meta",
                "payload": {
                    "run_id": self.session.run_id,
                    "policy": asdict(self.session.policy),
                },
            }) + "\n")
            f.write(json.dumps({
                "seq": self.session._seq,
                "type": "checkpoint",
                "payload": self.session._export_state(),
            }) + "\n")
        os.replace(tmp, self.path)

    # ---- reading ---------------------------------------------------------

    def entries(self) -> list[JournalEntry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(JournalEntry(seq=d["seq"], etype=d["type"], payload=d["payload"]))
        return out

    def restore(
        self,
        store: RawStore,
        *,
        upto_seq: int | None = None,
        tokenizer=None,
    ) -> ResearchSession:
        """Rebuild a session: meta header, latest checkpoint (<= upto_seq) if
        any, then replayed deltas. Delta-only journals replay from scratch."""
        entries = self.entries()
        if not entries or entries[0].etype != "meta":
            raise JournalError("journal missing meta header")
        meta = entries[0].payload
        limit = upto_seq if upto_seq is not None else entries[-1].seq

        base_idx = 0  # meta line -> fresh base
        for i, e in enumerate(entries):
            if e.etype == "checkpoint" and e.seq <= limit:
                base_idx = i

        if base_idx == 0:
            s = ResearchSession(
                run_id=meta["run_id"], store=store,
                policy=Policy(**meta["policy"]), tokenizer=tokenizer or count_tokens,
            )
        else:
            s = ResearchSession._from_state(entries[base_idx].payload, store, tokenizer)

        for e in entries[base_idx + 1:]:
            if e.seq > limit:
                break
            if e.etype == "say":
                s.say(e.payload["role"], e.payload["content"], protected=e.payload["protected"])
            elif e.etype == "observe":
                oid = s.observe(e.payload["label"], e.payload["content"])
                if oid != e.payload["obs_id"]:
                    raise JournalError(
                        f"journal replay diverged: expected {e.payload['obs_id']!r}, got {oid!r}"
                    )
            elif e.etype == "atom":
                s.attach_scratch().add(
                    e.payload["kind"], e.payload["text"],
                    src=tuple(e.payload["src"]), conf=e.payload.get("conf"),
                )
            elif e.etype == "checkpoint":
                s = ResearchSession._from_state(e.payload, store, tokenizer or count_tokens)
            elif e.etype == "meta":
                continue
            else:
                raise JournalError(f"unknown event type {e.etype!r}")
        return s
