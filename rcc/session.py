"""ResearchSession: compiles a growing research transcript into active context.

Channels:
- trusted messages (system/user/assistant) -- always inline; `protected=True`
  items (constraints, policies) are never masked.
- tool observations / documents -- untrusted; bulky old ones are MASKed into
  stable reference stubs after being persisted to the immutable store. The
  original remains deterministically recoverable via expand().

Fail-open guarantee for availability: an observation is only masked when its
content is present AND hash-verifiable in the store. If the store is missing
or corrupt at compile time the verbatim text stays inline instead.

Security invariants (tested):
- untrusted observations never become instructions: expand() re-wraps them in
  explicit untrusted markers;
- cross-run references are rejected;
- protected channel survives every compile unchanged and unmasked.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .dictenc import encode_json_objects
from .gate import Usage, break_even
from .router import TaskSignals, route
from .scratch import Atom, Scratch
from .store import RawStore, StoredRef, StoreError
from .tokens import TOKENIZER_ID, count_tokens

OBS_STUB = "[OBS {obs_id} label={label} sha={sha} tok={tok}]"  # compact machine ref

_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,63}$")

# Untrusted bodies must never close their own wrapper or open a trusted-looking
# section. Rendered views neutralize these pseudo-tag prefixes; raw bytes stay
# untouched in the store (storage-lossless is unaffected).
_WRAPPER_TAGS = (
    "OBSERVATION",
    "UNTRUSTED_OBSERVATION",
    "CONSTRAINT",
    "SCRATCH",
    "SYSTEM",
    "USER",
    "ASSISTANT",
    "TOOL",
)
_PSEUDO_TAG_RE = re.compile(
    r"<(/?(?:" + "|".join(_WRAPPER_TAGS) + r")\b)", re.IGNORECASE
)


def _sanitize_untrusted(text: str) -> str:
    return _PSEUDO_TAG_RE.sub(lambda m: "&lt;" + m.group(1), text)


class SessionError(Exception):
    pass


class CrossRunReferenceError(SessionError):
    pass


@dataclass(frozen=True)
class Policy:
    """Compaction policy. `enabled=False` reproduces baseline behaviour."""

    enabled: bool = True          # feature flag for the whole compiler
    keep_recent: int = 4          # most recent distinct observations stay inline
    min_mask_tokens: int = 150    # smaller observations always stay inline
    mask_duplicates: bool = True  # repeat occurrences become stubs immediately
    router_enabled: bool = False  # Phase 5.2: deterministic scratch-mode router
    gate: str = "window"          # Phase 7: "window" (default) | "breakeven"
    expected_reads_per_turn: float = 0.05   # q for break-even gate
    horizon_turns: int = 20       # N estimate for break-even gate
    encode_jsonl: bool = False    # Phase 6: lossless JSON-object dictionary encoding


@dataclass
class MessageItem:
    role: str            # system | user | assistant | tool
    content: str
    protected: bool = False   # never masked; part of the protected exact channel


@dataclass
class ObservationItem:
    obs_id: str          # stable within run: "obs-0007"
    ref: StoredRef
    label: str
    n_tokens: int


@dataclass
class Turn:
    item: MessageItem | ObservationItem
    seq: int


@dataclass
class CompileMetrics:
    tokenizer: str = TOKENIZER_ID
    total_tokens: int = 0
    inline_observation_tokens: int = 0
    stub_observation_tokens: int = 0
    message_tokens: int = 0
    scratch_tokens: int = 0
    scratch_mode: str | None = None
    jsonl_encoded: int = 0
    observations_masked: int = 0
    duplicates_stubbed: int = 0
    failopen_inline: int = 0     # would have been masked but kept for safety


@dataclass
class CompiledContext:
    text: str
    metrics: CompileMetrics


@dataclass
class ResearchSession:
    run_id: str
    store: RawStore
    policy: Policy = field(default_factory=Policy)
    # Token counter used for all budget decisions. Defaults to the
    # deterministic approximation; benchmarks may inject an exact tokenizer.
    tokenizer: Callable[[str], int] = field(default=count_tokens)

    def __post_init__(self) -> None:
        from .store import _check_component

        _check_component(self.run_id, "run_id")  # run_id becomes a directory name
        self._seq = 0
        self._obs_counter = 0
        self._turns: list[Turn] = []
        self._by_id: dict[str, ObservationItem] = {}
        self._by_sha: dict[str, ObservationItem] = {}
        self.scratch: Scratch | None = None  # attached lazily via attach_scratch()
        self.journal = None                  # attached via attach_journal()
        self.recent_failures = 0             # explicit signal for the router
        self.injection_suspected = False     # explicit signal for the router

    def note_failure(self) -> None:
        self.recent_failures += 1

    def flag_injection(self) -> None:
        self.injection_suspected = True

    def attach_journal(self, path: Path | str):
        """Bind an append-only checkpoint+delta journal to this session."""
        from .journal import Journal

        if self.journal is None:
            self.journal = Journal(path, self)
        return self.journal

    def attach_scratch(self) -> Scratch:
        """Bind a symbolic machine-state (RIR/1) scratchpad to this session."""
        if self.scratch is None:
            from .scratch import Scratch

            self.scratch = Scratch(self)
        return self.scratch

    # ---- ingestion -------------------------------------------------

    def say(self, role: str, content: str, *, protected: bool = False) -> None:
        if role not in _ALLOWED_ROLES:
            raise SessionError(
                f"unsafe role {role!r}: must be one of {sorted(_ALLOWED_ROLES)}"
            )
        self._turns.append(Turn(MessageItem(role=role, content=content, protected=protected), self._seq))
        self._seq += 1
        if self.journal is not None:
            self.journal.log_say(role, content, protected)

    def observe(self, label: str, content: str) -> str:
        """Register a tool/document observation.

        Identical content maps to ONE canonical stable obs_id (dedup by
        content hash); repeat occurrences append further turns referencing it.
        Labels render inside stubs/wrappers, so they are restricted to a safe
        charset (no whitespace, brackets or angle brackets).
        """
        if not isinstance(label, str) or not _SAFE_LABEL_RE.match(label):
            raise SessionError(
                f"unsafe observation label {label!r}: must match "
                "[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,63}"
            )
        ref = self.store.put(self.run_id, "observation", content, meta={"label": label})
        item = self._by_sha.get(ref.sha256)
        if item is None:
            self._obs_counter += 1
            obs_id = f"obs-{self._obs_counter:04d}"
            item = ObservationItem(
                obs_id=obs_id,
                ref=ref,
                label=label,
                n_tokens=self.tokenizer(content),
            )
            self._by_sha[ref.sha256] = item
        self._by_id[item.obs_id] = item
        self._turns.append(Turn(item, self._seq))
        self._seq += 1
        if self.journal is not None:
            self.journal.log_observe(label, content, item.obs_id)
        return item.obs_id

    # ---- compilation -----------------------------------------------

    def _distinct_recent_ids(self) -> list[str]:
        seen: list[str] = []
        for t in reversed(self._turns):
            if isinstance(t.item, ObservationItem) and t.item.obs_id not in seen:
                seen.append(t.item.obs_id)
                if len(seen) >= self.policy.keep_recent:
                    break
        return seen

    def _stub(self, it: ObservationItem) -> str:
        return OBS_STUB.format(
            obs_id=it.obs_id, label=it.label, sha=it.ref.short_sha, tok=it.n_tokens
        )

    def _mask_candidate(self, it: ObservationItem, dup_occurrence: bool,
                        recent: list[str], seq: int = 0) -> bool:
        """Policy decision only -- does not consider store availability."""
        if not self.policy.enabled or it.ref.size_bytes == 0:
            return False
        if it.n_tokens < self.policy.min_mask_tokens:
            return False  # stub would cost nearly as much as the content
        if dup_occurrence and self.policy.mask_duplicates:
            passes_window = True
        elif it.obs_id in recent:
            passes_window = False
        else:
            passes_window = True
        if not passes_window:
            return False
        if self.policy.gate == "breakeven":
            decision = break_even(Usage(
                content_tokens=it.n_tokens,
                stub_tokens=self.tokenizer(self._stub(it)),
                remaining_turns=max(1, self.policy.horizon_turns - seq),
                expected_reads_per_turn=self.policy.expected_reads_per_turn,
            ))
            return decision.mask
        return True

    def _safe_to_mask(self, it: ObservationItem) -> bool:
        try:
            self.store.get_text(it.ref)
            return True
        except StoreError:
            return False

    def compile(self) -> CompiledContext:
        m = CompileMetrics()
        mode = None
        if self.policy.router_enabled:
            mode = route(TaskSignals(
                n_observations=len(self._by_id),
                n_atoms=len(self.scratch) if self.scratch else 0,
                n_conflicts=(sum(1 for a in self.scratch.atoms() if a.kind == "C")
                             if self.scratch else 0),
                recent_failures=self.recent_failures,
                injection_suspected=self.injection_suspected,
            ))
        m.scratch_mode = mode
        recent = self._distinct_recent_ids()
        seen_once: set[str] = set()
        parts: list[str] = []

        for t in self._turns:
            if isinstance(t.item, MessageItem):
                tag = "CONSTRAINT" if t.item.protected else t.item.role.upper()
                parts.append(f"<{tag}>\n{t.item.content}\n</{tag}>")
                m.message_tokens += self.tokenizer(t.item.content) + 2
                continue

            it: ObservationItem = t.item
            dup_occurrence = it.obs_id in seen_once
            seen_once.add(it.obs_id)

            want_mask = self._mask_candidate(it, dup_occurrence, recent, seq=t.seq)
            if want_mask and not self._safe_to_mask(it):
                # fail-open: keep verbatim rather than emit an unrecoverable stub
                m.failopen_inline += 1
                want_mask = False

            if not want_mask:
                try:
                    body = self.store.get_text(it.ref)
                except StoreError:
                    # evidence unavailable at assembly time: keep a visible,
                    # honest placeholder instead of crashing or inventing content
                    parts.append(
                        f"<OBSERVATION_UNAVAILABLE id={it.obs_id} label={it.label} "
                        f"sha={it.ref.short_sha} />"
                    )
                    m.failopen_inline += 1
                    continue
                if self.policy.encode_jsonl:
                    try:
                        enc = encode_json_objects(body)
                    except Exception:  # noqa: BLE001 -- fail-open: keep verbatim
                        enc = None
                    if enc is not None:
                        body = enc[0]
                        m.jsonl_encoded += 1
                parts.append(
                    f"<OBSERVATION id={it.obs_id} label={it.label} sha={it.ref.short_sha}>\n"
                    + _sanitize_untrusted(body)
                    + "\n</OBSERVATION>"
                )
                m.inline_observation_tokens += self.tokenizer(body) + 6
                continue

            parts.append(self._stub(it))
            stub_tok = self.tokenizer(parts[-1])
            m.stub_observation_tokens += stub_tok
            if dup_occurrence:
                m.duplicates_stubbed += 1
            else:
                m.observations_masked += 1

        # Machine state (RIR/1) renders LAST: appends preserve the prefix.
        # DIRECT suppresses visibility only -- atoms remain in state/journal.
        if self.scratch is not None and mode != "DIRECT":
            block = self.scratch.render()
            if block:
                parts.append(block)
                m.scratch_tokens += self.tokenizer(block)

        m.total_tokens = (
            m.message_tokens
            + m.inline_observation_tokens
            + m.stub_observation_tokens
            + m.scratch_tokens
        )
        return CompiledContext(text="\n".join(parts), metrics=m)

    # ---- observability ------------------------------------------------

    def timeline(self) -> list[dict]:
        """Human-auditable per-turn record: what is inline, what was masked.

        This is the minimal live observability surface: a caller (or human)
        can inspect what the model will see at each step without waiting for
        the end of the run.
        """
        recent = self._distinct_recent_ids()
        seen_once: set[str] = set()
        out: list[dict] = []
        for t in self._turns:
            if isinstance(t.item, MessageItem):
                out.append({
                    "seq": t.seq, "kind": "message", "role": t.item.role,
                    "protected": t.item.protected,
                    "tokens": self.tokenizer(t.item.content),
                })
                continue
            it = t.item
            dup = it.obs_id in seen_once
            seen_once.add(it.obs_id)
            want_mask = self._mask_candidate(it, dup, recent, seq=t.seq)
            if want_mask and not self._safe_to_mask(it):
                action = "failopen_inline"
            elif want_mask:
                action = "masked_stub"
            else:
                action = "inline"
            out.append({
                "seq": t.seq, "kind": "observation", "obs_id": it.obs_id,
                "label": it.label, "action": action, "tokens": it.n_tokens,
                "duplicate_occurrence": dup,
            })
        return out

    # ---- expansion --------------------------------------------------

    def expand(self, obs_id: str) -> str:
        """Deterministically recover original observation content.

        Untrusted content is re-wrapped so it can never merge into the
        instruction channel.
        """
        if obs_id.count("-") != 1 or not obs_id.startswith("obs-"):
            raise SessionError(f"malformed reference {obs_id!r}")
        it = self._by_id.get(obs_id)
        if it is None:
            raise SessionError(f"unknown reference {obs_id!r}")
        if it.ref.run_id != self.run_id:
            raise CrossRunReferenceError(
                f"reference {obs_id} belongs to run {it.ref.run_id}, session is {self.run_id}"
            )
        content = self.store.get_text(it.ref)  # hash-verified, fail-closed
        return (
            f"<UNTRUSTED_OBSERVATION id={obs_id} label={it.label} "
            f"sha={it.ref.sha256}>\n{_sanitize_untrusted(content)}\n</UNTRUSTED_OBSERVATION>"
        )

    # ---- persistence (resume support) -------------------------------

    def _export_state(self) -> dict:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "tokenizer_id": getattr(self.tokenizer, "__name__", "custom"),
            "policy": asdict(self.policy),
            "seq": self._seq,
            "obs_counter": self._obs_counter,
            "observations": [
                {"obs_id": i.obs_id, "ref_json": i.ref.to_json(), "label": i.label}
                for i in self._by_id.values()
            ],
            "turns": [
                {"kind": "message", "role": t.item.role, "content": t.item.content,
                 "protected": t.item.protected, "seq": t.seq}
                if isinstance(t.item, MessageItem)
                else {"kind": "observation", "obs_id": t.item.obs_id, "seq": t.seq}
                for t in self._turns
            ],
            "scratch": [
                {"aid": a.aid, "kind": a.kind, "text": a.text,
                 "src": list(a.src), "conf": a.conf}
                for a in (self.scratch.atoms() if self.scratch else [])
            ],
        }

    def save(self, path: Path | str) -> None:
        payload = self._export_state()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, p)

    @classmethod
    def _from_state(
        cls,
        d: dict,
        store: RawStore,
        tokenizer: Callable[[str], int] | None = None,
    ) -> ResearchSession:
        if d.get("schema_version") != 1:
            raise SessionError(f"unsupported schema_version {d.get('schema_version')!r}")
        s = cls(run_id=d["run_id"], store=store, policy=Policy(**d["policy"]),
                tokenizer=tokenizer or count_tokens)
        s._seq = d["seq"]
        s._obs_counter = d["obs_counter"]
        for o in d["observations"]:
            ref = StoredRef.from_json(o["ref_json"])
            n_tokens = s.tokenizer(store.get_text(ref)) if store.has(ref) else 0
            item = ObservationItem(
                obs_id=o["obs_id"], ref=ref, label=o["label"], n_tokens=n_tokens,
            )
            s._by_id[o["obs_id"]] = item
            s._by_sha[ref.sha256] = item  # dedup index must survive resume
        for t in d["turns"]:
            if t["kind"] == "message":
                s._turns.append(Turn(
                    MessageItem(role=t["role"], content=t["content"], protected=t["protected"]),
                    t["seq"]))
            else:
                s._turns.append(Turn(s._by_id[t["obs_id"]], t["seq"]))
        if d.get("scratch"):
            sc = s.attach_scratch()
            for a in d["scratch"]:
                # validation ran at add()-time; store is immutable so it holds
                sc._atoms.append(Atom(
                    aid=a["aid"], kind=a["kind"], text=a["text"],
                    src=tuple(a["src"]), conf=a.get("conf"),
                ))
                sc._counter[a["kind"]] = max(sc._counter.get(a["kind"], 0),
                                             int(a["aid"][2:] or 0))
        return s

    @classmethod
    def load(
        cls,
        path: Path | str,
        store: RawStore,
        tokenizer: Callable[[str], int] | None = None,
    ) -> ResearchSession:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls._from_state(d, store, tokenizer)
