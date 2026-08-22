"""Proof-carrying provenance: answers must link back to raw evidence.

Identity separation (never conflate):
- blob_id        unique bytes (content-addressed, sha256)
- occurrence_id  one appearance of a blob in a source/time
- span_id        exact range inside an occurrence
- claim_id       normalized assertion derived from evidence
- latent_id      machine representation of an evidence set (workspace state)
- state_id       version of the working model state

A latent vector alone is NOT evidence. Every material final conclusion must
carry a ProvenanceLink whose blob refs resolve in the RawStore.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


def blob_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Span:
    occurrence_id: str
    start: int
    end: int          # exclusive; byte or char offsets by occurrence convention

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"empty span {self}")


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class ProvenanceLink:
    latent_id: str
    state_id: str
    claims: tuple[Claim, ...]
    extra: dict = field(default_factory=dict)

    def claim_ids(self) -> tuple[str, ...]:
        return tuple(c.claim_id for c in self.claims)


def make_latent_id(workspace_fingerprint: str, problem_id: str) -> str:
    h = hashlib.sha256(f"{problem_id}|{workspace_fingerprint}".encode()).hexdigest()
    return "lat-" + h[:16]


def link_answer_to_evidence(
    latent_id: str,
    state_id: str,
    claims_with_spans: list[tuple[str, str, list[tuple[str, int, int]]]],
) -> ProvenanceLink:
    """Build a link from (claim_id, claim_text, [(occurrence_id, start, end)])."""
    claims = tuple(
        Claim(
            claim_id=cid,
            text=text,
            spans=tuple(Span(oid, s, e) for oid, s, e in spans),
        )
        for cid, text, spans in claims_with_spans
    )
    return ProvenanceLink(latent_id=latent_id, state_id=state_id, claims=claims)


def verify_link(link: ProvenanceLink, occurrences: dict[str, str]) -> dict:
    """Check every span resolves against known occurrence texts.

    `occurrences` maps occurrence_id -> exact text. Returns a machine-checkable
    verdict; missing occurrences or out-of-range/absent claim text fail.
    """
    problems: list[str] = []
    for c in link.claims:
        for sp in c.spans:
            occ = occurrences.get(sp.occurrence_id)
            if occ is None:
                problems.append(f"{c.claim_id}: unknown occurrence {sp.occurrence_id}")
                continue
            if sp.end > len(occ):
                problems.append(f"{c.claim_id}: span {sp.start}:{sp.end} out of range")
                continue
            if c.text and c.text not in occ[sp.start:sp.end]:
                problems.append(f"{c.claim_id}: text not found inside cited span")
    return {"ok": not problems, "problems": problems}


def link_to_json(link: ProvenanceLink) -> str:
    return json.dumps({
        "latent_id": link.latent_id,
        "state_id": link.state_id,
        "claims": [
            {"claim_id": c.claim_id, "text": c.text,
             "spans": [{"occurrence_id": s.occurrence_id, "start": s.start,
                        "end": s.end} for s in c.spans]}
            for c in link.claims
        ],
    }, sort_keys=True)
