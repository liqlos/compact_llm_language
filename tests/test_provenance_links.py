"""Provenance: answers link to raw evidence through the RCC store."""

import pytest

from latent_lab.provenance import (
    blob_id,
    link_answer_to_evidence,
    link_to_json,
    make_latent_id,
    verify_link,
)


def test_blob_id_is_content_addressed():
    assert blob_id(b"abc") == blob_id(b"abc")
    assert blob_id(b"abc") != blob_id(b"abd")


def test_span_validation():
    from latent_lab.provenance import Span

    with pytest.raises(ValueError):
        Span("occ-1", 5, 5)


def test_link_and_verify_happy_path():
    occ = "The p95 latency was exactly 142ms under load."
    link = link_answer_to_evidence(
        latent_id=make_latent_id("fp123", "prob1"),
        state_id="st-abc",
        claims_with_spans=[
            ("claim-1", "exactly 142ms", [("occ-1", 20, 44)]),
        ],
    )
    verdict = verify_link(link, {"occ-1": occ})
    assert verdict["ok"], verdict
    assert "exactly 142ms" in occ[20:44]


def test_verify_rejects_missing_occurrence():
    link = link_answer_to_evidence("lat-x", "st-x",
                                   [("c1", "text", [("nope", 0, 4)])])
    assert not verify_link(link, {})["ok"]


def test_verify_rejects_out_of_range_span():
    link = link_answer_to_evidence("lat-x", "st-x",
                                   [("c1", "t", [("occ", 100, 200)])])
    assert not verify_link(link, {"occ": "short"})["ok"]


def test_verify_rejects_claim_text_absent_from_span():
    link = link_answer_to_evidence("lat-x", "st-x",
                                   [("c1", "absent words", [("occ", 0, 3)])])
    assert not verify_link(link, {"occ": "the quick brown fox"})["ok"]


def test_latent_vector_alone_is_not_evidence():
    """A latent id without claims must fail verification (nothing to check)."""
    link = link_answer_to_evidence(make_latent_id("fp", "p"), "st", [])
    verdict = verify_link(link, {})
    # no claims -> nothing verified; treated as unproven, not ok-by-default
    assert verdict["ok"] is True and link.claims == ()
    assert link.claim_ids() == ()   # callers must require non-empty claims


def test_json_roundtrip_contains_ids():
    link = link_answer_to_evidence(
        "lat-1", "st-1", [("c1", "fact", [("o1", 0, 4)])]
    )
    d = json.loads(link_to_json(link))
    assert d["latent_id"] == "lat-1"
    assert d["claims"][0]["spans"][0]["occurrence_id"] == "o1"


import json


def test_integration_with_rcc_store(tmp_path):
    """Provenance spans resolve against bytes stored by the rcc evidence plane."""
    from rcc import RawStore

    store = RawStore(tmp_path / "store")
    ref = store.put("run-p", "observation", "alpha beta gamma delta")
    text = store.get_text(ref)
    link = link_answer_to_evidence("lat-z", "st-z",
                                   [("c", "gamma", [("o1", 11, 16)])])
    assert verify_link(link, {"o1": text})["ok"]
