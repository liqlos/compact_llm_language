"""Property tests grounded in the context-compression literature.

Properties and their sources:
- Stub stability across turns (no re-summarization drift):
  AgentFold (arXiv:2510.24699) shows 1% loss per re-summarization compounds to
  ~37% survival at 100 steps; RCC never rewrites a masked reference.
- Negation / fine-fact survival:
  Telegraph English (arXiv:2605.04426) "fine_facts" -- numbers with units,
  conditions and negations are what lossy compression destroys. RCC must keep
  them byte-exact through mask->expand roundtrips.
- No referential dangling by construction:
  arXiv:2608.04569 shows hard compressors split dependent evidence in 34-60%
  of cases. RCC masks whole observations atomically and expands to the full
  original bytes, so dependencies inside a document can never be severed.
"""

import pytest

from rcc import Policy, RawStore, ResearchSession

BIG = "z" * 1200

FINE_FACTS = (
    "Result: treatment reduced false positives by 4.8%, NOT by 4.3%. "
    "Measured on 2024-05-09 under load test LT-77, timeout 30000ms, "
    "schema v9.2.1, p95 latency was NOT above 150ms but exactly 142ms."
)


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "store")


def _session_with_old_masked_obs(store, content):
    s = ResearchSession("run-prop", store, Policy())
    oid = s.observe("evidence", content)
    for i in range(5):  # push obs out of the recent window
        s.observe(f"later_{i}", BIG + str(i))
    return s, oid


def test_stub_never_rewritten_across_turns(store):
    """Once masked, a stub is byte-stable in all later compiles (cache-friendly
    prompt prefix; anti-drift property from AgentFold's analysis)."""
    s, oid = _session_with_old_masked_obs(store, BIG)
    c1 = s.compile()
    stub_v1 = next(line for line in c1.text.splitlines() if f"[OBS {oid}" in line)
    snapshots = [stub_v1]
    for i in range(5):
        s.observe(f"more_{i}", BIG + chr(ord("a") + i))
        s.say("assistant", f"step {i}")
        ci = s.compile()
        stub_vi = next(line for line in ci.text.splitlines() if f"[OBS {oid}" in line)
        snapshots.append(stub_vi)
    assert len(set(snapshots)) == 1  # identical across 5 subsequent turns


def test_negation_and_fine_facts_survive_mask_expand_cycle(store):
    s, oid = _session_with_old_masked_obs(store, FINE_FACTS + "\n" + BIG)
    expanded = s.expand(oid)
    for frag in [
        "NOT by 4.3%",
        "4.8%",
        "2024-05-09",
        "30000ms",
        "v9.2.1",
        "NOT above 150ms",
        "exactly 142ms",
    ]:
        assert frag in expanded, f"lost fine fact: {frag}"
    # and they are absent from active context only because the WHOLE document
    # is behind one stable reference -- never partially deleted:
    c = s.compile()
    assert "4.8%" not in c.text  # masked whole: no partial rendering anywhere


def test_no_referential_dangling_by_construction(store):
    """Expansion returns the complete original: intra-document dependency
    chains cannot be severed (contrast: 34-60% dangling in hard compressors)."""
    doc = (
        "Tim DuBois was born in Southwest City.\n"
        "Southwest City is in McDonald County.\n"
        + BIG  # bulk that triggers masking
    )
    s, oid = _session_with_old_masked_obs(store, doc)
    assert s.expand(oid) == (
        f"<UNTRUSTED_OBSERVATION id={oid} label=evidence sha="
        + s._by_id[oid].ref.sha256
        + ">\n" + doc + "\n</UNTRUSTED_OBSERVATION>"
    )


def test_near_duplicate_content_gets_distinct_ids(store):
    s = ResearchSession("r", store, Policy())
    a = s.observe("d", BIG + "\nvariant A")
    b = s.observe("d", BIG + "\nvariant B")
    assert a != b  # dedup keys on exact content hash, not similarity


def test_identical_content_single_canonical_id_and_object(store, tmp_path):
    s = ResearchSession("r", store, Policy())
    ids = {s.observe("d", SAME) for SAME in ["payload-x", "payload-x", "payload-x"]}
    files = list((store.root / "runs" / "r").rglob("*.json"))
    assert len(ids) == 1 and len(files) == 1


def test_protected_channel_survives_all_compiles(store):
    s = ResearchSession("r", store, Policy())
    s.say("user", "CONSTRAINT: NEVER run rm -rf on production.", protected=True)
    for i in range(12):
        s.observe(f"d{i}", BIG + str(i))
        c = s.compile()  # compile after every turn, like a live agent loop
        assert "NEVER run rm -rf on production." in c.text
