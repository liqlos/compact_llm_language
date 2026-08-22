"""Tests for the RIR/1 symbolic scratch layer and its exactness guard."""

import pytest

from rcc import Policy, RawStore, ResearchSession, ScratchError

BIG = "w" * 1200
FACT_DOC = (
    "Release notes: PostgreSQL 16.3 released 2024-05-09. Default port 5432. "
    "p95 latency measured 142ms under load test LT-77, timeout set to 30000ms, "
    "schema version v9.2.1."
) + "\n" + BIG


@pytest.fixture
def session(tmp_path):
    store = RawStore(tmp_path / "s")
    s = ResearchSession("run-rir", store, Policy())
    oid = s.observe("release_notes", FACT_DOC)
    for i in range(5):  # push it out of the recent window (masked)
        s.observe(f"filler_{i}", BIG + str(i))
    return s, oid


def test_atom_ids_deterministic_per_kind(session):
    s, oid = session
    sc = s.attach_scratch()
    assert sc.add("F", "default port is 5432", src=(oid,)) == "f01"
    assert sc.add("F", "timeout set to 30000ms", src=(oid,)) == "f02"
    assert sc.add("Q", "writer lock ordering?") == "q01"
    assert len(sc) == 3


def test_fabricated_number_rejected(session):
    s, _ = session
    sc = s.attach_scratch()
    with pytest.raises(ScratchError, match="not verbatim"):
        sc.add("F", "p95 latency was 143ms", src=("obs-0001",))


def test_altered_decimal_rejected_fine_facts_case(session):
    """The Telegraph-English fine_facts trap: 4.8 vs 4.3 must never drift."""
    s, _ = session
    sc = s.attach_scratch()
    with pytest.raises(ScratchError):
        sc.add("F", "latency exactly 14.2ms", src=("obs-0001",))


def test_verbatim_numbers_accepted(session):
    s, _ = session
    sc = s.attach_scratch()
    aids = [
        sc.add("F", "released 2024-05-09", src=("obs-0001",)),
        sc.add("F", "port 5432", src=("obs-0001",)),
        sc.add("F", "schema version v9.2.1", src=("obs-0001",)),
        # negation over an already-sourced number survives:
        sc.add("F", "p95 latency measured 142ms, NOT above it", src=("obs-0001",)),
    ]
    assert len(set(aids)) == 4


def test_guard_rejects_unsourced_threshold(session):
    """Writing 'not above 150ms' when evidence says 142ms fabricates a
    threshold -- exactly the drift the guard exists to stop."""
    s, _ = session
    sc = s.attach_scratch()
    with pytest.raises(ScratchError):
        sc.add("F", "p95 NOT above 150ms", src=("obs-0001",))


def test_numeric_without_provenance_rejected(session):
    s, _ = session
    sc = s.attach_scratch()
    with pytest.raises(ScratchError, match="provenance"):
        sc.add("F", "something cost 42 tokens")


def test_unknown_and_cross_run_sources_rejected(session, tmp_path):
    s, _ = session
    sc = s.attach_scratch()
    with pytest.raises(ScratchError):
        sc.add("N", "check obs-9999")
    other_store = RawStore(tmp_path / "other")
    other = ResearchSession("run-other", other_store)
    other_oid = other.observe("d", FACT_DOC)
    item = other._by_id[other_oid]
    s._by_id[other_oid] = item  # forge foreign reference
    with pytest.raises(ScratchError, match="cross-run"):
        sc.add("F", "port 5432", src=(other_oid,))


def test_validation_works_after_masking(session):
    """Sources resolve through the store even when masked from active context."""
    s, _ = session
    c = s.compile()
    assert "[OBS obs-0001" in c.text  # masked...
    sc = s.attach_scratch()
    sc.add("F", "p95 latency measured 142ms", src=("obs-0001",))  # ...still verifiable


def test_render_appends_prefix_stable(session):
    s, _ = session
    sc = s.attach_scratch()
    before = s.compile().text
    sc.add("F", "released 2024-05-09", src=("obs-0001",))
    mid = s.compile().text
    sc.add("Q", "lock ordering?")
    after = s.compile().text
    assert mid.startswith(before)          # appends only: prefix never moves
    assert after.startswith(mid)
    assert "<SCRATCH format=RIR/1>" in mid
    assert "</SCRATCH>" not in after       # no closing tag: pure suffix appends
    assert "F f01 released 2024-05-09 @obs-0001" in mid
    assert "Q q01 lock ordering?" in after


def test_metrics_count_scratch_tokens(session):
    s, _ = session
    base = s.compile()
    assert base.metrics.scratch_tokens == 0
    sc = s.attach_scratch()
    sc.add("F", "port 5432", src=("obs-0001",))
    c = s.compile()
    assert c.metrics.scratch_tokens > 0
    assert c.metrics.total_tokens > base.metrics.total_tokens


def test_save_load_roundtrip_preserves_atoms(session, tmp_path):
    s, _ = session
    sc = s.attach_scratch()
    sc.add("F", "port 5432", src=("obs-0001",), conf=0.98)
    sc.add("N", "verify lock order via simulation")
    state = tmp_path / "state.json"
    s.save(state)

    s2 = ResearchSession.load(state, s.store)
    assert len(s2.scratch) == 2
    assert s2.scratch.render() == sc.render()


def test_empty_scratch_not_rendered(session):
    s, _ = session
    s.attach_scratch()  # attached but no atoms yet
    c = s.compile()
    assert "SCRATCH" not in c.text
