"""Checkpoint/delta journal tests: resume, rollback, compaction."""

import pytest

from rcc import Policy, RawStore, ResearchSession
from rcc.journal import JournalError

BIG = "q" * 1200


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "s")


def _build(session):
    oid = session.observe("notes", "port 5432\n" + BIG)
    for i in range(5):
        session.observe(f"filler_{i}", BIG + str(i))
    session.say("user", "summarize")
    sc = session.attach_scratch()
    sc.add("F", "port is 5432", src=(oid,))
    return oid


def test_restore_equals_uninterrupted_run(store, tmp_path):
    a = ResearchSession("run-j", store, Policy())
    _build(a)
    expected = a.compile()

    b = ResearchSession("run-j", store, Policy())
    j = b.attach_journal(tmp_path / "journal.jsonl")
    _build(b)
    restored = j.restore(store)

    assert restored.compile().text == expected.text
    assert restored.compile().metrics.total_tokens == expected.metrics.total_tokens
    assert len(restored.scratch) == 1
    assert restored.expand("obs-0001").startswith("<UNTRUSTED_OBSERVATION id=obs-0001")


def test_rollback_to_checkpoint(store, tmp_path):
    s = ResearchSession("run-r", store, Policy())
    j = s.attach_journal(tmp_path / "j.jsonl")
    s.observe("d0", BIG + " zero")
    ckpt_seq = j.checkpoint()
    mid_state = s.compile()
    # divergence after checkpoint
    s.observe("d1", BIG + " one")
    s.say("assistant", "wrong path")
    late_state = s.compile()

    rolled = j.restore(store, upto_seq=ckpt_seq)
    assert rolled.compile().text == mid_state.text
    assert rolled.compile().text != late_state.text
    assert len(rolled._turns) == 1


def test_compact_folds_history_and_still_restores(store, tmp_path):
    s = ResearchSession("run-c", store, Policy())
    j = s.attach_journal(tmp_path / "j.jsonl")
    _build(s)
    n_before = len(j.entries())
    assert n_before >= 7

    reference = s.compile()
    j.compact()
    assert len(j.entries()) == 2
    assert j.entries()[-1].etype == "checkpoint"

    restored = j.restore(store)
    assert restored.compile().text == reference.text


def test_replay_divergence_detected(store, tmp_path):
    """Deterministic observe() must reproduce identical IDs; tampering fails."""
    s = ResearchSession("run-d", store, Policy())
    j = s.attach_journal(tmp_path / "j.jsonl")
    s.observe("a", BIG)
    entries = j.entries()
    entries[-1].payload["obs_id"] = "obs-4242"  # forged log
    j.entries = lambda: entries  # type: ignore[method-assign]
    with pytest.raises(AssertionError):
        j.restore(store)


def test_tampered_header_rejected(store, tmp_path):
    j = ResearchSession("x", store).attach_journal(tmp_path / "j.jsonl")
    lines = j.path.read_text().splitlines()
    import json as _json
    d = _json.loads(lines[0]); d["type"] = "bogus"; lines[0] = _json.dumps(d)
    j.path.write_text("\n".join(lines))
    with pytest.raises(JournalError, match="meta"):
        j.restore(store)


def test_journal_off_by_default_no_overhead(store):
    s = ResearchSession("y", store)
    assert s.journal is None
