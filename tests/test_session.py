import json

import pytest

from rcc import (
    CrossRunReferenceError,
    Policy,
    RawStore,
    ResearchSession,
    SessionError,
    StoreError,
)

BIG = "x" * 1200  # ~300 approx tokens, exceeds min_mask_tokens


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "store")


def make_session(store, run_id="run-1", policy=None):
    return ResearchSession(run_id=run_id, store=store, policy=policy or Policy())


def test_recent_window_stays_inline(store):
    s = make_session(store, policy=Policy(keep_recent=2))
    for i in range(5):
        s.observe(f"o{i}", f"{BIG} {i}")
        s.say("user", f"q{i}")
    c = s.compile()
    assert "o3" in c.text and "o4" in c.text      # recent: inline
    assert "[OBS obs-0001" in c.text               # old: stub


def test_small_observations_never_masked(store):
    s = make_session(store)
    for i in range(8):
        s.observe(f"tiny{i}", "small note")
    c = s.compile()
    assert "[OBS obs-" not in c.text
    assert c.metrics.observations_masked == 0


def test_duplicates_stubbed_immediately(store):
    s = make_session(store)
    s.observe("sql", BIG)          # obs-0001 first occurrence -> inline (recent)
    s.say("user", "again?")
    s.observe("sql", BIG)          # identical content -> dup occurrence
    c = s.compile()
    assert c.metrics.duplicates_stubbed == 1
    assert "[OBS obs-0001" in c.text  # dup references the CANONICAL id


def test_stub_is_compact_and_stable(store):
    s = make_session(store)
    s.observe("doc", BIG)
    for i in range(5):
        s.observe(f"later{i}", BIG + str(i))  # pushes obs-0001 out of window
    c1 = s.compile()
    c2 = s.compile()
    assert c1.text == c2.text  # deterministic assembly
    assert "[OBS obs-0001" in c1.text
    assert "tok=" in c1.text and "sha=" in c1.text


def test_expand_roundtrip_exact(store):
    s = make_session(store)
    content = f"{BIG}\nexact value 142ms"
    oid = s.observe("notes", content)
    for i in range(6):
        s.say("user", f"q{i}")
    expanded = s.expand(oid)
    assert "exact value 142ms" in expanded  # byte-exact recovery


def test_expand_unknown_and_malformed(store):
    s = make_session(store)
    with pytest.raises(SessionError):
        s.expand("obs-9999")
    with pytest.raises(SessionError):
        s.expand("not-a-ref")


def test_cross_run_reference_rejected(tmp_path):
    store = RawStore(tmp_path / "s")
    a = ResearchSession(run_id="alpha", store=store)
    oid = a.observe("d", BIG)
    b = ResearchSession(run_id="beta", store=store)
    # forge a reference from another run by loading state manually:
    item = a._by_id[oid]
    b._by_id[oid] = item
    with pytest.raises(CrossRunReferenceError):
        b.expand(oid)


def test_cross_run_reference_rejected_again_before_compile(tmp_path):
    store = RawStore(tmp_path / "s")
    a = ResearchSession(run_id="alpha", store=store)
    oid = a.observe("d", BIG)
    b = ResearchSession(run_id="beta", store=store)
    b.observe("own", BIG + "beta")
    b._turns[-1].item = a._by_id[oid]  # forge persisted/in-memory contamination
    with pytest.raises(CrossRunReferenceError):
        b.compile()


def test_corrupted_persisted_top_level_run_id_fails_closed(store, tmp_path):
    s = ResearchSession("original-run", store)
    s.observe("doc", BIG)
    state = tmp_path / "state.json"
    s.save(state)
    payload = json.loads(state.read_text(encoding="utf-8"))
    payload["run_id"] = "forged-run"
    state.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CrossRunReferenceError):
        ResearchSession.load(state, store)


def test_persisted_token_accounting_is_rechecked_and_reproducible(store, tmp_path):
    s = ResearchSession("token-run", store, Policy(keep_recent=1))
    for index in range(4):
        s.observe(f"doc-{index}", BIG + str(index))
    before = s.compile()
    state = tmp_path / "state.json"
    s.save(state)
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["tokenizer_id"] == "rcc-approx-1"
    assert all(item["n_tokens"] > 0 for item in payload["observations"])
    assert all(item["tokenizer_id"] == "rcc-approx-1" for item in payload["observations"])
    resumed = ResearchSession.load(state, store)
    after = resumed.compile()
    assert after.text == before.text
    assert after.metrics == before.metrics


def test_tampered_persisted_token_count_fails_closed(store, tmp_path):
    s = ResearchSession("token-run", store)
    s.observe("doc", BIG)
    state = tmp_path / "state.json"
    s.save(state)
    payload = json.loads(state.read_text(encoding="utf-8"))
    payload["observations"][0]["n_tokens"] += 1
    state.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionError, match="token count mismatch"):
        ResearchSession.load(state, store)


@pytest.mark.parametrize(
    ("target", "field", "message"),
    [
        ("top", "tokenizer_id", "missing tokenizer identity"),
        ("observation", "tokenizer_id", "tokenizer identity mismatch"),
        ("observation", "n_tokens", "invalid n_tokens"),
    ],
)
def test_missing_persisted_token_metadata_fails_closed(store, tmp_path, target, field, message):
    s = ResearchSession("token-run", store)
    s.observe("doc", BIG)
    state = tmp_path / f"missing-{target}-{field}.json"
    s.save(state)
    payload = json.loads(state.read_text(encoding="utf-8"))
    if target == "top":
        del payload[field]
    else:
        del payload["observations"][0][field]
    state.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionError, match=message):
        ResearchSession.load(state, store)


def test_missing_raw_object_does_not_become_zero_tokens(store, tmp_path):
    s = ResearchSession("token-run", store)
    oid = s.observe("doc", BIG)
    state = tmp_path / "state.json"
    s.save(state)
    store._object_path(s._by_id[oid].ref).unlink()
    with pytest.raises(StoreError, match="missing object"):
        ResearchSession.load(state, store)


def test_tokenizer_identity_mismatch_fails_closed(store, tmp_path):
    def word_counter(text: str) -> int:
        return len(text.split())

    s = ResearchSession(
        "custom-tokenizer-run",
        store,
        tokenizer=word_counter,
        tokenizer_id="test:words:v1",
    )
    s.observe("doc", "one two three")
    state = tmp_path / "state.json"
    s.save(state)
    same = ResearchSession.load(
        state,
        store,
        tokenizer=word_counter,
        tokenizer_id="test:words:v1",
    )
    assert same.compile().metrics.tokenizer == "test:words:v1"
    with pytest.raises(SessionError, match="tokenizer identity mismatch"):
        ResearchSession.load(
            state,
            store,
            tokenizer=word_counter,
            tokenizer_id="test:words:v2",
        )


def test_failopen_when_store_object_missing(store):
    s = make_session(store)
    for i in range(6):
        s.observe(f"doc{i}", f"{BIG} {i}")
    # destroy all stored objects behind the compiler's back
    import shutil

    shutil.rmtree(store.root / "runs")
    c = s.compile()  # must not raise: availability is preserved
    assert "[OBS obs-" not in c.text              # no fake recoverable stubs
    assert "<OBSERVATION_UNAVAILABLE" in c.text   # honest placeholder instead
    assert c.metrics.failopen_inline >= 2         # safety fallback engaged


def test_feature_flag_disabled_equals_baseline(store):
    content = BIG
    off = make_session(store, policy=Policy(enabled=False))
    off.observe("doc", content)
    for i in range(6):
        off.say("user", f"q{i}")
    c = off.compile()
    assert "[OBS obs-" not in c.text
    assert c.metrics.total_tokens > count_of(content)


def count_of(t):
    from rcc import count_tokens

    return count_tokens(t)


def test_protected_constraint_never_masked(store):
    s = make_session(store)
    s.say("user", "NEVER delete production data.", protected=True)
    for i in range(10):
        s.observe(f"d{i}", BIG)
    c = s.compile()
    assert "NEVER delete production data." in c.text
