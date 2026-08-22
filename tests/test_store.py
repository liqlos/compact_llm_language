import pytest

from rcc import HashMismatchError, RawStore, StoredRef


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "store")


def test_roundtrip_text(store):
    ref = store.put("r1", "observation", "hello world")
    assert store.get_text(ref) == "hello world"


def test_content_addressing_and_dedup(store):
    a = store.put("r1", "observation", "same content")
    b = store.put("r1", "observation", "same content")
    assert a.sha256 == b.sha256
    # only one object file exists
    files = list((store.root / "runs" / "r1").rglob(f"{a.sha256}.json"))
    assert len(files) == 1


def test_unicode_roundtrip(store):
    text = "héllo — ünïcode ✓ 日本語"
    ref = store.put("r1", "observation", text)
    assert store.get_text(ref) == text


def test_binary_roundtrip(store):
    blob = bytes(range(256))
    ref = store.put("r1", "observation", blob)
    assert store.get(ref) == blob


def test_tamper_detection_fails_closed(store):
    import json

    ref = store.put("r1", "observation", "original")
    path = next((store.root / "runs" / "r1").rglob(f"{ref.sha256}.json"))
    env = json.loads(path.read_text())
    env["content_b64"] = __import__("base64").b64encode(b"tampered").decode()
    path.write_text(json.dumps(env))
    with pytest.raises(HashMismatchError):
        store.get(ref)


def test_missing_object(store):
    ref = StoredRef(run_id="r1", kind="observation", sha256="0" * 64, size_bytes=3)
    assert not store.has(ref)
    assert store.verify(ref) is False


def test_ref_json_roundtrip(store):
    ref = store.put("run-9", "doc", "x" * 10)
    assert StoredRef.from_json(ref.to_json()) == ref


def test_run_isolation_at_store_level(store):
    """Same content in different runs lives under separate run prefixes."""
    r1 = store.put("alpha", "observation", "shared")
    r2 = store.put("beta", "observation", "shared")
    assert r1.sha256 == r2.sha256
    assert r1.run_id != r2.run_id
    assert (store.root / "runs" / "alpha").exists()
    assert (store.root / "runs" / "beta").exists()
