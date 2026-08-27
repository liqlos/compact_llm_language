"""Dictionary encoding tests: structurally lossless JSON codec."""

import json
import random

import pytest

from rcc import Policy, RawStore, ResearchSession
from rcc.dictenc import decode, encode_json_objects

ROWS = [
    {"id": i, "status": "paid" if i % 2 else "open",
     "amount": f"{i * 13}.{i % 100:02d}", "ts": f"2026-07-{i % 28 + 1:02d}"}
    for i in range(1, 9)
]
JSONL = "\n".join(__import__("json").dumps(r, separators=(",", ":")) for r in ROWS)


def test_roundtrip_lossless():
    enc = encode_json_objects(JSONL)
    assert enc is not None
    text, meta = enc
    assert meta["n_objects"] == 8
    assert decode(text) == ROWS


def test_numbers_and_strings_preserved_exactly():
    text, _ = encode_json_objects(JSONL)
    decoded = decode(text)
    assert decoded[0]["amount"] == "13.01"      # string stays a string
    assert isinstance(decoded[0]["amount"], str)
    assert decoded[0]["id"] == 1                # int stays an int
    assert isinstance(decoded[0]["id"], int)
    assert all(d["ts"].startswith("2026-07-") for d in decoded)


def test_rejects_mixed_or_short_input():
    assert encode_json_objects('{"a":1}') is None                      # too few
    mixed = '{"a":1}\n{"a":2}\n{"b":3}\n{"a":4}'                       # keys differ
    assert encode_json_objects(mixed) is None
    not_json = "hello\nworld\nfoo\nbar"
    assert encode_json_objects(not_json) is None
    arrays = "[1]\n[2]\n[3]"
    assert encode_json_objects(arrays) is None
    nonfinite = '{"a":NaN}\n{"a":NaN}\n{"a":NaN}'
    assert encode_json_objects(nonfinite) is None


def test_decoder_rejects_nonfinite_or_malformed_structural_payloads():
    with pytest.raises(ValueError):
        decode('["rcc.dict.v2",["a"],[[NaN]]]')
    with pytest.raises(ValueError, match="arity"):
        decode('["rcc.dict.v2",["a","b"],[[1]]]')
    with pytest.raises(ValueError, match="schema"):
        decode('["rcc.dict.v2",["a","a"],[[1,2]]]')


def test_escapes_separators_safely():
    rows = [{"k": f"a|b{i}", "n": i} for i in range(4)]
    text = "\n".join(__import__("json").dumps(r) for r in rows)
    enc = encode_json_objects(text)
    assert enc is not None
    assert decode(enc[0]) == rows


def test_structural_roundtrip_handles_all_json_edge_cases():
    keys = [
        "",
        "key\nwith newline|and\\backslash/quote\"雪",
        "arbitrary:[]=,{} key",
    ]
    values = [
        "line one\nline two",
        r"literal\n and slash/ and backslash\\ and | separator",
        'quotes "single\' and Unicode 日本語 ✓',
        "",
        {"nested": [1, -2.5, True, False, None, {"x|y": "z\\n"}]},
        [0, 1.25, -3, True, None],
        42,
        -0.125,
        True,
        False,
        None,
    ]
    rows = [
        {keys[0]: values[index % len(values)],
         keys[1]: values[(index + 1) % len(values)],
         keys[2]: values[(index + 2) % len(values)]}
        for index in range(24)
    ]
    text = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
    encoded = encode_json_objects(text)
    assert encoded is not None
    assert encoded[1]["roundtrip_guarantee"] == "structural-json-equality"
    assert decode(encoded[0]) == rows


def test_deterministic_structural_roundtrip_property():
    rng = random.Random(20260827)
    atoms = [None, True, False, "", "\n", r"\n", "\\", "|", '"', "Юникод", 0, -7, 2.5]
    for case in range(64):
        keys = [f"long repeated key {case} {suffix}" for suffix in ("\n|", "\\/\"", "雪")]
        rows = []
        for row_index in range(12):
            rows.append({
                keys[0]: rng.choice(atoms),
                keys[1]: [rng.choice(atoms), {"nested": rng.choice(atoms)}],
                keys[2]: {"i": row_index, "ok": bool(row_index % 2), "none": None},
            })
        text = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        encoded = encode_json_objects(text)
        assert encoded is not None, case
        assert decode(encoded[0]) == rows


def test_unicode_roundtrip():
    rows = [{"name": "日本語 ✓ — т", "v": 1}] * 1 + [
        {"name": "x", "v": 2}, {"name": "y", "v": 3}]
    text = "\n".join(__import__("json").dumps(r, ensure_ascii=False) for r in rows)
    enc = encode_json_objects(text)
    assert enc is not None and decode(enc[0]) == rows


def test_encodes_only_when_smaller():
    # values made of pipe characters force heavy escaping: the encoded rows
    # grow longer than the raw objects -> encoder must decline.
    rows = [{"k": "||||||||||"} for _ in range(3)]
    text = "\n".join(__import__("json").dumps(r) for r in rows)
    assert encode_json_objects(text) is None


# ---- session integration -----------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "s")


def _session(store, policy):
    s = ResearchSession("run-e", store, policy)
    s.observe("orders_jsonl", JSONL)
    for i in range(5):
        s.observe(f"f_{i}", "\n".join(["filler"] * 120))
        s.say("assistant", f"step {i}")
    return s


def test_flag_on_encodes_inline(store):
    s = _session(store, Policy(encode_jsonl=True))
    c = s.compile()
    assert c.metrics.jsonl_encoded >= 1
    assert '["rcc.dict.v2",["id","status","amount","ts"]' in c.text
    assert c.text.count('"status"') == 1       # repeated keys stated once


def test_flag_off_verbatim(store):
    s = _session(store, Policy())
    c = s.compile()
    assert c.metrics.jsonl_encoded == 0
    assert '"status":"paid"' in c.text or '"status"' in c.text


def test_encoded_inline_still_cheaper(store):
    on = _session(store, Policy(encode_jsonl=True)).compile().metrics
    off = _session(store, Policy()).compile().metrics
    assert on.total_tokens < off.total_tokens
