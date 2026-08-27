"""Structurally lossless dictionary encoding for repetitive JSON objects.

Targets JSON-lines style payloads where every object repeats the same keys
(tool results, test-run records, transaction dumps). The schema is stated
once; objects become positional rows. Fully reversible via decode().

Non-conforming input returns None -- callers keep the verbatim text
(no lossy fallback for prose, per master plan Priority 6).
"""

from __future__ import annotations

import json


FORMAT = "rcc.dict.v2"


def _reject_nonfinite(value: str):
    raise ValueError(f"non-finite JSON number {value!r} is unsupported")


def _try_obj(line: str):
    try:
        o = json.loads(line, parse_constant=_reject_nonfinite)
        return o if isinstance(o, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def encode_json_objects(text: str, min_objects: int = 3) -> tuple[str, dict] | None:
    """Encode consecutive same-schema JSON object lines.

    Returns (encoded_text, meta) or None if not applicable/beneficial.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < min_objects:
        return None
    objs = [_try_obj(ln) for ln in lines]
    if any(o is None for o in objs) or len({tuple(o.keys()) for o in objs}) != 1:
        return None

    keys = list(objs[0].keys())
    # A JSON array is the framing layer.  Values are never escaped by custom
    # delimiter rules, so strings containing newlines, backslashes, pipes, or
    # any other JSON character remain unambiguous.
    payload = [FORMAT, keys, [[obj[key] for key in keys] for obj in objs]]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    meta = {
        "format": FORMAT,
        "n_objects": len(objs),
        "keys": keys,
        "raw_chars": len(text),
        "encoded_chars": len(encoded),
        "roundtrip_guarantee": "structural-json-equality",
    }
    # never claim a win that is not there
    if len(encoded) >= len(text):
        return None
    return encoded, meta


def decode(encoded: str) -> list[dict]:
    """Decode ``rcc.dict.v2`` into structurally equal JSON objects."""
    try:
        payload = json.loads(encoded, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("not an RCC dictionary payload") from exc
    if not isinstance(payload, list) or len(payload) != 3 or payload[0] != FORMAT:
        raise ValueError("unsupported RCC dictionary format")
    keys, rows = payload[1], payload[2]
    if (
        not isinstance(keys, list)
        or not all(isinstance(key, str) for key in keys)
        or len(keys) != len(set(keys))
        or not isinstance(rows, list)
    ):
        raise ValueError("invalid RCC dictionary schema")
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(keys):
            raise ValueError("row/schema arity mismatch")
        out.append(dict(zip(keys, row, strict=True)))
    return out
