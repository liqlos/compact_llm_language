"""Lossless dictionary encoding for repetitive structured output (Phase 6).

Targets JSON-lines style payloads where every object repeats the same keys
(tool results, test-run records, transaction dumps). The schema is stated
once; objects become positional rows. Fully reversible via decode().

Non-conforming input returns None -- callers keep the verbatim text
(no lossy fallback for prose, per master plan Priority 6).
"""

from __future__ import annotations

import json


def _try_obj(line: str):
    try:
        o = json.loads(line)
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
    header = "SCHEMA " + "|".join(keys)
    rows = []
    for o in objs:
        vals = []
        for k in keys:
            # quote EVERY value: json.dumps guarantees exact type round-trip
            sv = json.dumps(o[k], ensure_ascii=False)
            sv = sv.replace("\n", "\\n").replace("|", "\\|")
            vals.append(sv)
        rows.append("ROW " + "|".join(vals))
    encoded = "\n".join([header, *rows])
    meta = {
        "n_objects": len(objs),
        "keys": keys,
        "raw_chars": len(text),
        "encoded_chars": len(encoded),
    }
    # never claim a win that is not there
    if len(encoded) >= len(text):
        return None
    return encoded, meta


def decode(encoded: str) -> list[dict]:
    """Exact inverse of encode_json_objects."""
    lines = [ln for ln in encoded.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("SCHEMA "):
        raise ValueError("not an RCC-encoded payload")
    keys = lines[0][len("SCHEMA "):].split("|")
    out = []
    for ln in lines[1:]:
        if not ln.startswith("ROW "):
            raise ValueError(f"unexpected line {ln[:20]!r}")
        parts = _split_row(ln[4:])
        if len(parts) != len(keys):
            raise ValueError("row/schema arity mismatch")
        rec = {}
        for k, raw in zip(keys, parts):
            v = raw.replace("\\|", "|").replace("\\n", "\n")
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass  # plain string value
            rec[k] = v
        out.append(rec)
    return out


def _split_row(row: str) -> list[str]:
    parts, buf, esc = [], [], False
    for ch in row:
        if esc:
            buf.append("\\" + ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "|":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts
