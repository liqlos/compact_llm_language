"""Immutable, content-addressed raw evidence store.

Design:
- Content is stored once per run keyed by SHA-256; identical payloads dedupe.
- Objects are never mutated after write (write-once files).
- Every read verifies the content hash; tampering fails closed.
- Run isolation: objects live under runs/<run_id>/ and a StoredRef carries the
  run_id it was created in. Cross-run resolution is rejected at the reference
  layer (rcc.session), not here.

Fail-closed policy: any hash mismatch raises HashMismatchError rather than
returning unverified bytes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


class StoreError(Exception):
    pass


class HashMismatchError(StoreError):
    """Stored content no longer matches its content address (tamper/corruption)."""


# Path-component safety: run_id/kind become directory names, so they must be
# a single safe component (no separators, no traversal, no dot-only names).
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _check_component(value: str, what: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT_RE.match(value):
        raise StoreError(
            f"unsafe {what} {value!r}: must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}"
        )
    return value


@dataclass(frozen=True)
class StoredRef:
    run_id: str
    kind: str
    sha256: str
    size_bytes: int

    @property
    def short_sha(self) -> str:
        return self.sha256[:8]

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "kind": self.kind,
                "sha256": self.sha256,
                "size_bytes": self.size_bytes,
            },
            sort_keys=True,
        )

    @staticmethod
    def from_json(raw: str) -> StoredRef:
        d = json.loads(raw)
        return StoredRef(
            run_id=d["run_id"],
            kind=d["kind"],
            sha256=d["sha256"],
            size_bytes=d["size_bytes"],
        )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RawStore:
    """Write-once object store rooted at a directory."""

    def __init__(self, root: os.PathLike[str] | str):
        self.root = Path(root)

    def _object_path(self, ref: StoredRef) -> Path:
        _check_component(ref.run_id, "run_id")
        _check_component(ref.kind, "kind")
        if not re.fullmatch(r"[0-9a-f]{64}", ref.sha256):
            raise StoreError(f"unsafe sha256 {ref.sha256!r}")
        return (
            self.root
            / "runs"
            / ref.run_id
            / ref.kind
            / ref.sha256[:2]
            / f"{ref.sha256}.json"
        )

    def put(self, run_id: str, kind: str, content: bytes | str, meta: dict | None = None) -> StoredRef:
        _check_component(run_id, "run_id")
        _check_component(kind, "kind")
        if isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = content
        digest = sha256_hex(data)
        ref = StoredRef(run_id=run_id, kind=kind, sha256=digest, size_bytes=len(data))
        path = self._object_path(ref)
        if path.exists():
            return ref  # write-once: identical content already stored
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "meta": meta or {},
            "content_b64": base64.b64encode(data).decode("ascii"),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(envelope), encoding="utf-8")
        os.replace(tmp, path)  # atomic publish
        return ref

    def get(self, ref: StoredRef) -> bytes:
        path = self._object_path(ref)
        if not path.exists():
            raise StoreError(f"missing object {ref.sha256} for run {ref.run_id}")
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            data = base64.b64decode(envelope["content_b64"], validate=True)
            if not isinstance(data, bytes):
                raise TypeError("decoded payload is not bytes")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError,
                binascii.Error, UnicodeDecodeError) as e:
            # corruption must fail closed as a typed error so callers can
            # fail-open at the availability layer (session.compile)
            raise StoreError(f"corrupt object envelope for {ref.sha256}: {e!r}") from e
        actual = sha256_hex(data)
        if actual != ref.sha256:
            raise HashMismatchError(
                f"hash mismatch for {ref.sha256}: got {actual}"
            )
        return data

    def get_text(self, ref: StoredRef) -> str:
        return self.get(ref).decode("utf-8")

    def has(self, ref: StoredRef) -> bool:
        return self._object_path(ref).exists()

    def verify(self, ref: StoredRef) -> bool:
        try:
            self.get(ref)
            return True
        except StoreError:
            return False
