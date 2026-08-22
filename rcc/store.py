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

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


class StoreError(Exception):
    pass


class HashMismatchError(StoreError):
    """Stored content no longer matches its content address (tamper/corruption)."""


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
        return (
            self.root
            / "runs"
            / ref.run_id
            / ref.kind
            / ref.sha256[:2]
            / f"{ref.sha256}.json"
        )

    def put(self, run_id: str, kind: str, content: bytes | str, meta: dict | None = None) -> StoredRef:
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
            "content_b64": __import__("base64").b64encode(data).decode("ascii"),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(envelope), encoding="utf-8")
        os.replace(tmp, path)  # atomic publish
        return ref

    def get(self, ref: StoredRef) -> bytes:
        path = self._object_path(ref)
        if not path.exists():
            raise StoreError(f"missing object {ref.sha256} for run {ref.run_id}")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        import base64

        data = base64.b64decode(envelope["content_b64"])
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
