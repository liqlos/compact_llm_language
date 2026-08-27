"""Deterministic token estimation.

This is NOT a provider tokenizer. It is a stable, dependency-free approximation
of BPE behaviour (GPT-2 style segmentation: contractions, letter chunks ~4 chars,
one token per digit, punctuation, whitespace runs). Because it is deterministic
and consistent, *deltas* between configurations are meaningful even though
absolute counts are approximate. Real tokenizer integration is tracked in the
implementation plan as a follow-up.

Label every persisted metric with `tokenizer="rcc-approx-1"`.
"""

from __future__ import annotations

import re

TOKENIZER_ID = "rcc-approx-1"
EXACT_DEFAULT_TOKENIZER_ID = "tiktoken:o200k_base"

_SEGMENT_RE = re.compile(
    r"'(?:[sdmt]|ll|ve|re)|[A-Za-z]+|[0-9]|[^\sA-Za-z0-9]+|\s+"
)


def count_tokens(text: str) -> int:
    """Estimate tokens deterministically. Never raises on normal input."""
    if not text:
        return 0
    total = 0
    for seg in _SEGMENT_RE.findall(text):
        c = seg[0]
        if c.isspace():
            # GPT-style: a single leading space merges with the next word.
            total += max(1, len(seg) // 8)
        elif c.isdigit():
            total += len(seg)  # one token per digit
        elif c.isalpha():
            total += max(1, -(-len(seg) // 4))  # ceil(len/4) BPE-ish chunks
        else:
            total += len(seg)  # punctuation splits roughly 1:1
    return total


# ---- exact tokenizers (optional dependency) -------------------------
# tiktoken is an OPTIONAL dev dependency: the core stays stdlib-only and
# deterministic; benchmarks and audits may request exact counts instead.

_EXACT_ENCODERS: dict[str, object] = {}


def available_encoders() -> list[str]:
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        return []
    return ["o200k_base", "cl100k_base"]


def count_tokens_exact(text: str, encoding: str = "o200k_base") -> int:
    """Exact BPE count via tiktoken. Raises ImportError if not installed."""
    if not text:
        return 0
    enc = _EXACT_ENCODERS.get(encoding)
    if enc is None:
        import tiktoken

        enc = tiktoken.get_encoding(encoding)
        _EXACT_ENCODERS[encoding] = enc
    return len(enc.encode(text))


def tokenizer_identity(counter, explicit: str | None = None) -> str:
    """Return a stable persisted identity for a token-counting callable.

    Custom counters should pass an explicit identity.  The callable fallback
    is deterministic enough for validation and, unlike ``__name__`` alone,
    includes its defining module.
    """
    known = None
    if counter is count_tokens:
        known = TOKENIZER_ID
    elif counter is count_tokens_exact:
        known = EXACT_DEFAULT_TOKENIZER_ID
    if known is not None:
        if explicit is not None and explicit != known:
            raise ValueError(
                f"tokenizer identity {explicit!r} conflicts with known counter {known!r}"
            )
        return known
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("tokenizer identity must be a non-empty string")
        return explicit
    declared = getattr(counter, "rcc_tokenizer_id", None)
    if isinstance(declared, str) and declared.strip():
        return declared
    module = getattr(counter, "__module__", None)
    qualname = getattr(counter, "__qualname__", None)
    if not module or not qualname:
        raise ValueError("custom tokenizer requires an explicit identity")
    return f"callable:{module}.{qualname}"
