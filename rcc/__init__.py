"""Research Context Compiler (RCC).

Compiles long research/coding transcripts into compact active context backed
by an immutable, content-addressed evidence store, plus a symbolic
machine-state scratch layer (RIR/1).
"""

from .scratch import Atom, Scratch, ScratchError
from .session import (
    CompiledContext,
    CompileMetrics,
    CrossRunReferenceError,
    ObservationItem,
    Policy,
    ResearchSession,
    SessionError,
)
from .store import HashMismatchError, RawStore, StoredRef, StoreError
from .tokens import TOKENIZER_ID, count_tokens, count_tokens_exact

__all__ = [
    "TOKENIZER_ID",
    "Atom",
    "CompileMetrics",
    "CompiledContext",
    "CrossRunReferenceError",
    "HashMismatchError",
    "ObservationItem",
    "Policy",
    "RawStore",
    "ResearchSession",
    "Scratch",
    "ScratchError",
    "SessionError",
    "StoreError",
    "StoredRef",
    "count_tokens",
    "count_tokens_exact",
]
