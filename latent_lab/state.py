"""RCCModelState — versioned model/runtime-specific state ABI.

NOT portable across models, revisions, quantizations or runtimes until
portability is explicitly proven. Every instance records the full identity
header so misuse fails loudly.

Vectors are plain tuples of floats for the stdlib mock backend; real backends
store opaque handles (torch tensors / MLX arrays) in the same fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

STATE_SCHEMA_VERSION = 2

Vec = tuple[float, ...]


def _vec_hash(vs: list[Vec]) -> str:
    h = hashlib.sha256()
    for v in vs:
        h.update(json.dumps(v, allow_nan=False).encode())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Workspace:
    """Fixed-size latent workspace (problem-conditioned, not learned constants)."""

    memory: list[Vec]     # M persistent slots
    working: list[Vec]    # H reasoning slots
    readout: list[Vec]    # R readout slots

    def __post_init__(self) -> None:
        if not self.memory or not self.working or not self.readout:
            raise ValueError("workspace requires >=1 slot in each of M/H/R")

    def fingerprint(self) -> str:
        return _vec_hash(self.memory + self.working + self.readout)


@dataclass(frozen=True)
class CacheHandle:
    """Opaque attention-KV / DeltaNet recurrent+conv state container."""

    kv_layers: dict[int, object] = field(default_factory=dict)
    deltanet_recurrent: dict[int, object] = field(default_factory=dict)
    deltanet_conv: dict[int, object] = field(default_factory=dict)

    def clone(self) -> CacheHandle:
        return CacheHandle(
            kv_layers=dict(self.kv_layers),
            deltanet_recurrent=dict(self.deltanet_recurrent),
            deltanet_conv=dict(self.deltanet_conv),
        )


@dataclass(frozen=True)
class RCCModelState:
    """One versioned snapshot of everything a latent step consumes/produces."""

    # identity header (ABI)
    schema_version: int
    repo_id: str
    revision: str
    config_hash: str
    tokenizer_hash: str
    runtime: str
    runtime_version: str
    dtype: str
    quantization: str | None

    # architecture + recurrence position
    layer_types: tuple[str, ...]
    interval: tuple[int, int]           # [lo, hi) layer indices, contiguous
    latent_step_index: int              # how many steps already applied

    # machine state
    workspace: Workspace
    cache: CacheHandle
    controller_state: Vec               # small fixed vector (halt logit etc.)

    # provenance handles into the evidence plane (blob/claim ids)
    provenance_refs: tuple[str, ...] = ()

    @staticmethod
    def create(info, workspace: Workspace, cache: CacheHandle,
               interval: tuple[int, int], *,
               controller_state: Vec = (0.0,),
               provenance_refs: tuple[str, ...] = ()) -> RCCModelState:
        lo, hi = interval
        if not (0 <= lo < hi <= len(info.layer_types)):
            raise ValueError(f"interval {interval} outside 0..{len(info.layer_types)}")
        return RCCModelState(
            schema_version=STATE_SCHEMA_VERSION,
            repo_id=info.repo_id,
            revision=info.revision,
            config_hash=info.config_hash,
            tokenizer_hash=info.tokenizer_hash,
            runtime=info.runtime,
            runtime_version=info.runtime_version,
            dtype=info.dtype,
            quantization=info.quantization,
            layer_types=tuple(info.layer_types),
            interval=(lo, hi),
            latent_step_index=0,
            workspace=workspace,
            cache=cache,
            controller_state=tuple(controller_state),
            provenance_refs=tuple(provenance_refs),
        )

    def with_workspace(self, workspace: Workspace) -> RCCModelState:
        return replace(self, workspace=workspace,
                       latent_step_index=self.latent_step_index + 1)

    def state_id(self) -> str:
        """Stable id for this exact state content (for logs/benchmarks)."""
        h = hashlib.sha256()
        h.update(json.dumps([
            self.repo_id, self.revision, self.config_hash, self.dtype,
            self.quantization, self.interval, self.latent_step_index,
            self.workspace.fingerprint(),
        ], sort_keys=True).encode())
        return "st-" + h.hexdigest()[:16]
