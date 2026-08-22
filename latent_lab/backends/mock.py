"""Deterministic mock backend — a tiny "hybrid" machine, no ML dependencies.

Purpose: unit-test the whole latent pipeline (contextualize -> K latent
steps -> controller -> decode) with exact reproducibility, and give the
recurrence loop something structurally honest to run against. This is NOT a
language model and proves nothing about Qwen behaviour; it only verifies the
plumbing and the no-decode contract.

Mechanics (pure python, dim d=8):
- layer types: 2 natural groups of ("gdn","gdn","gdn","attn") = 8 layers
- contextualize: hash-seeded pseudo-random orthogonal-ish map of the payload
  into M/H/R slots + per-layer cache handles
- latent_step: fixed contraction over the selected interval's layers:
      W' = tanh(W @ R_i + b_i) for gdn layers (recurrent state update),
      plus attention-style mixing across slots on the attn layer.
  Interval choice therefore causally changes results.
- readout features[0] = confidence grows as workspace converges toward the
  problem's fixed point, so the rule controller halts deterministically.
"""

from __future__ import annotations

import hashlib
import math

from ..intervals import LayerInterval
from ..protocols import ModelInfo, ProblemInput, Readout
from ..state import CacheHandle, RCCModelState, Vec, Workspace

_DIM = 8
_LAYER_TYPES: tuple[str, ...] = ("gdn", "gdn", "gdn", "attn") * 2


def _seeded_matrix(seed: str) -> list[list[float]]:
    h = hashlib.sha256(seed.encode()).digest()
    vals: list[float] = []
    while len(vals) < _DIM * _DIM:
        h = hashlib.sha256(h).digest()
        vals.extend(b / 255.0 - 0.5 for b in h)
    return [vals[i * _DIM:(i + 1) * _DIM] for i in range(_DIM)]


def _matvec(m: list[list[float]], v: Vec) -> Vec:
    return tuple(
        math.tanh(sum(m[i][j] * v[j] for j in range(_DIM)))
        for i in range(_DIM)
    )


def _mix(vs: list[Vec]) -> Vec:
    n = len(vs)
    return tuple(
        math.tanh(sum(v[j] for v in vs) / max(1, n))
        for j in range(_DIM)
    )


class MockHybridBackend:
    """Deterministic hybrid-layout mock implementing LatentBackend."""

    def __init__(self,
                 interval: LayerInterval | None = None,
                 m_slots: int = 2, h_slots: int = 2, r_slots: int = 1):
        self.interval = interval or LayerInterval(0, 4, "early")
        self.m, self.h, self.r = m_slots, h_slots, r_slots
        self._decode_calls = 0
        self._info = ModelInfo(
            repo_id="mock/hybrid-tiny",
            revision="mock-1",
            config_hash=hashlib.sha256(str(_LAYER_TYPES).encode()).hexdigest()[:16],
            tokenizer_hash="none-mock",
            runtime="mock",
            runtime_version="1.0",
            dtype="fp64-pyfloat",
            quantization=None,
            layer_types=_LAYER_TYPES,
        )

    # ---- protocol ------------------------------------------------------

    def info(self) -> ModelInfo:
        return self._info

    def workspace_dims(self) -> tuple[int, int, int]:
        return self.m, self.h, self.r

    @property
    def decode_calls(self) -> int:
        return self._decode_calls

    # ---- pipeline ------------------------------------------------------

    def contextualize(self, problem: ProblemInput) -> RCCModelState:
        if not isinstance(problem, ProblemInput):
            raise TypeError(
                f"contextualize takes ProblemInput, got {type(problem).__name__}; "
                "telemetry/text must never enter the cognition path"
            )
        seed = f"{self._info.repo_id}|{problem.problem_id}"
        mem = [_matvec(_seeded_matrix(f"{seed}|M{i}"), _pool(problem.payload))
               for i in range(self.m)]
        work = [_matvec(_seeded_matrix(f"{seed}|H{i}"), _pool(problem.payload))
                for i in range(self.h)]
        read = [(0.0,) * _DIM for _ in range(self.r)]
        cache = CacheHandle(
            kv_layers={i: None for i in range(len(_LAYER_TYPES))},
            deltanet_recurrent={i: None for i, t in enumerate(_LAYER_TYPES)
                                if t == "gdn"},
            deltanet_conv={i: None for i, t in enumerate(_LAYER_TYPES)
                           if t == "gdn"},
        )
        return RCCModelState.create(
            self._info,
            Workspace(memory=mem, working=work, readout=read),
            cache,
            (self.interval.lo, self.interval.hi),
            provenance_refs=problem.evidence_refs,
        )

    def latent_step(self, state: RCCModelState) -> RCCModelState:
        lo, hi = state.interval
        ws = state.workspace
        memory, working = ws.memory, ws.working
        for li in range(lo, hi):
            kind = state.layer_types[li]
            if kind == "gdn":
                mat = _seeded_matrix(f"{state.config_hash}|step{li}")
                working = [_matvec(mat, w) for w in working]
            elif kind == "attn":
                ctx = _mix(memory + working)
                working = [
                    tuple(w[j] + 0.25 * ctx[j] for j in range(_DIM))
                    for w in working
                ]
            else:
                raise ValueError(f"unknown layer type {kind!r}")
        # convergence proxy: readout confidence = 1 - distance to fixed point
        prev = ws.working
        delta = sum(
            abs(a - b) for pw, cw in zip(prev, working) for a, b in zip(pw, cw)
        )
        conf = 1.0 / (1.0 + delta)
        readout = [(conf,) + (0.0,) * (_DIM - 1)]
        new_ws = Workspace(memory=memory, working=working, readout=readout)
        return state.with_workspace(new_ws)

    def readout(self, state: RCCModelState) -> Readout:
        feats = tuple(state.workspace.readout[0]) if state.workspace.readout \
            else (0.0,) * _DIM
        handle = {
            "kind": "answer_handle",
            "workspace_fp": state.workspace.fingerprint(),
            "steps": state.latent_step_index,
            "interval": list(state.interval),
        }
        return Readout(features=feats, answer_handle=handle)

    def decode(self, state: RCCModelState, answer_handle: object) -> str:
        self._decode_calls += 1
        h = dict(answer_handle)  # type: ignore[arg-type]
        return (
            f"ANSWER[{h['kind']} steps={h['steps']} "
            f"interval={h['interval'][0]}:{h['interval'][1]} "
            f"fp={h['workspace_fp'][:8]}]"
        )


def _pool(payload: tuple[Vec, ...]) -> Vec:
    if not payload:
        return (0.0,) * _DIM
    out = [0.0] * _DIM
    for v in payload:
        for j in range(min(_DIM, len(v))):
            out[j] += v[j]
    n = len(payload)
    return tuple(x / n for x in out)
