"""Benchmark runner: harness self-checks + honest task evaluation.

`selfcheck` verifies the causal plumbing that anti-cheating controls rely on.
It works identically for the mock and (later) real Qwen backends; a real
backend must pass selfcheck before any quality/latency number is reported.

`evaluate` scores decoded answers with exact scorers. When no decoding policy
exists for a backend, tasks are recorded with success=None — never faked.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from ..backends.mock import MockHybridBackend
from ..controller import RuleController
from ..intervals import LayerInterval, candidate_intervals
from ..protocols import LatentBackend, ProblemInput, check_backend
from ..recurrence import run_latent_loop
from ..state import Workspace
from .metrics import SelfCheckReport, TaskMetrics, manifest
from .tasks import TaskInstance


def _zero_workspace(state):
    ws = state.workspace
    zero = (0.0,) * len(ws.memory[0])
    return replace(state, workspace=Workspace(
        memory=[zero] * len(ws.memory),
        working=[zero] * len(ws.working),
        readout=list(ws.readout),
    ))


def _fingerprint(backend: LatentBackend, state) -> str:
    return backend.readout(state).answer_handle["workspace_fp"]


def selfcheck(backend: LatentBackend, sink=None) -> SelfCheckReport:
    """Causal/anti-cheat mechanics. Cheap; run before every real benchmark."""
    t0 = time.perf_counter()
    check_backend(backend)
    rep = SelfCheckReport(backend=backend.info().repo_id)

    vec = tuple((i % 7) / 7 for i in range(_dim_for(backend)))
    prob = ProblemInput("selfcheck", (vec,), evidence_refs=("blob-selfcheck",))
    st0 = backend.contextualize(prob)

    # K causality: K=0 vs K=2 must differ
    r0 = run_latent_loop(backend, st0, k_steps=0)
    r2 = run_latent_loop(backend, st0, k_steps=2)
    rep.k_causal_differs = (
        _fingerprint(backend, r0.final_state) != _fingerprint(backend, r2.final_state)
    )

    # zero-state ablation changes the readout
    rz = run_latent_loop(backend, _zero_workspace(st0), k_steps=2)
    rep.zero_state_changes_readout = (
        _fingerprint(backend, rz.final_state) != _fingerprint(backend, r2.final_state)
    )

    # swapped states between problems change the readout
    other = backend.contextualize(ProblemInput("selfcheck-other", (vec[::-1],)))
    ro_other = run_latent_loop(backend, other, k_steps=2)
    swapped = replace(other, workspace=r2.final_state.workspace)
    ro_swapped = backend.readout(swapped)
    rep.swapped_state_changes_readout = (
        ro_swapped.answer_handle["workspace_fp"]
        != _fingerprint(backend, ro_other.final_state)
    )

    # truncated depth differs from full depth
    rt = run_latent_loop(backend, st0, k_steps=1)
    rep.truncated_depth_changes_readout = (
        _fingerprint(backend, rt.final_state) != _fingerprint(backend, r2.final_state)
    )

    # interval sensitivity across the standard candidate set
    for iv in candidate_intervals(st0.layer_types):
        if iv.lo == 0 and iv.hi == len(st0.layer_types) and len(
            candidate_intervals(st0.layer_types)
        ) > 1:
            pass  # full-decoder control included but not required to differ
        b2: LatentBackend
        if isinstance(backend, MockHybridBackend):
            b2 = MockHybridBackend(LayerInterval(iv.lo, iv.hi, iv.name))
            s2 = b2.contextualize(prob)
            rr = run_latent_loop(b2, s2, k_steps=2)
            rep.interval_sensitivity[iv.name] = (
                _fingerprint(b2, rr.final_state) != _fingerprint(backend, r2.final_state)
            )

    # determinism: identical rerun -> identical state id
    again = backend.contextualize(prob)
    r_again = run_latent_loop(backend, again, k_steps=2)
    rep.deterministic_rerun = (
        r_again.final_state.state_id() == r2.final_state.state_id()
    )
    rep.elapsed_seconds = time.perf_counter() - t0
    return rep


def _dim_for(backend: LatentBackend) -> int:
    info = backend.info()
    del info
    return 8  # mock dims; real backends read config hidden size instead


def evaluate(
    backend: LatentBackend,
    tasks: list[TaskInstance],
    *,
    k_steps: int,
    interval: LayerInterval | None,
    answer_fn: Callable[[object, TaskInstance], str] | None,
    scorer: Callable[[str, TaskInstance], float],
    config_name: str | None = None,
    payload_fn: Callable[[TaskInstance], tuple] | None = None,
    controller: RuleController | None = None,
) -> list[TaskMetrics]:
    """Evaluate tasks end-to-end. success=None when scoring impossible."""
    cfg = config_name or f"latent_k{k_steps}_interval_{interval.name if interval else 'ctx'}"
    out: list[TaskMetrics] = []
    for task in tasks:
        t0 = time.perf_counter()
        payload = tuple() if payload_fn is None else payload_fn(task)
        prob = ProblemInput(
            problem_id=task.task_id,
            payload=payload,
            evidence_refs=tuple(f"blob-{k}" for k, _ in task.facts),
        )
        state = backend.contextualize(prob)
        loop = run_latent_loop(backend, state, k_steps=k_steps,
                               controller=controller)
        wall = time.perf_counter() - t0
        if answer_fn is None:
            out.append(TaskMetrics(
                task_id=task.task_id, config=cfg, success=None,
                wall_seconds=wall, latent_steps=loop.steps_executed,
                error="no decode policy for this backend; not scored",
            ))
            continue
        answer = answer_fn(loop.final_state, task)
        score = scorer(answer, task)
        out.append(TaskMetrics(
            task_id=task.task_id, config=cfg, success=score >= 1.0,
            wall_seconds=wall, latent_steps=loop.steps_executed,
        ))
    return out


def save_report(obj: dict, path) -> None:
    import json

    obj = dict(obj)
    obj.setdefault("manifest", manifest())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
