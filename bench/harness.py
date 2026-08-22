"""Benchmark harness: replays scenarios in baseline vs compiled mode.

Modes:
- baseline : Policy(enabled=False) -- every observation inline forever
- compiled : default Policy          -- dedup + masking + recoverable refs

Token counting is pluggable: the deterministic rcc-approx-1 estimator by
default, or an exact tiktoken encoding via `tokenizer=count_tokens_exact`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bench.scenarios import SCENARIOS, Scenario
from rcc import Policy, RawStore, ResearchSession
from rcc.tokens import TOKENIZER_ID


@dataclass
class RunResult:
    scenario: str
    mode: str
    tokenizer_id: str
    total_tokens_all_turns: int   # sum of active-context tokens after each op
    peak_active_tokens: int       # max active context at any point
    final_tokens: int
    inline_observation_tokens: int
    stub_observation_tokens: int
    observations_masked: int
    duplicates_stubbed: int
    failopen_inline: int
    quality: dict = field(default_factory=dict)
    turn_tokens: list[int] = field(default_factory=list)  # active ctx after each op
    compile_ms_total: float = 0.0
    compile_calls: int = 0


def run_scenario(
    sc: Scenario,
    *,
    enabled: bool,
    root: Path,
    tokenizer: Callable[[str], int] | None = None,
) -> RunResult:
    mode = "compiled" if enabled else "baseline"
    tok_name = getattr(tokenizer, "__name__", "count_tokens") if tokenizer else TOKENIZER_ID
    run_dir = root / sc.name / mode
    store = RawStore(run_dir)
    session = ResearchSession(run_id=f"{sc.name}-{mode}", store=store,
                              policy=Policy(enabled=enabled), tokenizer=tokenizer or _default_tok())
    ops = sc.ops_factory()

    total = 0
    peak = 0
    final = None
    compile_ms = 0.0
    compile_calls = 0
    series: list[int] = []
    for op in ops:
        if op[0] == "save":
            session.save(Path(op[1]))
            continue
        if op[0] == "load":
            session = ResearchSession.load(Path(op[1]), store, tokenizer=session.tokenizer)
            continue
        if op[0] == "say":
            session.say(op[1], op[2], protected=bool(op[3]) if len(op) > 3 else False)
        elif op[0] == "observe":
            session.observe(op[1], op[2])
        t0 = time.perf_counter()
        c = session.compile()
        compile_ms += (time.perf_counter() - t0) * 1000
        compile_calls += 1
        total += c.metrics.total_tokens
        peak = max(peak, c.metrics.total_tokens)
        series.append(c.metrics.total_tokens)
        final = c

    quality = sc.verify(session, final) if sc.verify else {}
    m = final.metrics
    return RunResult(
        scenario=sc.name,
        mode=mode,
        tokenizer_id=tok_name,
        total_tokens_all_turns=total,
        peak_active_tokens=peak,
        final_tokens=m.total_tokens,
        inline_observation_tokens=m.inline_observation_tokens,
        stub_observation_tokens=m.stub_observation_tokens,
        observations_masked=m.observations_masked,
        duplicates_stubbed=m.duplicates_stubbed,
        failopen_inline=m.failopen_inline,
        quality=quality,
        turn_tokens=series,
        compile_ms_total=round(compile_ms, 2),
        compile_calls=compile_calls,
    )


def _default_tok():
    from rcc import count_tokens

    return count_tokens


def run_suite(root: Path, tokenizer: Callable[[str], int] | None = None) -> list[tuple[RunResult, RunResult]]:
    root.mkdir(parents=True, exist_ok=True)
    pairs = []
    for sc in SCENARIOS:
        base = run_scenario(sc, enabled=False, root=root, tokenizer=tokenizer)
        comp = run_scenario(sc, enabled=True, root=root, tokenizer=tokenizer)
        pairs.append((base, comp))
    return pairs


def format_report(pairs: list[tuple[RunResult, RunResult]]) -> str:
    lines = []
    header = (
        f"{'scenario':<14} {'metric':<26} {'baseline':>10} {'compiled':>10} {'delta':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for b, c in pairs:
        def row(metric: str, bv: int, cv: int, scenario: str = b.scenario) -> str:
            d = f"{cv - bv:+,}"
            return f"{scenario:<14} {metric:<26} {bv:>10,} {cv:>10,} {d:>9}"

        lines.append(row("total tokens (all turns)", b.total_tokens_all_turns, c.total_tokens_all_turns))
        lines.append(row("peak active tokens", b.peak_active_tokens, c.peak_active_tokens))
        lines.append(row("final context tokens", b.final_tokens, c.final_tokens))
        lines.append(row("inline observation tok", b.inline_observation_tokens, c.inline_observation_tokens))
        lines.append(row("stub tokens", b.stub_observation_tokens, c.stub_observation_tokens))
        lines.append(row("obs masked (count)", b.observations_masked, c.observations_masked))
        lines.append(row("dups stubbed (count)", b.duplicates_stubbed, c.duplicates_stubbed))
        for k, v in c.quality.items():
            lines.append(f"{c.scenario:<14} quality:{k:<18} {'-':>10} {v!s:>10} {'':>9}")
        lines.append("")
    return "\n".join(lines)


def results_json(pairs: list[tuple[RunResult, RunResult]]) -> str:
    return json.dumps(
        [asdict(r) for pair in pairs for r in pair], indent=2
    )
