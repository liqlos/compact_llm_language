"""CLI: run the live-model evaluation suite.

Offline (no provider, deterministic fixture):

    uv run python -m evals.run_live_eval --out .rcc_eval/results.json

Real run (any OpenAI-compatible endpoint; local servers need no key):

    RCC_EVAL_BASE_URL=http://127.0.0.1:11434/v1 RCC_EVAL_MODEL=llama3.1:8b \
        uv run python -m evals.run_live_eval --provider openai-compat

    RCC_EVAL_BASE_URL=https://api.openai.com/v1 RCC_EVAL_MODEL=gpt-4o-mini \
        RCC_EVAL_API_KEY=... uv run python -m evals.run_live_eval --provider openai-compat

Expansion channels (--expansion): `tool` (default) lets the model request
evidence via EXPAND lines; `closed` removes that option entirely -- compiled
fact recall is never scored there, only honest unavailability (EF3); `both`
runs each task in both channels as separate, non-comparable measurements.
Results are versioned machine-readable JSON including policy/tokenizer/model
params, per-result context hashes, raw responses and errors; a hard call
budget isolates partial results when exhausted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.scenarios import SCENARIOS
from evals.harness import RIR_CASE, run_suite
from evals.provider import FakeProvider, OpenAICompatClient
from rcc import count_tokens


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="live-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=("fake", "openai-compat"), default="fake",
                   help="fake = deterministic fixture; openai-compat = live endpoint")
    p.add_argument("--base-url", default=None,
                   help="endpoint base URL (or RCC_EVAL_BASE_URL)")
    p.add_argument("--model", default=None, help="model name (or RCC_EVAL_MODEL)")
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--max-completion-tokens", type=int, default=512)
    p.add_argument("--max-expand-rounds", type=int, default=1,
                   help="bounded expansion rounds per task (tool channel)")
    p.add_argument("--expansion", choices=("tool", "closed", "both"), default="tool",
                   help="expansion channel; arms are never compared cross-channel")
    p.add_argument("--max-calls", type=int, default=None,
                   help="hard total-call budget (default: computed suite bound)")
    p.add_argument("--scenarios", nargs="*", default=None,
                   help="subset of scenario names (default: all six)")
    p.add_argument("--out", default=".rcc_eval/results.json",
                   help="where to write machine-readable results")
    p.add_argument("--check", action="store_true",
                   help="only ping the provider once and exit (no suite run)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.provider == "openai-compat":
        try:
            client = OpenAICompatClient.from_env(
                args.base_url, args.model,
                timeout_s=args.timeout_s,
                max_completion_tokens=args.max_completion_tokens,
            )
        except Exception as e:  # noqa: BLE001 -- CLI boundary
            print(f"error: provider misconfigured: {e}", file=sys.stderr)
            return 2
        if args.check:
            try:
                client.check()
            except Exception as e:  # noqa: BLE001 -- CLI boundary
                print(f"error: provider check failed: {e}", file=sys.stderr)
                return 2
            print(f"ok: {client.model} @ {client.base_url} answered")
            return 0
    else:
        if args.check:
            print("ok: fake provider is always available")
            return 0
        client = FakeProvider()

    unknown = set(args.scenarios or []) - ({sc.name for sc in SCENARIOS} | {RIR_CASE.name})
    if unknown:
        known = ", ".join(sorted({sc.name for sc in SCENARIOS} | {RIR_CASE.name}))
        print(f"error: unknown scenarios {sorted(unknown)}; known: {known}",
              file=sys.stderr)
        return 2

    report = run_suite(
        client,
        workdir=Path(".rcc_eval/work"),
        tokenizer=count_tokens,
        max_expand_rounds=args.max_expand_rounds,
        scenario_filter=args.scenarios,
        expansion=args.expansion,
        max_calls=args.max_calls,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json(), encoding="utf-8")

    for arm, agg in report.summary_by_arm().items():
        print(f"{arm:<16} ef1={agg['mean_ef1_recall']:.2f} "
              f"ef2={agg['mean_ef2_numeric_integrity']:.2f} "
              f"crit_fail={agg['critical_fails']} "
              f"ctx_tok={agg['total_context_tokens']:,} "
              f"expand={agg['expand_reads']} err={agg['errors']}")
    for c in report.control_results():
        print(f"control[{c['scenario']}/{c['mode']}/{c['expansion']}] "
              f"sha={c['context_sha16']} error={c['error']}")
    print(f"calls spent: {report.config['budget_spent']}/{report.config['budget_max_calls']}")
    print(f"results: {out}")
    failed = sum(1 for r in report.results if r.error)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
