"""CLI: python -m bench.run_bench [--root DIR] [--json OUT] [--exact]"""

from __future__ import annotations

import argparse
import json as _json
from dataclasses import asdict
from pathlib import Path

from bench.harness import format_report, run_suite


def main() -> None:
    ap = argparse.ArgumentParser(description="RCC benchmark suite")
    ap.add_argument("--root", default=".rcc_bench", help="scratch dir for stores")
    ap.add_argument("--json", default=None, help="write raw results JSON here")
    ap.add_argument(
        "--exact", action="store_true",
        help="also run the suite with the exact tiktoken o200k_base counter",
    )
    args = ap.parse_args()

    root = Path(args.root)
    pairs = run_suite(root)
    print(format_report(pairs))
    print('tokenizer: rcc-approx-1 (deterministic approximation; deltas meaningful, absolutes approximate)')

    all_pairs = [("approx", pairs)]
    if args.exact:
        from rcc.tokens import count_tokens_exact

        exact_pairs = run_suite(root / "exact", tokenizer=count_tokens_exact)
        print("\n=== EXACT tokenizer: o200k_base (tiktoken) ===")
        print(format_report(exact_pairs))
        all_pairs.append(("exact-o200k_base", exact_pairs))

    if args.json:
        payload = [
            {**asdict(r), "suite": suite_name}
            for suite_name, pair_list in all_pairs
            for base, comp in pair_list
            for r in (base, comp)
        ]
        Path(args.json).write_text(_json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
