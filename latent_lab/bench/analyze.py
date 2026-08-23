"""Aggregate all gate-run JSONs into the decision-gate report tables.

Usage: python -m latent_lab.bench.analyze --results-dir .rcc_work/remote_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_all(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(d.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception as e:
            out[p.stem] = {"error": str(e)}
    return out


def textual_table(data: dict) -> list[dict]:
    rows = []
    for name, r in data.items():
        if not isinstance(r, dict) or "baseline" not in r:
            continue
        rows.append({
            "run": name,
            "mode": r["baseline"],
            "split": r.get("split"),
            "acc": r.get("accuracy"),
            "nonterm": f"{r.get('non_termination_count')}/{r.get('n_examples')}",
            "wall_s": r.get("wall_seconds"),
        })
    return sorted(rows, key=lambda x: (x["split"] or "", x["mode"]))


def latent_train_table(data: dict) -> list[dict]:
    rows = []
    for name, r in data.items():
        if not isinstance(r, dict) or "config" not in r or "best_val_acc" not in r:
            continue
        c = r["config"]
        rows.append({
            "adapter": name,
            "mode": c.get("mode"), "interval": tuple(c.get("interval", ())),
            "k": c.get("k"), "seed": c.get("seed"),
            "best_val_acc": r.get("best_val_acc"),
            "best_step": r.get("best_step"),
            "final_loss": round(r.get("final_train_loss", float("nan")), 3),
            "wall_min": round((r.get("wall_seconds") or 0) / 60, 1),
        })
    return sorted(rows, key=lambda x: (x["mode"], x["k"], x["seed"]))


def eval_table(data: dict) -> list[dict]:
    rows = []
    for name, r in data.items():
        if not isinstance(r, dict) or "results" not in r:
            continue
        cfg = r.get("config", {})
        for ab, res in r["results"].items():
            rows.append({
                "eval": name,
                "adapter_cfg": f"{cfg.get('mode')}|K={cfg.get('k')}|s{cfg.get('seed')}",
                "split": r.get("split"),
                "ablation": ab,
                "acc": res.get("accuracy"),
                "by_depth": res.get("by_depth"),
            })
    return sorted(rows, key=lambda x: (x["adapter_cfg"], x["split"],
                                       str(x["ablation"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = load_all(Path(args.results_dir))

    lines = []

    def emit(title, rows, cols):
        lines.append(f"\n## {title}")
        if not rows:
            lines.append("(none)")
            return
        lines.append("\t".join(cols))
        for row in rows:
            lines.append("\t".join(str(row.get(c)) for c in cols))

    emit("TEXTUAL BASELINES", textual_table(data),
         ["run", "mode", "split", "acc", "nonterm", "wall_s"])
    emit("LATENT TRAINING", latent_train_table(data),
         ["adapter", "mode", "interval", "k", "seed", "best_val_acc",
          "best_step", "final_loss", "wall_min"])
    emit("LATENT EVALS + ABLATIONS", eval_table(data),
         ["adapter_cfg", "split", "ablation", "acc", "by_depth"])

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report)


if __name__ == "__main__":
    main()
