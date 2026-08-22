"""Latent baseline training + evaluation driver over the behavioral suite.

Trains (frozen backbone + interval LoRA + zero-init step clock) to produce
gold answers after K latent steps, then evaluates exact candidate ranking
on validation/test splits with causal ablations.

Configurations:
  D  --interval full     (loop = entire decoder, Coconut-style feedback)
  E  --interval mid      (localized loop; default [12,18) on 24L)
  F  k_steps=0           (tail-only control, tail-adjacent LoRA still trains)

Usage (train):
  python -m latent_lab.bench.latent_run train --k 4 --interval mid \
      --steps 800 --out .rcc_work/latent_E_k4
Usage (eval):
  python -m latent_lab.bench.latent_run eval --adapter .rcc_work/latent_E_k4 \
      --split test_id [--ablate zero_state] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
INTERVALS = {"mid": (12, 18), "full": (0, 24), "head": (18, 24)}

ANSWER_PREFIX = " "  # prompt ends with "Answer:" -> continuation " <ans>"


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def _gpu_mem_report(device: str) -> dict:
    import torch
    out: dict = {"peak_rss_mib": round(peak_rss_mib(), 1)}
    if device.startswith("cuda"):
        out["cuda_peak_alloc_mib"] = round(
            torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        out["cuda_peak_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 2 ** 20, 1)
    elif device.startswith("mps"):
        try:
            out["mps_current_mib"] = round(
                torch.mps.current_allocated_memory() / 2 ** 20, 1)
        except Exception:
            pass
    return out


def load_model(device="mps"):
    import torch
    import transformers

    from latent_lab.backends.gdn_patch import install
    install()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID,
                                                     revision=REVISION)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16).eval()
    model.to(device)
    return model, tok


class SuiteTensors:
    """Tokenized suite with cached candidate token ids."""

    def __init__(self, tok, examples):
        self.examples = examples
        self.prompt_ids = []
        self.answer_ids = []
        self.cand_ids = []
        for ex in examples:
            p = tok(ex.prompt, return_tensors="pt", return_dict=True
                    ).input_ids
            self.prompt_ids.append(p)
            a = tok(ANSWER_PREFIX + ex.answer, add_special_tokens=False,
                    return_tensors="pt", return_dict=True).input_ids
            self.answer_ids.append(a)
            cs = tuple(tok(ANSWER_PREFIX + c, add_special_tokens=False)
                       .input_ids for c in ex.candidates)
            self.cand_ids.append(cs)

    def __len__(self):
        return len(self.examples)


def evaluate(rec, data: SuiteTensors, k_steps, indices, *, ablate=None,
             tag="", limit=None):
    """Ranking accuracy over selected examples; per-example records."""
    device = next(rec.model.parameters()).device
    t0 = time.perf_counter()
    records = []
    for i in (indices[:limit] if limit else indices):
        ex = data.examples[i]
        partner = None
        if ablate and ablate.get("swap_state"):
            j = (i + 1) % len(data.examples)
            partner = data.prompt_ids[j].to(device)
        order, scores, rep = rec.rank_candidates(
            data.prompt_ids[i].to(device), data.cand_ids[i], k_steps,
            ablate=ablate, partner_input_ids=partner)
        pred_rank = order.index(0) if 0 in order else -1
        records.append({
            "ex_id": ex.ex_id, "family": ex.family, "depth": ex.depth,
            "correct": 1.0 if pred_rank == 0 else 0.0,
            "rank_of_gold": pred_rank, "n_candidates": len(order),
        })
    acc = sum(r["correct"] for r in records) / max(1, len(records))
    by_depth = {}
    for r in records:
        by_depth.setdefault(r["depth"], []).append(r["correct"])
    by_depth = {d: round(sum(v) / len(v), 4) for d, v in sorted(by_depth.items())}
    by_family = {}
    for r in records:
        by_family.setdefault(r["family"], []).append(r["correct"])
    by_family = {f: round(sum(v) / len(v), 4)
                 for f, v in sorted(by_family.items())}
    return {
        "tag": tag, "ablate": ablate or {}, "k_steps": k_steps,
        "n": len(records), "accuracy": round(acc, 4),
        "by_depth": by_depth, "by_family": by_family,
        "seconds": round(time.perf_counter() - t0, 1),
        "records": records,
    }


def cmd_train(args):
    import torch

    from latent_lab.backends.localized import LocalizedRecurrence
    from latent_lab.bench.suite import build_suite

    torch.manual_seed(args.seed)
    device = args.device
    model, tok = load_model(device)
    interval = INTERVALS[args.interval]
    rec = LocalizedRecurrence(model, None, interval=interval, max_k=args.max_k,
                              lora_r=args.lora_r, grad_checkpoint=True)
    rec.clock.to(device)
    params = [p for p in rec.trainable_parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    suite = build_suite()
    train = SuiteTensors(tok, list(suite.train))
    val = SuiteTensors(tok, list(suite.validation))
    val_idx = list(range(len(val.examples)))
    n_val_eval = min(len(val_idx), args.val_examples)

    order = list(range(len(train)))
    history = []
    best = {"acc": -1.0, "state": None, "step": -1}
    t0 = time.perf_counter()
    base_lr = args.lr

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return base_lr * (step + 1) / args.warmup
        return base_lr

    perm = None
    final_loss = float("nan")
    for step in range(args.steps):
        if step % len(order) == 0 or perm is None:
            perm = torch.randperm(len(order)).tolist()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        i = perm[step % len(order)]
        loss = rec.loss_on_example(train.prompt_ids[i].to(device),
                                   train.answer_ids[i].to(device),
                                   args.k, detach_z0=args.detach_z0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        final_loss = float(loss.detach())
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step} loss {final_loss:.4f} lr {lr_at(step):.2e} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            ev = evaluate(rec, val, args.k, val_idx, tag="val",
                          limit=n_val_eval)
            ev["step"] = step + 1
            ev.pop("records")
            history.append(ev)
            print(f"  val acc {ev['accuracy']} @step {step+1}", flush=True)
            if ev["accuracy"] > best["acc"]:
                best = {
                    "acc": ev["accuracy"], "step": step + 1,
                    "state": rec.adapter_state_dict(),
                }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], out / "best_params.pt")
    report = {
        "config": {
            "mode": ("D-full" if args.interval == "full" else
                     "F-control" if args.k == 0 else "E-localized"),
            "interval": list(interval), "k": args.k,
            "lora_r": args.lora_r, "lr": args.lr, "steps": args.steps,
            "seed": args.seed, "max_k": args.max_k,
            "detach_z0": args.detach_z0, "device": device,
            "train_examples": len(train), "grad_checkpoint": True,
        },
        "model": MODEL_ID, "revision": REVISION,
        "suite_sha256": suite.manifest()["sha256"],
        "best_val_acc": best["acc"], "best_step": best["step"],
        "val_history": history,
        "final_train_loss": final_loss,
        "gpu_mem": _gpu_mem_report(device),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "peak_rss_mib": round(peak_rss_mib(), 1),
        "platform": platform.platform(),
    }
    with open(out / "train_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"[train done] best_val={best['acc']} @step {best['step']} "
          f"-> {out}")


def load_adapter_state(path):
    import torch

    return torch.load(Path(path) / "best_params.pt", map_location="cpu")


def cmd_eval(args):
    import torch

    from latent_lab.backends.localized import LocalizedRecurrence
    from latent_lab.bench.suite import build_suite

    torch.manual_seed(args.seed)
    device = args.device
    cfg = json.load(open(Path(args.adapter) / "train_report.json")
                    )["config"]
    model, tok = load_model(device)
    interval = tuple(cfg["interval"])
    rec = LocalizedRecurrence(model, None, interval=interval,
                              max_k=cfg["max_k"], lora_r=cfg["lora_r"],
                              grad_checkpoint=False)
    state = load_adapter_state(args.adapter)
    rec.load_adapter_state(state)

    suite = build_suite()
    split_name = args.split
    data = SuiteTensors(tok, list(getattr(suite, split_name)))
    idx = list(range(len(data.examples)))
    k = args.k if args.k is not None else cfg["k"]

    results = {}
    ablation = None
    if args.ablate:
        ablation = {args.ablate: {"zero_state": True, "bypass_interval": True,
                                  "clocks_off": True}.get(args.ablate, True)}
        if args.ablate == "clocks_off":
            ablation = {"clocks": "off"}
        elif args.ablate == "reverse_clocks":
            ablation = {"clocks": "reverse"}
        elif args.ablate.startswith("shuffle"):
            perm = args.ablate.split(":", 1)[1]
            ablation = {"clocks": f"shuffle_perm:{perm}"}
        elif args.ablate == "truncate_half":
            ablation = {"truncate_k": max(0, k // 2)}
        elif args.ablate == "swap_state":
            ablation = {"swap_state": True}
        elif args.ablate == "noise_state":
            ablation = {"noise_state": True}
    res = evaluate(rec, data, k, idx, ablate=ablation,
                   tag=f"{cfg['mode']}|{split_name}|{args.ablate or 'clean'}|K={k}",
                   limit=args.limit)
    results[args.ablate or "clean"] = res
    print(json.dumps({k2: v for k2, v in res.items() if k2 != "records"},
                     indent=1))

    if args.out:
        payload = {
            "adapter": args.adapter, "split": split_name,
            "config": cfg, "model": MODEL_ID, "revision": REVISION,
            "suite_sha256": suite.manifest()["sha256"],
            "device": device, "seed": args.seed,
            "results": results,
            "peak_rss_mib": round(peak_rss_mib(), 1),
            "platform": platform.platform(),
        }
        Path(args.out).write_text(json.dumps(payload, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--k", type=int, default=4)
    tr.add_argument("--interval", default="mid",
                    choices=list(INTERVALS))
    tr.add_argument("--steps", type=int, default=600)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--lora-r", type=int, default=8)
    tr.add_argument("--max-k", type=int, default=16)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--eval-every", type=int, default=100)
    tr.add_argument("--val-examples", type=int, default=28)
    tr.add_argument("--warmup", type=int, default=30)
    tr.add_argument("--clip", type=float, default=0.5)
    tr.add_argument("--detach-z0", action="store_true")
    tr.add_argument("--device", default="mps")
    tr.add_argument("--out", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--adapter", required=True)
    ev.add_argument("--split", default="test_id")
    ev.add_argument("--k", type=int, default=None)
    ev.add_argument("--ablate", default=None)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--device", default="mps")
    ev.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
