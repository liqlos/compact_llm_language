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
import sys
import time
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"

EVAL_ABLATIONS = ("zero_state", "bypass_interval", "clocks_off",
                  "reverse_clocks", "truncate_half", "swap_state",
                  "noise_state")


def interval_from_spec(spec: str, n_layers: int) -> tuple[int, int]:
    """'mid'/'full'/'head' proportional to depth, or explicit 'lo,hi'."""
    if "," in spec:
        lo, hi = (int(x) for x in spec.split(","))
        return (lo, hi)
    if spec == "full":
        return (0, n_layers)
    if spec == "mid":
        return (n_layers // 2, n_layers * 3 // 4)
    if spec == "head":
        return (n_layers * 3 // 4, n_layers)
    raise ValueError(f"unknown interval spec {spec}")

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
        except Exception:  # noqa: BLE001, S110 — memory probe is best-effort
            pass
    return out


def load_model(device="mps", model_id=None, revision=None):
    import torch
    import transformers

    from latent_lab.backends.gdn_patch import install
    from latent_lab.train.checkpointing import require_pinned_revision
    install()

    model_id = model_id or DEFAULT_MODEL_ID
    # Only revision=None selects the pinned default; falsey values such as
    # "", False or 0 are validated and rejected as-is, BEFORE any Hugging
    # Face contact: only immutable 40-hex commit revisions pass.
    revision = require_pinned_revision(
        DEFAULT_REVISION if revision is None else revision)
    tok = transformers.AutoTokenizer.from_pretrained(model_id,
                                                     revision=revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16).eval()
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


def build_eval_record(ex, order, scores) -> dict:
    """One lossless eval record: raw scores/order + gold/candidate identity.

    Derived fields (rank_of_gold/correct) are convenience only — the raw
    finite candidate scores in SCORED order, the model's ordering, the full
    candidate set, the answer and its candidate index are all retained so
    corrected scoring can be independently recomputed later.
    """
    if len(scores) != len(ex.candidates):
        raise ValueError(
            f"{ex.ex_id}: {len(scores)} scores for "
            f"{len(ex.candidates)} candidates")
    bad = [i for i, s in enumerate(scores)
           if s is None or not isinstance(s, (int, float))
           or s != s or s in (float("inf"), float("-inf"))]
    if bad:
        raise ValueError(f"{ex.ex_id}: non-finite raw scores at {bad}")
    gold_idx = ex.candidates.index(ex.answer)
    pred_rank = order.index(gold_idx) if gold_idx in order else -1
    return {
        "ex_id": ex.ex_id, "family": ex.family, "depth": ex.depth,
        "candidates": list(ex.candidates), "answer": ex.answer,
        "gold_candidate_index": gold_idx,
        "scores_raw": [float(s) for s in scores],
        "score_order": list(order),
        "rank_of_gold": pred_rank,
        "correct": 1.0 if pred_rank == 0 else 0.0,
        "n_candidates": len(order),
    }


def rescore_records(records) -> float:
    """Independently recompute accuracy from RAW record fields alone.

    Verifies that derived rank/correct agree with a fresh computation from
    scores_raw/score_order/gold_candidate_index, so any future scorer fix
    can be re-applied to persisted evidence without re-running the model.
    """
    correct = 0
    for r in records:
        scores = r["scores_raw"]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if list(order) != list(r["score_order"]):
            raise ValueError(
                f"{r.get('ex_id')}: score_order disagrees with scores_raw; "
                "evidence inconsistent")
        gold = r["gold_candidate_index"]
        rank = order.index(gold) if gold in order else -1
        if rank != r["rank_of_gold"] or \
                (1.0 if rank == 0 else 0.0) != r["correct"]:
            raise ValueError(
                f"{r.get('ex_id')}: derived fields disagree with raw "
                "scores; evidence inconsistent")
        correct += 1 if rank == 0 else 0
    return correct / max(1, len(records))


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
        records.append(build_eval_record(ex, order, scores))
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


def _dependency_versions() -> dict:
    import torch

    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    return out


def recipe_from_config(cfg: dict, suite_sha256: str) -> dict:
    """Canonical exact-identity recipe implied by cfg (re-exported)."""
    from latent_lab.train.checkpointing import recipe_from_config
    return recipe_from_config(cfg, suite_sha256)


def _mark_run_fatal(out: Path, e: BaseException) -> None:
    """Atomically mark the run fatal; no success artifact may follow."""
    from latent_lab.train.checkpointing import write_run_status
    write_run_status(out, "fatal",
                     command=" ".join(sys.argv),
                     error_type=type(e).__name__, error=str(e))


def cmd_train(args):
    from latent_lab.train.checkpointing import (
        FatalRunInvalidError, require_pinned_revision, write_run_status)

    # fail closed BEFORE loading/training/saving on a mutable revision
    revision = require_pinned_revision(args.revision)
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_run_status(out, "running",
                     command=" ".join(sys.argv),
                     model=args.model, revision=revision, seed=args.seed)
    try:
        _train_inner(args, out, device, revision)
    except FatalRunInvalidError as e:
        # explicit fatal status; NO success artifact may exist afterwards
        _mark_run_fatal(out, e)
        raise
    except Exception as e:
        # an unexpected crash must never leave the run marked 'running'
        # (which readers could confuse with an active/incomplete run);
        # it is marked fatal atomically and re-raised unchanged
        _mark_run_fatal(out, e)
        raise
    write_run_status(out, "complete",
                     command=" ".join(sys.argv))


def _train_inner(args, out: Path, device: str, revision: str):
    import torch

    from latent_lab.backends.localized import LocalizedRecurrence
    from latent_lab.bench.suite import build_suite
    from latent_lab.train.checkpointing import (
        BestCheckpointTracker, guarded_optimizer_step, sha256_file,
        write_train_generation)

    model, tok = load_model(device, args.model, revision)
    interval = interval_from_spec(
        args.interval, model.config.num_hidden_layers)
    rec = LocalizedRecurrence(model, None, interval=interval, max_k=args.max_k,
                              lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                              grad_checkpoint=True)
    rec.clock.to(device)
    params = [p for p in rec.trainable_parameters()]
    if args.optimizer != "adamw":
        raise ValueError(f"unsupported optimizer {args.optimizer!r}; "
                         "supported: ['adamw']")
    opt = torch.optim.AdamW(params, lr=args.lr,
                            weight_decay=args.weight_decay)

    suite = build_suite()
    train = SuiteTensors(tok, list(suite.train))
    val = SuiteTensors(tok, list(suite.validation))
    val_idx = list(range(len(val.examples)))
    n_val_eval = min(len(val_idx), args.val_examples)

    order = list(range(len(train)))
    history = []
    tracker = BestCheckpointTracker()
    t0 = time.perf_counter()
    base_lr = args.lr

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return base_lr * (step + 1) / args.warmup
        if args.lr_schedule == "cosine":
            import math
            t = (step - args.warmup) / max(1, args.steps - args.warmup)
            return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
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
        # fail-stop: any fault here raises FatalRunInvalidError and kills
        # the run — never retried against the mutated optimizer
        guarded_optimizer_step(opt, loss.detach(), params, args.clip)
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
            if tracker.update(ev["accuracy"], rec.adapter_state_dict(),
                              step=step + 1):
                print(f"  new best {ev['accuracy']} @step {step+1}",
                      flush=True)

    if not tracker.has_best():
        from latent_lab.train.checkpointing import EmptyCheckpointError
        raise EmptyCheckpointError(
            "no finite validation checkpoint was accepted; refusing to "
            "report or save final state")

    mode = ("D-full" if args.interval == "full"
            else "F-control" if args.k == 0 else "E-localized")
    suite_sha = suite.manifest()["sha256"]
    cfg = {
        "mode": mode,
        "model": args.model, "revision": revision,
        "interval": list(interval), "k": args.k,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lr": args.lr, "steps": args.steps,
        "seed": args.seed, "max_k": args.max_k,
        "optimizer": args.optimizer, "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule, "warmup": args.warmup,
        "clip": args.clip, "detach_z0": args.detach_z0,
        "grad_checkpoint": True,
        "device": device,
        "train_examples": len(train),
        "label": getattr(args, "label", None),
        "suite_sha256": suite_sha,
    }
    recipe = recipe_from_config(cfg, suite_sha)

    # reload the SELECTED BEST before reporting/saving — final state is
    # never silently used as evidence
    rec.load_adapter_state(tracker.best_state())
    bundle_path = out / "best_params.pt"
    bundle = rec.export_adapter_bundle(
        bundle_path, model_id=args.model, revision=revision,
        config=cfg,
        metrics={"best_val_acc": tracker.best_score})
    report = {
        "config": cfg,
        "model": args.model, "revision": revision,
        "suite_sha256": suite_sha,
        "recipe": recipe,
        "checkpoint_content_digest": bundle["content_digest"],
        "checkpoint_sha256": sha256_file(bundle_path),
        "precision": {
            "backbone_dtype": str(next(model.parameters()).dtype),
            "trainables_dtype": "torch.float32",
        },
        "best_val_acc": tracker.best_score,
        "best_step": tracker.best_step,
        "val_history": history,
        "final_train_loss": final_loss,
        "gpu_mem": _gpu_mem_report(device),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "peak_rss_mib": round(peak_rss_mib(), 1),
        "platform": platform.platform(),
    }
    # report + manifest promoted as ONE coherent generation; the manifest
    # (written last, atomically) carries the digests of both files and is
    # the commit marker any reader must verify
    manifest = {
        "kind": "latent_lab.train_generation", "status": "complete",
        "argv": list(sys.argv),
        "command": " ".join(sys.argv),
        "dependencies": _dependency_versions(),
        "precision": report["precision"],
        "seed": args.seed,
        "label": getattr(args, "label", None),
        "identity": {"model_id": args.model, "revision": revision},
        "recipe": recipe,
        "suite_sha256": suite_sha,
        "checkpoint_content_digest": bundle["content_digest"],
        "checkpoint_sha256": sha256_file(bundle_path),
        "wall_seconds": report["wall_seconds"],
    }
    write_train_generation(out, manifest=manifest, report=report)
    print(f"[train done] best_val={tracker.best_score} "
          f"@step {tracker.best_step} -> {out}")


def parse_ablation_cli(name, k_steps):
    """Strict CLI ablation parser: unknown modes are REJECTED, never run
    silently clean. Returns the latent_steps ablation dict."""
    if name is None:
        return None
    if name == "clocks_off":
        return {"clocks": "off"}
    if name == "reverse_clocks":
        return {"clocks": "reverse"}
    if name.startswith("shuffle_clocks:"):
        perm = name.split(":", 1)[1]
        return {"clocks": f"shuffle_perm:{perm}"}  # validated downstream
    if name == "truncate_half":
        return {"truncate_k": max(0, k_steps // 2)}
    if name in ("zero_state", "bypass_interval", "swap_state", "noise_state"):
        return {name: True}
    raise ValueError(
        f"unknown ablation {name!r}; supported: {sorted(EVAL_ABLATIONS)} "
        "or 'shuffle_clocks:i,j,...'")


def cmd_eval(args):
    import torch

    from latent_lab.backends.localized import LocalizedRecurrence
    from latent_lab.bench.suite import build_suite
    from latent_lab.train.checkpointing import (
        AdapterBundleError,
        AdapterBundleIdentityError,
        atomic_write_json,
        load_adapter_bundle,
        recipe_from_config,
        require_pinned_revision,
        verify_generation,
    )

    torch.manual_seed(args.seed)
    device = args.device
    report = json.loads(
        (Path(args.adapter) / "train_report.json").read_text())
    cfg = report["config"]
    model_id = cfg.get("model")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(
            f"adapter {args.adapter} carries no immutable model identity; "
            "refusing to evaluate")
    try:
        # fail closed BEFORE any model load on a mutable/missing revision
        revision = require_pinned_revision(cfg.get("revision"))
    except AdapterBundleError as e:
        raise ValueError(
            f"adapter {args.adapter} carries no immutable pinned model "
            "revision; refusing to evaluate") from e

    suite = build_suite()
    suite_sha = suite.manifest()["sha256"]

    # Identity-validate + digest-verify the on-disk generation and bundle
    # BEFORE any model/tokenizer load: a tampered adapter must never
    # trigger an arbitrary model fetch prior to rejection.
    manifest = verify_generation(args.adapter)
    report_recipe = report.get("recipe")
    recipe = recipe_from_config(cfg, suite_sha)
    if report_recipe != recipe:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: train_report recipe {report_recipe} "
            f"is not the canonical recipe of its own config "
            f"(config_sha256 {recipe['config_sha256']}); refusing to "
            "evaluate")
    if manifest.get("recipe") != recipe:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: run_manifest recipe "
            f"{manifest.get('recipe')} disagrees with the canonical "
            f"recipe (config_sha256 {recipe['config_sha256']}); refusing "
            "to evaluate")
    if manifest.get("suite_sha256") != suite_sha:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: manifest suite_sha256 "
            f"{manifest.get('suite_sha256')!r} != current suite "
            f"{suite_sha!r}; refusing to evaluate")
    m_ident = manifest.get("identity") or {}
    if m_ident.get("model_id") != model_id \
            or require_pinned_revision(m_ident.get("revision")) != revision:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: manifest identity {m_ident} "
            f"disagrees with the report identity ({model_id!r}, "
            f"{revision!r})")
    m_seed = manifest.get("seed")
    if m_seed != cfg.get("seed"):
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: manifest seed {m_seed!r} != config "
            f"seed {cfg.get('seed')!r}")
    state = load_adapter_bundle(Path(args.adapter) / "best_params.pt",
                                model_id=model_id, revision=revision,
                                recipe=recipe)

    model, tok = load_model(device, model_id, revision)
    interval = tuple(cfg["interval"])
    rec = LocalizedRecurrence(model, None, interval=interval,
                              max_k=cfg["max_k"], lora_r=cfg["lora_r"],
                              lora_alpha=float(cfg.get("lora_alpha", 16.0)),
                              grad_checkpoint=False)
    rec.load_adapter_state(state)

    split_name = args.split
    data = SuiteTensors(tok, list(getattr(suite, split_name)))
    idx = list(range(len(data.examples)))
    k = args.k if args.k is not None else cfg["k"]

    ablation = parse_ablation_cli(args.ablate, k)
    res = evaluate(rec, data, k, idx, ablate=ablation,
                   tag=f"{cfg['mode']}|{split_name}|{args.ablate or 'clean'}|K={k}",
                   limit=args.limit)
    results = {args.ablate or "clean": res}
    # prove the persisted evidence is independently rescorable right now
    rescore_records(res["records"])
    print(json.dumps({k2: v for k2, v in res.items() if k2 != "records"},
                     indent=1))

    if args.out:
        payload = {
            "status": "complete",
            "adapter": args.adapter, "split": split_name,
            "config": cfg, "model": cfg.get("model"),
            "revision": cfg.get("revision"),
            "identity": {
                "model_id": model_id, "revision": revision,
                "suite_sha256": suite_sha,
                "tokenizer_class": type(tok).__name__,
                "interval": list(interval), "max_k": cfg["max_k"],
                "k_steps": k,
                "ablation": args.ablate or "clean",
                "checkpoint_content_digest":
                    report.get("checkpoint_content_digest"),
            },
            "suite_sha256": suite_sha,
            "device": device, "seed": args.seed,
            "results": results,
            "peak_rss_mib": round(peak_rss_mib(), 1),
            "platform": platform.platform(),
        }
        atomic_write_json(args.out, payload)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--k", type=int, default=4)
    tr.add_argument("--interval", default="mid",
                    help="'mid'|'full'|'head'|'lo,hi'")
    tr.add_argument("--steps", type=int, default=600)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--lora-r", type=int, default=8)
    tr.add_argument("--lora-alpha", type=float, default=16.0)
    tr.add_argument("--max-k", type=int, default=16)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--eval-every", type=int, default=100)
    tr.add_argument("--val-examples", type=int, default=28)
    tr.add_argument("--warmup", type=int, default=30)
    tr.add_argument("--lr-schedule", default="constant",
                    choices=["constant", "cosine"])
    tr.add_argument("--optimizer", default="adamw")
    tr.add_argument("--weight-decay", type=float, default=0.01)
    tr.add_argument("--clip", type=float, default=0.5)
    tr.add_argument("--detach-z0", action="store_true")
    tr.add_argument("--label", default=None,
                    help="preregistered run label (bound into evidence; "
                    "never derived from output path)")
    tr.add_argument("--device", default="mps")
    tr.add_argument("--model", default=DEFAULT_MODEL_ID)
    tr.add_argument("--revision", default=DEFAULT_REVISION)
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
