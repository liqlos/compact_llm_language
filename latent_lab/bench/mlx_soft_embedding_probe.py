"""MLX soft/off-vocabulary embedding blocker probe (Apple Silicon).

Reproduces or refutes runtime hazards of feeding non-token embeddings into a
hybrid Qwen on MLX (docs BLOCKERS.md B6):

  E1 baseline token path            — prefill/decode wall-clock
  E2 exact vocabulary embeddings    — logits identical to token path? speed?
  E3 slightly perturbed embeddings  — divergence + speed
  E4 zero/random latent embedding   — distribution shift + derailing + speed
  E5 prefix-cache reuse             — does the hybrid cache actually skip work?

Honesty rules: unsupported operations are reported as {"supported": false},
never hidden. This probe measures RUNTIME behaviour only; it makes no claim
about reasoning quality.

Usage:
  python -m latent_lab.bench.mlx_soft_embedding_probe \
      --out latent_lab/bench/results/mlx_soft_embedding_probe.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--decode-steps", type=int, default=32)
    args = ap.parse_args()

    report: dict = {
        "probe": "rcc_mlx_soft_embedding_probe",
        "version": 1,
        "model": args.model,
        "platform": platform.platform(),
        "macos": platform.mac_ver()[0],
        "status": "running",
        "experiments": {},
    }
    try:
        _run(args, report)
    except Exception as e:  # noqa: BLE001 — always persist a report
        report["status"] = "error"
        report["error"] = f"{type(e).__name__}: {e}"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    print(f"[mlx_probe] status={report['status']} -> {args.out}")
    return 0 if report["status"] == "ok" else 3


def _run(args, report: dict) -> None:
    import mlx.core as mx

    import importlib.metadata as im

    from mlx_lm import load

    report["versions"] = {
        "mlx": im.version("mlx"),
        "mlx-lm": im.version("mlx-lm"),
    }

    model, tokenizer = load(args.model)
    tm = getattr(getattr(model, "language_model", model), "model", model)
    embed = tm.embed_tokens

    experiments = report["experiments"]

    # ---- structure ------------------------------------------------------
    cache_classes = [type(c).__name__ for c in model.make_cache()]
    experiments["structure"] = {
        "n_layers": len(tm.layers),
        "cache_classes_per_layer_group": cache_classes[:8],
        "input_embeddings_supported": "input_embeddings"
        in (_sig_params(tm),),
    }
    experiments["structure"]["input_embeddings_supported"] = (
        experiments["structure"]["input_embeddings_supported"]
    )

    ids_full = mx.array([[*range(1000, 1000 + args.prompt_tokens)]])
    emb_std = float(mx.std(embed(ids_full)))

    def prefill_and_decode(feed_fn, label: str, cache, decode_steps: int):
        """Common timing harness. feed_fn(cache) -> logits for last position."""
        t0 = time.perf_counter()
        logits = feed_fn(cache)
        mx.eval(logits)
        prefill_s = time.perf_counter() - t0
        n_prefill = args.prompt_tokens
        tok = mx.argmax(logits[:, -1], axis=-1)[:, None]
        generated = []
        t0 = time.perf_counter()
        for _ in range(decode_steps):
            step_logits = tm(inputs=tok, cache=cache)
            mx.eval(step_logits)
            generated.append(int(tok.item()))
            tok = mx.argmax(step_logits[:, -1], axis=-1)[:, None]
        decode_s = time.perf_counter() - t0
        return {
            "label": label,
            "prefill_seconds": round(prefill_s, 4),
            "prefill_tok_per_s": round(n_prefill / prefill_s, 1),
            "decode_seconds": round(decode_s, 4),
            "decode_tok_per_s": round(decode_steps / decode_s, 1),
            "generated_first10": generated[:10],
        }

    def token_feed(cache):
        return tm(inputs=ids_full, cache=cache)

    def embed_feed(embeds):
        def feed(cache):
            return tm(inputs=None, cache=cache, input_embeddings=embeds)
        return feed

    # warmup (compile/graph caches)
    w = model.make_cache()
    mx.eval(token_feed(w))

    # ---- E1 baseline -----------------------------------------------------
    base_cache = model.make_cache()
    e1 = prefill_and_decode(token_feed, "tokens", base_cache, args.decode_steps)

    # reference logits (no cache mutation afterwards matters not; fresh below)
    ref_cache = model.make_cache()
    ref_logits = token_feed(ref_cache)
    mx.eval(ref_logits)

    # ---- E2 exact vocab embeddings ---------------------------------------
    e2c = model.make_cache()
    try:
        e2 = prefill_and_decode(
            embed_feed(embed(ids_full)), "exact_vocab_embeds", e2c,
            args.decode_steps,
        )
        e2_cache = model.make_cache()
        emb_logits = embed_feed(embed(ids_full))(e2_cache)
        mx.eval(emb_logits)
        e2["logits_equal_token_path"] = bool(
            mx.allclose(emb_logits, ref_logits, atol=1e-4).item()
        )
        e2["max_abs_logit_diff"] = float(mx.max(mx.abs(emb_logits - ref_logits)).item())
        e2["same_continuation_as_baseline"] = (
            e2["generated_first10"] == e1["generated_first10"]
        )
    except Exception as ex:  # noqa: BLE001
        e2 = {"supported": False, "error": f"{type(ex).__name__}: {ex}"}
    experiments["E2_exact_vocab_embeds"] = e2

    # ---- E3 perturbed embeddings ------------------------------------------
    noise = mx.random.normal(embed(ids_full).shape) * (0.01 * emb_std)
    e3c = model.make_cache()
    try:
        e3 = prefill_and_decode(
            embed_feed(embed(ids_full) + noise), "perturbed_embeds", e3c,
            args.decode_steps,
        )
        p_cache = model.make_cache()
        p_logits = embed_feed(embed(ids_full) + noise)(p_cache)
        mx.eval(p_logits)
        e3["kl_vs_token_path"] = _kl(p_logits, ref_logits)
        e3["decode_slowdown_vs_E2"] = round(
            e2["decode_seconds"] / e3["decode_seconds"], 2
        ) if isinstance(e2.get("decode_seconds"), float) else None
    except Exception as ex:  # noqa: BLE001
        e3 = {"supported": False, "error": f"{type(ex).__name__}: {ex}"}
    experiments["E3_perturbed_embeds"] = e3

    # ---- E4 zero/random off-manifold token ---------------------------------
    e4 = {}
    for variant, vec in (
        ("zero", mx.zeros((1, 1, embed(ids_full).shape[-1]))),
        ("random", mx.random.normal((1, 1, embed(ids_full).shape[-1])) * emb_std),
    ):
        c = model.make_cache()
        try:
            base_logits = token_feed(c)
            mx.eval(base_logits)
            lat_logits = tm(inputs=None, cache=c, input_embeddings=vec)
            mx.eval(lat_logits)
            nxt = int(mx.argmax(lat_logits[:, -1], axis=-1).item())
            ref_nxt = int(mx.argmax(ref_logits[:, -1], axis=-1).item())
            cont = []
            tok = mx.array([[nxt]])
            t0 = time.perf_counter()
            for _ in range(args.decode_steps):
                lg = tm(inputs=tok, cache=c)
                mx.eval(lg)
                cont.append(int(tok.item()))
                tok = mx.argmax(lg[:, -1], axis=-1)[:, None]
            decode_s = time.perf_counter() - t0
            e4[variant] = {
                "supported": True,
                "next_token_diverges_from_baseline": bool(nxt != ref_nxt),
                "kl_next_token_vs_baseline": _kl(lat_logits, ref_logits),
                "continuation_matches_baseline": cont == e1["generated_first10"]
                [: len(cont)],
                "post_offmanifold_decode_tok_per_s": round(
                    args.decode_steps / decode_s, 1),
                "baseline_decode_tok_per_s": e1["decode_tok_per_s"],
            }
        except Exception as ex:  # noqa: BLE001
            e4[variant] = {"supported": False,
                           "error": f"{type(ex).__name__}: {ex}"}
    experiments["E4_zero_random_latent_token"] = e4

    # ---- E5 prefix cache reuse ---------------------------------------------
    e5: dict = {"supported": False}
    try:
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

        c = model.make_cache()
        mx.eval(token_feed(c))                       # fill with prompt
        reusable = can_trim_prompt_cache(c)
        e5["can_trim"] = bool(reusable)
        if reusable:
            ok = trim_prompt_cache(c, args.prompt_tokens - 8)
            suffix = ids_full[:, -8:]
            t0 = time.perf_counter()
            lg = tm(inputs=suffix, cache=c)
            mx.eval(lg)
            reuse_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            full_cache = model.make_cache()
            lg2 = tm(inputs=ids_full, cache=full_cache)
            mx.eval(lg2)
            full_s = time.perf_counter() - t0
            e5.update({
                "supported": True,
                "trim_returned": bool(ok),
                "reused_prefix_8tok_prefill_seconds": round(reuse_s, 5),
                "full_reprefill_64tok_seconds": round(full_s, 5),
                "speedup": round(full_s / max(1e-9, reuse_s * 8), 2),
            })
    except Exception as ex:  # noqa: BLE001
        e5 = {"supported": False, "error": f"{type(ex).__name__}: {ex}"}
    experiments["E5_prefix_cache_reuse"] = e5

    # ---- E6 ordering control ----------------------------------------------
    # If plain-token decode ALSO slows by now, earlier slowdowns are
    # process-order effects, NOT off-manifold embedding effects.
    e6c = model.make_cache()
    e6 = prefill_and_decode(token_feed, "baseline_recheck_at_end",
                            e6c, args.decode_steps)
    experiments["E6_baseline_recheck"] = {
        "decode_tok_per_s": e6["decode_tok_per_s"],
        "prefill_tok_per_s": e6["prefill_tok_per_s"],
        "note": "compare against E1/E2 before attributing any slowdown "
                "to off-vocabulary inputs",
    }

    experiments["E1_baseline_tokens"] = e1
    report["status"] = "ok"


def _kl(logits_a, logits_b) -> float:
    import mlx.core as mx

    def log_softmax(x):
        x = x.astype(mx.float32)
        shifted = x - mx.max(x, axis=-1, keepdims=True)
        return shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))

    log_pa = log_softmax(logits_a[:, -1])
    log_pb = log_softmax(logits_b[:, -1])
    kl = mx.sum(mx.exp(log_pa) * (log_pa - log_pb), axis=-1)
    mx.eval(kl)
    return float(kl.item())


def _sig_params(obj) -> list[str]:
    import inspect

    try:
        return list(inspect.signature(obj.__call__).parameters)
    except (ValueError, TypeError):
        return []


if __name__ == "__main__":
    raise SystemExit(main())
