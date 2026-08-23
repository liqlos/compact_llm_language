"""MLX internal-recurrence probe (Apple-silicon only).

Question: can localized internal recurrence bypass the slow PUBLIC
inputs_embeds path measured in mlx_soft_embedding_probe.py by composing
decoder layers directly inside the process?

Checks:
 1. manual all-layer composition reproduces the public forward logits
 2. localized loop: interval layers applied K times to the last-position
    hidden stream with per-layer caches (no vocabulary decode anywhere)
 3. latency: per-latent-step (manual layer calls) vs public
    input_embeddings decode step vs plain token decode step
 4. honest report JSON; unsupported pieces surfaced, never hidden

Usage:
  python -m latent_lab.bench.mlx_internal_recurrence_probe \
      --out .rcc_work/mlx_internal_probe.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-snapshot", default=None,
                    help="path to HF snapshot dir (default: local Qwen3.5-0.8B)")
    ap.add_argument("--interval", default="12,18")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snap = args.model_snapshot or sorted(glob.glob(
        "/Users/aleksei/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/"
        "snapshots/*"))[0]
    report = {"probe": "mlx_internal_recurrence", "model": snap,
              "status": "running"}
    try:
        _run(snap, args, report)
    except Exception as e:  # noqa: BLE001 — always write the report
        import traceback
        report["status"] = "error"
        report["error"] = f"{type(e).__name__}: {e}"
        report["trace"] = traceback.format_exc()[-2000:]
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"[mlx_internal] status={report['status']} -> {args.out}")
    return 0 if report["status"] == "ok" else 3


def _run(snap, args, report):
    import mlx.core as mx
    from mlx_lm import load

    lo, hi = (int(x) for x in args.interval.split(","))
    k = args.k
    t0 = time.perf_counter()
    model, tok = load(snap)
    inner = None
    for n, mod in model.named_modules():
        if hasattr(mod, "layers") and hasattr(mod, "norm"):
            inner = mod
            break
    layers = inner.layers
    L = len(layers)
    norm = inner.norm
    tie = getattr(getattr(model, "args", None), "tie_word_embeddings",
                  True)

    def to_logits(hidden):
        # tied embeddings: lm_head == embed_tokens.as_linear
        if tie:
            return inner.embed_tokens.as_linear(hidden)
        return inner.lm_head(hidden)
    head = to_logits
    report["load_seconds"] = round(time.perf_counter() - t0, 1)
    report["n_layers"] = L
    report["layer_class"] = type(layers[0]).__name__

    text = ("A machine starts in amber. amber --x--> birch, birch --y--> "
            "cedar, cedar --x--> amber.\nIt reads x y x. In which state "
            "does it end?\nAnswer:")
    ids = mx.array(tok.encode(text))[None]
    n_prompt = ids.shape[1]

    # ---- reference: public forward ------------------------------------
    def make_cache():
        if hasattr(model, "make_cache"):
            return model.make_cache()
        if hasattr(inner, "make_cache"):
            return inner.make_cache()
        from mlx_lm.models import cache as _cache
        return _cache.make_cache(model)
    c_ref = make_cache()
    logits_ref = model(ids, cache=c_ref)
    report["public_forward_ok"] = True

    # ---- 1. manual composition == public forward ----------------------
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    c_man = make_cache()
    h = inner.embed_tokens(ids)
    fa_idx = next(i for i, l in enumerate(layers) if not l.is_linear)
    fa_mask = create_attention_mask(h, c_man[fa_idx])
    ssm_mask = create_ssm_mask(h, c_man)
    for i, layer in enumerate(layers):
        h = layer(h, mask=(ssm_mask if layer.is_linear else fa_mask),
                  cache=c_man[i])
    logits_man = head(norm(h))
    same = bool(mx.allclose(logits_ref, logits_man, atol=0, rtol=0)) or \
        float(mx.abs(logits_ref - logits_man).max()) < 1e-4
    report["manual_composition_maxdiff"] = float(
        mx.abs(logits_ref - logits_man).max())
    report["manual_composition_close"] = bool(same)

    # ---- 2+3. localized loop on last position --------------------------
    z = h[:, -1:, :]
    # per-step timing: interval layers on a 1-token stream w/ live caches
    c_loop = make_cache()
    hh = inner.embed_tokens(ids)
    fa_mask2 = create_attention_mask(hh, c_loop[fa_idx])
    ssm_mask2 = create_ssm_mask(hh, c_loop)
    for i, layer in enumerate(layers):
        hh = layer(hh, mask=(ssm_mask2 if layer.is_linear else fa_mask2),
                   cache=c_loop[i])
    z = hh[:, -1:, :]

    def one_step(zv, pos_mask=None):
        out = zv
        for i in range(lo, hi):
            out = layers[i](out, mask=pos_mask, cache=c_loop[i])
        return out

    # correctness-ish: logits after tail remain finite & argmax stable
    outs = []
    t0 = time.perf_counter()
    for _ in range(k):
        z = one_step(z)
        outs.append(float(mx.abs(z).max()))
    loop_s = time.perf_counter() - t0
    ht = z
    for i in range(hi, L):
        ht = layers[i](ht, mask=None, cache=c_loop[i])
    logits_latent = head(norm(ht[:, -1:, :]))
    ref_next = logits_ref[:, -1, :]
    top_overlap = int(mx.argmax(logits_latent[0]) == mx.argmax(ref_next))
    report["localized_loop"] = {
        "k": k, "interval": [lo, hi],
        "seconds_per_step_ms": round(loop_s / k * 1000, 2),
        "z_max_per_step": [round(v, 3) for v in outs[-3:]],
        "final_top1_matches_no_loop_reference": bool(top_overlap),
        "logits_finite": bool(mx.isfinite(logits_latent).all().item()),
        "note": "top-1 comparison is informational; latent steps CHANGE the "
                "state by design, so divergence from the no-loop reference "
                "is expected, not an error",
    }

    # ---- 3b. public input_embeddings decode step latency ---------------
    emb_last = inner.embed_tokens(ids[:, -1:])
    c_pub = make_cache()
    model(ids, cache=c_pub)
    t0 = time.perf_counter()
    reps = max(4, k)
    for _ in range(reps):
        _ = model(None, cache=c_pub, input_embeddings=emb_last)
    pub_s = time.perf_counter() - t0
    report["public_inputs_embeds_step_ms"] = round(pub_s / reps * 1000, 2)

    # ---- 3c. plain token decode step latency ---------------------------
    c_tok = make_cache()
    model(ids, cache=c_tok)
    next_tok = mx.array([[13]])
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(next_tok, cache=c_tok)
    tok_s = time.perf_counter() - t0
    report["plain_token_step_ms"] = round(tok_s / reps * 1000, 2)

    ratio = report["public_inputs_embeds_step_ms"] / max(
        1e-6, report["localized_loop"]["seconds_per_step_ms"])
    report["verdict"] = {
        "internal_recurrence_avoids_public_path": True,
        "public_vs_internal_latency_ratio": round(ratio, 2),
        "interpretation": (
            "manual per-layer composition runs the SAME weights without "
            "ever calling generate()/input_embeddings; latency ratio shows "
            "whether that also avoids the measured slowdown"),
    }
    report["status"] = "ok"


if __name__ == "__main__":
    raise SystemExit(main())
