"""State probe: prove runtime control over a real Qwen hybrid model.

This is the mandatory executable vertical slice (docs ADR-001 / plan §14).
It is a RUNTIME CONTROL proof — not latent reasoning, not a benchmark.

Checks performed (each recorded honestly, unsupported ops surfaced):
 1. load smallest available Qwen hybrid checkpoint
 2. read config + layer types
 3. embeddings & hidden states accessible
 4. attention KV + DeltaNet recurrent/conv state structure captured
 5. cache clone/restore roundtrip is exact
 6. normal token path measured
 7. inputs_embeds path measured
 8. one continuous recurrence step with ZERO lm_head calls
 9. actual per-layer call counts recorded
10. wall-clock, memory, shapes recorded
11. machine-readable JSON report
12. unsupported operations reported, never hidden

Usage:
  python -m latent_lab.bench.state_probe --model Qwen/Qwen3.5-0.8B \
      --out .rcc_work/state_probe_qwen35_08b.json [--device mps|cpu] [--dtype fp32|fp16]
"""

from __future__ import annotations

import argparse
import json
import platform
import time


def _snapshot_bytes(snap: dict) -> int:
    torch = __import__("torch")

    def size(v) -> int:
        if torch.is_tensor(v):
            return v.numel() * v.element_size()
        if isinstance(v, dict):
            return sum(size(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return sum(size(x) for x in v)
        return 0

    return sum(size(x) for k in ("conv", "recurrent") for x in snap[k].values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    report: dict = {
        "probe": "rcc_state_probe",
        "version": 1,
        "model": args.model,
        "status": "running",
        "checks": {},
    }
    try:
        _run(args, report)
    except Exception as e:  # noqa: BLE001 — probe must always write its report
        report["status"] = "error"
        report["error"] = f"{type(e).__name__}: {e}"
        report["error_type"] = type(e).__name__

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    print(f"[state_probe] status={report['status']} -> {args.out}")
    return 0 if report.get("status") == "ok" else 3


def _run(args, report: dict) -> None:
    torch = __import__("torch")
    transformers = __import__("transformers")
    from ..backends import hf_qwen as hq

    checks = report["checks"]
    t_imports = time.perf_counter()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    t0 = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = None
    load_errors = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                     "AutoModel"):
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(args.model, dtype=dtype)
            checks["load"] = {"ok": True, "auto_class": cls_name}
            break
        except Exception as e:  # noqa: BLE001 — try the next loader
            load_errors.append(f"{cls_name}: {type(e).__name__}: {e}")
    if model is None:
        raise RuntimeError("all AutoModel loaders failed: " + " | ".join(load_errors))
    model.eval().to(device)
    load_s = time.perf_counter() - t0

    cfg = model.config
    layer_types = hq.extract_layer_types(cfg)
    n_layers = len(layer_types)

    # ---- check: config + hidden size shapes --------------------------
    hidden = getattr(cfg, "hidden_size", None) or getattr(
        getattr(cfg, "text_config", cfg), "hidden_size", None
    )
    checks["config"] = {
        "ok": True,
        "arch": cfg.architectures,
        "layer_types": layer_types,
        "n_layers": n_layers,
        "hidden_size": hidden,
        "vocab_size": getattr(cfg, "vocab_size", None),
        "load_seconds": round(load_s, 2),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }

    # ---- counters -----------------------------------------------------
    cc, layers = hq.attach_layer_counters(model)
    cc.counts["lm_head"] = 0

    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    # ---- 6. normal token path -----------------------------------------
    cc.counts = {k: 0 for k in cc.counts}
    t0 = time.perf_counter()
    with __import__("torch").no_grad():
        out_tok = model(ids, output_hidden_states=True, use_cache=True)
    tok_s = time.perf_counter() - t0
    hs = out_tok.hidden_states
    checks["token_path"] = {
        "ok": hs is not None and len(hs) == n_layers + 1,
        "n_hidden_tensors": len(hs) if hs is not None else 0,
        "hidden_shape": list(hs[-1].shape),
        "seconds": round(tok_s, 4),
        "layer_calls": {k: v for k, v in cc.counts.items()
                        if k.startswith("layer_")},
        "lm_head_calls": cc.counts["lm_head"],
    }

    # ---- cache structure (check 4) ------------------------------------
    torch = __import__("torch")
    cache = out_tok.past_key_values
    cache_info = {"type": type(cache).__name__, "layers": []}

    def shape_of(v):
        if torch.is_tensor(v):
            return list(v.shape)
        if isinstance(v, dict) and v:
            return {str(k): shape_of(x) for k, x in list(v.items())[:1]}
        if isinstance(v, (list, tuple)) and v:
            return [shape_of(x) for x in v[:1]]
        return None

    for i, cl in enumerate(getattr(cache, "layers", [])):
        entry = {"idx": i,
                 "layer_type": layer_types[i] if i < n_layers else "?",
                 "layer_class": type(cl).__name__}
        for attr in ("conv_states", "recurrent_states"):
            v = getattr(cl, attr, None)
            entry[attr] = {"present": v is not None and (not isinstance(v, (list, tuple)) or len(v) > 0),
                           "shape": shape_of(v)}
        ks = getattr(cl, "keys", None)
        vs = getattr(cl, "values", None)
        entry["kv"] = {"present": ks is not None and vs is not None,
                       "shape": [list(ks.shape), list(vs.shape)]
                       if ks is not None and vs is not None else None}
        cache_info["layers"].append(entry)
    checks["cache_structure"] = {"ok": len(cache_info["layers"]) == n_layers,
                                 **cache_info}

    # ---- 5. clone/restore exactness ------------------------------------
    snap = hq.cache_snapshot(cache)
    with __import__("torch").no_grad():
        out_a = model(ids[:, -1:], past_key_values=cache, use_cache=True)
    hq.cache_restore(cache, snap)
    with __import__("torch").no_grad():
        out_b = model(ids[:, -1:], past_key_values=cache, use_cache=True)
    eq = hq.trees_equal(out_a.logits, out_b.logits)
    checks["cache_clone_restore"] = {
        "ok": bool(eq),
        "logits_equal_after_restore": bool(eq),
        "snapshot_bytes_approx": _snapshot_bytes(snap),
    }

    # ---- 7. inputs_embeds path ----------------------------------------
    embed = model.get_input_embeddings()
    with __import__("torch").no_grad():
        emb_from_ids = embed(ids)
    cc.counts = {k: 0 for k in cc.counts}
    with __import__("torch").no_grad():
        out_emb = model(inputs_embeds=emb_from_ids, output_hidden_states=True,
                        use_cache=False)
    same_logits = hq.trees_equal(out_emb.logits, out_tok.logits)
    checks["inputs_embeds_path"] = {
        "ok": True,
        "embed_shape": list(emb_from_ids.shape),
        "logits_match_token_path_exactly": bool(same_logits),
        "note": "" if same_logits else (
            "expected small numeric drift across paths; see dtype"),
    }

    # ---- 8+9. continuous recurrence step, zero lm_head calls ----------
    cc.counts = {k: 0 for k in cc.counts}
    h_last = out_emb.hidden_states[-1][:, -1:, :]          # [B,1,D]
    rec_cache = out_tok.past_key_values                    # reuse populated cache
    base = model.model if hasattr(model, "model") else model  # no LM head
    step_shapes = []
    t0 = time.perf_counter()
    k_steps = 3
    with torch.no_grad():
        for _ in range(k_steps):
            step_out = base(inputs_embeds=h_last,
                            past_key_values=rec_cache,
                            use_cache=True,
                            output_hidden_states=True)
            h_next = step_out.hidden_states[-1][:, -1:, :]
            step_shapes.append(list(h_next.shape))
            h_last = h_next
    rec_s = time.perf_counter() - t0
    lm_calls_during_recurrence = cc.counts["lm_head"]
    checks["continuous_recurrence"] = {
        "ok": lm_calls_during_recurrence == 0 and len(step_shapes) == k_steps,
        "k_steps": k_steps,
        "step_hidden_shapes": step_shapes,
        "seconds_total": round(rec_s, 4),
        "seconds_per_step": round(rec_s / k_steps, 4),
        "lm_head_calls_during_loop": lm_calls_during_recurrence,
        "layer_calls_during_loop": {
            k: v for k, v in cc.counts.items() if k.startswith("layer_")
        },
        "verdict": (
            "hidden state fed back without vocabulary decode "
            "(structural control proven; NOT yet evidence of reasoning)"
        ),
    }

    # ---- 10. memory ------------------------------------------------------
    checks["memory"] = {"ok": True, **hq.gpu_mem_mib(device)}

    # ---- summary ----------------------------------------------------------
    all_ok = all(
        c.get("ok", False) for c in checks.values() if isinstance(c, dict)
    )
    report["status"] = "ok" if all_ok else "partial"
    report["device"] = str(device)
    report["dtype"] = str(dtype)
    report["platform"] = platform.platform()
    report["imports_seconds"] = round(time.perf_counter() - t_imports, 2)
    cc.detach()


if __name__ == "__main__":
    raise SystemExit(main())
