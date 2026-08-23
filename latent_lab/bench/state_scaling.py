"""State scaling measurement: how big is the machine state, really?

Measures per-layer and total bytes of:
  * DeltaNet recurrent state
  * DeltaNet conv state
  * full-attention KV cache
for Qwen3.5-0.8B / -2B at a fixed prompt length, plus the size of one
serialized RCCModelState workspace under the latent_lab ABI.

Usage:
  python -m latent_lab.bench.state_scaling --models 0.8B 2B \
      --prompt-tokens 128 --out .rcc_work/state_scaling.json
"""

from __future__ import annotations

import argparse
import json


def measure(model_id: str, rev: str, prompt_tokens: int) -> dict:
    import torch
    import transformers

    from latent_lab.backends.hf_qwen import extract_layer_types

    tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=rev)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=rev, dtype=torch.bfloat16).eval()
    ids = torch.randint(100, 1000, (1, prompt_tokens))
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache = out.past_key_values

    def nbytes(v):
        t = torch.as_tensor(v) if not torch.is_tensor(v) else v
        return t.numel() * t.element_size()

    per_layer = []
    lt = extract_layer_types(model.config)
    tot = {"recurrent": 0, "conv": 0, "kv": 0}
    for i, cl in enumerate(cache.layers):
        entry = {"idx": i, "type": lt[i] if i < len(lt) else "?",
                 "bytes": {}}
        for attr in ("recurrent_states", "conv_states"):
            v = getattr(cl, attr, None)
            if v is None:
                continue
            vals = (list(v.values()) if isinstance(v, dict)
                    else list(v) if isinstance(v, (list, tuple)) else [v])
            n = sum(nbytes(x) for x in vals if torch.is_tensor(x))
            key = "recurrent" if "recurrent" in attr else "conv"
            entry["bytes"][key] = n
            tot[key] += n
        ks, vs = getattr(cl, "keys", None), getattr(cl, "values", None)
        if ks is not None and vs is not None:
            n = nbytes(ks) + nbytes(vs)
            entry["bytes"]["kv"] = n
            tot["kv"] += n
        if entry["bytes"]:
            per_layer.append(entry)

    # RCCModelState workspace under the latent_lab ABI (fp32 tuples):
    # memory slots M + working H + readout R vectors of hidden size,
    # plus small controller vector; caches stored by reference (opaque).
    cfg = model.config.text_config if hasattr(model.config, "text_config") \
        else model.config
    hidden = getattr(cfg, "hidden_size", 0)
    ws = {"M": 8, "H": 8, "R": 4}
    workspace_bytes_fp32 = sum(ws.values()) * hidden * 4
    del model, out, cache, tok
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "model": model_id, "revision": rev,
        "dtype": "bfloat16", "prompt_tokens": prompt_tokens,
        "layer_types_count": {t: lt.count(t) for t in set(lt)},
        "total_cache_bytes": sum(tot.values()),
        "total_by_kind_mib": {k: round(v / 2 ** 20, 3)
                              for k, v in tot.items()},
        "per_layer_sample": per_layer[:3],
        "rcc_model_state_workspace_bytes_fp32": workspace_bytes_fp32,
        "workspace_slots": ws,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".rcc_work/state_scaling.json")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    args = ap.parse_args()
    results = []
    for tag, mid in [("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B"),
                     ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B")]:
        try:
            results.append(measure(mid, "main", args.prompt_tokens))
        except Exception as e:  # noqa: BLE001
            results.append({"model": mid, "error": str(e)})
    payload = {"probe": "state_scaling", "results": results}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps(payload, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
