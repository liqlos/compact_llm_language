"""Real Qwen hybrid backend helpers (Transformers).

Import-safe without torch/transformers: heavy imports happen inside
functions. These helpers are shared with bench.state_probe so there is no
dead code. Nothing here claims latent reasoning — this is runtime control
plumbing whose correctness the state probe must establish first.
"""

from __future__ import annotations

import hashlib
import json

GDN_KEYS = ("gated_deltanet", "gdn", "linear_attention", "linear_attn", "delta_net")
ATTN_KEYS = ("full_attention", "attention")


def require_torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "torch is required for real-model probes. Install optional group: "
            "`uv sync --group lab` (see pyproject.toml [dependency-groups])"
        ) from e
    import torch

    return torch


def require_transformers(min_version=(5, 3)):
    try:
        import transformers
    except ImportError as e:
        raise RuntimeError(
            "transformers is required for real-model probes. "
            "`uv sync --group lab`"
        ) from e
    from packaging.version import parse

    got = parse(transformers.__version__)
    want = parse(".".join(map(str, min_version)))
    if got < want:
        raise RuntimeError(
            f"transformers {transformers.__version__} too old for qwen3_5 "
            f"(need >= {want}); GDN cached-forward fix was PR #45513"
        )
    return transformers


def extract_layer_types(config) -> list[str]:
    """Read hybrid layer types from a qwen3_5-family config.

    Returns e.g. ["gdn","gdn","gdn","attn", ...]. Raises when the config
    does not describe a known hybrid layout (honest failure).
    """
    lt = getattr(config, "layer_types", None) or getattr(
        config.text_config, "layer_types", None
    )
    if lt is None:
        raise ValueError(
            f"{type(config).__name__} has no layer_types — not a hybrid config?"
        )
    types: list[str] = []
    for t in lt:
        s = str(t).lower()
        if any(k in s for k in GDN_KEYS):
            types.append("gdn")
        elif any(k in s for k in ATTN_KEYS):
            types.append("attn")
        else:
            types.append(s)
    return types


def config_hash(config) -> str:
    js = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(js.encode()).hexdigest()[:16]


class CallCounter:
    """Forward-hook based call counter for decoder layers / lm_head."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self._handles: list = []

    def attach(self, module, name: str) -> None:
        from torch import nn

        self.counts.setdefault(name, 0)

        def hook(_m, _inp, _out):
            self.counts[name] += 1

        self._handles.append(module.register_forward_hook(hook))
        assert isinstance(module, nn.Module)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def attach_layer_counters(model) -> tuple[CallCounter, list]:
    """Count per-decoder-layer forward calls; also arm an lm_head counter."""
    cc = CallCounter()
    layers = None
    for obj in (model, getattr(model, "model", None)):
        cand = getattr(obj, "layers", None) if obj is not None else None
        if cand is not None:
            layers = cand
            break
    if layers is None:
        raise RuntimeError("could not locate decoder layers on the model")
    for i, layer in enumerate(layers):
        cc.attach(layer, f"layer_{i}")
    out_emb = model.get_output_embeddings()
    if out_emb is not None:
        cc.attach(out_emb, "lm_head")
    return cc, layers


def cache_snapshot(cache) -> dict:
    """Copy conv/recurrent/KV state tensors of a Qwen3_5DynamicCache.

    transformers >= 5.x layout: cache.layers[i] is a DynamicLayer (attention
    KV: .keys/.values) or LinearAttentionLayer (GDN: .conv_states[state_idx]
    / .recurrent_states[state_idx] lists).
    """
    snap: dict[str, dict] = {"conv": {}, "recurrent": {}, "kv": {}}
    torch = require_torch()
    for i, layer in enumerate(cache.layers):
        conv = getattr(layer, "conv_states", None)
        rec = getattr(layer, "recurrent_states", None)
        if isinstance(conv, dict) and conv:
            snap["conv"][i] = {k: _clone_tree(v) for k, v in conv.items()}
        elif isinstance(conv, (list, tuple)) and conv or torch.is_tensor(conv):
            snap["conv"][i] = _clone_tree(conv)
        if isinstance(rec, dict) and rec:
            snap["recurrent"][i] = {k: _clone_tree(v) for k, v in rec.items()}
        elif isinstance(rec, (list, tuple)) and rec or torch.is_tensor(rec):
            snap["recurrent"][i] = _clone_tree(rec)
        ks = getattr(layer, "keys", None)
        vs = getattr(layer, "values", None)
        if ks is not None and vs is not None:
            snap["kv"][i] = (_clone_tree(ks), _clone_tree(vs))
    return snap


def cache_restore(cache, snap: dict) -> None:
    torch = require_torch()

    def restore_like(slot, value):
        """Write cloned tensors back into the cache's live containers."""
        if isinstance(slot, dict) and isinstance(value, dict):
            for k, v in value.items():
                if k in slot:
                    slot[k] = _clone_tree(v)
        elif isinstance(slot, (list, tuple)) and isinstance(value, (list, tuple)):
            for j in range(min(len(slot), len(value))):
                slot[j] = _clone_tree(value[j])

    for i, layer in enumerate(cache.layers):
        if i in snap["conv"]:
            conv = getattr(layer, "conv_states", None)
            v = snap["conv"][i]
            if torch.is_tensor(conv):
                layer.conv_states = _clone_tree(v)
            else:
                restore_like(conv, v)
        if i in snap["recurrent"]:
            rec = getattr(layer, "recurrent_states", None)
            v = snap["recurrent"][i]
            if torch.is_tensor(rec):
                layer.recurrent_states = _clone_tree(v)
            else:
                restore_like(rec, v)
        kv = snap["kv"].get(i)
        if kv is not None:
            layer.keys = _clone_tree(kv[0])
            layer.values = _clone_tree(kv[1])


def _clone_tree(obj):
    torch = require_torch()
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_tree(x) for x in obj)
    return obj


def trees_equal(a, b) -> bool:
    torch = require_torch()
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            trees_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(trees_equal(x, y) for x, y in zip(a, b))
    return a == b


def gpu_mem_mib(device: str) -> dict:
    torch = require_torch()
    out: dict = {}
    if device.startswith("mps") and hasattr(torch, "mps"):
        try:
            out["mps_current_mib"] = torch.mps.current_allocated_memory() / 2**20
        except Exception:
            pass
    import resource

    out["rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return out
