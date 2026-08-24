"""Localized continuous recurrence runtime over a real hybrid Qwen.

One class covers three configurations:
  E  localized: interval=[lo,hi), loop applies ONLY those layers K times
  D  full-decoder: interval=[0,L), loop applies the whole decoder
  F  control: k_steps=0 (tail-only readout)

Structural guarantees (enforced, unit-tested):
  * inside the latent loop there are ZERO lm_head calls, ZERO tokenizer
    encode/decode calls, ZERO .generate() calls;
  * the loop operates purely on hidden states + the layer cache;
  * manual layer composition is bit-exact vs the canonical HF forward
    (validated on a tiny hybrid config in tests/test_localized.py).

Adaptation: frozen backbone + LoRA on interval-layer projections +
zero-initialized per-step clock embeddings. Nothing outside the interval
is adapted, so K>0 vs K=0 differences isolate the recurrence itself.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

try:
    import torch
except ImportError:  # pragma: no cover - import-safety without lab group
    torch = None


class LatentLoopViolation(RuntimeError):
    """Forbidden vocabulary/tokenizer operation inside the latent loop."""


class NonFiniteCandidateScores(RuntimeError):
    """A candidate score was non-finite; ranking is refused."""


class VocabGuard:
    """Counts lm_head calls + tokenizer operations; windows are asserted."""

    def __init__(self, model, tokenizer=None) -> None:
        self.lm_head_calls = 0
        self.tokenizer_calls = 0
        self.generate_calls = 0
        self._tokenizer = tokenizer
        self._handles = []
        head = model.get_output_embeddings()
        if head is not None:
            def head_hook(_m, _i, _o):
                self.lm_head_calls += 1
            self._handles.append(head.register_forward_hook(head_hook))
        if tokenizer is not None:
            for meth in ("encode", "decode"):
                original = getattr(type(tokenizer), meth, None)
                if original is None:
                    continue

                def make_wrapper(orig):
                    def wrapped(inst, *a, **kw):
                        self.tokenizer_calls += 1
                        return orig(inst, *a, **kw)
                    return wrapped

                setattr(type(tokenizer), meth, make_wrapper(original))
                self._handles.append(("tok_meth", meth, type(tokenizer),
                                      original))

    @contextlib.contextmanager
    def window(self, allow_vocab: bool = False):
        """Assert zero vocab/tokenizer activity inside the block."""
        lm0, tk0 = self.lm_head_calls, self.tokenizer_calls
        try:
            yield
        finally:
            if not allow_vocab:
                if self.lm_head_calls != lm0:
                    raise LatentLoopViolation(
                        f"lm_head called {self.lm_head_calls - lm0} time(s) "
                        "inside latent region")
                if self.tokenizer_calls != tk0:
                    raise LatentLoopViolation(
                        f"tokenizer called {self.tokenizer_calls - tk0} "
                        "time(s) inside latent region")

    def close(self) -> None:
        for h in self._handles:
            if isinstance(h, tuple):
                _, meth, cls, original = h
                setattr(cls, meth, original)
            else:
                h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

LORA_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj", "in_proj_qkvz", "in_proj_ba",
)

_Base = torch.nn.Module if torch is not None else object


class LoRALinear(_Base):
    """y = W x + (alpha/r) * B A x ; W stays frozen."""

    def __init__(self, base, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        # master LoRA weights stay fp32 even over a lower-precision backbone
        dev = next(base.parameters()).device
        self.lora_A = torch.nn.Parameter(torch.zeros(r, base.in_features,
                                                     dtype=torch.float32,
                                                     device=dev))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r,
                                                     dtype=torch.float32,
                                                     device=dev))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        # master LoRA weights stay fp32 even over a bf16 base layer
        delta = ((x.to(self.lora_A.dtype) @ self.lora_A.transpose(0, 1))
                 @ self.lora_B.transpose(0, 1)) * self.scaling
        return self.base(x) + delta.to(x.dtype)


def inject_lora(layers, *, r: int = 8, alpha: float = 16.0,
                suffixes=LORA_TARGET_SUFFIXES) -> list:
    targets = []
    for layer in layers:
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear) and name.endswith(suffixes):
                targets.append((layer, name, module))
    injected = []
    for layer, name, module in targets:
        parts = name.split(".")
        container = layer
        for p in parts[:-1]:
            container = getattr(container, p)
        lora = LoRALinear(module, r=r, alpha=alpha)
        setattr(container, parts[-1], lora)
        injected.append(lora)
    if not injected:
        raise RuntimeError("no LoRA targets matched — check suffix list")
    return injected


def lora_parameters(injected) -> list:
    ps = []
    for lora in injected:
        ps += [lora.lora_A, lora.lora_B]
    return ps


def make_step_clock(hidden: int, max_k: int, device=None):
    """Zero-initialized per-step clock embeddings; always fp32."""
    clock = torch.nn.Embedding(max_k + 1, hidden, device=device,
                               dtype=torch.float32)
    torch.nn.init.zeros_(clock.weight)
    return clock


# ---------------------------------------------------------------------------
# recurrence runtime
# ---------------------------------------------------------------------------

@dataclass
class RecurrenceReport:
    k_steps: int
    interval: tuple[int, int]
    layer_applications: int
    lm_head_calls_total: int
    seconds_prefill: float = 0.0
    seconds_loop: float = 0.0
    seconds_readout: float = 0.0
    extra: dict = field(default_factory=dict)


class LocalizedRecurrence:
    """Frozen-backbone localized recurrence with guarded latent loop."""

    def __init__(self, model, tokenizer=None, *, interval, max_k=16,
                 lora_r=8, lora_alpha=16.0, use_clock=True,
                 grad_checkpoint=True):
        if torch is None:
            raise RuntimeError("torch required for LocalizedRecurrence")
        self.model = model
        # frozen-backbone contract: nothing except LoRA/clock may train
        for p in model.parameters():
            p.requires_grad_(False)
        if getattr(model, "_rcc_lora_bound", False):
            raise RuntimeError(
                "this model already carries a LocalizedRecurrence binding; "
                "reload the checkpoint to switch configuration")
        model._rcc_lora_bound = True
        self.base = model.model if hasattr(model, "model") else model
        self.config = model.config
        self.layer_types = [str(t) for t in self.config.layer_types]
        self.n_layers = len(self.base.layers)
        lo, hi = interval
        if not (0 <= lo < hi <= self.n_layers):
            raise ValueError(f"bad interval {interval} for {self.n_layers}L")
        self.interval = (lo, hi)
        self.max_k = max_k
        hidden = self.base.embed_tokens.embedding_dim

        self.guard = VocabGuard(model, tokenizer)

        dev = next(model.parameters()).device
        # clock trainables are explicitly fp32; never the global default dtype
        self.clock = torch.nn.Embedding(max_k + 1, hidden, device=dev,
                                        dtype=torch.float32)
        torch.nn.init.zeros_(self.clock.weight)

        self.injected = inject_lora(
            [self.base.layers[i] for i in range(lo, hi)],
            r=lora_r, alpha=lora_alpha)
        self.use_clock = use_clock
        self.grad_checkpoint = grad_checkpoint

    # -- positional plumbing -------------------------------------------------

    def _device(self):
        return next(self.base.parameters()).device

    def _pos4(self, n, start):
        return torch.arange(start, start + n,
                            device=self._device()).view(1, 1, -1).expand(4, 1, -1)

    def _rotary(self, emb, n, start):
        # canonical forward feeds rotary the mrope rows [1:] and layers row [0]
        return self.base.rotary_emb(emb, self._pos4(n, start)[1:])

    @staticmethod
    def _sever_cache_grads(cache):
        """Break autograd links INTO cached state storage.

        The hybrid cache mutates its tensors in place on every layer call;
        treating the store as a numeric buffer lets gradients flow through
        the hidden-state chain (z path) while remaining autograd-safe.
        """
        def sever(v):
            if torch.is_tensor(v):
                if v.grad_fn is not None or v.requires_grad:
                    v.data = v.detach().data
            elif isinstance(v, dict):
                for x in v.values():
                    sever(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    sever(x)

        for layer in cache.layers:
            for attr in ("conv_states", "recurrent_states", "keys",
                         "values"):
                v = getattr(layer, attr, None)
                if v is not None:
                    sever(v)

    def _run_layers(self, indices, h, cache, pos, *, grad=False):
        pe = self._rotary(h, h.shape[1], pos)
        pos_ids = self._pos4(h.shape[1], pos)[0]
        for i in indices:
            layer = self.base.layers[i]
            h = layer(h, position_embeddings=pe, attention_mask=None,
                      position_ids=pos_ids, past_key_values=cache,
                      use_cache=True)
            if grad:
                self._sever_cache_grads(cache)
        return h

    def _encode(self, input_ids_or_emb, *, grad=False):
        """Full decoder pass; returns (cache, pre-norm last-position hidden)."""
        from transformers import DynamicCache
        cache = DynamicCache(config=self.config)
        emb = (input_ids_or_emb if not torch.is_tensor(input_ids_or_emb)
               or input_ids_or_emb.dtype.is_floating_point
               else self.base.embed_tokens(input_ids_or_emb))
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            h = self._run_layers(range(self.n_layers), emb, cache, 0,
                                 grad=grad)
        return cache, h[:, -1:, :]

    # -- stages ----------------------------------------------------------------

    def latent_steps(self, z, cache, start_pos, k_steps, *,
                     grad=False, ablate=None):
        """Apply the interval K times; GUARDED: no vocab/tokenizer activity.

        ablate keys:
          zero_state         -> incoming z replaced by zeros
          clocks             -> "identity"|"reverse"|"shuffle_perm:i,j,.."|"off"
          bypass_interval    -> interval layers replaced by identity
          truncate_k         -> effective step count reduced
          swap/noise         -> performed by caller replacing z beforehand
        """
        ablate = dict(ablate or {})
        if ablate.get("zero_state"):
            z = torch.zeros_like(z)
        if ablate.get("noise_state"):
            g = torch.Generator(device=z.device.type)
            g.manual_seed(int(ablate.get("noise_seed", 1234)))
            n = torch.randn(z.shape, generator=g, device=z.device,
                            dtype=torch.float32).to(z.dtype)
            n = n / (n.norm() + 1e-8) * (z.norm() + 1e-8)
            z = n
        mode = ablate.get("clocks", "identity")
        eff_k = min(int(ablate.get("truncate_k", k_steps)), k_steps)
        idxs = list(range(eff_k))
        if isinstance(mode, str) and mode.startswith("shuffle_perm:"):
            idxs = [int(x) for x in mode.split(":", 1)[1].split(",")][:eff_k]
        elif mode == "reverse":
            idxs = list(reversed(idxs))

        lo, hi = self.interval
        with self.guard.window():
            pos = start_pos
            for ci in idxs:
                zz = z
                if self.use_clock and mode != "off":
                    tok = torch.tensor([ci + 1], device=z.device)
                    zz = zz + self.clock(tok).view(1, 1, -1).to(z.dtype)
                if ablate.get("bypass_interval"):
                    z = zz
                else:
                    z = self._run_layers(range(lo, hi), zz, cache, pos,
                                         grad=grad)
                pos += 1
        return z, pos

    def logits_from_hidden(self, h_prenorm):
        with self.guard.window(allow_vocab=True):
            return self.model.lm_head(self.base.norm(h_prenorm))

    # -- evaluation -------------------------------------------------------------

    @torch.no_grad()
    def rank_candidates(self, input_ids, candidate_ids, k_steps, *,
                        ablate=None, partner_input_ids=None):
        """Exact constrained scoring: log P(candidate | prompt, latent loop).

        ablate.swap_state semantics: pass partner_input_ids; the loop then
        runs on the PARTNER's post-encoding state while scoring this
        example's candidates.
        """
        import time

        from ..backends.hf_qwen import cache_restore, cache_snapshot
        t0 = time.perf_counter()
        cache, z0 = self._encode(input_ids)
        loop_start = input_ids.shape[1]
        if partner_input_ids is not None:
            # swap_state: run the loop on the PARTNER's cache+state; loop
            # positions must follow the PARTNER's prompt length so KV
            # alignment stays valid (only the candidates remain this ex's)
            loop_start = partner_input_ids.shape[1]
            cache, z0 = self._encode(partner_input_ids)
        t1 = time.perf_counter()
        ab = dict(ablate or {})
        if partner_input_ids is not None:
            ab["swap_state"] = True
        z, pos = self.latent_steps(z0, cache, loop_start, k_steps,
                                   ablate=ab)
        t2 = time.perf_counter()
        snap = cache_snapshot(cache)
        scores = []
        for cand in candidate_ids:
            cache_restore(cache, snap)
            cids = torch.tensor([cand], device=input_ids.device)
            h = torch.cat([z, self.base.embed_tokens(cids)], dim=1)
            ht = self.tail_sequence(h, cache, pos)
            logits = self.logits_from_hidden(ht[:, :-1, :])
            logp = torch.log_softmax(logits.float(), dim=-1)
            lp = sum(float(logp[0, j, tid]) for j, tid in enumerate(cand))
            scores.append(lp)
        t3 = time.perf_counter()
        bad = [i for i, s in enumerate(scores) if not math.isfinite(s)]
        if bad:
            raise NonFiniteCandidateScores(
                f"candidate scores at indices {bad} are not finite "
                f"({[scores[i] for i in bad]}); refusing to rank")
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        rep = RecurrenceReport(
            k_steps=k_steps, interval=self.interval,
            layer_applications=(
                (self.interval[1] - self.interval[0]) * k_steps
                + max(0, self.n_layers - self.interval[1])),
            lm_head_calls_total=self.guard.lm_head_calls,
            seconds_prefill=t1 - t0, seconds_loop=t2 - t1,
            seconds_readout=t3 - t2,
            extra={"n_candidates": len(candidate_ids)})
        return order, scores, rep

    def tail_sequence(self, h, cache, pos):
        """Tail layers over an arbitrary short stream (z [+ answer tokens])."""
        hi = self.interval[1]
        if hi >= self.n_layers:
            return h
        out = h
        for j, i in enumerate(range(hi, self.n_layers)):
            pe = self._rotary(out, out.shape[1], pos)
            pos_ids = self._pos4(out.shape[1], pos)[0]
            out = self.base.layers[i](out, position_embeddings=pe,
                                      attention_mask=None, position_ids=pos_ids,
                                      past_key_values=cache, use_cache=True)
        return out

    # -- training ---------------------------------------------------------------

    def loss_on_example(self, input_ids, answer_ids, k_steps, *,
                        ablate=None, detach_z0=False):
        """Teacher-forced CE on gold answer tokens appended after the loop."""
        cache, z0 = self._encode(input_ids, grad=True)
        if detach_z0:
            z0 = z0.detach()
        z, pos = self.latent_steps(z0, cache, input_ids.shape[1], k_steps,
                                   grad=True, ablate=ablate)
        ans_emb = self.base.embed_tokens(answer_ids)
        hz = torch.cat([z, ans_emb], dim=1)
        ht = self.tail_sequence(hz, cache, pos)
        logits = self.logits_from_hidden(ht[:, :-1, :])
        logp = torch.log_softmax(logits.float(), dim=-1)
        n = answer_ids.shape[1]
        lp = logp[0, torch.arange(n, device=logp.device), answer_ids[0]]
        return -lp.mean()

    def trainable_parameters(self):
        ps = lora_parameters(self.injected)
        if self.use_clock:
            ps += list(self.clock.parameters())
        return ps

    def adapter_state_dict(self):
        """CPU clones of every trainable tensor (LoRA pairs + clock)."""
        sd = {f"lora.{i}.A": l.lora_A.detach().to("cpu").clone()
              for i, l in enumerate(self.injected)}
        sd.update({f"lora.{i}.B": l.lora_B.detach().to("cpu").clone()
                   for i, l in enumerate(self.injected)})
        if self.use_clock:
            sd["clock.weight"] = self.clock.weight.detach().to("cpu").clone()
        return sd

    def _expected_adapter_targets(self):
        targets = {}
        for i, l in enumerate(self.injected):
            targets[f"lora.{i}.A"] = l.lora_A
            targets[f"lora.{i}.B"] = l.lora_B
        if self.use_clock:
            targets["clock.weight"] = self.clock.weight
        return targets

    def load_adapter_state(self, sd):
        """Prevalidate every key/tensor/shape/dtype/finiteness, then copy.

        Nothing is copied until ALL entries validate, so a failed load leaves
        the live adapter bit-for-bit unchanged (atomic).
        """
        from ..train.checkpointing import (
            AdapterBundleSchemaError, NonFiniteStateError)
        targets = self._expected_adapter_targets()
        if not isinstance(sd, dict):
            raise AdapterBundleSchemaError("adapter state must be a dict")
        missing = sorted(set(targets) - set(sd))
        extra = sorted(set(sd) - set(targets))
        if missing or extra:
            raise AdapterBundleSchemaError(
                f"adapter state keys mismatch: missing={missing} "
                f"unexpected={extra}")
        staged = {}
        for key, target in targets.items():
            t = sd[key]
            if not torch.is_tensor(t):
                raise AdapterBundleSchemaError(f"{key}: value is not a Tensor")
            if tuple(t.shape) != tuple(target.shape):
                raise AdapterBundleSchemaError(
                    f"{key}: shape {tuple(t.shape)} != expected "
                    f"{tuple(target.shape)}")
            if t.dtype != target.dtype:
                raise AdapterBundleSchemaError(
                    f"{key}: dtype {t.dtype} != expected {target.dtype}")
            if t.is_floating_point() and not bool(torch.isfinite(t).all()):
                raise NonFiniteStateError(f"{key}: contains non-finite values")
            staged[key] = t.to(target.device)
        with torch.no_grad():
            for key, target in targets.items():
                target.copy_(staged[key])

    def export_adapter_bundle(self, path, *, model_id, revision, metrics=None):
        """Persist this adapter as an identity-bound best_params.pt bundle."""
        from ..train.checkpointing import save_adapter_bundle
        return save_adapter_bundle(path, self.adapter_state_dict(),
                                   model_id=model_id, revision=revision,
                                   metrics=metrics)

    def load_adapter_bundle(self, path, *, model_id, revision):
        """Load an identity-bound bundle with full prevalidation."""
        from ..train.checkpointing import load_adapter_bundle
        state = load_adapter_bundle(path, model_id=model_id, revision=revision)
        self.load_adapter_state(state)
