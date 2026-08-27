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
zero-initialized per-step clock embeddings.  By default the same adapter is
active in every stage.  The recurrence-only policy instead activates LoRA only
inside latent steps, leaving prompt prefill and answer scoring on the frozen
base model so K=0 is an adapter-free control.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

try:
    import torch
except ImportError:  # pragma: no cover - import-safety without lab group
    torch = None


def _require_torch(operation: str):
    """Fail clearly when a tensor runtime is used without the lab extra."""
    if torch is None:
        raise RuntimeError(
            f"torch required for {operation}; install the 'lab' dependency group"
        )
    return torch


def _torch_no_grad(function):
    """Apply ``torch.no_grad`` without making torch an import dependency."""
    if torch is None:
        return function
    return torch.no_grad()(function)


class LatentLoopViolation(RuntimeError):
    """Forbidden vocabulary/tokenizer operation inside the latent loop."""


class NonFiniteCandidateScores(RuntimeError):
    """A candidate score was non-finite; ranking is refused."""


class AmbiguousTopTie(RuntimeError):
    """The preregistered primary score has more than one exact winner."""

    def __init__(self, message, *, candidate_details=(), report=None):
        super().__init__(message)
        self.candidate_details = tuple(candidate_details)
        self.report = report
        self.details = dict(report.extra) if report is not None else {}
        self.raw_sums = tuple(
            detail.raw_sum_logprob for detail in self.candidate_details)


class VocabGuard:
    """Scoped guard for vocabulary, tokenizer, and generation operations.

    Tokenizer special methods can only be intercepted on their class.  The
    wrappers therefore exist only for the duration of ``window`` and only act
    on this guard's exact tokenizer instance; no persistent global monkeypatch
    is left behind.
    """

    TOKENIZER_METHODS = (
        "__call__", "encode", "decode", "batch_decode",
        "apply_chat_template",
    )

    def __init__(self, model, tokenizer=None) -> None:
        self.lm_head_calls = 0
        self.tokenizer_calls = 0
        self.generate_calls = 0
        self.tokenizer_operations: dict[str, int] = {
            name: 0 for name in self.TOKENIZER_METHODS
        }
        self._model = model
        self._tokenizer = tokenizer

    @contextlib.contextmanager
    def window(self, allow_vocab: bool = False):
        """Guard one block; ``allow_vocab`` permits only the output head."""
        restores = []

        def patch_target_method(target, name, *, operation, counter):
            cls = type(target)
            original = getattr(cls, name, None)
            if original is None:
                return
            owned = name in cls.__dict__

            def wrapped(inst, *args, **kwargs):
                if inst is target:
                    if counter == "tokenizer":
                        self.tokenizer_calls += 1
                        self.tokenizer_operations[operation] += 1
                        raise LatentLoopViolation(
                            f"tokenizer {operation} called inside latent region")
                    self.generate_calls += 1
                    raise LatentLoopViolation(
                        "model.generate called inside latent region")
                return original(inst, *args, **kwargs)

            setattr(cls, name, wrapped)
            restores.append((cls, name, original, owned))

        handle = None
        try:
            if self._tokenizer is not None:
                for method in self.TOKENIZER_METHODS:
                    patch_target_method(
                        self._tokenizer, method, operation=method,
                        counter="tokenizer")
            patch_target_method(
                self._model, "generate", operation="generate",
                counter="generate")

            head = self._model.get_output_embeddings()
            if head is not None:
                def head_hook(_module, _inputs, _output):
                    self.lm_head_calls += 1
                    if not allow_vocab:
                        raise LatentLoopViolation(
                            "output head called inside latent region")

                handle = head.register_forward_hook(head_hook)
            yield
        finally:
            if handle is not None:
                handle.remove()
            for cls, name, original, owned in reversed(restores):
                if owned:
                    setattr(cls, name, original)
                else:
                    delattr(cls, name)

    def close(self) -> None:
        """Compatibility no-op: guard instrumentation is scoped per window."""


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

LORA_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    # Qwen3.5 GatedDeltaNet projections in transformers>=5.3.
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    # Retain the earlier fused spellings for compatible Qwen revisions.
    "in_proj", "in_proj_qkvz", "in_proj_ba",
)
RECURRENCE_ONLY_LORA_MODE_SUFFIX = "+recurrence-only-lora"

_Base = torch.nn.Module if torch is not None else object


class LoRALinear(_Base):
    """y = W x + (alpha/r) * B A x ; W stays frozen."""

    supports_runtime_toggle = True

    def __init__(self, base, r: int, alpha: float):
        _require_torch("LoRALinear")
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.enabled = True
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
        if not self.enabled:
            return self.base(x)
        # master LoRA weights stay fp32 even over a bf16 base layer
        delta = ((x.to(self.lora_A.dtype) @ self.lora_A.transpose(0, 1))
                 @ self.lora_B.transpose(0, 1)) * self.scaling
        return self.base(x) + delta.to(x.dtype)


def inject_lora(layers, *, r: int = 8, alpha: float = 16.0,
                suffixes=LORA_TARGET_SUFFIXES) -> list:
    _require_torch("inject_lora")
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
    _require_torch("make_step_clock")
    clock = torch.nn.Embedding(max_k + 1, hidden, device=device,
                               dtype=torch.float32)
    torch.nn.init.zeros_(clock.weight)
    return clock


# ---------------------------------------------------------------------------
# ablation parsing (strict; fail-closed)
# ---------------------------------------------------------------------------

CLOCK_MODES = ("identity", "off", "reverse")
KNOWN_ABLATION_KEYS = ("zero_state", "noise_state", "noise_seed", "clocks",
                       "bypass_interval", "truncate_k", "swap_state",
                       "reset_state", "reset_cache")


def parse_clock_mode(mode, k_steps: int):
    """Parse the 'clocks' ablation value; return the concrete step index list.

    Accepts:
      "identity" -> [0..k)
      "off"      -> [0..k) (all loop layers still run; step embeddings omitted)
      "reverse"  -> [k-1..0]
      "shuffle_perm:i,j,..." -> an EXPLICIT full unique permutation of
          range(k_steps). Anything else (wrong length, repeats, omissions,
          out-of-range, non-integer) is rejected. A full permutation with
          k_steps entries keeps compute matched to the clean run.
    Unknown modes raise ValueError — silently running clean is forbidden.
    """
    if not isinstance(mode, str):
        raise ValueError(f"unknown clocks mode {mode!r}")
    if mode in CLOCK_MODES:
        idxs = list(range(k_steps))
        return list(reversed(idxs)) if mode == "reverse" else idxs
    if mode.startswith("shuffle_perm:"):
        raw = mode.split(":", 1)[1]
        parts = raw.split(",") if raw else []
        try:
            perm = [int(x) for x in parts]
        except ValueError as e:
            raise ValueError(
                f"shuffle_perm entries must be integers: {mode!r}") from e
        if sorted(perm) != list(range(k_steps)):
            raise ValueError(
                f"shuffle_perm {perm!r} is not a full unique permutation of "
                f"range({k_steps}); compute would not match the clean run")
        return perm
    raise ValueError(
        f"unknown clocks mode {mode!r}; expected one of "
        f"{CLOCK_MODES} or 'shuffle_perm:i,j,...' over range({k_steps})")


def validate_ablation(ablate, k_steps: int) -> dict:
    """Strictly validate an ablation spec against known keys/modes."""
    ab = dict(ablate or {})
    unknown = sorted(set(ab) - set(KNOWN_ABLATION_KEYS))
    if unknown:
        raise ValueError(
            f"unknown ablation key(s) {unknown}; supported: "
            f"{sorted(KNOWN_ABLATION_KEYS)}")
    if "clocks" in ab and ab["clocks"] is not None:
        parse_clock_mode(ab["clocks"], k_steps)
    tk = ab.get("truncate_k")
    if tk is not None and (isinstance(tk, bool) or not isinstance(tk, int)
                           or not 0 <= tk <= k_steps):
        raise ValueError(f"truncate_k must be an int in [0, {k_steps}]")
    if "zero_state" in ab and not isinstance(ab["zero_state"], bool):
        raise ValueError("zero_state must be a boolean")
    if "reset_state" in ab and not isinstance(ab["reset_state"], bool):
        raise ValueError("reset_state must be a boolean")
    if "reset_cache" in ab and not isinstance(ab["reset_cache"], bool):
        raise ValueError("reset_cache must be a boolean")
    for key in ("noise_state", "swap_state"):
        if key in ab and not isinstance(ab[key], bool):
            raise ValueError(f"{key} must be a boolean")
    active_state_controls = [
        key for key in ("zero_state", "noise_state", "swap_state",
                        "reset_state") if ab.get(key)]
    if len(active_state_controls) > 1:
        raise ValueError(
            f"state interventions are mutually exclusive: "
            f"{active_state_controls}")
    if ab.get("reset_cache") and any(
            ab.get(key) for key in ("zero_state", "noise_state", "swap_state")):
        raise ValueError(
            "reset_cache may only be combined with reset_state; "
            "zero_state/noise_state/swap_state remain mutually exclusive")
    if "bypass_interval" in ab \
            and not isinstance(ab["bypass_interval"], bool):
        raise ValueError("bypass_interval must be a boolean")
    return ab


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


@dataclass(frozen=True)
class CandidateScoreDetail:
    """Raw, independently aggregatable evidence for one candidate."""

    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    raw_sum_logprob: float
    length_normalized_logprob: float

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


class LocalizedRecurrence:
    """Frozen-backbone localized recurrence with guarded latent loop."""

    RUNTIME_CONTRACT_VERSION = "localized_recurrence.runtime.v3"
    TRAINING_GRADIENT_SEMANTICS = (
        "hidden_state_chain_bptt_with_detached_cache_recurrence"
    )

    def __init__(self, model, tokenizer=None, *, interval, max_k=16,
                 lora_r=8, lora_alpha=16.0, use_clock=True,
                 recurrence_only_lora=False):
        _require_torch("LocalizedRecurrence")
        if not isinstance(recurrence_only_lora, bool):
            raise ValueError("recurrence_only_lora must be a boolean")
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
        if "linear_attention" in self.layer_types:
            # The upstream pure-torch GDN kernels use graph-invalid in-place
            # accumulation.  Runtime construction installs the tested
            # out-of-place equivalents so direct API users get the same
            # gradient semantics as the CLI loader.
            from .gdn_patch import install as install_gdn_patch
            install_gdn_patch()
        self.n_layers = len(self.base.layers)
        lo, hi = interval
        if not (0 <= lo < hi <= self.n_layers):
            raise ValueError(f"bad interval {interval} for {self.n_layers}L")
        self.interval = (lo, hi)
        if isinstance(max_k, bool) or not isinstance(max_k, int) or max_k < 0:
            raise ValueError("max_k must be a non-negative integer")
        self.max_k = max_k
        hidden = self.base.embed_tokens.embedding_dim

        self.guard = VocabGuard(model, tokenizer)

        dev = next(model.parameters()).device
        # clock trainables are explicitly fp32; never the global default dtype
        self.clock = make_step_clock(hidden, max_k, device=dev)

        self.injected = inject_lora(
            [self.base.layers[i] for i in range(lo, hi)],
            r=lora_r, alpha=lora_alpha)
        self.use_clock = use_clock
        self.recurrence_only_lora = recurrence_only_lora
        if self.recurrence_only_lora:
            # Disabled is the fail-safe resting state.  Only latent_steps opens
            # a narrowly scoped adapter-active window.
            self._set_lora_enabled(False)

    def runtime_contract(self) -> dict:
        """Executable scientific metadata; generic RCC ABI is not this path."""
        return {
            "contract_version": self.RUNTIME_CONTRACT_VERSION,
            "evidence_runtime": "LocalizedRecurrence",
            "generic_state_controller_abi": "SCAFFOLD_NOT_EVIDENCE_PATH",
            "training_gradient_semantics": self.TRAINING_GRADIENT_SEMANTICS,
            "cache_gradient_semantics": "detached_after_each_layer_application",
            "adapter_activation_policy": (
                "recurrence_only" if self.recurrence_only_lora else "all_stages"),
            "prefill_adapter_active": not self.recurrence_only_lora,
            "recurrence_adapter_active": True,
            "candidate_adapter_active": not self.recurrence_only_lora,
            "gradient_checkpointing": "UNSUPPORTED_ABSENT",
            "candidate_scoring": "autoregressive_raw_per_token_logprobs",
            "same_adapter_supported_k": list(range(self.max_k + 1)),
        }

    def _set_lora_enabled(self, enabled: bool) -> None:
        """Toggle every injected adapter or fail before executing a stage."""
        if not isinstance(enabled, bool):
            raise TypeError("LoRA enabled state must be a boolean")
        for adapter in self.injected:
            if not isinstance(adapter, LoRALinear) \
                    or not getattr(adapter, "supports_runtime_toggle", False) \
                    or not hasattr(adapter, "enabled"):
                raise RuntimeError(
                    "recurrence-only LoRA requires runtime-toggleable "
                    "LoRALinear adapters")
        for adapter in self.injected:
            adapter.enabled = enabled
        if any(adapter.enabled is not enabled for adapter in self.injected):
            raise RuntimeError("LoRA activation toggle did not take effect")

    @contextlib.contextmanager
    def _lora_scope(self, enabled: bool):
        if not self.recurrence_only_lora:
            yield
            return
        previous = tuple(adapter.enabled for adapter in self.injected)
        self._set_lora_enabled(enabled)
        try:
            yield
        finally:
            for adapter, state in zip(self.injected, previous):
                adapter.enabled = state
            if any(adapter.enabled is not state
                   for adapter, state in zip(self.injected, previous)):
                raise RuntimeError("failed to restore LoRA activation state")

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
                return v.detach()
            elif isinstance(v, dict):
                for key, value in v.items():
                    v[key] = sever(value)
                return v
            elif isinstance(v, list):
                for index, value in enumerate(v):
                    v[index] = sever(value)
                return v
            elif isinstance(v, tuple):
                return tuple(sever(value) for value in v)
            return v

        for layer in cache.layers:
            for attr in ("conv_states", "recurrent_states", "keys",
                         "values"):
                v = getattr(layer, attr, None)
                if v is not None:
                    setattr(layer, attr, sever(v))

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
        """Prefill the full cache and return the interval-boundary state.

        Localized recurrence must start/read out at the interval's output
        boundary.  Returning the final decoder state here would feed a
        final-layer representation back into middle layers and would also
        make K=0 re-apply the tail to an already-final state.
        """
        from transformers import DynamicCache
        cache = DynamicCache(config=self.config)
        emb = (input_ids_or_emb if not torch.is_tensor(input_ids_or_emb)
               or input_ids_or_emb.dtype.is_floating_point
               else self.base.embed_tokens(input_ids_or_emb))
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx, self._lora_scope(False):
            hi = self.interval[1]
            h = self._run_layers(range(hi), emb, cache, 0, grad=grad)
            boundary = h[:, -1:, :]
            if hi < self.n_layers:
                self._run_layers(range(hi, self.n_layers), h, cache, 0,
                                 grad=grad)
        return cache, boundary

    # -- stages ----------------------------------------------------------------

    def latent_steps(self, z, cache, start_pos, k_steps, *,
                     grad=False, ablate=None):
        """Apply the interval K times; GUARDED: no vocab/tokenizer activity.

        ablate keys (strictly validated; unknown keys/modes are fatal):
          zero_state         -> post-recurrence readout carrier z zeroed
          noise_state        -> post-recurrence readout carrier replaced by
                                deterministic norm-matched noise
          clocks             -> "identity"|"off"|"reverse"|
                                "shuffle_perm:i,j,.." (full unique perm of
                                range(k_steps); compute-matched)
          bypass_interval    -> interval layers replaced by identity
          truncate_k         -> effective step count reduced
          swap_state         -> caller replaces post-recurrence z with the
                                 partner's post-recurrence carrier
          reset_state        -> caller restores z0 after compute-matched loop
          reset_cache        -> caller restores the prompt-only cache after the
                                compute-matched loop, before readout
        """
        if isinstance(k_steps, bool) or not isinstance(k_steps, int) \
                or not 0 <= k_steps <= self.max_k:
            raise ValueError(f"k_steps must be an int in [0, {self.max_k}]")
        ab = validate_ablation(ablate, k_steps)
        mode = ab.get("clocks", "identity")
        eff_k = min(int(ab.get("truncate_k", k_steps)), k_steps)
        # validated against the FULL requested loop; truncation is explicit
        idxs = parse_clock_mode(mode, k_steps)[:eff_k]

        lo, hi = self.interval
        with self.guard.window(), self._lora_scope(True):
            pos = start_pos
            for ci in idxs:
                zz = z
                if self.use_clock and mode != "off":
                    tok = torch.tensor([ci + 1], device=z.device)
                    zz = zz + self.clock(tok).view(1, 1, -1).to(z.dtype)
                if ab.get("bypass_interval"):
                    z = zz
                else:
                    z = self._run_layers(range(lo, hi), zz, cache, pos,
                                         grad=grad)
                pos += 1
        # Causal state interventions happen after the requested recurrence,
        # immediately before readout.  Ablating z0 would let the loop rebuild
        # information from the prompt cache and could falsely look state-free.
        if ab.get("zero_state"):
            z = torch.zeros_like(z)
        if ab.get("noise_state"):
            g = torch.Generator(device=z.device.type)
            g.manual_seed(int(ab.get("noise_seed", 1234)))
            n = torch.randn(z.shape, generator=g, device=z.device,
                            dtype=torch.float32).to(z.dtype)
            n = n / (n.norm() + 1e-8) * (z.norm() + 1e-8)
            z = n
        return z, pos

    def logits_from_hidden(self, h_prenorm):
        with self.guard.window(allow_vocab=True):
            return self.model.lm_head(self.base.norm(h_prenorm))

    # -- evaluation -------------------------------------------------------------

    def _score_candidate_tokens(self, z, cache, pos, candidate_ids, *,
                                grad=False):
        """Autoregressively score one non-empty candidate from raw token ids.

        The recurrence output predicts the first token.  Every preceding
        answer token then traverses the complete decoder before predicting the
        next token.  This is essential for interval=(0, L), where the old
        batch-style tail path accidentally sent answer embeddings through zero
        decoder layers.
        """
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1 \
                or candidate_ids.shape[1] == 0:
            raise ValueError("candidate_ids must have shape [1, n] with n > 0")

        hi = self.interval[1]
        if hi < self.n_layers:
            with self._lora_scope(False):
                current = self._run_layers(
                    range(hi, self.n_layers), z, cache, pos, grad=grad)
            next_pos = pos + 1
            readout_tail_layers = self.n_layers - hi
        else:
            current = z
            next_pos = pos
            readout_tail_layers = 0

        # Keep a live, fixed readout carrier from the completed recurrence.  The
        # first answer token keeps the established direct readout above; every
        # later token re-enters the decoder from embeddings and receives this
        # carrier at the same output-boundary where recurrence produced it.
        carrier = z
        token_logprobs = []
        for token_index in range(candidate_ids.shape[1]):
            logits = self.logits_from_hidden(current[:, -1:, :])
            logp = torch.log_softmax(logits.float(), dim=-1)
            target = candidate_ids[0, token_index]
            token_logprobs.append(logp[0, 0, target])
            if token_index + 1 == candidate_ids.shape[1]:
                continue
            previous = candidate_ids[:, token_index:token_index + 1]
            hidden = self.base.embed_tokens(previous)
            with self._lora_scope(False):
                current = self._run_layers(
                    range(hi), hidden, cache, next_pos, grad=grad)
                current = current + carrier
                current = self._run_layers(
                    range(hi, self.n_layers), current, cache, next_pos,
                    grad=grad)
            next_pos += 1

        return torch.stack(token_logprobs), {
            "readout_tail_layer_applications": readout_tail_layers,
            "answer_context_full_decoder_layer_applications": (
                self.n_layers * max(0, candidate_ids.shape[1] - 1)
            ),
        }

    @_torch_no_grad
    def score_candidates(self, input_ids, candidate_ids, k_steps, *,
                         ablate=None, partner_input_ids=None):
        """Return raw per-token candidate evidence and measured compute."""
        import time

        from ..backends.hf_qwen import cache_restore, cache_snapshot

        if not candidate_ids:
            raise ValueError("candidate set must be non-empty")

        lm0 = self.guard.lm_head_calls
        tk0 = self.guard.tokenizer_calls
        gen0 = self.guard.generate_calls
        decode0 = (self.guard.tokenizer_operations["decode"]
                   + self.guard.tokenizer_operations["batch_decode"])
        cuda_device = (input_ids.device
                       if input_ids.device.type == "cuda" else None)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
        t0 = time.perf_counter()
        ab = validate_ablation(ablate or {}, k_steps)
        if ab.get("swap_state") and partner_input_ids is None:
            raise ValueError(
                "swap_state requires partner_input_ids; refusing a mislabeled "
                "clean run")
        if partner_input_ids is not None and not ab.get("swap_state"):
            raise ValueError(
                "partner_input_ids requires swap_state; refusing an unlabeled "
                "state intervention")

        # Cache/prompt context always belongs to the evaluated example.
        # Carrier interventions preserve the target recurrence cache.  The
        # orthogonal reset_cache arm alone snapshots prompt-only cache state.
        cache, z0 = self._encode(input_ids)
        prompt_cache = cache_snapshot(cache) if ab.get("reset_cache") else None
        partner_prefill_tokens = 0
        carrier_intervention = bool(
            ab.get("zero_state") or ab.get("noise_state")
            or ab.get("swap_state") or ab.get("reset_state"))
        loop_start = input_ids.shape[1]
        partner_cache = None
        partner_z0 = None
        if partner_input_ids is not None:
            partner_cache, partner_z0 = self._encode(partner_input_ids)
            partner_prefill_tokens = int(partner_input_ids.shape[1])
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        t1 = time.perf_counter()

        z, pos = self.latent_steps(
            z0, cache, loop_start, k_steps, ablate=ab)
        if partner_input_ids is not None:
            partner_ab = dict(ab)
            partner_ab.pop("swap_state", None)
            partner_z, _partner_pos = self.latent_steps(
                partner_z0, partner_cache, partner_input_ids.shape[1],
                k_steps, ablate=partner_ab)
            z = partner_z
        if ab.get("reset_state"):
            # Position/compute-matched no-recurrence carrier control: execute
            # the full loop, then discard zK and read out the original z0.
            z = z0
        if prompt_cache is not None:
            # Orthogonal cache control: recurrence really executed and pos
            # remains prompt_len + effective_k, but readout sees prompt cache.
            cache_restore(cache, prompt_cache)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        t2 = time.perf_counter()
        snap = cache_snapshot(cache)
        details = []
        candidate_compute = []
        for candidate in candidate_ids:
            cache_restore(cache, snap)
            ids = torch.as_tensor(
                candidate, device=input_ids.device, dtype=torch.long).view(1, -1)
            if ids.shape[1] == 0:
                raise ValueError("candidate token sequences must be non-empty")
            token_tensors, counters = self._score_candidate_tokens(
                z, cache, pos, ids)
            token_logprobs = tuple(float(value) for value in token_tensors)
            if any(not math.isfinite(value) for value in token_logprobs):
                bad = [i for i, value in enumerate(token_logprobs)
                       if not math.isfinite(value)]
                raise NonFiniteCandidateScores(
                    f"candidate {len(details)} token logprobs at positions "
                    f"{bad} are not finite; refusing to aggregate")
            # Must match the canonical v3 scorer bit-for-bit.  Built-in sum
            # can round differently from math.fsum and fabricate/remove an
            # exact top tie in real-model evidence.
            raw_sum = math.fsum(token_logprobs)
            details.append(CandidateScoreDetail(
                token_ids=tuple(int(value) for value in candidate),
                token_logprobs=token_logprobs,
                raw_sum_logprob=raw_sum,
                length_normalized_logprob=raw_sum / len(token_logprobs),
            ))
            candidate_compute.append(counters)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        t3 = time.perf_counter()

        effective_k = min(int(ab.get("truncate_k", k_steps)), k_steps)
        recurrence_multiplier = 2 if partner_input_ids is not None else 1
        recurrence_apps = recurrence_multiplier * (
            0 if ab.get("bypass_interval") else
            (self.interval[1] - self.interval[0]) * effective_k
        )
        target_prefill_tokens = int(input_ids.shape[1])
        prefill_apps = self.n_layers * (
            target_prefill_tokens + partner_prefill_tokens)
        readout_tail_apps = sum(
            item["readout_tail_layer_applications"]
            for item in candidate_compute)
        answer_decoder_apps = sum(
            item["answer_context_full_decoder_layer_applications"]
            for item in candidate_compute)
        candidate_apps = readout_tail_apps + answer_decoder_apps
        total_apps = prefill_apps + recurrence_apps + candidate_apps

        raw_sums = [detail.raw_sum_logprob for detail in details]
        if cuda_device is not None:
            peak_memory_bytes = int(
                torch.cuda.max_memory_allocated(cuda_device))
        else:
            # Torch exposes no resettable MPS peak allocator counter.  Do not
            # mislabel a current-allocation/RSS sample as peak VRAM.
            peak_memory_bytes = None
        wall_seconds = t3 - t0
        measured_compute = {
            "prefill_layers": prefill_apps,
            "recurrence_interval_applications": recurrence_apps,
            "k_loops": recurrence_multiplier * effective_k,
            "candidate_tail_layers": candidate_apps,
            "lm_head_calls": self.guard.lm_head_calls - lm0,
            "tokenizer_calls": self.guard.tokenizer_calls - tk0,
            "decode_calls": (
                self.guard.tokenizer_operations["decode"]
                + self.guard.tokenizer_operations["batch_decode"] - decode0),
            "wall_seconds": wall_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "successful_task": True,
            # This object is sealed into each latent_eval.v3 record hash.  It
            # prevents a clean record from being relabeled as an intervention
            # by changing only the surrounding eval envelope.
            "eval_ablation": dict(ab),
            "adapter_activation_policy": (
                "recurrence_only" if self.recurrence_only_lora else "all_stages"),
            "prefill_adapter_active": not self.recurrence_only_lora,
            "recurrence_adapter_active": True,
            "candidate_adapter_active": not self.recurrence_only_lora,
        }
        rep = RecurrenceReport(
            k_steps=k_steps,
            interval=self.interval,
            layer_applications=total_apps,
            lm_head_calls_total=self.guard.lm_head_calls,
            seconds_prefill=t1 - t0,
            seconds_loop=t2 - t1,
            seconds_readout=t3 - t2,
            extra={
                "n_candidates": len(candidate_ids),
                "successful_candidates": len(details),
                "failed_candidates": 0,
                "candidate_token_logprobs": [
                    list(detail.token_logprobs) for detail in details],
                "candidate_token_counts": [
                    detail.token_count for detail in details],
                "candidate_raw_sum_logprobs": raw_sums,
                "candidate_length_normalized_logprobs": [
                    detail.length_normalized_logprob for detail in details],
                "prefill_tokens": target_prefill_tokens,
                "partner_state_prefill_tokens": partner_prefill_tokens,
                "latent_state_source": (
                    "partner" if partner_input_ids is not None else "target"),
                "prompt_cache_source": "target",
                "latent_state_ablation_stage": (
                    "post_recurrence_readout"
                    if carrier_intervention else None),
                "recurrence_cache_ablation_stage": (
                    "post_recurrence_pre_readout"
                    if ab.get("reset_cache") else None),
                "recurrence_cache_at_readout": (
                    "target_prompt_only"
                    if (ab.get("reset_cache") or effective_k == 0
                        or ab.get("bypass_interval")) else
                    "prompt_plus_target_recurrence"),
                "recurrence_start_position": loop_start,
                "readout_position": pos,
                "prefill_layer_applications": prefill_apps,
                "recurrence_interval_layer_applications": recurrence_apps,
                "recurrence_effective_k": effective_k,
                "candidate_readout_tail_layer_applications": readout_tail_apps,
                "candidate_full_decoder_layer_applications": answer_decoder_apps,
                "candidate_layer_applications": candidate_apps,
                "total_layer_token_applications": total_apps,
                "lm_head_calls": measured_compute["lm_head_calls"],
                "tokenizer_calls": measured_compute["tokenizer_calls"],
                "generate_calls": self.guard.generate_calls - gen0,
                "tokenizer_decode_calls": measured_compute["decode_calls"],
                "wall_clock_seconds": wall_seconds,
                "peak_memory_bytes": measured_compute["peak_memory_bytes"],
                "adapter_activation_policy":
                    measured_compute["adapter_activation_policy"],
                "prefill_adapter_active":
                    measured_compute["prefill_adapter_active"],
                "recurrence_adapter_active":
                    measured_compute["recurrence_adapter_active"],
                "candidate_adapter_active":
                    measured_compute["candidate_adapter_active"],
                "training_gradient_semantics": (
                    self.TRAINING_GRADIENT_SEMANTICS),
                "compute": measured_compute,
            },
        )
        return details, rep

    @_torch_no_grad
    def rank_candidates(self, input_ids, candidate_ids, k_steps, *,
                        ablate=None, partner_input_ids=None):
        """Length-normalized primary ranking over raw scoring evidence.

        ablate.swap_state semantics: pass partner_input_ids; the loop then
        runs on the PARTNER's post-encoding state while scoring this
        example's candidates.

        The second return value remains raw summed log-probabilities for API
        compatibility.  Exact top ties fail closed; non-top ties are ordered
        by candidate token content only for byte-stable serialization.
        """
        details, rep = self.score_candidates(
            input_ids, candidate_ids, k_steps, ablate=ablate,
            partner_input_ids=partner_input_ids)
        scores = [detail.raw_sum_logprob for detail in details]
        primary = [detail.length_normalized_logprob for detail in details]
        best = max(primary)
        tied_top = [index for index, value in enumerate(primary)
                    if value == best]
        rep.extra["primary_score_definition"] = (
            "mean_candidate_token_logprob_v1")
        rep.extra["primary_scores"] = primary
        rep.extra["exact_top_tie_indices"] = (
            tied_top if len(tied_top) > 1 else [])
        if len(tied_top) != 1:
            raise AmbiguousTopTie(
                f"exact top tie under mean_candidate_token_logprob_v1 at "
                f"candidate indices {tied_top}; refusing index tie-break",
                candidate_details=details, report=rep)
        order = sorted(
            range(len(details)),
            key=lambda index: (-primary[index], details[index].token_ids),
        )
        return order, scores, rep

    def tail_sequence(self, h, cache, pos, *, grad=False):
        """Tail layers over an arbitrary short stream (z [+ answer tokens])."""
        hi = self.interval[1]
        if hi >= self.n_layers:
            return h
        with self._lora_scope(False):
            return self._run_layers(
                range(hi, self.n_layers), h, cache, pos, grad=grad)

    # -- training ---------------------------------------------------------------

    def loss_on_example(self, input_ids, answer_ids, k_steps, *,
                        ablate=None, detach_z0=False):
        """Teacher-forced CE on gold answer tokens appended after the loop."""
        from ..backends.hf_qwen import cache_restore, cache_snapshot

        ab = validate_ablation(ablate or {}, k_steps)
        if self.recurrence_only_lora and k_steps == 0:
            raise ValueError(
                "recurrence-only LoRA has no trainable path at K=0; "
                "evaluate K=0 from a K>0-trained adapter instead")
        eval_only = [key for key in ("swap_state", "reset_state", "reset_cache")
                     if ab.get(key)]
        if eval_only:
            raise ValueError(
                f"evaluation-only ablation(s) {eval_only} are unsupported in "
                "loss_on_example")
        cache, z0 = self._encode(input_ids, grad=True)
        if detach_z0:
            z0 = z0.detach()
        state_intervention = bool(ab.get("zero_state") or ab.get("noise_state"))
        prompt_cache = cache_snapshot(cache) if state_intervention else None
        z, pos = self.latent_steps(z0, cache, input_ids.shape[1], k_steps,
                                   grad=True, ablate=ab)
        if prompt_cache is not None:
            cache_restore(cache, prompt_cache)
        token_logprobs, _counters = self._score_candidate_tokens(
            z, cache, pos, answer_ids, grad=True)
        return -token_logprobs.mean()

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
        """Prevalidate, pre-stage, then mutate targets in one safe commit.

        Every incoming tensor is converted to the exact target
        device/dtype BEFORE the first mutation (staged candidates); all
        targets are snapshotted and, if any copy raises late, every target
        is restored bit-exactly from its own pre-mutation snapshot before
        AdapterBundleError propagates. Validation covers exact keys,
        tensor type, shape, dtype and finiteness.
        """
        from ..train.checkpointing import (
            AdapterBundleError, AdapterBundleSchemaError, NonFiniteStateError)
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
            if t.is_floating_point() or t.is_complex():
                if not bool(torch.isfinite(t).all()):
                    raise NonFiniteStateError(
                        f"{key}: contains non-finite values")
            staged[key] = t.to(device=target.device, dtype=target.dtype)
        snapshots = {key: target.detach().clone()
                     for key, target in targets.items()}
        try:
            with torch.no_grad():
                for key, target in targets.items():
                    target.copy_(staged[key])
        except BaseException as e:
            with torch.no_grad():
                for key, target in targets.items():
                    try:
                        target.copy_(snapshots[key])
                    except BaseException:
                        # a persistently failing copy implementation must
                        # not corrupt already-restored targets: rebind the
                        # pristine snapshot onto the live Parameter instead
                        target.data = snapshots[key].data.to(
                            device=target.device, dtype=target.dtype)
            raise AdapterBundleError(
                "adapter load failed mid-copy; all targets rolled back "
                f"bit-exactly ({e})") from e

    # -- identity-bound bundles ------------------------------------------------

    def adapter_recipe(self, *, config: dict) -> dict:
        """The canonical recipe this runtime binds into exported bundles.

        The recipe is derived from the caller's FULL training config via
        the strict canonical builder; runtime state (interval/max_k/rank/
        alpha/mode) must agree with it exactly or identity fails closed —
        a runtime configured differently from its claimed config can
        never mint an adapter bundle.
        """
        from ..train.checkpointing import (
            AdapterBundleIdentityError, recipe_from_config)
        recipe = recipe_from_config(config,
                                    suite_sha256=config["suite_sha256"])
        lo, hi = self.interval
        runtime_alpha = float(self.injected[0].scaling
                              * self.injected[0].lora_A.shape[0])
        drift = []
        claimed_policy = config.get("recurrence_only_lora", False)
        if not isinstance(claimed_policy, bool):
            drift.append("recurrence_only_lora is not a boolean")
            claimed_policy = None
        mode_has_policy = recipe["mode"].endswith(
            RECURRENCE_ONLY_LORA_MODE_SUFFIX)
        if claimed_policy is not None and claimed_policy != mode_has_policy:
            drift.append(
                f"mode {recipe['mode']!r} does not seal "
                f"recurrence_only_lora={claimed_policy}")
        if claimed_policy is not None \
                and claimed_policy != self.recurrence_only_lora:
            drift.append(
                f"recurrence_only_lora {claimed_policy} != runtime "
                f"{self.recurrence_only_lora}")
        stored_contract = config.get("runtime_contract")
        if claimed_policy and stored_contract is None:
            drift.append("recurrence-only LoRA lacks runtime_contract metadata")
        elif stored_contract is not None \
                and stored_contract != self.runtime_contract():
            drift.append("runtime_contract != live runtime contract")
        if list(recipe["interval"]) != [lo, hi]:
            drift.append(f"interval {recipe['interval']} != runtime "
                         f"{[lo, hi]}")
        if int(recipe["max_k"]) != int(self.max_k):
            drift.append(f"max_k {recipe['max_k']} != runtime "
                         f"{int(self.max_k)}")
        if int(recipe["lora_r"]) != int(self.injected[0].lora_A.shape[0]):
            drift.append(f"lora_r {recipe['lora_r']} != runtime "
                         f"{self.injected[0].lora_A.shape[0]}")
        if float(recipe["lora_alpha"]) != runtime_alpha:
            drift.append(f"lora_alpha {recipe['lora_alpha']} != runtime "
                         f"{runtime_alpha}")
        if drift:
            raise AdapterBundleIdentityError(
                "training config disagrees with live runtime: "
                + "; ".join(drift))
        return recipe

    def export_adapter_bundle(self, path, *, model_id, revision,
                              config, metrics=None) -> dict:
        """Persist this adapter as an identity-bound best_params.pt bundle.

        ``config`` is the FULL training config (including mode and
        suite_sha256); every training-semantic field is canonically bound.
        """
        from ..train.checkpointing import save_adapter_bundle
        return save_adapter_bundle(
            path, self.adapter_state_dict(), model_id=model_id,
            revision=revision,
            recipe=self.adapter_recipe(config=config),
            metrics=metrics)

    def load_adapter_bundle(self, path, *, model_id, revision, config):
        """Load an identity-bound bundle; recipe/identity fully prevalidated."""
        from ..train.checkpointing import load_adapter_bundle
        state = load_adapter_bundle(path, model_id=model_id, revision=revision,
                                    recipe=self.adapter_recipe(config=config))
        self.load_adapter_state(state)
