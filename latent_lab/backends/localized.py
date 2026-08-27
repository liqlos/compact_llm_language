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
zero-initialized per-step clock embeddings.  The same adapter is active during
prompt prefill, recurrence, and answer scoring; recurrence-dose comparisons
therefore require evaluating one fixed adapter at every K.
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
# ablation parsing (strict; fail-closed)
# ---------------------------------------------------------------------------

CLOCK_MODES = ("identity", "off", "reverse")
KNOWN_ABLATION_KEYS = ("zero_state", "noise_state", "noise_seed", "clocks",
                       "bypass_interval", "truncate_k", "swap_state")


def parse_clock_mode(mode, k_steps: int):
    """Parse the 'clocks' ablation value; return the concrete step index list.

    Accepts:
      "identity" -> [0..k)
      "off"      -> [] (no step embeddings applied)
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

    RUNTIME_CONTRACT_VERSION = "localized_recurrence.runtime.v1"
    TRAINING_GRADIENT_SEMANTICS = (
        "hidden_state_chain_bptt_with_detached_cache_recurrence"
    )

    def __init__(self, model, tokenizer=None, *, interval, max_k=16,
                 lora_r=8, lora_alpha=16.0, use_clock=True):
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

    def runtime_contract(self) -> dict:
        """Executable scientific metadata; generic RCC ABI is not this path."""
        return {
            "contract_version": self.RUNTIME_CONTRACT_VERSION,
            "evidence_runtime": "LocalizedRecurrence",
            "generic_state_controller_abi": "SCAFFOLD_NOT_EVIDENCE_PATH",
            "training_gradient_semantics": self.TRAINING_GRADIENT_SEMANTICS,
            "cache_gradient_semantics": "detached_after_each_layer_application",
            "prefill_adapter_active": True,
            "gradient_checkpointing": "UNSUPPORTED_ABSENT",
            "candidate_scoring": "autoregressive_raw_per_token_logprobs",
            "same_adapter_supported_k": list(range(self.max_k + 1)),
        }

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

        ablate keys (strictly validated; unknown keys/modes are fatal):
          zero_state         -> incoming z zeroed AND all prompt-derived
                                cache state (conv/recurrent/KV) zeroed,
                                so no prompt latent survives the reset
          clocks             -> "identity"|"off"|"reverse"|
                                "shuffle_perm:i,j,.." (full unique perm of
                                range(k_steps); compute-matched)
          bypass_interval    -> interval layers replaced by identity
          truncate_k         -> effective step count reduced
          swap/noise         -> performed by caller replacing z beforehand
        """
        if isinstance(k_steps, bool) or not isinstance(k_steps, int) \
                or not 0 <= k_steps <= self.max_k:
            raise ValueError(f"k_steps must be an int in [0, {self.max_k}]")
        ab = validate_ablation(ablate, k_steps)
        if ab.get("zero_state"):
            from ..backends.hf_qwen import cache_zero_prompt_state
            z = torch.zeros_like(z)
            cache_zero_prompt_state(cache)  # erase ALL prompt-derived state
        if ab.get("noise_state"):
            g = torch.Generator(device=z.device.type)
            g.manual_seed(int(ab.get("noise_seed", 1234)))
            n = torch.randn(z.shape, generator=g, device=z.device,
                            dtype=torch.float32).to(z.dtype)
            n = n / (n.norm() + 1e-8) * (z.norm() + 1e-8)
            z = n
        mode = ab.get("clocks", "identity")
        eff_k = min(int(ab.get("truncate_k", k_steps)), k_steps)
        # validated against the FULL requested loop; truncation is explicit
        idxs = parse_clock_mode(mode, k_steps)[:eff_k]

        lo, hi = self.interval
        with self.guard.window():
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
            current = self._run_layers(
                range(hi, self.n_layers), z, cache, pos, grad=grad)
            next_pos = pos + 1
            readout_tail_layers = self.n_layers - hi
        else:
            current = z
            next_pos = pos
            readout_tail_layers = 0

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
            current = self._run_layers(
                range(self.n_layers), hidden, cache, next_pos, grad=grad)
            next_pos += 1

        return torch.stack(token_logprobs), {
            "readout_tail_layer_applications": readout_tail_layers,
            "answer_context_full_decoder_layer_applications": (
                self.n_layers * max(0, candidate_ids.shape[1] - 1)
            ),
        }

    @torch.no_grad()
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
        t0 = time.perf_counter()
        source_ids = (partner_input_ids
                      if partner_input_ids is not None else input_ids)
        cache, z0 = self._encode(source_ids)
        loop_start = source_ids.shape[1]
        t1 = time.perf_counter()
        ab = validate_ablation(ablate or {}, k_steps)
        if partner_input_ids is not None:
            ab["swap_state"] = True
        z, pos = self.latent_steps(z0, cache, loop_start, k_steps,
                                   ablate=ab)
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
            raw_sum = sum(token_logprobs)
            details.append(CandidateScoreDetail(
                token_ids=tuple(int(value) for value in candidate),
                token_logprobs=token_logprobs,
                raw_sum_logprob=raw_sum,
                length_normalized_logprob=raw_sum / len(token_logprobs),
            ))
            candidate_compute.append(counters)
        t3 = time.perf_counter()

        effective_k = min(int(ab.get("truncate_k", k_steps)), k_steps)
        recurrence_apps = (
            0 if ab.get("bypass_interval") else
            (self.interval[1] - self.interval[0]) * effective_k
        )
        prefill_apps = self.n_layers * int(source_ids.shape[1])
        readout_tail_apps = sum(
            item["readout_tail_layer_applications"]
            for item in candidate_compute)
        answer_decoder_apps = sum(
            item["answer_context_full_decoder_layer_applications"]
            for item in candidate_compute)
        candidate_apps = readout_tail_apps + answer_decoder_apps
        total_apps = prefill_apps + recurrence_apps + candidate_apps

        raw_sums = [detail.raw_sum_logprob for detail in details]
        wall_seconds = t3 - t0
        measured_compute = {
            "prefill_layers": prefill_apps,
            "recurrence_interval_applications": recurrence_apps,
            "k_loops": effective_k,
            "candidate_tail_layers": candidate_apps,
            "lm_head_calls": self.guard.lm_head_calls - lm0,
            "tokenizer_calls": self.guard.tokenizer_calls - tk0,
            "decode_calls": (
                self.guard.tokenizer_operations["decode"]
                + self.guard.tokenizer_operations["batch_decode"] - decode0),
            "wall_seconds": wall_seconds,
            "peak_memory_bytes": None,
            "successful_task": True,
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
                "prefill_tokens": int(source_ids.shape[1]),
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
                "prefill_adapter_active": True,
                "training_gradient_semantics": (
                    self.TRAINING_GRADIENT_SEMANTICS),
                "compute": measured_compute,
            },
        )
        return details, rep

    @torch.no_grad()
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
        rep.extra["exact_top_tie_indices"] = tied_top
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
        return self._run_layers(
            range(hi, self.n_layers), h, cache, pos, grad=grad)

    # -- training ---------------------------------------------------------------

    def loss_on_example(self, input_ids, answer_ids, k_steps, *,
                        ablate=None, detach_z0=False):
        """Teacher-forced CE on gold answer tokens appended after the loop."""
        cache, z0 = self._encode(input_ids, grad=True)
        if detach_z0:
            z0 = z0.detach()
        z, pos = self.latent_steps(z0, cache, input_ids.shape[1], k_steps,
                                   grad=True, ablate=ablate)
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
