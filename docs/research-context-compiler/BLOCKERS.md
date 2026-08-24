# Blockers — measured runtime facts

Updated: 2026-08-22. Each entry: claim, measurement, status.

## B1 — Local hardware below original assumption

- Claim (old plan): Apple Silicon M4, 32 GB unified memory.
- Measurement (2026-08-22): Apple M1 Pro, 16 GB RAM (`sysctl hw.memsize`
  = 17179869184), ~64 GB disk free of 926 GB (93% used).
- Consequence: 27B training locally impossible; even 27B quantized inference
  is tight. Proxy experiments: feasible (Qwen3.5-0.8B ≈ 1.75 GB bf16).
- Status: RECORDED; ladder adjusted in VISION.md.

## B2 — No ML stack installed

- Claim: latent probe can run immediately.
- Measurement: `import torch` / `import mlx` fail in project venv
  (Python 3.14, uv-managed). tiktoken present.
- Plan: add optional dependency group `lab` (torch, transformers,
  accelerate); keep stdlib core importable without it.
- Status: BLOCKED → resolved this session if install succeeds.

## B3 — No Qwen weights cached

- Measurement: `~/.cache/huggingface/hub` contains no Qwen models.
- Plan: download Qwen/Qwen3.5-0.8B (~1.75 GB bf16, verified via HF API)
  only after disk check; never auto-download 27B.
- Status: pending.

## B4 — Transformers GDN cached-forward bug window

- Research finding (PR #45513, fixed Apr 2026): between v5.2.0 and the fix,
  Gated DeltaNet layers ignored cached recurrent state unless seq_len==1,
  silently corrupting multi-token cached forwards for qwen3_5-family models.
- Consequence: pin transformers to a version including the fix; state probe
  must assert recurrence correctness across multi-token cached forwards.
- Status: MITIGATION PLANNED.

## B5 — MPS dtype pitfalls

- Research findings: bf16 has no hardware path on Apple Silicon (up to ~10×
  slowdowns); fp16 attention NaN issues on macOS ≥14.5; torch 2.13.0 fixes a
  macOS 27 beta MPS corruption bug.
- Consequence: probe runs fp32 on CPU and fp32/fp16 on MPS, records both;
  never reports bf16 MPS numbers as performance evidence.
- Status: MITIGATION PLANNED.

## B6 — MLX-LM hybrid support & inputs_embeds path

- Research status (2026-08-22): mlx-lm qwen3_5 support exists per docs;
  off-vocabulary embedding behaviour on hybrid cache unverified locally.
- **MEASURED 2026-08-22** (`latent_lab/bench/results/mlx_soft_embedding_probe.json`,
  mlx 0.32.1 / mlx-lm 0.31.3 / macOS 26.4.1 / M1 Pro 16 GB,
  Qwen3.5-0.8B bf16):
  - exact vocabulary embeddings via `input_embeddings`: logits bit-identical
    to token path, same speed (E2) — public soft-prompt entry point works;
  - off-manifold embeddings (perturbed/zero/random): ~3.5–5× slowdown of the
    pass AND of subsequent normal-token decode; ordering control (E6) recovers
    full speed ⇒ effect is input-specific, suspected Metal subnormal handling;
  - prefix-cache trim reuse: `can_trim_prompt_cache=False` for hybrid
    ArraysCache ⇒ no public prompt-prefix reuse in mlx-lm 0.31.3.
- Consequence: (a) internal localized recurrence must bypass the slow
  off-manifold path or flush subnormals; (b) MLX latency benchmarking is
  BLOCKED on cache-reuse + kernel mitigation; architecture results must be
  separated from this runtime defect.
- Status: MEASURED; two concrete mitigations identified.

## B7 — Environment-dependent test counts

- Claim (README): "93 tests".
- Measurement: 93 passed with tiktoken installed (this env). Previous review
  observed 85 passed + 2 skipped modules without tiktoken.
- Consequence: README/plan now phrase counts as environment-dependent.
- Status: FIXED (docs).

## B8 — State probe: RESOLVED (runtime control proven)

- Claim: we cannot control hidden states/recurrent caches of a hybrid Qwen
  without token generation.
- Measurement 2026-08-22 (`latent_lab/bench/results/state_probe_{cpu,mps_fp16}.json`,
  Qwen/Qwen3.5-0.8B rev 2fc0636, transformers 5.15.1, torch 2.13.0):
  - config layer_types parsed (24 layers, 3×GDN+1×attn × 6);
  - KV shapes [1,2,pos,256]; GDN conv+recurrent states ≈19.8 MB (fp16) per
    prompt, snapshot/restore roundtrip EXACT (logits equal);
  - `inputs_embeds` path matches token path;
  - continuous recurrence K=3 through base model with ZERO lm_head calls
    (no vocabulary decode inside loop);
  - wall-clock per recurrence step: 5.9 s CPU fp32 → **16.9 ms MPS fp16**;
    RSS ≤ 3.7 GB.
- Status: MODEL_VERIFIED on the 0.8B proxy — localized-recurrence control
  points all reachable. NOT evidence of reasoning quality.

## B9 — Guarded optimizer step: paid Vast/CUDA training blocked pending A/B

- Claim: `guarded_optimizer_step` (latent_lab/train/checkpointing.py) is
  safe to wrap around paid Vast/CUDA training runs.
- Measurement: none on CUDA. The transaction's cost is bounded by design —
  no CPU serialization, no per-leaf copies; exactly ONE on-device clone per
  unique backing storage plus metadata/finiteness reads (complex-aware) —
  but finiteness reductions synchronize per state/grad tensor, so the
  step-time impact on CUDA is UNKNOWN until measured.
- Consequence: paid Vast/CUDA training REMAINS BLOCKED on this guard until
  a target-CUDA A/B shows <= 5% median AND p95 step-time regression versus
  the unguarded baseline. No CUDA numbers are claimed or fabricated here;
  CPU-only tests prove correctness only.
- Status: BLOCKED (documentation of boundary; correctness suite: 41
  runtime-integrity tests green on CPU).
