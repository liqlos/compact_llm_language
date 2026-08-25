# Historical blockers and measured runtime facts

Snapshot date: 2026-08-24. This is an evidence ledger, not the current task
queue. Recheck environment-dependent claims before acting; current maturity and
gates live in `LATENT_ROADMAP.md`.

Execution environment (owner decision 2026-08-24): serious 4B/27B training
and CUDA experiment matrices run on rented Vast.ai GPUs behind the fail-closed
no-spend READY gate; this laptop runs unit/smoke tests, artifact validation,
and orchestration only; final 27B validation runs later on an owner-supplied
server. MLX entries below are preserved as measured historical evidence — MLX
on this laptop is not the release gate (VISION.md).

## B1 — Local hardware below original assumption

- Claim (old plan): Apple Silicon M4, 32 GB unified memory.
- Measurement (2026-08-22): Apple M1 Pro, 16 GB RAM (`sysctl hw.memsize`
  = 17179869184), ~64 GB disk free of 926 GB (93% used).
- Consequence: 27B training locally impossible; even 27B quantized inference
  is tight. Proxy experiments: feasible (Qwen3.5-0.8B ≈ 1.75 GB bf16).
- Status: RECORDED; ladder adjusted in VISION.md.

## B2 — ML stack was absent (RESOLVED)

- Claim: latent probe can run immediately.
- Measurement: `import torch` / `import mlx` fail in project venv
  (Python 3.14, uv-managed). tiktoken present.
- Plan: add optional dependency group `lab` (torch, transformers,
  accelerate); keep stdlib core importable without it.
- Current check (2026-08-25): both `torch` and `mlx` resolve in the project
  environment. Status: RESOLVED.

## B3 — No Qwen weights cached in the 2026-08-22 snapshot

- Measurement: `~/.cache/huggingface/hub` contains no Qwen models.
- Plan: download Qwen/Qwen3.5-0.8B (~1.75 GB bf16, verified via HF API)
  only after disk check; never auto-download 27B.
- Status: HISTORICAL; cache state is volatile and must be checked before a run.

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

## B7 — Environment-dependent test counts (RESOLVED DOC CLAIM)

- Historical measurement: 93 passed with tiktoken installed; an earlier review
  observed 85 passed + 2 skipped modules without it.
- Consequence: durable docs no longer use that count as current evidence;
  verification reports record the exact suite result for their own revision.
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
