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
- Consequence: dedicated soft-embedding blocker probe required before any
  MLX-based latency claims (`latent_lab/bench/mlx_soft_embedding_probe.py`).
  Until then no MLX speed claims.
- Status: NOT MEASURED locally (no mlx installed).

## B7 — Environment-dependent test counts

- Claim (README): "93 tests".
- Measurement: 93 passed with tiktoken installed (this env). Previous review
  observed 85 passed + 2 skipped modules without tiktoken.
- Consequence: README/plan now phrase counts as environment-dependent.
- Status: FIXED (docs).
