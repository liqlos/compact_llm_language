"""Latent baseline training + evaluation driver over the behavioral suite.

Trains (frozen backbone + interval LoRA + zero-init step clock) to produce
gold answers after K latent steps, then evaluates exact candidate ranking
on validation/test splits with causal ablations.

Configurations:
  D  --interval full     (loop = entire decoder, Coconut-style feedback)
  E  --interval mid      (localized loop; default [12,18) on 24L)
  F  k_steps=0           (tail-only control, tail-adjacent LoRA still trains)

Usage (train):
  python -m latent_lab.bench.latent_run train --k 4 --interval mid \
      --steps 800 --out .rcc_work/latent_E_k4
Usage (eval):
  python -m latent_lab.bench.latent_run eval --adapter .rcc_work/latent_E_k4 \
      --split test_id [--ablate zero_state] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

from latent_lab.backends.localized import (
    CANDIDATE_CE_MODE_SUFFIX,
    NEUTRAL_DELTA_MODE_SUFFIX,
    RECURRENCE_ONLY_LORA_MODE_SUFFIX,
    TRAINING_OBJECTIVES,
    training_objective_from_config,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"

EVAL_ABLATIONS = ("zero_state", "bypass_interval", "clocks_off",
                   "reverse_clocks", "truncate_half", "swap_state",
                   "noise_state", "readout_reset_to_z0",
                   "cache_reset_to_prompt", "full_state_reset_to_z0")


def interval_from_spec(spec: str, n_layers: int) -> tuple[int, int]:
    """'mid'/'full'/'head' proportional to depth, or explicit 'lo,hi'."""
    if "," in spec:
        lo, hi = (int(x) for x in spec.split(","))
        return (lo, hi)
    if spec == "full":
        return (0, n_layers)
    if spec == "mid":
        return (n_layers // 2, n_layers * 3 // 4)
    if spec == "head":
        return (n_layers * 3 // 4, n_layers)
    raise ValueError(f"unknown interval spec {spec}")

ANSWER_PREFIX = " "  # prompt ends with "Answer:" -> continuation " <ans>"


def peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return peak_bytes / 2 ** 20


def _gpu_mem_report(device: str) -> dict:
    import torch
    out: dict = {"peak_rss_mib": round(peak_rss_mib(), 1)}
    if device.startswith("cuda"):
        out["cuda_peak_alloc_mib"] = round(
            torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        out["cuda_peak_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 2 ** 20, 1)
    elif device.startswith("mps"):
        try:
            out["mps_current_mib"] = round(
                torch.mps.current_allocated_memory() / 2 ** 20, 1)
        except Exception:  # noqa: BLE001, S110 — memory probe is best-effort
            pass
    return out


def load_model(device="mps", model_id=None, revision=None):
    import torch
    import transformers

    from latent_lab.backends.gdn_patch import install
    from latent_lab.train.checkpointing import require_pinned_revision
    install()

    model_id = model_id or DEFAULT_MODEL_ID
    # Only revision=None selects the pinned default; falsey values such as
    # "", False or 0 are validated and rejected as-is, BEFORE any Hugging
    # Face contact: only immutable 40-hex commit revisions pass.
    revision = require_pinned_revision(
        DEFAULT_REVISION if revision is None else revision)
    tok = transformers.AutoTokenizer.from_pretrained(model_id,
                                                     revision=revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16).eval()
    model.to(device)
    return model, tok


class SuiteTensors:
    """Tokenized suite with cached candidate token ids."""

    def __init__(self, tok, examples):
        self.examples = examples
        self.prompt_ids = []
        self.answer_ids = []
        self.cand_ids = []
        for ex in examples:
            p = tok(ex.prompt, return_tensors="pt", return_dict=True
                    ).input_ids
            self.prompt_ids.append(p)
            a = tok(ANSWER_PREFIX + ex.answer, add_special_tokens=False,
                    return_tensors="pt", return_dict=True).input_ids
            self.answer_ids.append(a)
            cs = tuple(tok(ANSWER_PREFIX + c, add_special_tokens=False)
                       .input_ids for c in ex.candidates)
            self.cand_ids.append(cs)

    def __len__(self):
        return len(self.examples)


def derive_gold_index(candidates, answer, *, ex_id="") -> int:
    """Derive the UNIQUE gold candidate position from contents alone.

    Gold identity comes from the answer/candidate strings, never from a
    trusted supplied index: an answer absent from the candidate set
    (missing) or present more than once (ambiguous/duplicated) fails
    closed instead of scoring against a guessable position.
    """
    occurrences = [i for i, c in enumerate(candidates) if c == answer]
    if len(occurrences) != 1:
        state = "missing from" if not occurrences else "duplicated in"
        raise ValueError(
            f"{ex_id}: gold answer {answer!r} {state} candidates "
            f"{list(candidates)!r}; gold identity is ambiguous — "
            "refusing to score")
    return occurrences[0]


def build_eval_record(ex, order, scores) -> dict:
    """One lossless eval record: raw scores/order + gold/candidate identity.

    Derived fields (rank_of_gold/correct) are convenience only — the raw
    finite candidate scores in SCORED order, the model's ordering, the full
    candidate set, the answer and its candidate index are all retained so
    corrected scoring can be independently recomputed later.
    """
    if len(scores) != len(ex.candidates):
        raise ValueError(
            f"{ex.ex_id}: {len(scores)} scores for "
            f"{len(ex.candidates)} candidates")
    bad = [i for i, s in enumerate(scores)
           if s is None or not isinstance(s, (int, float))
           or s != s or s in (float("inf"), float("-inf"))]
    if bad:
        raise ValueError(f"{ex.ex_id}: non-finite raw scores at {bad}")
    gold_idx = derive_gold_index(ex.candidates, ex.answer, ex_id=ex.ex_id)
    pred_rank = order.index(gold_idx) if gold_idx in order else -1
    return {
        "ex_id": ex.ex_id, "family": ex.family, "depth": ex.depth,
        "candidates": list(ex.candidates), "answer": ex.answer,
        "gold_candidate_index": gold_idx,
        "scores_raw": [float(s) for s in scores],
        "score_order": list(order),
        "rank_of_gold": pred_rank,
        "correct": 1.0 if pred_rank == 0 else 0.0,
        "n_candidates": len(order),
    }


def rescore_records(records) -> float:
    """Independently recompute accuracy from RAW record fields alone.

    Gold identity is RE-DERIVED from answer/candidates and the persisted
    gold_candidate_index is accepted only when it names that same unique
    position: a missing, duplicated/ambiguous or substituted gold fails
    closed instead of being scored from a trusted supplied index. Also
    verifies that derived rank/correct agree with a fresh computation
    from scores_raw/score_order, so any future scorer fix can be
    re-applied to persisted evidence without re-running the model.
    """
    correct = 0
    for r in records:
        scores = r["scores_raw"]
        candidates = r["candidates"]
        if len(candidates) != len(scores):
            raise ValueError(
                f"{r.get('ex_id')}: candidates vs scores length mismatch")
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if list(order) != list(r["score_order"]):
            raise ValueError(
                f"{r.get('ex_id')}: score_order disagrees with scores_raw; "
                "evidence inconsistent")
        gold = derive_gold_index(candidates, r.get("answer"),
                                 ex_id=r.get("ex_id"))
        claimed = r.get("gold_candidate_index")
        if isinstance(claimed, bool) or not isinstance(claimed, int) \
                or claimed != gold:
            raise ValueError(
                f"{r.get('ex_id')}: gold_candidate_index {claimed!r} does "
                f"not identify the answer {r.get('answer')!r} in "
                f"candidates (unique position {gold}); missing/substituted"
                " gold identity rejected")
        rank = order.index(gold)
        if rank != r["rank_of_gold"] or \
                (1.0 if rank == 0 else 0.0) != r["correct"]:
            raise ValueError(
                f"{r.get('ex_id')}: derived fields disagree with raw "
                "scores; evidence inconsistent")
        correct += 1 if rank == 0 else 0
    return correct / max(1, len(records))


def evaluate(rec, data: SuiteTensors, k_steps, indices, *, ablate=None,
             tag="", limit=None):
    """Historical summed-score evaluator.

    This path is retained only to read/quarantine legacy artifacts.  New
    validation, checkpoint selection, and final evidence must use
    :func:`evaluate_v3`.
    """
    device = next(rec.model.parameters()).device
    t0 = time.perf_counter()
    records = []
    for i in (indices[:limit] if limit else indices):
        ex = data.examples[i]
        partner = None
        if ablate and ablate.get("swap_state"):
            j = (i + 1) % len(data.examples)
            partner = data.prompt_ids[j].to(device)
        order, scores, rep = rec.rank_candidates(
            data.prompt_ids[i].to(device), data.cand_ids[i], k_steps,
            ablate=ablate, partner_input_ids=partner)
        records.append(build_eval_record(ex, order, scores))
    acc = sum(r["correct"] for r in records) / max(1, len(records))
    by_depth = {}
    for r in records:
        by_depth.setdefault(r["depth"], []).append(r["correct"])
    by_depth = {d: round(sum(v) / len(v), 4) for d, v in sorted(by_depth.items())}
    by_family = {}
    for r in records:
        by_family.setdefault(r["family"], []).append(r["correct"])
    by_family = {f: round(sum(v) / len(v), 4)
                 for f, v in sorted(by_family.items())}
    return {
        "tag": tag, "ablate": ablate or {}, "k_steps": k_steps,
        "n": len(records), "accuracy": round(acc, 4),
        "by_depth": by_depth, "by_family": by_family,
        "seconds": round(time.perf_counter() - t0, 1),
        "records": records,
    }


_V3_EVIDENCE_IDENTITY_KEYS = frozenset((
    "run_id", "recipe_hash", "model_id", "model_revision", "adapter_id",
    "checkpoint_id", "checkpoint_content_hash", "suite_id",
    "suite_version", "suite_hash", "recurrence_config",
))


def _v3_identity(identity) -> dict:
    if not isinstance(identity, dict):
        raise ValueError("v3 evidence_identity must be an object")
    if set(identity) != set(_V3_EVIDENCE_IDENTITY_KEYS):
        missing = sorted(_V3_EVIDENCE_IDENTITY_KEYS - set(identity))
        extra = sorted(set(identity) - _V3_EVIDENCE_IDENTITY_KEYS)
        raise ValueError(
            f"v3 evidence_identity keys mismatch; missing={missing}, "
            f"extra={extra}")
    return dict(identity)


def build_v3_runtime_record(ex, *, split, k_steps, evidence_identity,
                            details, runtime_order=None,
                            runtime_raw_sums=None,
                            runtime_tie_error=None):
    """Convert one runtime raw-detail snapshot into canonical v3 evidence.

    The converter performs no model work.  It independently recomputes every
    score/ranking/verdict through :mod:`eval_v3` and requires the runtime's
    duplicated sums, normalized scores, primary definition, and ordering to
    agree exactly.  This is the shared validation/final/offline boundary.
    """
    from latent_lab.bench.eval_v3 import (
        PRIMARY_SCORE_DEFINITION, build_eval_record)

    ident = _v3_identity(evidence_identity)
    declared_split = getattr(ex, "split", split)
    if declared_split != split:
        raise ValueError(
            f"{ex.ex_id}: example belongs to {declared_split!r}, not "
            f"requested split {split!r}")
    if not isinstance(details, dict):
        raise ValueError("runtime score details must be an object")
    rows = details.get("candidate_token_logprobs")
    compute = details.get("compute")
    record = build_eval_record(
        run_id=ident["run_id"], recipe_hash=ident["recipe_hash"],
        model_id=ident["model_id"], model_revision=ident["model_revision"],
        adapter_id=ident["adapter_id"], checkpoint_id=ident["checkpoint_id"],
        checkpoint_content_hash=ident["checkpoint_content_hash"],
        suite_id=ident["suite_id"], suite_version=ident["suite_version"],
        suite_hash=ident["suite_hash"], example_id=ex.ex_id, split=split,
        family=ex.family, prompt=ex.prompt, candidates=ex.candidates,
        candidate_permutation_seed=ex.candidate_permutation_seed,
        candidate_permutation=ex.candidate_permutation,
        gold_answer=ex.answer, per_token_logprobs=rows, k=k_steps,
        recurrence_config=ident["recurrence_config"], compute=compute,
    )
    duplicates = (
        ("candidate_token_counts", record["token_counts"]),
        ("candidate_raw_sum_logprobs", record["raw_summed_logprobs"]),
        ("candidate_length_normalized_logprobs", record["normalized_scores"]),
    )
    for field, expected in duplicates:
        if field not in details or list(details[field]) != expected:
            raise ValueError(
                f"runtime {field} disagrees with canonical v3 raw rescore")
    if details.get("primary_score_definition") != PRIMARY_SCORE_DEFINITION:
        raise ValueError("runtime primary score definition is not canonical v3")
    expected_top_tie = (record["ties"][0]
                        if record["error_status"] == "AMBIGUOUS_TOP_TIE"
                        else [])
    claimed_top_tie = details.get("exact_top_tie_indices")
    if not isinstance(claimed_top_tie, list) \
            or sorted(claimed_top_tie) != sorted(expected_top_tie):
        raise ValueError(
            "runtime exact-top-tie claim disagrees with canonical v3: "
            f"claimed={claimed_top_tie!r} expected={expected_top_tie!r} "
            f"normalized_scores={record['normalized_scores']!r}")
    if runtime_tie_error is not None \
            and bool(runtime_tie_error) != bool(expected_top_tie):
        raise ValueError(
            "runtime tie control flow disagrees with canonical v3")
    primary_scores = details.get("primary_scores")
    if primary_scores is not None \
            and list(primary_scores) != record["normalized_scores"]:
        raise ValueError("runtime primary_scores disagree with canonical v3")
    if runtime_raw_sums is not None \
            and list(runtime_raw_sums) != record["raw_summed_logprobs"]:
        raise ValueError("runtime raw sums disagree with canonical v3")
    if runtime_order is not None:
        runtime_order = list(runtime_order)
        if sorted(runtime_order) != list(range(len(record["candidates"]))):
            raise ValueError("runtime ranking is not a candidate permutation")
        # Non-leading exact ties may be serialized differently by a runtime
        # that has token ids but not candidate strings.  They cannot change
        # the verdict.  A different unique leader is always a hard mismatch.
        if record["status"] == "ok" \
                and runtime_order[0] != record["ranking"][0]:
            raise ValueError("runtime winner disagrees with canonical v3")
        if not record["ties"] and runtime_order != record["ranking"]:
            raise ValueError("runtime ranking disagrees with canonical v3")
    return record


def evaluate_v3(rec, data: SuiteTensors, k_steps, indices, *,
                evidence_identity, split, ablate=None, tag="", limit=None):
    """Canonical online evaluation used by validation and final scoring."""
    from latent_lab.bench.eval_v3 import aggregate_records

    device = next(rec.model.parameters()).device
    t0 = time.perf_counter()
    records = []
    for i in (indices[:limit] if limit else indices):
        ex = data.examples[i]
        partner = None
        if ablate and ablate.get("swap_state"):
            j = (i + 1) % len(data.examples)
            partner = data.prompt_ids[j].to(device)
        try:
            order, raw_sums, report = rec.rank_candidates(
                data.prompt_ids[i].to(device), data.cand_ids[i], k_steps,
                ablate=ablate, partner_input_ids=partner)
            details = report.extra
            tie_error = False
        except Exception as exc:  # exact runtime tie is evidence, not a crash
            if type(exc).__name__ != "AmbiguousTopTie" \
                    or not isinstance(getattr(exc, "details", None), dict):
                raise
            order = None
            raw_sums = getattr(exc, "raw_sums", None)
            details = exc.details
            tie_error = True
        records.append(build_v3_runtime_record(
            ex, split=split, k_steps=k_steps,
            evidence_identity=evidence_identity, details=details,
            runtime_order=order, runtime_raw_sums=raw_sums,
            runtime_tie_error=tie_error))
    metrics = aggregate_records(records)
    return {
        "schema_version": metrics["schema_version"],
        "tag": tag,
        "ablate": ablate or {},
        "k_steps": k_steps,
        "n": len(records),
        "metrics": metrics,
        "seconds": time.perf_counter() - t0,
        "records": records,
    }


def select_v3_checkpoint(history):
    """Canonical checkpoint selection; legacy accuracy entries are rejected."""
    from latent_lab.bench.eval_v3 import select_best_checkpoint
    return select_best_checkpoint(history)


def rescore_v3_records(records):
    """Canonical offline/report rescore from persisted raw fields only."""
    from latent_lab.bench.eval_v3 import aggregate_records
    return aggregate_records(records)


def canonical_v3_history_entry(step, records):
    """Persist raw validation records beside their recomputed summary."""
    from latent_lab.bench.eval_v3 import checkpoint_history_entry

    retained = list(records)
    entry = checkpoint_history_entry(step, retained)
    entry["records"] = retained
    return entry


def select_v3_checkpoint_from_raw_history(history, *, expected_identity=None):
    """Select only after independently rebuilding every v3 summary.

    A summary-only or tampered validation history cannot select a checkpoint.
    The same canonical selector is then used by training and the no-spend
    gate.
    """
    from latent_lab.bench.eval_v3 import (
        EvalV3Error,
        aggregate_records,
        validate_record_against_current_suite,
    )

    canonical = []
    for index, entry in enumerate(history or ()):
        if not isinstance(entry, dict):
            raise ValueError(f"validation history[{index}] is not an object")
        records = entry.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(
                f"validation history[{index}] has no raw v3 records")
        bound_records = []
        for record_index, record in enumerate(records):
            try:
                bound_records.append(validate_record_against_current_suite(
                    record, expected_split="validation"))
            except EvalV3Error as exc:
                raise ValueError(
                    f"validation history[{index}] record[{record_index}] "
                    f"is not canonical behavioral-v3 validation evidence: "
                    f"{exc}") from exc
        recomputed = aggregate_records(bound_records)
        if expected_identity is not None:
            if not isinstance(expected_identity, dict):
                raise ValueError("expected validation identity must be an object")
            record = bound_records[0]
            actual_identity = {
                "run_id": record["run_id"],
                "recipe_hash": record["recipe_hash"],
                "model_id": record["model_identity"]["model_id"],
                "model_revision": record["model_identity"]["revision"],
                "adapter_id": record["checkpoint_identity"]["adapter_id"],
                "suite_id": record["suite_identity"]["suite_id"],
                "suite_version": record["suite_identity"]["version"],
                "suite_hash": record["suite_identity"]["sha256"],
                "split": record["split"],
                "k": record["k"],
                "recurrence_config": record["recurrence_config"],
            }
            unknown = sorted(set(expected_identity) - set(actual_identity))
            if unknown:
                raise ValueError(
                    f"unknown expected validation identity keys {unknown}")
            for field, expected in expected_identity.items():
                if actual_identity[field] != expected:
                    raise ValueError(
                        f"validation history[{index}] {field} disagrees "
                        "with selected adapter identity")
        if entry.get("metrics") != recomputed:
            raise ValueError(
                f"validation history[{index}] summary disagrees with raw "
                "latent_eval.v3 records")
        canonical.append({"step": entry.get("step"), "metrics": recomputed})
    return select_v3_checkpoint(canonical)


def selected_v3_adapter_state_sha256(history, *, expected_step=None,
                                      expected_metric=None) -> str:
    """Return the state hash bound by the canonically selected raw records.

    Every record at the selected validation step must name the same adapter
    state.  This is deliberately distinct from the final bundle content
    digest, which also covers bundle metadata and metrics.
    """
    selected = select_v3_checkpoint_from_raw_history(history)
    if selected is None:
        raise ValueError("raw v3 history selects no checkpoint")
    if expected_step is not None and selected.step != expected_step:
        raise ValueError(
            "raw v3 history selected step disagrees with reported best_step")
    if expected_metric is not None and selected.metric != expected_metric:
        raise ValueError(
            "raw v3 history selected metric disagrees with reported best metric")
    entries = [entry for entry in history if entry.get("step") == selected.step]
    if len(entries) != 1:
        raise ValueError(
            "raw v3 history must contain exactly one selected-step entry")
    hashes = {
        record["checkpoint_identity"]["content_sha256"]
        for record in entries[0]["records"]
    }
    if len(hashes) != 1:
        raise ValueError(
            "selected-step raw records bind multiple adapter state hashes")
    return next(iter(hashes))


def _suite_v3_contract(suite) -> dict:
    manifest = suite.manifest()
    required = ("suite_identity", "suite_version", "suite_hash")
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(
            f"behavioral-v3 manifest missing identity fields {missing}")
    if manifest["suite_identity"] != "behavioral-v3" \
            or manifest["suite_version"] != 3:
        raise ValueError(
            f"new evidence requires behavioral-v3, got "
            f"{manifest['suite_identity']!r} v{manifest['suite_version']!r}")
    if suite.records_hash() != manifest["suite_hash"]:
        raise ValueError("behavioral-v3 manifest hash disagrees with records")
    return manifest


def _recipe_hash(recipe) -> str:
    from latent_lab.bench.eval_v3 import canonical_sha256
    return canonical_sha256(recipe)


def _adapter_policy_mode(mode: str, recurrence_only_lora: bool) -> str:
    if not isinstance(recurrence_only_lora, bool):
        raise ValueError("recurrence_only_lora must be a boolean")
    has_suffix = mode.endswith(RECURRENCE_ONLY_LORA_MODE_SUFFIX)
    if recurrence_only_lora:
        return mode if has_suffix else mode + RECURRENCE_ONLY_LORA_MODE_SUFFIX
    if has_suffix:
        raise ValueError(
            "mode claims recurrence-only LoRA but recurrence_only_lora is false")
    return mode


def _training_objective_mode(mode: str, training_objective: str) -> str:
    """Seal the objective before any recurrence-only activation suffix."""
    if training_objective not in TRAINING_OBJECTIVES:
        raise ValueError(
            f"training_objective must be one of {TRAINING_OBJECTIVES}")
    recurrence_suffix = ""
    if mode.endswith(RECURRENCE_ONLY_LORA_MODE_SUFFIX):
        mode = mode.removesuffix(RECURRENCE_ONLY_LORA_MODE_SUFFIX)
        recurrence_suffix = RECURRENCE_ONLY_LORA_MODE_SUFFIX
    suffix_count = mode.count(CANDIDATE_CE_MODE_SUFFIX)
    if suffix_count > 1 or (suffix_count == 1 and
                            not mode.endswith(CANDIDATE_CE_MODE_SUFFIX)):
        raise ValueError("mode carries a malformed candidate-CE suffix")
    if training_objective == "candidate_ce":
        if suffix_count == 0:
            mode += CANDIDATE_CE_MODE_SUFFIX
    elif suffix_count:
        raise ValueError(
            "mode claims candidate CE but training_objective is gold_nll")
    return mode + recurrence_suffix


def _neutral_delta_mode(mode: str, neutral_delta: bool) -> str:
    """Seal neutral-delta before objective and adapter-policy suffixes."""
    if not isinstance(neutral_delta, bool):
        raise ValueError("neutral_delta must be a boolean")
    recurrence_count = mode.count(RECURRENCE_ONLY_LORA_MODE_SUFFIX)
    if recurrence_count > 1 or (recurrence_count == 1 and not mode.endswith(
            RECURRENCE_ONLY_LORA_MODE_SUFFIX)):
        raise ValueError("mode carries a malformed recurrence-only suffix")
    recurrence_suffix = (RECURRENCE_ONLY_LORA_MODE_SUFFIX
                         if recurrence_count else "")
    core = mode.removesuffix(recurrence_suffix)
    objective_count = core.count(CANDIDATE_CE_MODE_SUFFIX)
    if objective_count > 1 or (objective_count == 1 and not core.endswith(
            CANDIDATE_CE_MODE_SUFFIX)):
        raise ValueError("mode carries a malformed candidate-CE suffix")
    objective_suffix = CANDIDATE_CE_MODE_SUFFIX if objective_count else ""
    core = core.removesuffix(objective_suffix)
    count = core.count(NEUTRAL_DELTA_MODE_SUFFIX)
    if count > 1 or (count == 1 and not core.endswith(
            NEUTRAL_DELTA_MODE_SUFFIX)):
        raise ValueError("mode carries a malformed neutral-delta suffix")
    if neutral_delta:
        base = core.removesuffix(NEUTRAL_DELTA_MODE_SUFFIX)
        if base != "D-full":
            raise ValueError("neutral_delta requires D-full mode")
        if not count:
            core += NEUTRAL_DELTA_MODE_SUFFIX
        return core + objective_suffix + recurrence_suffix
    if count:
        raise ValueError("mode claims neutral_delta but neutral_delta is false")
    return core + objective_suffix + recurrence_suffix


def _recurrence_only_lora_from_config(cfg: dict) -> bool:
    """Recover and cross-check the policy sealed into the recipe's mode."""
    value = cfg.get("recurrence_only_lora", False)
    if not isinstance(value, bool):
        raise ValueError("config.recurrence_only_lora must be a boolean")
    mode = cfg.get("mode")
    if not isinstance(mode, str):
        raise ValueError("config.mode must be a string")
    expected_mode = _adapter_policy_mode(
        mode.removesuffix(RECURRENCE_ONLY_LORA_MODE_SUFFIX), value)
    if mode != expected_mode:
        raise ValueError(
            "config mode and recurrence_only_lora policy disagree; refusing "
            "an unsealed adapter activation policy")

    contract = cfg.get("runtime_contract")
    if value and not isinstance(contract, dict):
        raise ValueError(
            "recurrence-only LoRA config lacks sealed runtime_contract metadata")
    if contract is not None:
        if not isinstance(contract, dict):
            raise ValueError("config.runtime_contract must be an object")
        expected = {
            "adapter_activation_policy": (
                "recurrence_only" if value else "all_stages"),
            "prefill_adapter_active": not value,
            "recurrence_adapter_active": True,
            "candidate_adapter_active": not value,
        }
        observed = {key: contract.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                "config.runtime_contract disagrees with adapter activation "
                f"policy: expected {expected}, got {observed}")
    return value


def _neutral_delta_from_config(cfg: dict) -> bool:
    """Recover neutral-delta and reject suffix/order/config disagreement."""
    value = cfg.get("neutral_delta", False)
    if not isinstance(value, bool):
        raise ValueError("config.neutral_delta must be a boolean")
    mode = cfg.get("mode")
    if not isinstance(mode, str):
        raise ValueError("config.mode must be a string")
    core = mode.removesuffix(RECURRENCE_ONLY_LORA_MODE_SUFFIX)
    if CANDIDATE_CE_MODE_SUFFIX in core \
            and not core.endswith(CANDIDATE_CE_MODE_SUFFIX):
        raise ValueError(
            "config.mode carries malformed neutral-delta suffix ordering")
    core = core.removesuffix(CANDIDATE_CE_MODE_SUFFIX)
    count = mode.count(NEUTRAL_DELTA_MODE_SUFFIX)
    sealed = count == 1 and core.endswith(NEUTRAL_DELTA_MODE_SUFFIX)
    if count > 1 or (count == 1 and not sealed):
        raise ValueError("config.mode carries a malformed neutral-delta suffix")
    if value != sealed:
        raise ValueError(
            "config mode and neutral_delta disagree; refusing an unsealed "
            "neutral recurrence architecture")
    if value:
        if cfg.get("recurrence_only_lora") is not True:
            raise ValueError(
                "neutral_delta requires recurrence_only_lora=true")
        base = core.removesuffix(NEUTRAL_DELTA_MODE_SUFFIX)
        if base != "D-full":
            raise ValueError("neutral_delta requires D-full mode")
        interval = cfg.get("interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2 \
                or interval[0] != 0:
            raise ValueError(
                "neutral_delta requires a full decoder interval beginning at 0")
    return value


def _recurrence_config(cfg: dict) -> dict:
    """Measured recurrence settings; unsupported flags cannot enter v3."""
    if cfg.get("grad_checkpoint") not in (None, False):
        raise ValueError("grad_checkpoint=true is unsupported")
    training_objective_from_config(cfg)
    recurrence_only_lora = _recurrence_only_lora_from_config(cfg)
    neutral_delta = _neutral_delta_from_config(cfg)
    return {
        "mode": cfg["mode"],
        "interval": list(cfg["interval"]),
        "trained_k": cfg["k"],
        "max_k": cfg["max_k"],
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "detach_z0": cfg["detach_z0"],
        "recurrence_only_lora": recurrence_only_lora,
        "neutral_delta": neutral_delta,
        "adapter_activation_policy": (
            "recurrence_only" if recurrence_only_lora else "all_stages"),
    }


def _training_run_id(out: Path, recipe_hash: str, seed: int, *,
                     model_id: str, revision: str) -> str:
    from latent_lab.bench.eval_v3 import canonical_sha256
    digest = canonical_sha256({
        "output_path": str(out.resolve()),
        "recipe_hash": recipe_hash,
        "seed": seed,
        "model_id": model_id,
        "model_revision": revision,
    })
    return f"latent-train-{digest[:24]}"


def _v3_evidence_identity(*, run_id, recipe, model_id, revision,
                          adapter_id, checkpoint_id,
                          checkpoint_content_hash, suite_manifest,
                          recurrence_config) -> dict:
    return _v3_identity({
        "run_id": run_id,
        "recipe_hash": _recipe_hash(recipe),
        "model_id": model_id,
        "model_revision": revision,
        "adapter_id": adapter_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_content_hash": checkpoint_content_hash,
        "suite_id": suite_manifest["suite_identity"],
        "suite_version": suite_manifest["suite_version"],
        "suite_hash": suite_manifest["suite_hash"],
        "recurrence_config": recurrence_config,
    })


def build_v3_eval_payload(*, adapter, split, config, model_id, revision,
                          suite_hash, tokenizer_class, interval, k_steps,
                          ablation, seed, checkpoint_content_digest,
                          result, device):
    """Canonical persisted eval envelope, validated from raw v3 records."""
    metrics = rescore_v3_records(result.get("records", ()))
    if result.get("metrics") != metrics:
        raise ValueError("eval result summary disagrees with raw v3 records")
    ablation_name = ablation or CLEAN_ABLATION
    return {
        "status": "complete",
        "adapter": adapter,
        "split": split,
        "config": config,
        "model": model_id,
        "revision": revision,
        "identity": {
            "model_id": model_id,
            "revision": revision,
            "suite_sha256": suite_hash,
            "tokenizer_class": tokenizer_class,
            "interval": list(interval),
            "max_k": config["max_k"],
            "k_steps": k_steps,
            "split": split,
            "seed": seed,
            "ablation": ablation_name,
            "checkpoint_content_digest": checkpoint_content_digest,
        },
        "suite_sha256": suite_hash,
        "device": device,
        "seed": seed,
        "results": {ablation_name: result},
        "peak_rss_mib": round(peak_rss_mib(), 1),
        "platform": platform.platform(),
    }


def _dependency_versions() -> dict:
    import torch

    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    return out


def recipe_from_config(cfg: dict, suite_sha256: str) -> dict:
    """Canonical exact-identity recipe implied by cfg (re-exported)."""
    from latent_lab.train.checkpointing import recipe_from_config
    return recipe_from_config(cfg, suite_sha256)


def mode_from_spec(interval_spec: str, k: int,
                   recurrence_only_lora: bool = False,
                   training_objective: str = "gold_nll",
                   neutral_delta: bool = False) -> str:
    """The preregistered mode implied by the interval spec and K."""
    base = ("D-full" if interval_spec == "full"
            else "F-control" if k == 0 else "E-localized")
    if neutral_delta and (interval_spec != "full" or
                          not recurrence_only_lora):
        raise ValueError(
            "neutral_delta requires interval='full' and recurrence-only LoRA")
    base = _neutral_delta_mode(base, neutral_delta)
    objective_mode = _training_objective_mode(base, training_objective)
    return _adapter_policy_mode(objective_mode, recurrence_only_lora)


def train_recipe_digest(*, mode, interval, k, max_k, lora_r, lora_alpha,
                         lr, steps, seed, optimizer, weight_decay,
                         lr_schedule, warmup, clip, detach_z0,
                         suite_sha256, grad_checkpoint=None,
                         recurrence_only_lora=False,
                         training_objective="gold_nll",
                         neutral_delta=False) -> str:
    """ONE canonical digest binding EVERY behavior-changing field.

    The single source of truth shared by the trainer and the paid
    driver's preregistration: any change to mode/interval/K/max_k/LoRA/
    LR/steps/seed/optimizer/weight-decay/schedule/warmup/clip/detach/
    suite produces a materially different config_sha256.  The removed
    ``grad_checkpoint`` keyword is accepted only as a false migration input;
    checkpointing.recipe_from_config rejects true and never persists it.
    """
    if neutral_delta and not recurrence_only_lora:
        raise ValueError("neutral_delta requires recurrence_only_lora=true")
    cfg = {
        "mode": _adapter_policy_mode(
            _training_objective_mode(
                _neutral_delta_mode(mode, neutral_delta), training_objective),
            recurrence_only_lora),
        "interval": list(interval), "k": k, "max_k": max_k,
        "lora_r": lora_r, "lora_alpha": lora_alpha,
        "lr": lr, "steps": steps, "seed": seed,
        "optimizer": optimizer, "weight_decay": weight_decay,
        "lr_schedule": lr_schedule, "warmup": warmup, "clip": clip,
        "detach_z0": detach_z0,
    }
    if grad_checkpoint is not None:
        cfg["grad_checkpoint"] = grad_checkpoint
    return recipe_from_config(cfg, suite_sha256)["config_sha256"]


def _training_loss_on_example(rec, train, index: int, *, device: str,
                              k_steps: int, training_objective: str,
                              detach_z0: bool):
    """Dispatch one example without trusting any supplied gold index."""
    prompt_ids = train.prompt_ids[index].to(device)
    if training_objective == "candidate_ce":
        ex = train.examples[index]
        gold_index = derive_gold_index(
            ex.candidates, ex.answer, ex_id=ex.ex_id)
        return rec.candidate_ce_loss_on_example(
            prompt_ids, train.cand_ids[index], gold_index, k_steps,
            detach_z0=detach_z0)
    if training_objective != "gold_nll":
        raise ValueError(
            f"training_objective must be one of {TRAINING_OBJECTIVES}")
    return rec.loss_on_example(
        prompt_ids, train.answer_ids[index].to(device), k_steps,
        detach_z0=detach_z0)


def _mark_run_fatal(out: Path, e: BaseException) -> None:
    """Atomically mark the run fatal; no success artifact may follow."""
    from latent_lab.train.checkpointing import write_run_status
    write_run_status(out, "fatal",
                     command=" ".join(sys.argv),
                     error_type=type(e).__name__, error=str(e))


def _quarantine_evidence_root(out: Path, e: BaseException) -> None:
    """Guarantee no complete-looking generation stays in the ACTIVE root.

    Step 0 ATOMICALLY poisons the ACTIVE completion mark IN PLACE before
    any best-effort artifact handling: from that moment the root can
    never validate as a complete generation (verify_generation /
    artifacts.validate_run reject non-complete status) even if every
    later cleanup/publication step also fails. Only then are fixed-name
    success artifacts quarantined; if ANY of those operations fails, the
    ENTIRE root is moved aside atomically so no validator can ever see
    it as active evidence again, and a fresh empty root is created for
    fatal-status publication.
    """
    from latent_lab.train.checkpointing import (
        quarantine_success_artifacts, write_run_status)
    try:
        write_run_status(out, "fatal",
                         command=" ".join(sys.argv),
                         error_type=type(e).__name__, error=str(e))
    except Exception:  # noqa: BLE001 - best-effort; the ladder below holds
        pass
    try:
        quarantine_success_artifacts(out)
    except Exception:
        target = out.with_name(f"{out.name}.invalid.{time.time_ns()}")
        os.replace(out, target)
        out.mkdir(parents=True, exist_ok=True)


def cmd_train(args):
    from latent_lab.train.checkpointing import (
        EvidenceLifecycleError, require_pinned_revision, write_run_status)

    # fail closed BEFORE loading/training/saving on a mutable revision
    revision = require_pinned_revision(args.revision)
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_run_status(out, "running",
                     command=" ".join(sys.argv),
                     model=args.model, revision=revision, seed=args.seed)
    try:
        _train_inner(args, out, device, revision)
        # The terminal complete-status transition is INSIDE the protected
        # success-publication lifecycle: a failure writing it must never
        # escape unhandled and leave status 'running' beside finished
        # evidence.
        write_run_status(out, "complete", command=" ".join(sys.argv))
    except BaseException as e:
        # Fail-stop publication hygiene. FIRST the active completion
        # mark is atomically poisoned in place (so the root can never
        # validate as complete even under total cleanup failure), THEN
        # every fixed-name success artifact is invalidated/quarantined
        # (escalating to whole-root quarantine if any removal fails),
        # and only then is fatal status published. If any step fails,
        # fail closed — raising an explicit lifecycle error that
        # preserves the ORIGINAL exception as cause (never replaced by
        # the secondary cleanup/reporting error).
        try:
            _quarantine_evidence_root(out, e)
        except BaseException as cleanup_err:
            raise EvidenceLifecycleError(
                f"run failed ({type(e).__name__}: {e}) AND success-artifact "
                f"quarantine failed ({type(cleanup_err).__name__}: "
                f"{cleanup_err}); failing closed — the evidence root at "
                f"{out} is untrusted and carries no complete generation"
            ) from e
        try:
            _mark_run_fatal(out, e)
        except BaseException as fatal_err:
            raise EvidenceLifecycleError(
                f"run failed ({type(e).__name__}: {e}) AND fatal-status "
                f"publication failed ({type(fatal_err).__name__}: "
                f"{fatal_err}); failing closed — the evidence root at "
                f"{out} carries no complete generation") from e
        raise


def _train_inner(args, out: Path, device: str, revision: str):
    import random

    import torch

    from latent_lab.backends.localized import LocalizedRecurrence
    from latent_lab.bench.suite_v3 import build_suite
    from latent_lab.train.checkpointing import (
        BestCheckpointTracker, adapter_state_sha256,
        guarded_optimizer_step, sha256_file, write_train_generation)

    # Seed before every stochastic construction: model load, LoRA init,
    # optimizer creation, and the torch.randperm training shuffle.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    neutral_delta = getattr(args, "neutral_delta", False)
    if not isinstance(neutral_delta, bool):
        raise ValueError("--neutral-delta must be a boolean flag")
    if neutral_delta and args.interval != "full":
        raise ValueError("--neutral-delta requires --interval full")
    # Neutral-delta is defined with adapters active only in its proposal loop.
    recurrence_only_lora = (
        getattr(args, "recurrence_only_lora", False) or neutral_delta)
    training_objective = getattr(args, "training_objective", "gold_nll")
    _training_objective_mode("D-full", training_objective)
    if recurrence_only_lora and args.k == 0:
        raise ValueError(
            "--recurrence-only-lora requires --k > 0 for training; "
            "evaluate K=0 from the trained K>0 adapter")

    model, tok = load_model(device, args.model, revision)
    interval = interval_from_spec(
        args.interval, model.config.num_hidden_layers)
    rec = LocalizedRecurrence(model, None, interval=interval, max_k=args.max_k,
                              lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                              recurrence_only_lora=recurrence_only_lora,
                              neutral_delta=neutral_delta)
    rec.clock.to(device)
    params = [p for p in rec.trainable_parameters()]
    if args.optimizer != "adamw":
        raise ValueError(f"unsupported optimizer {args.optimizer!r}; "
                         "supported: ['adamw']")
    opt = torch.optim.AdamW(params, lr=args.lr,
                            weight_decay=args.weight_decay)

    suite = build_suite()
    suite_manifest = _suite_v3_contract(suite)
    suite_sha = suite_manifest["suite_hash"]
    mode = mode_from_spec(
        args.interval, args.k, recurrence_only_lora=recurrence_only_lora,
        training_objective=training_objective, neutral_delta=neutral_delta)
    cfg = {
        "mode": mode,
        "model": args.model, "revision": revision,
        "interval": list(interval), "k": args.k,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lr": args.lr, "steps": args.steps,
        "seed": args.seed, "max_k": args.max_k,
        "optimizer": args.optimizer, "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule, "warmup": args.warmup,
        "clip": args.clip, "detach_z0": args.detach_z0,
        "training_objective": training_objective,
        "recurrence_only_lora": recurrence_only_lora,
        "neutral_delta": neutral_delta,
        "runtime_contract": rec.runtime_contract(),
        "device": device,
        "label": getattr(args, "label", None),
        "suite_sha256": suite_sha,
    }
    recipe = recipe_from_config(cfg, suite_sha)
    run_id = _training_run_id(
        out, _recipe_hash(recipe), args.seed,
        model_id=args.model, revision=revision)
    recurrence_config = _recurrence_config(cfg)
    train = SuiteTensors(tok, list(suite.train))
    val = SuiteTensors(tok, list(suite.validation))
    cfg["train_examples"] = len(train)
    val_idx = list(range(len(val.examples)))
    if args.val_examples is not None and args.val_examples < 1:
        raise ValueError("--val-examples must be >=1 or omitted for full validation")
    n_val_eval = (len(val_idx) if args.val_examples is None
                  else min(len(val_idx), args.val_examples))

    order = list(range(len(train)))
    history = []
    tracker = BestCheckpointTracker()
    t0 = time.perf_counter()
    base_lr = args.lr

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return base_lr * (step + 1) / args.warmup
        if args.lr_schedule == "cosine":
            import math
            t = (step - args.warmup) / max(1, args.steps - args.warmup)
            return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
        return base_lr

    perm = None
    final_loss = float("nan")
    for step in range(args.steps):
        if step % len(order) == 0 or perm is None:
            perm = torch.randperm(len(order)).tolist()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        i = perm[step % len(order)]
        loss = _training_loss_on_example(
            rec, train, i, device=device, k_steps=args.k,
            training_objective=training_objective,
            detach_z0=args.detach_z0)
        opt.zero_grad()
        loss.backward()
        # fail-stop: any fault here raises FatalRunInvalidError and kills
        # the run — never retried against the mutated optimizer
        guarded_optimizer_step(opt, loss.detach(), params, args.clip)
        final_loss = float(loss.detach())
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step} loss {final_loss:.4f} lr {lr_at(step):.2e} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            checkpoint_step = step + 1
            adapter_state = rec.adapter_state_dict()
            checkpoint_hash = adapter_state_sha256(adapter_state)
            evidence_identity = _v3_evidence_identity(
                run_id=run_id, recipe=recipe, model_id=args.model,
                revision=revision, adapter_id=run_id,
                checkpoint_id=f"step-{checkpoint_step}",
                checkpoint_content_hash=checkpoint_hash,
                suite_manifest=suite_manifest,
                recurrence_config=recurrence_config)
            ev = evaluate_v3(
                rec, val, args.k, val_idx,
                evidence_identity=evidence_identity, split="validation",
                tag=f"{mode}|validation|clean|K={args.k}",
                limit=n_val_eval)
            history.append(canonical_v3_history_entry(
                checkpoint_step, ev["records"]))
            selected = select_v3_checkpoint_from_raw_history(
                history,
                expected_identity={
                    "run_id": run_id,
                    "recipe_hash": _recipe_hash(recipe),
                    "model_id": args.model,
                    "model_revision": revision,
                    "adapter_id": run_id,
                    "suite_id": suite_manifest["suite_identity"],
                    "suite_version": suite_manifest["suite_version"],
                    "suite_hash": suite_sha,
                    "split": "validation",
                    "k": args.k,
                    "recurrence_config": recurrence_config,
                })
            if selected is None:
                raise ValueError(
                    "canonical v3 selector found no valid checkpoint")
            metric = ev["metrics"]["micro_accuracy"]
            print(f"  val micro_accuracy {metric} @step {checkpoint_step}",
                  flush=True)
            if selected.step == checkpoint_step and tracker.update(
                    selected.metric, adapter_state, step=checkpoint_step):
                print(f"  new best {selected.metric} @step {checkpoint_step}",
                      flush=True)

    if not tracker.has_best():
        from latent_lab.train.checkpointing import EmptyCheckpointError
        raise EmptyCheckpointError(
            "no finite validation checkpoint was accepted; refusing to "
            "report or save final state")

    # Reload the SELECTED BEST before reporting/saving — final state is
    # never silently used as evidence.  Re-hash the reloaded state and bind
    # it to the selected step's raw validation records before publication.
    rec.load_adapter_state(tracker.best_state())
    selected_raw_state_sha256 = selected_v3_adapter_state_sha256(
        history, expected_step=tracker.best_step,
        expected_metric=tracker.best_score)
    selected_adapter_state_sha256 = adapter_state_sha256(
        rec.adapter_state_dict())
    if selected_adapter_state_sha256 != selected_raw_state_sha256:
        raise ValueError(
            "reloaded best adapter state disagrees with selected-step raw "
            "validation evidence")
    bundle_path = out / "best_params.pt"
    bundle = rec.export_adapter_bundle(
        bundle_path, model_id=args.model, revision=revision,
        config=cfg,
        metrics={"best_val_acc": tracker.best_score})
    report = {
        "run_id": run_id,
        "adapter_id": run_id,
        "config": cfg,
        "model": args.model, "revision": revision,
        "suite_sha256": suite_sha,
        "suite_identity": suite_manifest["suite_identity"],
        "suite_version": suite_manifest["suite_version"],
        "recipe": recipe,
        "selection_provenance":
            "latent_eval.v3_recomputed_from_raw_validation_records",
        "best_val_metric_definition": "micro_accuracy",
        "selected_adapter_state_sha256": selected_adapter_state_sha256,
        "checkpoint_content_digest": bundle["content_digest"],
        "checkpoint_sha256": sha256_file(bundle_path),
        "precision": {
            "backbone_dtype": str(next(model.parameters()).dtype),
            "trainables_dtype": "torch.float32",
        },
        "trainable_precision": "fp32",
        "best_val_acc": tracker.best_score,
        "best_step": tracker.best_step,
        "val_history": history,
        "final_train_loss": final_loss,
        "gpu_mem": _gpu_mem_report(device),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "peak_rss_mib": round(peak_rss_mib(), 1),
        "platform": platform.platform(),
    }
    # report + manifest promoted as ONE coherent generation; the manifest
    # (written last, atomically) carries the digests of both files and is
    # the commit marker any reader must verify
    manifest = {
        "kind": "latent_lab.train_generation", "status": "complete",
        "argv": list(sys.argv),
        "command": " ".join(sys.argv),
        "dependencies": _dependency_versions(),
        "precision": report["precision"],
        "seed": args.seed,
        "label": getattr(args, "label", None),
        "identity": {"model_id": args.model, "revision": revision},
        "run_id": run_id,
        "recipe": recipe,
        "suite_identity": suite_manifest["suite_identity"],
        "suite_version": suite_manifest["suite_version"],
        "suite_sha256": suite_sha,
        "selected_adapter_state_sha256": selected_adapter_state_sha256,
        "checkpoint_content_digest": bundle["content_digest"],
        "checkpoint_sha256": sha256_file(bundle_path),
        "wall_seconds": report["wall_seconds"],
    }
    write_train_generation(out, manifest=manifest, report=report)
    print(f"[train done] best_val={tracker.best_score} "
          f"@step {tracker.best_step} -> {out}")


CLEAN_ABLATION = "clean"

# The canonical no-ablation marker persisted in eval payloads.
_CLEAN_SPECS = (None, CLEAN_ABLATION)


def normalize_ablation(name, k_steps=None):
    """ONE shared whitelist/normalizer for ablation names.

    Used verbatim by the CLI parser (parse_ablation_cli) and by the
    artifact validator (validate_eval) so an unknown mode such as
    'bogus' is rejected identically everywhere — never silently run
    clean or accepted as evidence. Returns the ablate spec dict, or
    None for the canonical clean run (name None or 'clean').
    """
    if name in _CLEAN_SPECS:
        return None
    if not isinstance(name, str):
        raise ValueError(f"unknown ablation {name!r}; supported: "
                         f"{sorted(EVAL_ABLATIONS)} "
                         "or 'shuffle_clocks:i,j,...'")
    if name == "clocks_off":
        return {"clocks": "off"}
    if name == "reverse_clocks":
        return {"clocks": "reverse"}
    if name.startswith("shuffle_clocks:"):
        perm = name.split(":", 1)[1]
        return {"clocks": f"shuffle_perm:{perm}"}  # validated downstream
    if name == "truncate_half":
        return {"truncate_k": max(0, k_steps // 2)}
    if name == "readout_reset_to_z0":
        return {"reset_state": True}
    if name == "cache_reset_to_prompt":
        return {"reset_cache": True}
    if name == "full_state_reset_to_z0":
        return {"reset_state": True, "reset_cache": True}
    if name in ("zero_state", "bypass_interval", "swap_state", "noise_state"):
        return {name: True}
    raise ValueError(
        f"unknown ablation {name!r}; supported: {sorted(EVAL_ABLATIONS)} "
        "or 'shuffle_clocks:i,j,...'")


def parse_ablation_cli(name, k_steps):
    """Strict CLI ablation parser: unknown modes are REJECTED, never run
    silently clean. Returns the latent_steps ablation dict."""
    return normalize_ablation(name, k_steps)


def cmd_eval(args):
    import torch

    from latent_lab.backends.localized import LocalizedRecurrence
    from latent_lab.bench.suite_v3 import build_suite
    from latent_lab.train.checkpointing import (
        AdapterBundleError,
        AdapterBundleIdentityError,
        adapter_state_sha256,
        atomic_write_json,
        load_adapter_bundle,
        recipe_from_config,
        require_pinned_revision,
        validate_selected_adapter_state_binding,
        verify_generation,
    )

    torch.manual_seed(args.seed)
    device = args.device
    from latent_lab.train.checkpointing import strict_json_loads
    report = strict_json_loads(
        (Path(args.adapter) / "train_report.json").read_text())
    cfg = report["config"]
    training_objective_from_config(cfg)
    recurrence_only_lora = _recurrence_only_lora_from_config(cfg)
    neutral_delta = _neutral_delta_from_config(cfg)
    model_id = cfg.get("model")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(
            f"adapter {args.adapter} carries no immutable model identity; "
            "refusing to evaluate")
    try:
        # fail closed BEFORE any model load on a mutable/missing revision
        revision = require_pinned_revision(cfg.get("revision"))
    except AdapterBundleError as e:
        raise ValueError(
            f"adapter {args.adapter} carries no immutable pinned model "
            "revision; refusing to evaluate") from e

    suite = build_suite()
    suite_manifest = _suite_v3_contract(suite)
    suite_sha = suite_manifest["suite_hash"]

    # Identity-validate + digest-verify the on-disk generation and bundle
    # BEFORE any model/tokenizer load: a tampered adapter must never
    # trigger an arbitrary model fetch prior to rejection.
    manifest = verify_generation(args.adapter)
    report_recipe = report.get("recipe")
    recipe = recipe_from_config(cfg, suite_sha)
    if report_recipe != recipe:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: train_report recipe {report_recipe} "
            f"is not the canonical recipe of its own config "
            f"(config_sha256 {recipe['config_sha256']}); refusing to "
            "evaluate")
    if manifest.get("recipe") != recipe:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: run_manifest recipe "
            f"{manifest.get('recipe')} disagrees with the canonical "
            f"recipe (config_sha256 {recipe['config_sha256']}); refusing "
            "to evaluate")
    if manifest.get("suite_sha256") != suite_sha:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: manifest suite_sha256 "
            f"{manifest.get('suite_sha256')!r} != current suite "
            f"{suite_sha!r}; refusing to evaluate")
    if report.get("suite_identity") != suite_manifest["suite_identity"] \
            or report.get("suite_version") != suite_manifest["suite_version"]:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: report is not bound to behavioral-v3; "
            "legacy/quarantine checkpoints cannot enter current evaluation")
    if report.get("selection_provenance") != \
            "latent_eval.v3_recomputed_from_raw_validation_records":
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: checkpoint selection was not proven "
            "from raw latent_eval.v3 validation records")
    run_id = report.get("run_id")
    adapter_id = report.get("adapter_id")
    if not isinstance(run_id, str) or not run_id \
            or not isinstance(adapter_id, str) or not adapter_id:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: missing v3 run/adapter identity")
    selected = select_v3_checkpoint_from_raw_history(
        report.get("val_history"),
        expected_identity={
            "run_id": run_id,
            "recipe_hash": _recipe_hash(recipe),
            "model_id": model_id,
            "model_revision": revision,
            "adapter_id": adapter_id,
            "suite_id": suite_manifest["suite_identity"],
            "suite_version": suite_manifest["suite_version"],
            "suite_hash": suite_sha,
            "split": "validation",
            "k": cfg["k"],
            "recurrence_config": _recurrence_config(cfg),
        })
    if selected is None or selected.step != report.get("best_step") \
            or selected.metric != report.get("best_val_acc"):
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: raw v3 validation history does not "
            "reproduce the reported selected checkpoint")
    selected_raw_state_sha256 = selected_v3_adapter_state_sha256(
        report.get("val_history"), expected_step=selected.step,
        expected_metric=selected.metric)
    m_ident = manifest.get("identity") or {}
    if m_ident.get("model_id") != model_id \
            or require_pinned_revision(m_ident.get("revision")) != revision:
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: manifest identity {m_ident} "
            f"disagrees with the report identity ({model_id!r}, "
            f"{revision!r})")
    m_seed = manifest.get("seed")
    if m_seed != cfg.get("seed"):
        raise AdapterBundleIdentityError(
            f"adapter {args.adapter}: manifest seed {m_seed!r} != config "
            f"seed {cfg.get('seed')!r}")
    state = load_adapter_bundle(Path(args.adapter) / "best_params.pt",
                                model_id=model_id, revision=revision,
                                recipe=recipe)
    validate_selected_adapter_state_binding(
        report=report, manifest=manifest,
        actual_state_sha256=adapter_state_sha256(state),
        raw_selected_state_sha256=selected_raw_state_sha256,
        where=f"adapter {args.adapter}")

    model, tok = load_model(device, model_id, revision)
    interval = tuple(cfg["interval"])
    rec = LocalizedRecurrence(model, None, interval=interval,
                               max_k=cfg["max_k"], lora_r=cfg["lora_r"],
                               lora_alpha=float(cfg.get("lora_alpha", 16.0)),
                               recurrence_only_lora=recurrence_only_lora,
                               neutral_delta=neutral_delta)
    stored_runtime_contract = cfg.get("runtime_contract")
    if stored_runtime_contract is not None \
            and stored_runtime_contract != rec.runtime_contract():
        raise AdapterBundleIdentityError(
            "adapter runtime_contract disagrees with the constructed "
            "LocalizedRecurrence runtime")
    rec.load_adapter_state(state)

    split_name = args.split
    if split_name not in suite.splits() or split_name == "train":
        raise ValueError(
            f"unsupported v3 eval split {split_name!r}; available: "
            f"{sorted(set(suite.splits()) - {'train'})}")
    data = SuiteTensors(tok, list(getattr(suite, split_name)))
    idx = list(range(len(data.examples)))
    k = args.k if args.k is not None else cfg["k"]
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= cfg["max_k"]:
        raise ValueError(f"eval K must satisfy 0 <= K <= {cfg['max_k']}")

    ablation = parse_ablation_cli(args.ablate, k)
    evidence_identity = _v3_evidence_identity(
        run_id=run_id, recipe=recipe, model_id=model_id, revision=revision,
        adapter_id=adapter_id, checkpoint_id=f"step-{selected.step}",
        checkpoint_content_hash=report["checkpoint_content_digest"],
        suite_manifest=suite_manifest,
        recurrence_config=_recurrence_config(cfg))
    res = evaluate_v3(
        rec, data, k, idx, evidence_identity=evidence_identity,
        split=split_name, ablate=ablation,
        tag=f"{cfg['mode']}|{split_name}|{args.ablate or 'clean'}|K={k}",
        limit=args.limit)
    results = {args.ablate or "clean": res}
    # prove the persisted evidence is independently rescorable right now
    if rescore_v3_records(res["records"]) != res["metrics"]:
        raise ValueError("online/offline latent_eval.v3 rescore disagreement")
    print(json.dumps({k2: v for k2, v in res.items() if k2 != "records"},
                     indent=1))

    if args.out:
        payload = build_v3_eval_payload(
            adapter=args.adapter, split=split_name, config=cfg,
            model_id=model_id, revision=revision, suite_hash=suite_sha,
            tokenizer_class=type(tok).__name__, interval=interval,
            k_steps=k, ablation=args.ablate, seed=args.seed,
            checkpoint_content_digest=report["checkpoint_content_digest"],
            result=res, device=device)
        atomic_write_json(args.out, payload)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--k", type=int, default=4)
    tr.add_argument("--interval", default="mid",
                    help="'mid'|'full'|'head'|'lo,hi'")
    tr.add_argument("--steps", type=int, default=600)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--lora-r", type=int, default=8)
    tr.add_argument("--lora-alpha", type=float, default=16.0)
    tr.add_argument("--max-k", type=int, default=16)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--eval-every", type=int, default=100)
    tr.add_argument(
        "--val-examples", type=int, default=None,
        help="validation cap; omitted means the complete behavioral-v3 "
             "validation split")
    tr.add_argument("--warmup", type=int, default=30)
    tr.add_argument("--lr-schedule", default="constant",
                    choices=["constant", "cosine"])
    tr.add_argument("--optimizer", default="adamw")
    tr.add_argument("--weight-decay", type=float, default=0.01)
    tr.add_argument("--clip", type=float, default=0.5)
    tr.add_argument("--detach-z0", action="store_true")
    tr.add_argument(
        "--training-objective", choices=TRAINING_OBJECTIVES,
        default="gold_nll",
        help="gold-token NLL (default) or listwise candidate CE aligned to "
             "evaluation scoring")
    tr.add_argument(
        "--recurrence-only-lora", action="store_true",
        help="activate LoRA only inside latent recurrence; prompt/readout use "
              "the frozen base model (K=0 training is unsupported)")
    tr.add_argument(
        "--neutral-delta", action="store_true",
        help="full-decoder ReZero delta recurrence at fixed prompt-end cache; "
             "implies recurrence-only LoRA")
    tr.add_argument("--label", default=None,
                    help="preregistered run label (bound into evidence; "
                    "never derived from output path)")
    tr.add_argument("--device", default="mps")
    tr.add_argument("--model", default=DEFAULT_MODEL_ID)
    tr.add_argument("--revision", default=DEFAULT_REVISION)
    tr.add_argument("--out", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--adapter", required=True)
    ev.add_argument(
        "--split", default="test_id",
        choices=("validation", "test_id", "test_ood_length",
                 "test_ood_semantic", "final_test"))
    ev.add_argument("--k", type=int, default=None)
    ev.add_argument("--ablate", default=None)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--device", default="mps")
    ev.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
