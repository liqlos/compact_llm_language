"""Canonical, independently rescorable evaluation contract for R1.

``latent_eval.v3`` is the only schema that may support a current measured
claim.  Candidate scoring is fixed before experiments: the arithmetic mean
of the candidate token log probabilities.  Raw per-token values and their
sum are retained so the verdict can be reproduced without the model.

The module is intentionally stdlib-only.  It is shared by online evaluation,
checkpoint selection, offline rescore, reports, and integrity gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "latent_eval.v3"
SUMMARY_SCHEMA_VERSION = "latent_eval.summary.v3"
SCORER_IMPLEMENTATION = "latent_lab.bench.eval_v3"
SCORER_VERSION = "3"
PRIMARY_SCORE_DEFINITION = "mean_candidate_token_logprob_v1"
TIE_POLICY = "exact_top_tie_is_error_v1"

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_NONTERMINATION = "nontermination"
ERROR_AMBIGUOUS_TOP_TIE = "AMBIGUOUS_TOP_TIE"

_SHA256_LEN = 64
_MODEL_KEYS = frozenset(("model_id", "revision"))
_CHECKPOINT_KEYS = frozenset(
    ("adapter_id", "checkpoint_id", "content_sha256")
)
_SUITE_KEYS = frozenset(("suite_id", "version", "sha256"))
_SCORER_KEYS = frozenset(("implementation", "version", "sha256"))
_REQUIRED_COMPUTE_KEYS = frozenset(
    (
        "prefill_layers",
        "recurrence_interval_applications",
        "k_loops",
        "candidate_tail_layers",
        "lm_head_calls",
        "tokenizer_calls",
        "decode_calls",
        "wall_seconds",
        "peak_memory_bytes",
        "successful_task",
    )
)

RECORD_FIELDS = frozenset(
    (
        "schema_version",
        "run_id",
        "recipe_hash",
        "model_identity",
        "checkpoint_identity",
        "suite_identity",
        "example_id",
        "split",
        "family",
        "prompt_hash",
        "candidates",
        "candidate_permutation_seed",
        "candidate_permutation",
        "gold_answer",
        "gold_index",
        "per_token_logprobs",
        "token_counts",
        "raw_summed_logprobs",
        "normalized_scores",
        "primary_score_definition",
        "ranking",
        "ties",
        "tie_policy",
        "predicted_answer",
        "correctness",
        "status",
        "error_status",
        "error_detail",
        "k",
        "recurrence_config",
        "compute",
        "scorer",
        "record_sha256",
    )
)

DERIVED_FIELDS = (
    "gold_index",
    "token_counts",
    "raw_summed_logprobs",
    "normalized_scores",
    "primary_score_definition",
    "ranking",
    "ties",
    "tie_policy",
    "predicted_answer",
    "correctness",
    "status",
    "error_status",
)


class EvalV3Error(ValueError):
    """A record cannot be current evidence."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def scorer_implementation_sha256() -> str:
    """Hash the actual versioned implementation, not a human label."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def scorer_identity() -> dict[str, str]:
    return {
        "implementation": SCORER_IMPLEMENTATION,
        "version": SCORER_VERSION,
        "sha256": scorer_implementation_sha256(),
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvalV3Error(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    text = _require_nonempty_str(value, field)
    if len(text) != _SHA256_LEN or any(c not in "0123456789abcdef" for c in text):
        raise EvalV3Error(f"{field} must be a lowercase 64-hex SHA-256")
    return text


def _require_exact_keys(value: Any, expected: frozenset[str], field: str) -> dict:
    if not isinstance(value, dict):
        raise EvalV3Error(f"{field} must be an object")
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise EvalV3Error(f"{field} keys mismatch; missing={missing}, extra={extra}")
    return value


def _require_json_finite(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvalV3Error(f"{field} contains NaN/Inf")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _require_json_finite(item, f"{field}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvalV3Error(f"{field} contains a non-string key")
            _require_json_finite(item, f"{field}.{key}")
        return
    raise EvalV3Error(f"{field} contains non-JSON value {type(value).__name__}")


def derive_gold_index(candidates: Sequence[str], gold_answer: str) -> int:
    matches = [i for i, candidate in enumerate(candidates) if candidate == gold_answer]
    if len(matches) != 1:
        state = "absent" if not matches else "duplicated"
        raise EvalV3Error(f"gold answer is {state} in candidates")
    return matches[0]


def _validate_candidates(candidates: Sequence[str]) -> tuple[str, ...]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise EvalV3Error("candidates must be a sequence")
    out = tuple(candidates)
    if len(out) < 2:
        raise EvalV3Error("at least two candidates are required")
    if any(not isinstance(candidate, str) or not candidate for candidate in out):
        raise EvalV3Error("candidates must be non-empty strings")
    if len(set(out)) != len(out):
        raise EvalV3Error("duplicate candidates are forbidden")
    return out


@dataclass(frozen=True)
class ScoreResult:
    gold_index: int
    token_counts: tuple[int, ...]
    raw_summed_logprobs: tuple[float, ...]
    normalized_scores: tuple[float, ...]
    ranking: tuple[int, ...]
    ties: tuple[tuple[int, ...], ...]
    predicted_index: int | None
    correctness: bool
    status: str
    error_status: str | None


def score_candidates(
    candidates: Sequence[str],
    gold_answer: str,
    per_token_logprobs: Sequence[Sequence[float]],
) -> ScoreResult:
    """Score one candidate set from raw token log probabilities.

    Exact top ties are measured errors.  Non-top ties are retained in the
    record and ordered by candidate contents solely to make the complete
    ranking byte-stable; they cannot affect correctness.
    """
    cands = _validate_candidates(candidates)
    _require_nonempty_str(gold_answer, "gold_answer")
    gold_index = derive_gold_index(cands, gold_answer)
    if isinstance(per_token_logprobs, (str, bytes)) or not isinstance(
        per_token_logprobs, Sequence
    ):
        raise EvalV3Error("per_token_logprobs must be a sequence")
    if len(per_token_logprobs) != len(cands):
        raise EvalV3Error("per_token_logprobs length must match candidates")

    normalized_rows: list[tuple[float, ...]] = []
    for i, row in enumerate(per_token_logprobs):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
            raise EvalV3Error(f"candidate {i} must retain at least one token logprob")
        values: list[float] = []
        for j, raw in enumerate(row):
            if not _is_finite_real(raw):
                raise EvalV3Error(f"per_token_logprobs[{i}][{j}] is not finite real")
            values.append(float(raw))
        normalized_rows.append(tuple(values))

    counts = tuple(len(row) for row in normalized_rows)
    sums = tuple(math.fsum(row) for row in normalized_rows)
    means = tuple(total / count for total, count in zip(sums, counts))
    ranking = tuple(
        sorted(
            range(len(cands)),
            key=lambda i: (-means[i], cands[i].encode("utf-8")),
        )
    )

    by_score: dict[float, list[int]] = {}
    for index, value in enumerate(means):
        by_score.setdefault(value, []).append(index)
    rank_position = {candidate_index: pos for pos, candidate_index in enumerate(ranking)}
    ties = tuple(
        tuple(sorted(group, key=rank_position.__getitem__))
        for _, group in sorted(by_score.items(), key=lambda item: -item[0])
        if len(group) > 1
    )
    leaders = tuple(i for i, value in enumerate(means) if value == means[ranking[0]])
    if len(leaders) > 1:
        return ScoreResult(
            gold_index=gold_index,
            token_counts=counts,
            raw_summed_logprobs=sums,
            normalized_scores=means,
            ranking=ranking,
            ties=ties,
            predicted_index=None,
            correctness=False,
            status=STATUS_ERROR,
            error_status=ERROR_AMBIGUOUS_TOP_TIE,
        )
    predicted = ranking[0]
    return ScoreResult(
        gold_index=gold_index,
        token_counts=counts,
        raw_summed_logprobs=sums,
        normalized_scores=means,
        ranking=ranking,
        ties=ties,
        predicted_index=predicted,
        correctness=predicted == gold_index,
        status=STATUS_OK,
        error_status=None,
    )


def _validate_identity_inputs(
    *,
    run_id: str,
    recipe_hash: str,
    model_id: str,
    model_revision: str,
    adapter_id: str,
    checkpoint_id: str,
    checkpoint_content_hash: str,
    suite_id: str,
    suite_version: int,
    suite_hash: str,
    example_id: str,
    split: str,
    family: str,
    candidate_permutation_seed: int,
    candidate_permutation: Sequence[int],
    candidates: Sequence[str],
    k: int,
    recurrence_config: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> None:
    for field, value in (
        ("run_id", run_id),
        ("model_id", model_id),
        ("model_revision", model_revision),
        ("adapter_id", adapter_id),
        ("checkpoint_id", checkpoint_id),
        ("suite_id", suite_id),
        ("example_id", example_id),
        ("split", split),
        ("family", family),
    ):
        _require_nonempty_str(value, field)
    _require_sha256(recipe_hash, "recipe_hash")
    _require_sha256(checkpoint_content_hash, "checkpoint_content_hash")
    _require_sha256(suite_hash, "suite_hash")
    if not _is_int(suite_version) or suite_version < 1:
        raise EvalV3Error("suite_version must be an int >= 1")
    if not _is_int(candidate_permutation_seed):
        raise EvalV3Error("candidate_permutation_seed must be an int")
    if not _is_int(k) or k < 0:
        raise EvalV3Error("k must be an int >= 0")
    cands = _validate_candidates(candidates)
    perm = list(candidate_permutation)
    if any(not _is_int(value) for value in perm) or sorted(perm) != list(
        range(len(cands))
    ):
        raise EvalV3Error("candidate_permutation must be a complete permutation")
    if not isinstance(recurrence_config, Mapping):
        raise EvalV3Error("recurrence_config must be an object")
    if "grad_checkpoint" in recurrence_config:
        raise EvalV3Error("unsupported grad_checkpoint must not appear in v3 evidence")
    _require_json_finite(dict(recurrence_config), "recurrence_config")
    if not isinstance(compute, Mapping):
        raise EvalV3Error("compute must be an object")
    missing_compute = sorted(_REQUIRED_COMPUTE_KEYS - set(compute))
    if missing_compute:
        raise EvalV3Error(f"compute is missing required counters {missing_compute}")
    _require_json_finite(dict(compute), "compute")
    for name in _REQUIRED_COMPUTE_KEYS - {"wall_seconds", "successful_task"}:
        value = compute[name]
        if value is not None and (not _is_int(value) or value < 0):
            raise EvalV3Error(f"compute.{name} must be null or int >= 0")
    if not _is_finite_real(compute["wall_seconds"]) or compute["wall_seconds"] < 0:
        raise EvalV3Error("compute.wall_seconds must be finite and >= 0")
    if not isinstance(compute["successful_task"], bool):
        raise EvalV3Error("compute.successful_task must be bool")


def _base_record(
    *,
    run_id: str,
    recipe_hash: str,
    model_id: str,
    model_revision: str,
    adapter_id: str,
    checkpoint_id: str,
    checkpoint_content_hash: str,
    suite_id: str,
    suite_version: int,
    suite_hash: str,
    example_id: str,
    split: str,
    family: str,
    prompt: str,
    candidates: Sequence[str],
    candidate_permutation_seed: int,
    candidate_permutation: Sequence[int],
    gold_answer: str,
    k: int,
    recurrence_config: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    _require_nonempty_str(prompt, "prompt")
    _require_nonempty_str(gold_answer, "gold_answer")
    _validate_identity_inputs(
        run_id=run_id,
        recipe_hash=recipe_hash,
        model_id=model_id,
        model_revision=model_revision,
        adapter_id=adapter_id,
        checkpoint_id=checkpoint_id,
        checkpoint_content_hash=checkpoint_content_hash,
        suite_id=suite_id,
        suite_version=suite_version,
        suite_hash=suite_hash,
        example_id=example_id,
        split=split,
        family=family,
        candidate_permutation_seed=candidate_permutation_seed,
        candidate_permutation=candidate_permutation,
        candidates=candidates,
        k=k,
        recurrence_config=recurrence_config,
        compute=compute,
    )
    cands = list(candidates)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "recipe_hash": recipe_hash,
        "model_identity": {"model_id": model_id, "revision": model_revision},
        "checkpoint_identity": {
            "adapter_id": adapter_id,
            "checkpoint_id": checkpoint_id,
            "content_sha256": checkpoint_content_hash,
        },
        "suite_identity": {
            "suite_id": suite_id,
            "version": suite_version,
            "sha256": suite_hash,
        },
        "example_id": example_id,
        "split": split,
        "family": family,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "candidates": cands,
        "candidate_permutation_seed": candidate_permutation_seed,
        "candidate_permutation": list(candidate_permutation),
        "gold_answer": gold_answer,
        "k": k,
        "recurrence_config": dict(recurrence_config),
        "compute": dict(compute),
        "scorer": scorer_identity(),
    }


def _seal_record(record: dict[str, Any]) -> dict[str, Any]:
    record["record_sha256"] = canonical_sha256(record)
    return record


def build_eval_record(
    *,
    per_token_logprobs: Sequence[Sequence[float]],
    **metadata: Any,
) -> dict[str, Any]:
    """Build and seal one score-bearing ``latent_eval.v3`` record."""
    record = _base_record(**metadata)
    result = score_candidates(
        record["candidates"], record["gold_answer"], per_token_logprobs
    )
    record.update(
        {
            "gold_index": result.gold_index,
            "per_token_logprobs": [list(map(float, row)) for row in per_token_logprobs],
            "token_counts": list(result.token_counts),
            "raw_summed_logprobs": list(result.raw_summed_logprobs),
            "normalized_scores": list(result.normalized_scores),
            "primary_score_definition": PRIMARY_SCORE_DEFINITION,
            "ranking": list(result.ranking),
            "ties": [list(group) for group in result.ties],
            "tie_policy": TIE_POLICY,
            "predicted_answer": (
                record["candidates"][result.predicted_index]
                if result.predicted_index is not None
                else None
            ),
            "correctness": result.correctness,
            "status": result.status,
            "error_status": result.error_status,
            "error_detail": None,
        }
    )
    return _seal_record(record)


def build_error_record(
    *,
    status: str,
    error_status: str,
    error_detail: str | None,
    **metadata: Any,
) -> dict[str, Any]:
    """Build a nontermination/runtime-error record without invented scores."""
    if status not in (STATUS_ERROR, STATUS_NONTERMINATION):
        raise EvalV3Error("error record status must be error or nontermination")
    _require_nonempty_str(error_status, "error_status")
    if error_status == ERROR_AMBIGUOUS_TOP_TIE:
        raise EvalV3Error("ambiguous ties must be built from retained raw scores")
    if error_detail is not None and not isinstance(error_detail, str):
        raise EvalV3Error("error_detail must be string or null")
    record = _base_record(**metadata)
    count = len(record["candidates"])
    record.update(
        {
            "gold_index": derive_gold_index(record["candidates"], record["gold_answer"]),
            "per_token_logprobs": [[] for _ in range(count)],
            "token_counts": [0] * count,
            "raw_summed_logprobs": [None] * count,
            "normalized_scores": [None] * count,
            "primary_score_definition": PRIMARY_SCORE_DEFINITION,
            "ranking": [],
            "ties": [],
            "tie_policy": TIE_POLICY,
            "predicted_answer": None,
            "correctness": False,
            "status": status,
            "error_status": error_status,
            "error_detail": error_detail,
        }
    )
    return _seal_record(record)


def _score_verdict(record: Mapping[str, Any]) -> dict[str, Any]:
    result = score_candidates(
        record["candidates"], record["gold_answer"], record["per_token_logprobs"]
    )
    return {
        "gold_index": result.gold_index,
        "token_counts": list(result.token_counts),
        "raw_summed_logprobs": list(result.raw_summed_logprobs),
        "normalized_scores": list(result.normalized_scores),
        "primary_score_definition": PRIMARY_SCORE_DEFINITION,
        "ranking": list(result.ranking),
        "ties": [list(group) for group in result.ties],
        "tie_policy": TIE_POLICY,
        "predicted_answer": (
            record["candidates"][result.predicted_index]
            if result.predicted_index is not None
            else None
        ),
        "correctness": result.correctness,
        "status": result.status,
        "error_status": result.error_status,
    }


def rescore_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute only the verdict fields from raw evidence."""
    if record.get("status") in (STATUS_ERROR, STATUS_NONTERMINATION) and record.get(
        "error_status"
    ) != ERROR_AMBIGUOUS_TOP_TIE:
        count = len(record.get("candidates", ()))
        return {
            "gold_index": derive_gold_index(record["candidates"], record["gold_answer"]),
            "token_counts": [0] * count,
            "raw_summed_logprobs": [None] * count,
            "normalized_scores": [None] * count,
            "primary_score_definition": PRIMARY_SCORE_DEFINITION,
            "ranking": [],
            "ties": [],
            "tie_policy": TIE_POLICY,
            "predicted_answer": None,
            "correctness": False,
            "status": record["status"],
            "error_status": record["error_status"],
        }
    return _score_verdict(record)


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Strict schema, identity, raw-value, and derived-verdict validation."""
    data = _require_exact_keys(record, RECORD_FIELDS, "record")
    if data["schema_version"] != SCHEMA_VERSION:
        raise EvalV3Error(f"schema_version must be {SCHEMA_VERSION!r}")
    _validate_identity_inputs(
        run_id=data["run_id"],
        recipe_hash=data["recipe_hash"],
        model_id=_require_exact_keys(data["model_identity"], _MODEL_KEYS, "model_identity")[
            "model_id"
        ],
        model_revision=data["model_identity"]["revision"],
        adapter_id=_require_exact_keys(
            data["checkpoint_identity"], _CHECKPOINT_KEYS, "checkpoint_identity"
        )["adapter_id"],
        checkpoint_id=data["checkpoint_identity"]["checkpoint_id"],
        checkpoint_content_hash=data["checkpoint_identity"]["content_sha256"],
        suite_id=_require_exact_keys(data["suite_identity"], _SUITE_KEYS, "suite_identity")[
            "suite_id"
        ],
        suite_version=data["suite_identity"]["version"],
        suite_hash=data["suite_identity"]["sha256"],
        example_id=data["example_id"],
        split=data["split"],
        family=data["family"],
        candidate_permutation_seed=data["candidate_permutation_seed"],
        candidate_permutation=data["candidate_permutation"],
        candidates=data["candidates"],
        k=data["k"],
        recurrence_config=data["recurrence_config"],
        compute=data["compute"],
    )
    _require_sha256(data["prompt_hash"], "prompt_hash")
    scorer = _require_exact_keys(data["scorer"], _SCORER_KEYS, "scorer")
    if scorer != scorer_identity():
        raise EvalV3Error("scorer identity/hash does not match canonical implementation")
    if data["primary_score_definition"] != PRIMARY_SCORE_DEFINITION:
        raise EvalV3Error("primary_score_definition is not canonical")
    if data["tie_policy"] != TIE_POLICY:
        raise EvalV3Error("tie_policy is not canonical")
    if data["error_detail"] is not None and not isinstance(data["error_detail"], str):
        raise EvalV3Error("error_detail must be string or null")

    expected = rescore_record(data)
    for field, value in expected.items():
        if data[field] != value:
            raise EvalV3Error(f"{field} disagrees with independent raw rescore")
    if data["status"] in (STATUS_ERROR, STATUS_NONTERMINATION) and data[
        "error_status"
    ] != ERROR_AMBIGUOUS_TOP_TIE:
        if data["per_token_logprobs"] != [[] for _ in data["candidates"]]:
            raise EvalV3Error("runtime-error record must not invent token logprobs")
    elif data["status"] not in (STATUS_OK, STATUS_ERROR):
        raise EvalV3Error("score-bearing record has invalid status")

    claimed_hash = _require_sha256(data["record_sha256"], "record_sha256")
    core = dict(data)
    del core["record_sha256"]
    if claimed_hash != canonical_sha256(core):
        raise EvalV3Error("record_sha256 mismatch")
    return dict(data)


@lru_cache(maxsize=1)
def _current_suite_binding() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the immutable benchmark identity used by current evidence.

    Schema-valid raw scores are not sufficient evidence: the record must
    describe the exact canonical example whose id it claims.  Keeping the
    lookup here gives artifact validation, checkpoint selection, offline
    rescore, and the no-spend gate one record-to-suite truth.
    """
    from latent_lab.bench.suite_v3 import (
        SUITE_IDENTITY,
        SUITE_VERSION,
        build_suite,
    )

    suite = build_suite()
    manifest = suite.manifest()
    suite_hash = suite.records_hash()
    if manifest.get("suite_identity") != SUITE_IDENTITY \
            or manifest.get("suite_version") != SUITE_VERSION \
            or manifest.get("suite_hash") != suite_hash:
        raise EvalV3Error("canonical behavioral-v3 suite is not self-consistent")
    examples = {
        example.ex_id: example
        for split_examples in suite.splits().values()
        for example in split_examples
    }
    identity = {
        "suite_id": SUITE_IDENTITY,
        "version": SUITE_VERSION,
        "sha256": suite_hash,
    }
    return identity, examples


def current_suite_identity() -> dict[str, Any]:
    """Return a copy of the immutable suite identity eligible as current."""
    identity, _ = _current_suite_binding()
    return dict(identity)


def validate_record_against_current_suite(
    record: Mapping[str, Any],
    *,
    expected_split: str | None = None,
) -> dict[str, Any]:
    """Bind a valid raw record to its exact behavioral-v3 example.

    The generic ``latent_eval.v3`` validator proves internal consistency.
    This evidence-bound validator additionally proves that suite identity,
    split membership, prompt, family, actual candidate order, permutation
    metadata, and gold fields came from the immutable current suite.
    """
    data = validate_record(record)
    suite_identity, examples = _current_suite_binding()
    if data["suite_identity"] != suite_identity:
        raise EvalV3Error(
            "suite_identity does not match the canonical behavioral-v3 suite")
    if expected_split is not None and data["split"] != expected_split:
        raise EvalV3Error(
            f"split {data['split']!r} does not match expected split "
            f"{expected_split!r}")

    example = examples.get(data["example_id"])
    if example is None:
        raise EvalV3Error(
            f"example_id {data['example_id']!r} is absent from behavioral-v3")
    expected_prompt_hash = hashlib.sha256(
        example.prompt.encode("utf-8")
    ).hexdigest()
    checks = (
        ("split", data["split"], example.split),
        ("family", data["family"], example.family),
        ("prompt_hash", data["prompt_hash"], expected_prompt_hash),
        ("candidates", data["candidates"], list(example.candidates)),
        (
            "candidate_permutation_seed",
            data["candidate_permutation_seed"],
            example.candidate_permutation_seed,
        ),
        (
            "candidate_permutation",
            data["candidate_permutation"],
            list(example.candidate_permutation),
        ),
        ("gold_answer", data["gold_answer"], example.answer),
        ("gold_index", data["gold_index"], example.gold_index),
    )
    mismatches = [field for field, actual, expected in checks
                  if actual != expected]
    if mismatches:
        raise EvalV3Error(
            f"example_id {data['example_id']!r} disagrees with canonical "
            "behavioral-v3 fields: " + ", ".join(mismatches))
    return data


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    validated = [validate_record(record) for record in records]
    if not validated:
        raise EvalV3Error("cannot aggregate zero records")
    example_ids = [record["example_id"] for record in validated]
    if len(set(example_ids)) != len(example_ids):
        raise EvalV3Error("duplicate example_id records are forbidden")
    identity_fields = ("run_id", "recipe_hash", "model_identity", "checkpoint_identity", "suite_identity", "split", "k", "recurrence_config", "scorer")
    for field in identity_fields:
        if any(record[field] != validated[0][field] for record in validated[1:]):
            raise EvalV3Error(f"aggregate mixes incompatible {field}")

    n = len(validated)
    hits = sum(record["correctness"] is True for record in validated)
    per_family_counts: dict[str, list[int]] = {}
    for record in validated:
        bucket = per_family_counts.setdefault(record["family"], [0, 0])
        bucket[0] += int(record["correctness"] is True)
        bucket[1] += 1
    per_family = {
        family: count / total
        for family, (count, total) in sorted(per_family_counts.items())
    }
    micro = hits / n
    macro = math.fsum(per_family.values()) / len(per_family)
    uniform_chance = math.fsum(1.0 / len(record["candidates"]) for record in validated) / n
    chance_normalized = (micro - uniform_chance) / (1.0 - uniform_chance)
    error_rate = sum(record["status"] != STATUS_OK for record in validated) / n
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "record_schema_version": SCHEMA_VERSION,
        "primary_score_definition": PRIMARY_SCORE_DEFINITION,
        "tie_policy": TIE_POLICY,
        "scorer": scorer_identity(),
        "run_id": validated[0]["run_id"],
        "recipe_hash": validated[0]["recipe_hash"],
        "suite_identity": validated[0]["suite_identity"],
        "split": validated[0]["split"],
        "k": validated[0]["k"],
        "n_examples": n,
        "micro_accuracy": micro,
        "macro_by_family_accuracy": macro,
        "per_family_accuracy": per_family,
        "mean_uniform_chance": uniform_chance,
        "chance_normalized_accuracy": chance_normalized,
        "nontermination_error_rate": error_rate,
        "record_hashes": [record["record_sha256"] for record in validated],
    }


def paired_comparison(
    treatment_records: Iterable[Mapping[str, Any]],
    control_records: Iterable[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Deterministic paired bootstrap over the exact same examples."""
    if not _is_int(bootstrap_seed):
        raise EvalV3Error("bootstrap_seed must be int")
    if not _is_int(bootstrap_samples) or bootstrap_samples < 1:
        raise EvalV3Error("bootstrap_samples must be int >= 1")
    if not _is_finite_real(confidence) or not 0 < float(confidence) < 1:
        raise EvalV3Error("confidence must be between 0 and 1")
    def by_example_id(
        records: Iterable[Mapping[str, Any]], label: str
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for raw_record in records:
            record = validate_record(raw_record)
            example_id = record["example_id"]
            if example_id in out:
                raise EvalV3Error(
                    f"paired {label} contains duplicate example_id {example_id!r}"
                )
            out[example_id] = record
        return out

    treatment = by_example_id(treatment_records, "treatment")
    control = by_example_id(control_records, "control")
    if set(treatment) != set(control) or not treatment:
        raise EvalV3Error("paired comparison requires the same non-empty example ids")
    ids = sorted(treatment)
    for example_id in ids:
        left, right = treatment[example_id], control[example_id]
        for field in (
            "run_id", "recipe_hash", "model_identity", "checkpoint_identity",
            "recurrence_config", "suite_identity", "split", "family",
            "prompt_hash", "candidates",
            "candidate_permutation_seed", "candidate_permutation",
            "gold_answer",
        ):
            if left[field] != right[field]:
                raise EvalV3Error(f"paired example {example_id} disagrees on {field}")
    deltas = [
        int(treatment[example_id]["correctness"] is True)
        - int(control[example_id]["correctness"] is True)
        for example_id in ids
    ]
    observed = math.fsum(deltas) / len(deltas)
    rng = random.Random(bootstrap_seed)
    estimates = sorted(
        math.fsum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    )
    alpha = (1.0 - float(confidence)) / 2.0
    low_index = max(0, math.floor(alpha * (bootstrap_samples - 1)))
    high_index = min(
        bootstrap_samples - 1, math.ceil((1.0 - alpha) * (bootstrap_samples - 1))
    )
    return {
        "method": "paired_percentile_bootstrap_v1",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "confidence": float(confidence),
        "n_pairs": len(deltas),
        "accuracy_delta": observed,
        "confidence_interval": [estimates[low_index], estimates[high_index]],
        "treatment_errors": sum(
            treatment[example_id]["status"] != STATUS_OK for example_id in ids
        ),
        "control_errors": sum(
            control[example_id]["status"] != STATUS_OK for example_id in ids
        ),
    }


@dataclass(frozen=True)
class SelectionResult:
    step: int
    metric: float
    n_considered: int
    n_rejected: int
    provenance: str = "latent_eval.v3_recomputed_from_record_summaries"


def checkpoint_history_entry(step: int, records: Iterable[Mapping[str, Any]]) -> dict:
    if not _is_int(step) or step < 0:
        raise EvalV3Error("checkpoint step must be int >= 0")
    return {"step": step, "metrics": aggregate_records(records)}


def select_best_checkpoint(
    history: Iterable[Mapping[str, Any]],
    *,
    metric_key: str = "micro_accuracy",
) -> SelectionResult | None:
    """Select only from canonical v3 summaries; earliest exact tie wins."""
    accepted: list[tuple[float, int]] = []
    rejected = 0
    for entry in history or ():
        if not isinstance(entry, Mapping):
            rejected += 1
            continue
        step = entry.get("step")
        metrics = entry.get("metrics")
        if not _is_int(step) or step < 0 or not isinstance(metrics, Mapping):
            rejected += 1
            continue
        if (
            metrics.get("schema_version") != SUMMARY_SCHEMA_VERSION
            or metrics.get("primary_score_definition") != PRIMARY_SCORE_DEFINITION
            or metrics.get("tie_policy") != TIE_POLICY
            or metrics.get("scorer") != scorer_identity()
        ):
            rejected += 1
            continue
        value = metrics.get(metric_key)
        if not _is_finite_real(value):
            rejected += 1
            continue
        accepted.append((float(value), step))
    if not accepted:
        return None
    best_metric = max(metric for metric, _ in accepted)
    best_step = min(step for metric, step in accepted if metric == best_metric)
    return SelectionResult(
        step=best_step,
        metric=best_metric,
        n_considered=len(accepted),
        n_rejected=rejected,
    )
