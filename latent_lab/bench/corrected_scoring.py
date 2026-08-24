"""Corrected gold-aware scoring + deterministic checkpoint selection.

Offline, stdlib-only primitives used by the no-spend integrity gate and by
future eval drivers:

* ``normalize_answer`` / ``gold_indices`` — whitespace/parser-tolerant gold
  alignment that never assumes candidate zero is gold and never silently
  scores ambiguous (duplicated) candidate sets.
* ``corrected_score`` — rank-of-gold from an explicit ranking order or from
  per-candidate scores; deterministic under candidate permutation when the
  score-to-candidate mapping is preserved.
* ``rescore_records`` — recompute corrected accuracy ONLY where raw scorer
  inputs were retained; anything less is reported as
  ``NON_RESCORABLE_MISSING_RAW_PREDICTION``, never relabelled as a rescore.
* ``select_best_checkpoint`` — recompute the best step from a validation
  history with finite-metric enforcement and earliest-step tie-breaking,
  ignoring any stored/poisoned ``best_val_acc``/``best_step`` fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MISSING_RAW_PREDICTION = "NON_RESCORABLE_MISSING_RAW_PREDICTION"

# Fail-closed per-record invalid statuses: an eval file containing any such
# record can NEVER aggregate to RESCORED_CORRECTED — it is evidence of a
# broken/ambiguous scorer capture, not a partially usable rescore.
INVALID_UNKNOWN_EXAMPLE = "INVALID_UNKNOWN_EXAMPLE"
INVALID_DUPLICATE_EXAMPLE_RECORDS = "INVALID_DUPLICATE_EXAMPLE_RECORDS"
INVALID_CONFLICTING_REPRESENTATIONS = "INVALID_CONFLICTING_REPRESENTATIONS"
INVALID_MALFORMED_CANDIDATE_SCORES = "INVALID_MALFORMED_CANDIDATE_SCORES"
INVALID_NONFINITE_CANDIDATE_SCORES = "INVALID_NONFINITE_CANDIDATE_SCORES"
INVALID_AMBIGUOUS_TOP_TIE = "INVALID_AMBIGUOUS_TOP_TIE"
INVALID_MALFORMED_RANKED_CANDIDATES = "INVALID_MALFORMED_RANKED_CANDIDATES"
INVALID_DUPLICATE_CANDIDATES = "INVALID_DUPLICATE_CANDIDATES"

FLAG_DUPLICATE_CANDIDATES = "DUPLICATE_CANDIDATES"
FLAG_GOLD_ABSENT = "GOLD_ABSENT"
FLAG_ORDER_NOT_PERMUTATION = "ORDER_NOT_PERMUTATION"
FLAG_NONFINITE_SCORES = "NONFINITE_SCORES"
FLAG_NORMALIZED_MATCH = "NORMALIZED_MATCH"

RAW_SCORER_FIELDS = (
    "candidate_scores",       # aligned per-candidate log-probs/scores
    "ranked_candidates",      # candidate contents in ranked order
    "predicted_answer",       # decoded answer string
)

# Lossless raw representations the corrected scorer accepts. A decoded
# ``predicted_answer`` string is NOT lossless (the ranking is unrecoverable),
# so it never satisfies the rescore contract.
LOSSLESS_RAW_FIELDS = ("candidate_scores", "ranked_candidates")


def normalize_answer(value: object) -> str:
    """Deterministic parser-tolerant normalization: str(), strip, collapse
    internal whitespace runs to single spaces. No case folding — suite
    answers are case-significant tokens."""
    return " ".join(str(value).split())


def gold_indices(candidates, answer) -> tuple[int, ...]:
    """All indices whose (normalized) candidate equals the normalized gold."""
    gold = normalize_answer(answer)
    return tuple(
        i for i, c in enumerate(candidates) if normalize_answer(c) == gold
    )


@dataclass(frozen=True)
class CorrectedScore:
    correct: bool
    rank_of_gold: int          # -1 when gold absent from the ranking
    flags: tuple[str, ...] = field(default=())


def corrected_score(candidates, answer, *, order=None, scores=None) -> CorrectedScore:
    """Gold-aware exact ranking score for one example.

    Exactly one of ``order``/``scores`` must be given. ``order`` is a
    candidate-index ranking (best first). ``scores`` are per-candidate
    values mapped BY POSITION IN ``candidates``; the effective order is
    ``sorted(range(n), key=(-score, index))`` after finiteness enforcement,
    which makes the verdict invariant to input permutation as long as each
    score stays attached to its candidate content.
    """
    n = len(candidates)
    flags: list[str] = []
    gidx = gold_indices(candidates, answer)
    if len(gidx) > 1:
        flags.append(FLAG_DUPLICATE_CANDIDATES)
    if not gidx:
        return CorrectedScore(correct=False, rank_of_gold=-1,
                              flags=(FLAG_GOLD_ABSENT,))

    if scores is not None:
        if len(scores) != n:
            raise ValueError("scores length != candidates length")
        vals: list[float] = []
        for s in scores:
            f = float(s)
            if not math.isfinite(f):
                flags.append(FLAG_NONFINITE_SCORES)
                f = -math.inf     # deterministic sink, never wins ties by value
            vals.append(f)
        eff_order = sorted(range(n), key=lambda i: (-vals[i], i))
    else:
        if order is None:
            raise ValueError("exactly one of order/scores is required")
        if sorted(order) != list(range(n)):
            raise ValueError("order is not a permutation of candidate indices")
        eff_order = list(order)

    # With duplicated candidates the ambiguity is flagged; conservatively
    # use the best-ranked gold occurrence.
    rank = min(eff_order.index(g) for g in gidx)
    if any(normalize_answer(candidates[i]) != str(answer).strip()
           and normalize_answer(candidates[i]) == normalize_answer(answer)
           for i in gidx):
        flags.append(FLAG_NORMALIZED_MATCH)
    return CorrectedScore(correct=(rank == 0), rank_of_gold=rank,
                          flags=tuple(sorted(set(flags))))


@dataclass(frozen=True)
class RescoreOutcome:
    status: str                       # RESCORED_CORRECTED | MISSING_RAW_PREDICTION | NO_RECORDS | INVALID_*
    n_records: int = 0
    corrected_accuracy: float | None = None
    detail: str | None = None
    flags: tuple[str, ...] = ()       # aggregated, never dropped on success


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _top_tie(vals: list[float]) -> bool:
    """True iff the maximum score is attained by more than one candidate —
    the top of the ranking is then ambiguous and the record is invalid."""
    if not vals:
        return False
    m = max(vals)
    return sum(1 for v in vals if v == m) > 1


def _record_raw_inputs(record: dict, candidates) -> bool:
    """True iff this record retained enough raw evidence to re-run the
    corrected scorer (per-candidate scores aligned to the example's
    candidates, or the full ranked candidate strings). A decoded
    ``predicted_answer`` alone is lossy and does NOT qualify."""
    if "candidate_scores" in record:
        cs = record["candidate_scores"]
        return isinstance(cs, (list, tuple)) and len(cs) == len(candidates)
    if "ranked_candidates" in record:
        rc = record["ranked_candidates"]
        return isinstance(rc, (list, tuple)) and len(rc) == len(candidates)
    return False


def rescore_records(records, examples_by_id) -> RescoreOutcome:
    """Recompute corrected accuracy over retained records — fail-closed.

    Contract enforced per record:
      * the example id must exist in the suite and appear at most once;
      * EXACTLY ONE lossless raw representation may be present
        (``candidate_scores`` or ``ranked_candidates``); several at once
        are a conflicting capture, none is NON_RESCORABLE;
      * scores must be numeric, aligned in length and fully finite; a top
        tie is ambiguous-invalid; duplicated candidate contents are
        ambiguous-invalid;
      * a ranking must be a full unique permutation of the example's
        candidates (unknown/partial/repeated entries are invalid).

    Any violation aborts aggregation with the specific INVALID_* status —
    never an uncaught exception, never a partial rescore.
    """
    if not records:
        return RescoreOutcome(status="NO_RECORDS")
    hits = 0
    flags: set[str] = set()
    seen_ex: set = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            return RescoreOutcome(
                status=INVALID_MALFORMED_CANDIDATE_SCORES,
                n_records=len(records),
                detail=f"record {i}: not an object")
        ex_id = rec.get("ex_id")
        ex = examples_by_id.get(ex_id)
        if ex is None:
            return RescoreOutcome(
                status=INVALID_UNKNOWN_EXAMPLE, n_records=len(records),
                detail=f"example {ex_id!r} absent from suite")
        if ex_id in seen_ex:
            return RescoreOutcome(
                status=INVALID_DUPLICATE_EXAMPLE_RECORDS,
                n_records=len(records),
                detail=f"example {ex_id!r} evaluated more than once")
        seen_ex.add(ex_id)
        cands = tuple(ex["candidates"]) if isinstance(ex, dict) \
            else tuple(ex.candidates)
        ans = ex["answer"] if isinstance(ex, dict) else ex.answer

        by_norm: dict[str, int] = {}
        for j, c in enumerate(cands):
            key = normalize_answer(c)
            if key in by_norm:
                return RescoreOutcome(
                    status=INVALID_DUPLICATE_CANDIDATES,
                    n_records=len(records),
                    detail=f"example {ex_id!r}: duplicate candidate {key!r}")
            by_norm[key] = j

        present = [f for f in LOSSLESS_RAW_FIELDS if f in rec]
        if not present:
            return RescoreOutcome(
                status=MISSING_RAW_PREDICTION, n_records=len(records),
                detail="records carry only derived correct/rank_of_gold")
        if len(present) > 1:
            return RescoreOutcome(
                status=INVALID_CONFLICTING_REPRESENTATIONS,
                n_records=len(records),
                detail=f"record {i}: multiple raw representations "
                       f"{sorted(present)}")

        if present[0] == "candidate_scores":
            cs_raw = rec["candidate_scores"]
            if not isinstance(cs_raw, (list, tuple)) or \
                    len(cs_raw) != len(cands) or \
                    not all(_is_number(v) for v in cs_raw):
                return RescoreOutcome(
                    status=INVALID_MALFORMED_CANDIDATE_SCORES,
                    n_records=len(records),
                    detail=f"record {i}: candidate_scores must be numeric "
                           f"and aligned with candidates")
            vals = [float(v) for v in cs_raw]
            if not all(math.isfinite(v) for v in vals):
                return RescoreOutcome(
                    status=INVALID_NONFINITE_CANDIDATE_SCORES,
                    n_records=len(records),
                    detail=f"record {i}: non-finite candidate_scores")
            if _top_tie(vals):
                return RescoreOutcome(
                    status=INVALID_AMBIGUOUS_TOP_TIE,
                    n_records=len(records),
                    detail=f"record {i}: top score attained by multiple "
                           f"candidates")
            cs = corrected_score(cands, ans, scores=cs_raw)
        else:
            rc = rec["ranked_candidates"]
            if not isinstance(rc, (list, tuple)) or \
                    len(rc) != len(cands) or \
                    not all(isinstance(x, str) for x in rc):
                return RescoreOutcome(
                    status=INVALID_MALFORMED_RANKED_CANDIDATES,
                    n_records=len(records),
                    detail=f"record {i}: ranked_candidates must be a full "
                           f"string ranking aligned with candidates")
            keys = [normalize_answer(x) for x in rc]
            if len(set(keys)) != len(keys) or \
                    any(k not in by_norm for k in keys):
                return RescoreOutcome(
                    status=INVALID_MALFORMED_RANKED_CANDIDATES,
                    n_records=len(records),
                    detail=f"record {i}: ranked_candidates repeat or "
                           f"introduce entries outside the example")
            order = [by_norm[k] for k in keys]
            cs = corrected_score(cands, ans, order=order)

        flags.update(cs.flags)
        hits += int(cs.correct)
    return RescoreOutcome(status="RESCORED_CORRECTED",
                          n_records=len(records),
                          corrected_accuracy=round(hits / len(records), 6),
                          flags=tuple(sorted(flags)))


@dataclass(frozen=True)
class SelectionResult:
    step: int
    metric: float
    n_considered: int
    n_rejected_nonfinite: int
    provenance: str = "recomputed_from_history"


def select_best_checkpoint(history, *, metric_key: str = "accuracy") -> SelectionResult | None:
    """Deterministically re-select the best validation entry.

    Ignores stored best_* fields entirely (they may be poisoned by a broken
    scorer); rejects non-finite metrics instead of skipping silently;
    breaks accuracy ties toward the EARLIEST step so later training cannot
    win on noise. Returns None when nothing finite remains.
    """
    considered, rejected = [], 0
    for h in history or ():
        try:
            v = float(h[metric_key])
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        if not math.isfinite(v):
            rejected += 1
            continue
        considered.append((v, int(h["step"])))
    if not considered:
        return None
    best_v = max(v for v, _ in considered)
    best_s = min(s for v, s in considered if v == best_v)
    return SelectionResult(step=best_s, metric=best_v,
                           n_considered=len(considered),
                           n_rejected_nonfinite=rejected)
