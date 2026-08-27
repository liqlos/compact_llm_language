"""Fail-closed salvage and classification for pre-v3 legacy evidence.

This module is **not** a current online scorer, checkpoint selector, or eval
driver API.  ``latent_lab.bench.eval_v3`` is the sole scoring truth for new
validation, selection, final evaluation, offline rescore, reports, and gates.
These stdlib-only helpers exist only to inspect historical artifacts that
predate ``latent_eval.v3`` and to demonstrate why they cannot become current
evidence merely because some raw candidate data survived.

LEGACY-ONLY contract:

* ``normalize_answer`` / ``gold_indices`` — whitespace/parser-tolerant gold
  alignment that never assumes candidate zero is gold.
* ``corrected_score`` — rank-of-gold from exactly ONE unambiguous raw
  prediction representation. Any evidence defect makes the verdict
  INVALID (``valid=False`` plus a machine-readable reason) instead of
  raising or silently scoring:
    - conflicting inputs (both ``order`` and ``scores``);
    - missing representation;
    - non-permutation / unknown / partial ``order`` lists;
    - scores that are not finite real numbers (NaN/Inf never sink to -inf);
    - a top-score tie between DIFFERENT candidates (array position is
      never used as a tiebreaker, so candidate permutation cannot change
      correctness);
    - duplicated (ambiguous) candidate sets.
  Gold absent from the ranking is a decisive, unambiguous INCORRECT — it
  stays a valid verdict flagged ``GOLD_ABSENT``.
* ``rescore_records`` — salvage a diagnostic fraction ONLY where every
  record retained exactly one usable raw representation; any malformed,
  ambiguous or invalid record fails the WHOLE file as
  ``INVALID_RECORDS``, and derived-only files stay
  ``NON_RESCORABLE_MISSING_RAW_PREDICTION``. Record-level flags are
  aggregated into ``flag_counts`` and can never be lost; no aggregation
  of invalid records can yield ``LEGACY_RAW_RESCORED``.  Even that success
  status has ``current_evidence_eligible=False`` and is mapped by the gate to
  ``HISTORICAL_UNBOUND_LEGACY_SCORER``.
* ``select_best_checkpoint`` — audit historical validation histories with
  finite-metric enforcement and earliest-step tie-breaking, ignoring any
  stored/poisoned ``best_val_acc``/``best_step`` fields.  It is not used by
  current latent_eval.v3 selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MISSING_RAW_PREDICTION = "NON_RESCORABLE_MISSING_RAW_PREDICTION"
INVALID_RECORDS = "INVALID_RECORDS"
NO_RECORDS = "NO_RECORDS"
LEGACY_RAW_RESCORED = "LEGACY_RAW_RESCORED"
LEGACY_EVIDENCE_SCOPE = "legacy_salvage_only"
CURRENT_EVIDENCE_ELIGIBLE = False

FLAG_DUPLICATE_CANDIDATES = "DUPLICATE_CANDIDATES"
FLAG_GOLD_ABSENT = "GOLD_ABSENT"
FLAG_ORDER_NOT_PERMUTATION = "ORDER_NOT_PERMUTATION"
FLAG_NONFINITE_SCORES = "NONFINITE_SCORES"
FLAG_NORMALIZED_MATCH = "NORMALIZED_MATCH"
FLAG_AMBIGUOUS_TIE = "AMBIGUOUS_TOP_TIE"
FLAG_CONFLICTING_INPUTS = "CONFLICTING_REPRESENTATIONS"
FLAG_NO_RAW_INPUTS = "NO_RAW_PREDICTION"
FLAG_SCORE_TYPE = "SCORE_NOT_A_REAL_NUMBER"
FLAG_SCORES_LENGTH = "SCORES_LENGTH_MISMATCH"

RAW_SCORER_FIELDS = (
    "candidate_scores",       # aligned per-candidate log-probs/scores
    "ranked_candidates",      # candidate contents in ranked order
    "predicted_answer",       # decoded answer string
)

MAX_DETAIL_PROBLEMS = 8


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
    rank_of_gold: int          # -1 when gold absent or verdict invalid
    flags: tuple[str, ...] = field(default=())
    valid: bool = True         # False => evidence defect, NEVER count this
    invalid_reason: str | None = None


def _invalid(reason: str, flags: tuple[str, ...]) -> CorrectedScore:
    return CorrectedScore(correct=False, rank_of_gold=-1,
                          flags=tuple(sorted(set(flags))),
                          valid=False, invalid_reason=reason)


def corrected_score(candidates, answer, *, order=None, scores=None) -> CorrectedScore:
    """Gold-aware exact ranking score for one example.

    Exactly one of ``order``/``scores`` must be supplied. Evidence defects
    return an INVALID CorrectedScore (``valid=False``); they never raise
    and never silently score. ``scores`` are per-candidate values mapped BY
    POSITION IN ``candidates``; a top-score tie between different
    candidates is ambiguous and invalid rather than resolved by index.
    """
    n = len(candidates)
    if n == 0:
        return _invalid("empty_candidate_set", ())
    norm = [normalize_answer(c) for c in candidates]
    if len(set(norm)) != n:
        return _invalid("duplicate_candidates", (FLAG_DUPLICATE_CANDIDATES,))
    if scores is not None and order is not None:
        return _invalid("conflicting_representations",
                        (FLAG_CONFLICTING_INPUTS,))

    gidx = gold_indices(candidates, answer)

    eff_order: list[int]
    if scores is not None:
        if not isinstance(scores, (list, tuple)) or len(scores) != n:
            return _invalid("scores_length_mismatch", (FLAG_SCORES_LENGTH,))
        vals: list[float] = []
        flags: list[str] = []
        for s in scores:
            if isinstance(s, bool) or not isinstance(s, (int, float)):
                flags.append(FLAG_SCORE_TYPE)
                continue
            f = float(s)
            if not math.isfinite(f):
                flags.append(FLAG_NONFINITE_SCORES)
            vals.append(f)
        if flags:
            primary = ("score_not_a_real_number" if FLAG_SCORE_TYPE in flags
                       else "nonfinite_score")
            return _invalid(primary, tuple(flags))
        top = max(vals)
        leaders = [i for i in range(n) if vals[i] == top]
        if len(leaders) > 1:
            # distinct candidates share the top score -> ambiguous no
            # matter what array position says; permutation-invariant.
            return _invalid("ambiguous_top_tie", (FLAG_AMBIGUOUS_TIE,))
        eff_order = sorted(range(n), key=lambda i: (-vals[i], i))
    elif order is not None:
        if (not isinstance(order, (list, tuple))
                or len(order) != n
                or any(isinstance(i, bool) or not isinstance(i, int)
                       for i in order)
                or sorted(order) != list(range(n))):
            return _invalid("order_not_permutation",
                            (FLAG_ORDER_NOT_PERMUTATION,))
        eff_order = list(order)
    else:
        return _invalid("no_raw_prediction", (FLAG_NO_RAW_INPUTS,))

    if not gidx:
        return CorrectedScore(correct=False, rank_of_gold=-1,
                              flags=(FLAG_GOLD_ABSENT,), valid=True)

    rank = min(eff_order.index(g) for g in gidx)
    out_flags = set()
    if any(norm[i] == normalize_answer(answer)
           and str(candidates[i]).strip() != str(answer).strip()
           for i in gidx):
        out_flags.add(FLAG_NORMALIZED_MATCH)
    return CorrectedScore(correct=(rank == 0), rank_of_gold=rank,
                          flags=tuple(sorted(out_flags)), valid=True)


@dataclass(frozen=True)
class RescoreOutcome:
    status: str       # LEGACY_RAW_RESCORED | MISSING_RAW_PREDICTION |
                      # NO_RECORDS | INVALID_RECORDS
    n_records: int = 0
    corrected_accuracy: float | None = None
    detail: str | None = None
    flag_counts: dict | None = None   # flag -> occurrences across records
    evidence_scope: str = field(default=LEGACY_EVIDENCE_SCOPE, init=False)
    current_evidence_eligible: bool = field(
        default=CURRENT_EVIDENCE_ELIGIBLE, init=False)


class _RecordError(Exception):
    pass


def rescore_records(records, examples_by_id) -> RescoreOutcome:
    """Salvage a legacy-only diagnostic over retained raw records.

    Fail-closed aggregation: ANY record that is malformed, references an
    unknown example, lacks a raw prediction, carries conflicting raw
    representations, or yields an invalid scorer verdict fails the whole
    file — a partial aggregate is never emitted, so invalid records can
    never produce ``LEGACY_RAW_RESCORED``. Record-level flags survive into
    ``flag_counts``. A successful result remains historical-only and cannot
    satisfy current-evidence gates.
    """
    if not isinstance(records, (list, tuple)):
        return RescoreOutcome(status=INVALID_RECORDS,
                              detail="records container is not a list")
    if not records:
        return RescoreOutcome(status=NO_RECORDS)

    hits = 0
    flag_counts: dict[str, int] = {}
    problems: list[str] = []

    def note_flag(f: str) -> None:
        flag_counts[f] = flag_counts.get(f, 0) + 1

    def fail_invalid(reason: str) -> RescoreOutcome:
        shown = problems[:MAX_DETAIL_PROBLEMS]
        detail = "; ".join(shown + ([reason] if reason else []))
        return RescoreOutcome(status=INVALID_RECORDS,
                              n_records=len(records),
                              detail=detail or None,
                              flag_counts=dict(flag_counts))

    for idx, rec in enumerate(records):
        try:
            if not isinstance(rec, dict):
                raise _RecordError("record is not an object")
            ex_id = rec.get("ex_id")
            ex = examples_by_id.get(ex_id) if isinstance(ex_id, str) else None
            if ex is None:
                return RescoreOutcome(
                    status=MISSING_RAW_PREDICTION, n_records=len(records),
                    detail=f"example {ex_id!r} absent from suite")
            cands = tuple(ex["candidates"]) if isinstance(ex, dict) \
                else tuple(ex.candidates)
            ans = ex["answer"] if isinstance(ex, dict) else ex.answer

            reps = [name for name in ("candidate_scores",
                                      "ranked_candidates") if name in rec]
            if len(reps) > 1:
                note_flag(FLAG_CONFLICTING_INPUTS)
                raise _RecordError(
                    f"conflicting raw representations {sorted(reps)}")
            if "candidate_scores" in rec:
                cs = corrected_score(cands, ans, scores=rec["candidate_scores"])
            elif "ranked_candidates" in rec:
                rc = rec["ranked_candidates"]
                if not isinstance(rc, (list, tuple)):
                    raise _RecordError("ranked_candidates is not a list")
                by_norm: dict[str, int] = {}
                for i, c in enumerate(cands):
                    by_norm.setdefault(normalize_answer(c), i)
                order: list[int] = []
                for x in rc:
                    key = normalize_answer(x)
                    if key not in by_norm:
                        raise _RecordError(
                            f"ranked candidate {x!r} not in candidate set")
                    order.append(by_norm[key])
                cs = corrected_score(cands, ans, order=order)
            elif "predicted_answer" in rec:
                pred = normalize_answer(rec["predicted_answer"])
                cs = CorrectedScore(correct=(pred == normalize_answer(ans)),
                                    rank_of_gold=-1)
            else:
                return RescoreOutcome(
                    status=MISSING_RAW_PREDICTION, n_records=len(records),
                    detail="records carry only derived correct/rank_of_gold")
            if not cs.valid:
                for f in cs.flags:
                    note_flag(f)
                raise _RecordError(
                    f"invalid scorer verdict: {cs.invalid_reason}")
            for f in cs.flags:
                note_flag(f)
            hits += int(cs.correct)
        except _RecordError as e:
            problems.append(f"record[{idx}]: {e}")
        except Exception as e:  # noqa: BLE001 — any surprise is invalidity
            problems.append(f"record[{idx}]: {type(e).__name__}: {e}")

    if problems:
        return fail_invalid("")
    return RescoreOutcome(status=LEGACY_RAW_RESCORED,
                          n_records=len(records),
                          corrected_accuracy=round(hits / len(records), 6),
                          flag_counts=dict(flag_counts) or None)


@dataclass(frozen=True)
class SelectionResult:
    step: int
    metric: float
    n_considered: int
    n_rejected_nonfinite: int
    provenance: str = "recomputed_from_history"


def select_best_checkpoint(history, *, metric_key: str = "accuracy") -> SelectionResult | None:
    """Deterministically audit a legacy best-validation entry.

    Ignores stored best_* fields entirely (they may be poisoned by a broken
    scorer); rejects entries whose metric is not a finite real number or
    whose step is not a non-negative integer (bool/float steps are
    rejected, never coerced); breaks accuracy ties toward the EARLIEST
    step so later training cannot win on noise. Returns None when nothing
    valid remains. This diagnostic never establishes current checkpoint
    selection provenance; current selection lives in ``eval_v3``.
    """
    considered, rejected = [], 0
    for h in history or ():
        if not isinstance(h, dict):
            rejected += 1
            continue
        step = h.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            rejected += 1
            continue
        raw = h.get(metric_key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            rejected += 1
            continue
        v = float(raw)
        if not math.isfinite(v):
            rejected += 1
            continue
        considered.append((v, step))
    if not considered:
        return None
    best_v = max(v for v, _ in considered)
    best_s = min(s for v, s in considered if v == best_v)
    return SelectionResult(step=best_s, metric=best_v,
                           n_considered=len(considered),
                           n_rejected_nonfinite=rejected)
