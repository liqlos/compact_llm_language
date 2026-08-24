"""Deterministic tests for gold scoring in latent_run.evaluate.

Gold candidate index must be derived from ex.candidates.index(ex.answer) and
rank_of_gold read off the returned order — never assuming candidate zero is
gold. Tests place gold at a nonzero candidate position.
"""

from types import SimpleNamespace

from latent_lab.bench.latent_run import evaluate
from latent_lab.bench.suite import Example


class _Ids:
    """Minimal stand-in for a token-id tensor (only .to is used)."""

    def to(self, device):
        return self


class _StubRec:
    """rank_candidates stub returning scripted candidate orders."""

    def __init__(self, orders):
        self.model = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device="cpu")]))
        self.orders = list(orders)

    def rank_candidates(self, input_ids, candidate_ids, k_steps, *,
                        ablate=None, partner_input_ids=None):
        order = self.orders.pop(0)
        scores = [float(len(order) - order.index(i)) for i in range(len(order))]
        return order, scores, None


def _ex(gold_pos, n_cands=4, ex_id="e0"):
    cands = tuple(f"c{i}" for i in range(n_cands))
    return Example(
        ex_id=ex_id, family="fsm", depth=3, prompt="p",
        answer=cands[gold_pos], candidates=cands, content_key="k",
    )


def _data(examples):
    return SimpleNamespace(
        examples=examples,
        prompt_ids=[_Ids() for _ in examples],
        cand_ids=[[_Ids() for _ in ex.candidates] for ex in examples],
    )


def test_gold_at_nonzero_position_top_ranked_is_correct():
    # Gold sits at candidate index 2 and the model ranks it first.
    ex = _ex(gold_pos=2)
    res = evaluate(_StubRec([[2, 0, 3, 1]]), _data([ex]), k_steps=1,
                   indices=[0], tag="t")
    assert len(res["records"]) == 1
    rec = res["records"][0]
    assert rec["correct"] == 1.0
    assert rec["rank_of_gold"] == 0
    assert rec["n_candidates"] == 4
    assert res["accuracy"] == 1.0


def test_gold_at_nonzero_position_non_top_ranked_is_incorrect():
    # Candidate 0 ranks first (old buggy assumption would score this as a
    # hit) but gold lives at candidate index 2 and lands at rank 1 only.
    ex = _ex(gold_pos=2)
    res = evaluate(_StubRec([[0, 2, 3, 1]]), _data([ex]), k_steps=1,
                   indices=[0], tag="t")
    rec = res["records"][0]
    assert rec["correct"] == 0.0
    assert rec["rank_of_gold"] == 1
    assert rec["n_candidates"] == 4
    assert res["accuracy"] == 0.0


def test_mixed_batch_aggregates_per_example_gold_position():
    # Two examples, both with gold at nonzero positions: one ranked second,
    # one whose gold lands at the last rank of a full permutation.
    exs = [_ex(gold_pos=1, ex_id="rank1"), _ex(gold_pos=3, ex_id="last")]
    orders = [[3, 1, 0, 2], [2, 1, 0, 3]]
    res = evaluate(_StubRec(orders), _data(exs), k_steps=1,
                   indices=[0, 1], tag="t")
    by_id = {r["ex_id"]: r for r in res["records"]}
    assert by_id["rank1"]["rank_of_gold"] == 1
    assert by_id["rank1"]["correct"] == 0.0
    assert by_id["last"]["rank_of_gold"] == 3
    assert by_id["last"]["correct"] == 0.0
    assert res["accuracy"] == 0.0

    # Same batch, orders flipped so each gold is ranked top: all correct.
    res2 = evaluate(_StubRec([[1, 3, 0, 2], [3, 0, 1, 2]]),
                    _data(exs), k_steps=1, indices=[0, 1], tag="t")
    assert res2["accuracy"] == 1.0
    assert all(r["rank_of_gold"] == 0 for r in res2["records"])


def test_records_retain_raw_scores_and_are_independently_rescorable():
    """Lossless raw evidence: scores_raw + score_order + candidate/gold
    identity must survive, and accuracy must be recomputable after an
    arbitrary consistent candidate permutation."""
    from latent_lab.bench.latent_run import rescore_records

    ex = _ex(gold_pos=2)
    res = evaluate(_StubRec([[0, 3, 2, 1]]), _data([ex]), k_steps=1,
                   indices=[0], tag="t")
    rec = res["records"][0]
    assert rec["scores_raw"] == [4.0, 1.0, 2.0, 3.0]
    assert rec["score_order"] == [0, 3, 2, 1]
    assert rec["candidates"] == ["c0", "c1", "c2", "c3"]
    assert rec["answer"] == "c2"
    assert rec["gold_candidate_index"] == 2

    # independent rescore agrees without touching derived fields first
    assert rescore_records([rec]) == 0.0

    # permute the candidate set consistently (scores/order/gold move with
    # the candidates): rescored accuracy is IDENTICAL — raw evidence is
    # sufficient to redo scoring under any corrected scorer
    m = {j: p for p, j in enumerate([3, 1, 0, 2])}  # old idx -> new position
    moved = {"candidates": [None] * 4, "scores_raw": [None] * 4}
    for j in range(4):
        moved["candidates"][m[j]] = rec["candidates"][j]
        moved["scores_raw"][m[j]] = rec["scores_raw"][j]
    moved["answer"] = rec["answer"]
    moved["gold_candidate_index"] = m[rec["gold_candidate_index"]]
    moved["score_order"] = [m[i] for i in rec["score_order"]]
    merged = {**rec, **moved}
    assert rescore_records([merged]) == rescore_records([rec])


def test_rescore_rejects_tampered_derived_fields():
    from latent_lab.bench.latent_run import rescore_records

    ex = _ex(gold_pos=2)
    res = evaluate(_StubRec([[0, 3, 2, 1]]), _data([ex]), k_steps=1,
                   indices=[0], tag="t")
    rec = dict(res["records"][0])
    assert rec["correct"] == 0.0
    tampered = {**rec, "correct": 1.0}  # claim correct against raw evidence
    import pytest
    with pytest.raises(ValueError):
        rescore_records([tampered])
    tampered_order = {**rec, "score_order": [1, 0, 2, 3]}
    with pytest.raises(ValueError):
        rescore_records([tampered_order])


# ---------------------------------------------------------------------------
# gold identity is re-derived from answer/candidates — a supplied index is
# never trusted (missing / duplicated / substituted gold fails closed)
# ---------------------------------------------------------------------------

def _gold_pos2_record():
    ex = _ex(gold_pos=2)
    return dict(evaluate(_StubRec([[0, 3, 2, 1]]), _data([ex]), k_steps=1,
                         indices=[0], tag="t")["records"][0])


def test_rescore_rejects_substituted_gold_index_even_when_self_consistent():
    """Stealthy substitution: the persisted index names a DIFFERENT
    candidate while rank_of_gold/correct/accuracy are rewritten to match
    it — index-trusting scoring would accept this; deriving gold from
    answer/candidates must not."""
    import pytest

    from latent_lab.bench.latent_run import rescore_records

    rec = _gold_pos2_record()          # candidates c0..c3, answer c2
    assert rec["answer"] == "c2"
    rec["gold_candidate_index"] = 0    # substitute the top-scored rival
    rec["rank_of_gold"] = 0            # ... and stay internally consistent
    rec["correct"] = 1.0
    with pytest.raises(ValueError, match="gold_candidate_index"):
        rescore_records([rec])


def test_rescore_rejects_missing_and_duplicated_gold():
    import pytest

    from latent_lab.bench.latent_run import rescore_records

    rec = _gold_pos2_record()
    missing = {**rec, "answer": "nope"}
    with pytest.raises(ValueError, match="missing from"):
        rescore_records([missing])

    dup = {**_gold_pos2_record(),
           "candidates": ["c0", "c1", "c2", "c2"],
           "scores_raw": [4.0, 1.0, 3.0, 2.0],
           "score_order": [0, 2, 3, 1]}
    with pytest.raises(ValueError, match="duplicated"):
        rescore_records([dup])


def test_rescore_rejects_missing_answer_field_and_bad_index_types():
    import pytest

    from latent_lab.bench.latent_run import rescore_records

    rec = _gold_pos2_record()
    no_answer = {k: v for k, v in rec.items() if k != "answer"}
    with pytest.raises(ValueError, match="missing from"):
        rescore_records([no_answer])

    bad_idx = {**rec, "gold_candidate_index": "2"}
    with pytest.raises(ValueError, match="gold_candidate_index"):
        rescore_records([bad_idx])
