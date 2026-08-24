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
    # one whose gold never appears in the returned order.
    exs = [_ex(gold_pos=1, ex_id="rank1"), _ex(gold_pos=3, ex_id="absent")]
    orders = [[3, 1, 0, 2], [2, 1, 0]]
    res = evaluate(_StubRec(orders), _data(exs), k_steps=1,
                   indices=[0, 1], tag="t")
    by_id = {r["ex_id"]: r for r in res["records"]}
    assert by_id["rank1"]["rank_of_gold"] == 1
    assert by_id["rank1"]["correct"] == 0.0
    assert by_id["absent"]["rank_of_gold"] == -1
    assert by_id["absent"]["correct"] == 0.0
    assert res["accuracy"] == 0.0

    # Same batch, orders flipped so each gold is ranked top: all correct.
    res2 = evaluate(_StubRec([[1, 3, 0, 2], [3, 0, 1, 2]]),
                    _data(exs), k_steps=1, indices=[0, 1], tag="t")
    assert res2["accuracy"] == 1.0
    assert all(r["rank_of_gold"] == 0 for r in res2["records"])
