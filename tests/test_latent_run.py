"""Deterministic scoring tests for latent_run.evaluate.

Gold must be located by answer identity (ex.candidates.index(ex.answer)),
never by assuming candidate index 0 is gold. All fixtures place gold at a
NONZERO candidate position.
"""

from types import SimpleNamespace

from latent_lab.bench.latent_run import evaluate
from latent_lab.bench.suite import Example


class _Ids:
    """Minimal stand-in for a tokenizer tensor: supports .to(device)."""

    def __init__(self, ids):
        self.ids = list(ids)

    def to(self, device):
        return self


class _Encoded:
    def __init__(self, ids):
        self.input_ids = _Ids(ids)


class _StubTok:
    """Whitespace-token -> stable int ids; no external deps."""

    def __init__(self):
        self._vocab = {}

    def __call__(self, text, return_tensors=None, return_dict=False,
                 add_special_tokens=True):
        out = []
        for w in text.split():
            if w not in self._vocab:
                self._vocab[w] = len(self._vocab) + 1
            out.append(self._vocab[w])
        return _Encoded(out)


class _FakeRec:
    """rank_candidates returns a scripted per-example candidate order."""

    def __init__(self, order_by_prompt):
        self.order_by_prompt = order_by_prompt
        self.model = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device="cpu")]))

    def rank_candidates(self, input_ids, candidate_ids, k_steps, *,
                        ablate=None, partner_input_ids=None):
        key = tuple(input_ids.ids)
        assert key in self.order_by_prompt, "unexpected prompt scored"
        n = len(candidate_ids)
        order = self.order_by_prompt[key]
        assert sorted(order) == list(range(n))
        return list(order), [0.0] * n, SimpleNamespace()


def _example(ex_id, answer, candidates):
    return Example(
        ex_id=ex_id, family="fsm", depth=3,
        prompt=f"prompt {ex_id} Answer:", answer=answer,
        candidates=tuple(candidates), content_key=f"ck:{ex_id}")


def _build(examples, orders):
    tok = _StubTok()
    prompts = []
    from latent_lab.bench.latent_run import SuiteTensors
    data = SuiteTensors(tok, examples)
    prompts = [tuple(t.ids) for t in data.prompt_ids]
    scripted = dict(zip(prompts, orders))
    rec = _FakeRec(scripted)
    return rec, data


def test_gold_at_nonzero_position_ranked_top_is_correct():
    # candidates ("b", "a", "c"), gold "a" at index 1; model ranks it top
    ex = _example("e1", "a", ("b", "a", "c"))
    rec, data = _build([ex], [[1, 0, 2]])
    out = evaluate(rec, data, k_steps=2, indices=[0])
    r = out["records"][0]
    assert r["correct"] == 1.0
    assert r["rank_of_gold"] == 0
    assert out["accuracy"] == 1.0
    assert r["n_candidates"] == 3


def test_gold_at_nonzero_position_not_top_is_incorrect():
    # same example; model ranks candidate 2 top, gold "a" lands at rank 2
    ex = _example("e1", "a", ("b", "a", "c"))
    rec, data = _build([ex], [[2, 0, 1]])
    out = evaluate(rec, data, k_steps=2, indices=[0])
    r = out["records"][0]
    assert r["correct"] == 0.0
    assert r["rank_of_gold"] == 2
    assert out["accuracy"] == 0.0


def test_mixed_orders_aggregate_and_per_example_records():
    # two examples, both with gold at index 1; one ranked top, one ranked last
    exs = [
        _example("e1", "x", ("y", "x", "z")),
        _example("e2", "q", ("r", "q", "p")),
    ]
    rec, data = _build(exs, [[1, 0, 2], [2, 1, 0]])
    out = evaluate(rec, data, k_steps=2, indices=[0, 1])
    by_ex = {r["ex_id"]: r for r in out["records"]}
    assert by_ex["e1"]["rank_of_gold"] == 0 and by_ex["e1"]["correct"] == 1.0
    assert by_ex["e2"]["rank_of_gold"] == 1 and by_ex["e2"]["correct"] == 0.0
    assert out["accuracy"] == round(1 / 2, 4)


def test_candidate_zero_ranked_first_with_nonzero_gold_is_wrong():
    # old defect would score this as correct: index 0 is top-ranked but
    # is NOT the gold answer
    ex = _example("e1", "gold", ("decoy", "gold", "other"))
    rec, data = _build([ex], [[0, 1, 2]])
    out = evaluate(rec, data, k_steps=2, indices=[0])
    r = out["records"][0]
    assert r["correct"] == 0.0
    assert r["rank_of_gold"] == 1
