"""Regression: special tokens must not corrupt answer parsing (4B gate bug)."""

from latent_lab.bench.text_baselines import parse_answer, score_example


class _Ex:
    def __init__(self, answer, candidates):
        self.answer = answer
        self.candidates = candidates


def test_im_end_suffix_does_not_break_match():
    ans, status = parse_answer("step\nAnswer: maple<|im_end|>\n<|endoftext|>")
    assert (ans, status) == ("maple", "ok")


def test_scoring_counts_correct_with_special_tokens():
    ex = _Ex("4", ("4", "5"))
    assert score_example("...\nAnswer: 4<|im_end|>", ex)["correct"] == 1.0


def test_truncation_still_nontermination():
    ex = _Ex("4", ("4", "5"))
    res = score_example("Answer: 4 but then keeps going", ex)
    assert res["status"] == "NON_TERMINATION" and res["correct"] == 0.0
