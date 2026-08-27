"""Regression: special tokens must not corrupt answer parsing (4B gate bug)."""

from latent_lab.bench.text_baselines import (
    DEFAULT_MAX_NEW_TOKENS,
    MODE_DESCRIPTIONS,
    parse_answer,
    score_example,
)


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


def test_answer_marker_must_be_the_final_content_line():
    ex = _Ex("4", ("4", "5"))
    res = score_example("Answer: 4\nCorrection: 5<|im_end|>", ex)
    assert res["correct"] == 0.0


def test_im_start_is_not_a_termination_marker():
    ex = _Ex("4", ("4", "5"))
    res = score_example("Answer: 4<|im_start|>", ex)
    assert res["status"] == "NON_TERMINATION"
    assert res["correct"] == 0.0


def test_preregistered_text_arms_are_distinct_and_small_budgeted():
    assert DEFAULT_MAX_NEW_TOKENS["A"] == 64
    assert DEFAULT_MAX_NEW_TOKENS["C"] == 64
    assert "direct" in MODE_DESCRIPTIONS["A"]
    assert "visible scratch" in MODE_DESCRIPTIONS["C"]
