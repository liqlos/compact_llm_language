import pytest

from rcc import count_tokens
from rcc.tokens import tokenizer_identity


def test_deterministic():
    s = "PostgreSQL 16.3 released 2024-05-09; p95=142ms"
    assert count_tokens(s) == count_tokens(s)


def test_monotonic_in_length():
    assert count_tokens("a b c") <= count_tokens("a b c d e f g h")


def test_empty():
    assert count_tokens("") == 0


def test_structured_text_counts_more_than_prose_per_char():
    # equal-length inputs: punctuation/digits tokenize denser than prose
    structured = "a|b|c|d|e|f|" * 10          # 120 chars -> dense tokens
    prose = "abcd efgh ijkl mnop " * 6        # 120 chars -> sparse tokens
    assert len(structured) == len(prose)
    assert count_tokens(structured) > count_tokens(prose)


def test_numbers_digit_by_digit():
    assert count_tokens("12345") == 5


def test_tokenizer_identity_is_stable_and_known_counter_cannot_be_mislabeled():
    assert tokenizer_identity(count_tokens) == "rcc-approx-1"
    with pytest.raises(ValueError, match="conflicts with known counter"):
        tokenizer_identity(count_tokens, "custom:versioned:v1")
