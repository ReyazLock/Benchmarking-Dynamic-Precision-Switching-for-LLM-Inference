import pytest

from apbench.metrics import contains_reference, exact_match, rouge_l, token_f1


def test_normalized_exact_match():
    assert exact_match("The Paris!", "Paris") == 1.0


def test_contains_reference():
    assert contains_reference("The answer is 42.", "42") == 1.0


def test_token_f1_partial_overlap():
    assert token_f1("red green", "red blue") == pytest.approx(0.5)


def test_rouge_l_sequence_order():
    assert rouge_l("red b c", "red x c") == pytest.approx(2 / 3)

def test_rouge_l_no_overlap():
    assert rouge_l("apple", "banana") == 0.0