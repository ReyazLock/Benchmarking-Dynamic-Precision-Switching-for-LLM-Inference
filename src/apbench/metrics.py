from __future__ import annotations

import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Callable

# --- Tunable settings -------------------------------------------------------
STEM_PLURALS = True          # "cells" matches "cell"; applied to both sides
CONCISE_EXTRA_TOKENS = 6     # concise_hit allows this many tokens beyond the reference
_NEGATORS = {"not", "no", "never", "isn", "aren", "wasn", "weren", "didn", "doesn",
             "don", "cannot", "without"}

# --- Normalisation ----------------------------------------------------------
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _canon_token(token: str) -> str:
    """Canonical form of one token: numbers by value, optional plural stripping."""
    if _NUMBER.fullmatch(token):
        try:
            return format(Decimal(token).normalize(), "f")   # 14.0 -> 14, 0.50 -> 0.5
        except InvalidOperation:
            return token
    if STEM_PLURALS and token.isalpha() and len(token) > 3 \
            and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", value)       # 1,100 -> 1100
    value = re.sub(r"(?<=\d)\.(?=\d)", "\x00", value)           # protect decimals
    value = "".join(" " if c in string.punctuation else c for c in value)
    value = value.replace("\x00", ".")
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(_canon_token(t) for t in value.split())


def _tokens(value: str) -> list[str]:
    return normalize_answer(value).split()


# --- Answer extraction ------------------------------------------------------
_QUESTION_CUT = re.compile(r"(?:^|\s)question\s*:", re.I)
_ANSWER_LINE = re.compile(r"^[ \t>*_]*(?:final |short )?answer[ \t]*:\s*(\S.*)$", re.I | re.M)
_INLINE_ANSWER = re.compile(r"\s(?:final |short )?answer\s*:", re.I)


def extract_answer(prediction: str) -> str:
    """The model's first answer: text before any invented follow-up question.

    1. Cut everything from the first "Question:" onward.
    2. If a line starts with "Answer:" (also "Final/Short answer:"), use its content.
    3. Otherwise use the first non-empty line.
    4. Drop anything after an inline "Answer:" label.
    """
    head = _QUESTION_CUT.split(prediction.strip())[0].strip()
    if not head:
        return ""
    match = _ANSWER_LINE.search(head)
    span = match.group(1) if match else head.splitlines()[0]
    return _INLINE_ANSWER.split(span)[0].strip()


# --- Matching helpers -------------------------------------------------------
def _references(reference: str) -> list[list[str]]:
    """Several acceptable answers can be written as 'Paris || City of Paris'."""
    parts = [p for p in (s.strip() for s in reference.split("||")) if p]
    return [t for t in (_tokens(p) for p in parts) if t]


def _find_phrase(pred: list[str], ref: list[str]) -> int:
    n = len(ref)
    for i in range(len(pred) - n + 1):
        if pred[i:i + n] == ref:
            return i
    return -1


def _is_hedged(pred: list[str], start: int, n: int) -> bool:
    """Reject 'not April', 'April or May' style answers."""
    before = pred[max(0, start - 3):start]
    if any(t in _NEGATORS for t in before):
        return True
    if start > 0 and pred[start - 1] == "or":
        return True
    return start + n < len(pred) and pred[start + n] == "or"


def _contains(pred: list[str], reference: str, guard: bool) -> bool:
    for ref in _references(reference):
        i = _find_phrase(pred, ref)
        if i >= 0 and not (guard and _is_hedged(pred, i, len(ref))):
            return True
    return False


def _f1(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return float(left == right)
    overlap = sum((Counter(left) & Counter(right)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(left), overlap / len(right)
    return 2 * precision * recall / (precision + recall)


def _lcs_f1(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return float(left == right)
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_token == right_token
                           else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(left), lcs / len(right)
    return 2 * precision * recall / (precision + recall)


# --- Metrics (all return floats) -------------------------------------------
def exact_match(prediction: str, reference: str) -> float:
    """Answer span equals a reference after normalisation."""
    span = _tokens(extract_answer(prediction))
    return float(any(span == ref for ref in _references(reference)))


def contains_reference(prediction: str, reference: str) -> float:
    """Correctness: reference appears as whole tokens in the answer span, not hedged."""
    return float(_contains(_tokens(extract_answer(prediction)), reference, guard=True))


def contains_reference_anywhere(prediction: str, reference: str) -> float:
    """Diagnostic: reference appears anywhere in the whole output (no hedge guard)."""
    return float(_contains(_tokens(prediction), reference, guard=False))


def concise_hit(prediction: str, reference: str) -> float:
    """Correct and short: contains_reference, span at most N tokens longer than reference."""
    span = _tokens(extract_answer(prediction))
    refs = _references(reference)
    if not refs or not _contains(span, reference, guard=True):
        return 0.0
    return float(len(span) <= min(len(r) for r in refs) + CONCISE_EXTRA_TOKENS)


def empty_answer(prediction: str, reference: str) -> float:
    """1.0 when no answer span could be extracted (model gave nothing usable)."""
    return float(not _tokens(extract_answer(prediction)))


def token_f1(prediction: str, reference: str) -> float:
    """Token F1 of the answer span against the best-matching reference."""
    span = _tokens(extract_answer(prediction))
    refs = _references(reference) or [[]]
    return max(_f1(span, ref) for ref in refs)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 of the answer span against the best-matching reference."""
    span = _tokens(extract_answer(prediction))
    refs = _references(reference) or [[]]
    return max(_lcs_f1(span, ref) for ref in refs)


METRICS: dict[str, Callable[[str, str], float]] = {
    # The extracted answer segment is exactly the same as the standardized token list of any of the reference answers
    # For strict accuracy rate of short answers
    "exact_match": exact_match,

    # Find the complete and continuous reference answer tokens in the answer fragment and conduct a negative/hesitation check
    # Allow for a relatively high hit rate with some explanatory text
    "contains_reference": contains_reference,

    # Find consecutive reference answer tokens in the complete output; do not check for negations
    # Whether the correct answer has ever appeared or not, should not be directly regarded as the accuracy rate
    "contains_reference_anywhere": contains_reference_anywhere,

    # First, ensure that contains_reference is met, then require that the answer length does not exceed "the shortest reference answer length + 6 tokens"
    "concise_hit": concise_hit,
    "empty_answer": empty_answer,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
}


def evaluate(prediction: str, reference: str, names: tuple[str, ...]) -> dict[str, float]:
    unknown = set(names) - METRICS.keys()
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
    return {name: METRICS[name](prediction, reference) for name in names}
