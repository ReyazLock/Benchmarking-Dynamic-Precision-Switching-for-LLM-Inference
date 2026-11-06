from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", value)       # 1,100 -> 1100
    value = re.sub(r"(?<=\d)\.(?=\d)", "\x00", value)           # protect decimals
    value = "".join(" " if c in string.punctuation else c for c in value)
    value = value.replace("\x00", ".")
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())

def extract_answer(prediction: str) -> str:
    text = re.sub(r"^\s*(final\s+)?answer\s*:\s*", "", prediction, flags=re.I)
    return re.split(r"\n|question\s*:|answer\s*:", text, flags=re.I)[0].strip()

def _has_phrase(pred_tokens, ref_tokens):
    n = len(ref_tokens)
    return n > 0 and any(pred_tokens[i:i+n] == ref_tokens
                         for i in range(len(pred_tokens) - n + 1))

def contains_reference(prediction, reference):          # answer span, token boundaries
    return float(_has_phrase(normalize_answer(extract_answer(prediction)).split(),
                             normalize_answer(reference).split()))

def contains_reference_anywhere(prediction, reference): # whole output (diagnostic)
    return float(_has_phrase(normalize_answer(prediction).split(),
                             normalize_answer(reference).split()))

def exact_match(prediction, reference):                  # strict, answer span
    return float(normalize_answer(extract_answer(prediction)) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    left = normalize_answer(prediction).split()
    right = normalize_answer(reference).split()
    if not left or not right:
        return float(left == right)
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(left)
    recall = lcs / len(right)
    return 2 * precision * recall / (precision + recall)


METRICS: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "contains_reference": contains_reference,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
}


def evaluate(prediction: str, reference: str, names: tuple[str, ...]) -> dict[str, float]:
    unknown = set(names) - METRICS.keys()
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
    return {name: METRICS[name](prediction, reference) for name in names}

