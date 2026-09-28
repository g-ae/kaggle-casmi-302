"""MRR@25, as in the competition (answers and predictions compared as InChIKey14)."""

from collections.abc import Mapping, Sequence

K = 25


def reciprocal_rank(ranked: Sequence[str], answer: str, k: int = K) -> float:
    for rank, key in enumerate(ranked[:k], start=1):
        if key == answer:
            return 1.0 / rank
    return 0.0


def mrr_at_k(predictions: Mapping[str, Sequence[str]], answers: Mapping[str, str], k: int = K) -> float:
    """predictions: molecule_id -> ranked InChIKey14s; answers: molecule_id -> InChIKey14.

    Molecules without a prediction score 0 (the real metric rejects such submissions).
    """
    if not answers:
        return 0.0
    return sum(reciprocal_rank(predictions.get(m, []), a, k) for m, a in answers.items()) / len(answers)
