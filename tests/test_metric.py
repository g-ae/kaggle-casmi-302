import pytest

from casmi_flyte.config import neutral_mass
from casmi_flyte.metric import mrr_at_k, reciprocal_rank


def test_reciprocal_rank():
    ranked = [f"K{i}" for i in range(30)]
    assert reciprocal_rank(ranked, "K0") == 1.0
    assert reciprocal_rank(ranked, "K1") == 0.5
    assert reciprocal_rank(ranked, "K24") == pytest.approx(0.04)
    assert reciprocal_rank(ranked, "K25") == 0.0  # beyond the top 25
    assert reciprocal_rank(ranked, "missing") == 0.0


def test_mrr():
    answers = {"m1": "A", "m2": "B", "m3": "C"}
    predictions = {"m1": ["A", "X"], "m2": ["X", "B"]}  # m3 missing -> 0
    assert mrr_at_k(predictions, answers) == pytest.approx((1 + 0.5 + 0) / 3)


def test_neutral_mass():
    # caffeine C8H10N4O2, monoisotopic 194.080376
    assert neutral_mass(195.087652, "[M+H]+") == pytest.approx(194.080376, abs=1e-5)
    assert neutral_mass(193.073100, "[M-H]-") == pytest.approx(194.080376, abs=1e-5)
    assert neutral_mass(217.069594, "[M+Na]+") == pytest.approx(194.080376, abs=1e-5)
