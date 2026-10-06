import math

import pytest

from src.evals.ranking_metrics import agreement, percentile, ranking_metrics


def test_known_ranking_and_duplicates():
    result = ranking_metrics(["wrong", "a", "a", "b"], {"a", "b"})
    assert result["recall@1"] == 0
    assert result["recall@5"] == 1
    assert result["mrr"] == 0.5
    assert result["ndcg@10"] == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))
    assert all(value == 0 for value in ranking_metrics([], {"a"}).values())
    assert all(value == 0 for value in ranking_metrics(["a"], set()).values())


def test_nearest_rank_percentiles():
    assert percentile(list(range(1, 101)), 0.5) == 50
    assert percentile(list(range(1, 101)), 0.95) == 95
    assert percentile([3], 0) == percentile([3], 1) == 3
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_human_agreement_is_not_fabricated_for_missing_labels():
    assert agreement([True, False], [True, False])["cohen_kappa"] == 1
    assert agreement([True, False], [False, True])["cohen_kappa"] == -1
    assert agreement([True], [True])["cohen_kappa"] is None
    with pytest.raises(ValueError):
        agreement([None], [True])
