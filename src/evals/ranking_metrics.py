"""Binary source-chunk ranking metrics; no token-overlap quality proxies."""

import math
from collections.abc import Sequence


def ranking_metrics(ranked_ids: Sequence[str], relevant_ids: set[str]) -> dict[str, float]:
    ranked = list(dict.fromkeys(ranked_ids))
    hits = [int(chunk_id in relevant_ids) for chunk_id in ranked]
    first = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits[:10], 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant_ids), 10) + 1))
    return {
        **{f"recall@{k}": sum(hits[:k]) / len(relevant_ids) if relevant_ids else 0.0 for k in (1, 5, 10)},
        "mrr": 1 / first if first else 0.0,
        "ndcg@10": dcg / ideal if ideal else 0.0,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile, including both endpoint quantiles."""
    if not values or not 0 <= quantile <= 1:
        raise ValueError("Require samples and a quantile between zero and one")
    return sorted(values)[max(0, math.ceil(quantile * len(values)) - 1)]


def agreement(human: Sequence[bool], judge: Sequence[bool]) -> dict[str, float | int | None]:
    if not human or len(human) != len(judge):
        raise ValueError("Require equally sized, nonempty human and judge labels")
    if any(type(value) is not bool for value in (*human, *judge)):
        raise ValueError("Labels must be actual human/judge booleans, not missing or coercible values")
    observed = sum(a == b for a, b in zip(human, judge)) / len(human)
    p_human = sum(human) / len(human)
    p_judge = sum(judge) / len(judge)
    expected = p_human * p_judge + (1 - p_human) * (1 - p_judge)
    return {"label_count": len(human), "agreement": observed,
            "cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 else None}
