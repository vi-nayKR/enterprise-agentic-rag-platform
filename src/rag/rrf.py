from typing import List, Dict, Any, Union
from src.rag.models import SearchResult

def reciprocal_rank_fusion(
    dense_results: List[Union[SearchResult, Dict[str, Any]]],
    sparse_results: List[Union[SearchResult, Dict[str, Any]]],
    k: int = 60
) -> List[SearchResult]:
    """
    Fuses dense vector search rankings and sparse lexical (BM25) search rankings
    using the Reciprocal Rank Fusion (RRF) algorithm:
    RRF_score(d) = sum(1.0 / (k + rank_i(d)))
    """
    scores: Dict[str, float] = {}
    obj_map: Dict[str, SearchResult] = {}

    def _to_search_result(item: Union[SearchResult, Dict[str, Any]]) -> SearchResult:
        if isinstance(item, SearchResult):
            return item
        return SearchResult(
            chunk_id=item.get("id") or item.get("chunk_id", "unknown"),
            document_id=item.get("document_id", "unknown"),
            text=item.get("text", ""),
            dense_score=float(item.get("score", 0.0) or item.get("dense_score", 0.0)),
            sparse_score=float(item.get("sparse_score", 0.0)),
            metadata=item.get("metadata", {})
        )

    # Process dense results
    for rank, item in enumerate(dense_results, start=1):
        res = _to_search_result(item)
        doc_id = res.chunk_id
        obj_map[doc_id] = res
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Process sparse results
    for rank, item in enumerate(sparse_results, start=1):
        res = _to_search_result(item)
        doc_id = res.chunk_id
        if doc_id not in obj_map:
            obj_map[doc_id] = res
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sort fused results by combined RRF score descending
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_results: List[SearchResult] = []
    for doc_id, score in sorted_docs:
        item = obj_map[doc_id]
        item.rrf_score = round(float(score), 6)
        fused_results.append(item)

    return fused_results
