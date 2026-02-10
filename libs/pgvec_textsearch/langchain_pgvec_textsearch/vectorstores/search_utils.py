"""Search utility functions (ranking, fusion)"""

from typing import Any, Sequence

from sqlalchemy import RowMapping


def reciprocal_rank_fusion(
    dense_results: Sequence[RowMapping],
    sparse_results: Sequence[RowMapping],
    *,
    id_column: str,
    fetch_top_k: int,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Combine dense and sparse results using Reciprocal Rank Fusion.

    RRF score = sum(weight / (k + rank)) for each result list.
    Higher RRF score = more relevant.

    Args:
        dense_results: Results from dense vector search (ordered by distance).
        sparse_results: Results from sparse BM25 search (ordered by bm25_score).
        id_column: Name of the ID column.
        fetch_top_k: Number of top results to return.
        rrf_k: RRF constant (default 60).
        dense_weight: Weight for dense results.
        sparse_weight: Weight for sparse results.

    Returns:
        Fused results sorted by RRF score descending.
    """
    # Build rank maps (rank 1 = best)
    dense_ranks: dict[str, int] = {}
    for rank, row in enumerate(dense_results, 1):
        dense_ranks[str(row[id_column])] = rank

    sparse_ranks: dict[str, int] = {}
    for rank, row in enumerate(sparse_results, 1):
        sparse_ranks[str(row[id_column])] = rank

    # Calculate RRF scores for all unique documents
    all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    scored: list[tuple[str, float]] = []
    for doc_id in all_ids:
        rrf_score = 0.0
        if doc_id in dense_ranks:
            rrf_score += dense_weight / (rrf_k + dense_ranks[doc_id])
        if doc_id in sparse_ranks:
            rrf_score += sparse_weight / (rrf_k + sparse_ranks[doc_id])
        scored.append((doc_id, rrf_score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Build result list preferring dense rows (has embedding)
    dense_map = {
        str(row[id_column]): dict(row) for row in dense_results
    }
    sparse_map = {
        str(row[id_column]): dict(row) for row in sparse_results
    }

    results: list[dict[str, Any]] = []
    for doc_id, rrf_score in scored[:fetch_top_k]:
        row_data = dense_map.get(doc_id) or sparse_map.get(doc_id, {})
        row_data["rrf_score"] = rrf_score
        results.append(row_data)

    return results
