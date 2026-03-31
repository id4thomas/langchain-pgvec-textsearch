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


def weighted_sum_ranking(
    dense_results: Sequence[RowMapping],
    sparse_results: Sequence[RowMapping],
    *,
    id_column: str,
    fetch_top_k: int,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Combine dense and sparse results using Weighted Sum Ranking.

    Normalizes scores from each result list to [0, 1] using min-max normalization,
    then combines them with weights. Higher weighted score = more relevant.

    Dense results use 'distance' (lower = better), so the scale is inverted.
    Sparse results use 'bm25_score' (negative BM25, lower = better), also inverted.

    Args:
        dense_results: Results from dense vector search (ordered by distance).
        sparse_results: Results from sparse BM25 search (ordered by bm25_score).
        id_column: Name of the ID column.
        fetch_top_k: Number of top results to return.
        dense_weight: Weight for dense results (default 0.7).
        sparse_weight: Weight for sparse results (default 0.3).

    Returns:
        Fused results sorted by weighted score descending.
    """
    scored: dict[str, dict[str, Any]] = {}

    # Normalize dense scores (distance: lower = better → invert)
    dense_list = [dict(row) for row in dense_results]
    if dense_list:
        scores = [row["distance"] for row in dense_list]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s if max_s != min_s else 1.0

        for item in dense_list:
            normalized = (item["distance"] - min_s) / rng
            doc_id = str(item[id_column])
            item["weighted_score"] = (1.0 - normalized) * dense_weight
            scored[doc_id] = item

    # Normalize sparse scores (bm25_score: lower = better → invert)
    sparse_list = [dict(row) for row in sparse_results]
    if sparse_list:
        scores = [row["bm25_score"] for row in sparse_list]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s if max_s != min_s else 1.0

        for item in sparse_list:
            normalized = (item["bm25_score"] - min_s) / rng
            sparse_score = (1.0 - normalized) * sparse_weight
            doc_id = str(item[id_column])
            if doc_id in scored:
                scored[doc_id]["weighted_score"] += sparse_score
            else:
                item["weighted_score"] = sparse_score
                scored[doc_id] = item

    ranked = sorted(
        scored.values(),
        key=lambda x: x["weighted_score"],
        reverse=True,
    )

    return ranked[:fetch_top_k]
