"""Hybrid search configuration for pg_textsearch (BM25 + pgvector)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from sqlalchemy import RowMapping

from .indexes import DistanceStrategy


def reciprocal_rank_fusion(
    dense_results: Sequence[RowMapping],
    sparse_results: Sequence[RowMapping],
    rrf_k: float = 60,
    fetch_top_k: int = 4,
    **kwargs: Any,
) -> Sequence[dict[str, Any]]:
    """
    Ranks documents using Reciprocal Rank Fusion (RRF) from dense and sparse search.

    For pg_textsearch hybrid search:
    - Dense results: pgvector HNSW search (lower distance = better for cosine/euclidean)
    - Sparse results: pg_textsearch BM25 search (lower <@> score = better, negated BM25)

    Args:
        dense_results: Results from pgvector dense search.
        sparse_results: Results from pg_textsearch BM25 sparse search.
        rrf_k: The RRF parameter k. Default: 60.
        fetch_top_k: Number of documents to return. Default: 4.

    Returns:
        A list of merged results sorted by RRF score in descending order.
    """
    distance_strategy = kwargs.get(
        "distance_strategy", DistanceStrategy.COSINE_DISTANCE
    )
    id_column = kwargs.get("id_column", "langchain_id")
    rrf_scores: dict[str, dict[str, Any]] = {}

    # Dense search: lower distance is better for COSINE/EUCLIDEAN
    # For INNER_PRODUCT, higher is better
    is_similarity_metric = distance_strategy == DistanceStrategy.INNER_PRODUCT
    sorted_dense = sorted(
        dense_results,
        key=lambda item: item["distance"],
        reverse=is_similarity_metric,
    )

    for rank, row in enumerate(sorted_dense):
        doc_id = str(row[id_column])
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = dict(row)
            rrf_scores[doc_id]["rrf_score"] = 0.0
        rrf_scores[doc_id]["rrf_score"] += 1.0 / (rank + 1 + rrf_k)

    # Sparse search (BM25): <@> returns negative BM25 score, lower is better
    # So we sort ascending (reverse=False)
    sorted_sparse = sorted(
        sparse_results,
        key=lambda item: item["bm25_score"],
        reverse=False,  # Lower <@> score = higher BM25 relevance
    )

    for rank, row in enumerate(sorted_sparse):
        doc_id = str(row[id_column])
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = dict(row)
            rrf_scores[doc_id]["rrf_score"] = 0.0
        rrf_scores[doc_id]["rrf_score"] += 1.0 / (rank + 1 + rrf_k)

    # Sort by RRF score descending
    ranked_results = sorted(
        rrf_scores.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:fetch_top_k]


def weighted_sum_ranking(
    dense_results: Sequence[RowMapping],
    sparse_results: Sequence[RowMapping],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    fetch_top_k: int = 4,
    **kwargs: Any,
) -> Sequence[dict[str, Any]]:
    """
    Ranks documents using weighted sum of normalized scores.

    Args:
        dense_results: Results from pgvector dense search.
        sparse_results: Results from pg_textsearch BM25 sparse search.
        dense_weight: Weight for dense search scores. Default: 0.5.
        sparse_weight: Weight for sparse search scores. Default: 0.5.
        fetch_top_k: Number of documents to return. Default: 4.

    Returns:
        A list of merged results sorted by weighted score in descending order.
    """
    id_column = kwargs.get("id_column", "langchain_id")
    distance_strategy = kwargs.get(
        "distance_strategy", DistanceStrategy.COSINE_DISTANCE
    )
    is_distance_metric = distance_strategy != DistanceStrategy.INNER_PRODUCT

    weighted_scores: dict[str, dict[str, Any]] = {}

    # Normalize dense scores
    dense_list = [dict(row) for row in dense_results]
    if dense_list:
        scores = [row["distance"] for row in dense_list]
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        for item in dense_list:
            normalized = (item["distance"] - min_score) / score_range
            if is_distance_metric:
                item["normalized_score"] = (1.0 - normalized) * dense_weight
            else:
                item["normalized_score"] = normalized * dense_weight
            doc_id = str(item[id_column])
            weighted_scores[doc_id] = item
            weighted_scores[doc_id]["weighted_score"] = item["normalized_score"]

    # Normalize sparse scores (BM25: lower <@> = better)
    sparse_list = [dict(row) for row in sparse_results]
    if sparse_list:
        scores = [row["bm25_score"] for row in sparse_list]
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        for item in sparse_list:
            # Lower <@> score is better, so invert
            normalized = (item["bm25_score"] - min_score) / score_range
            item["normalized_score"] = (1.0 - normalized) * sparse_weight
            doc_id = str(item[id_column])
            if doc_id in weighted_scores:
                weighted_scores[doc_id]["weighted_score"] += item["normalized_score"]
            else:
                weighted_scores[doc_id] = item
                weighted_scores[doc_id]["weighted_score"] = item["normalized_score"]

    ranked_results = sorted(
        weighted_scores.values(),
        key=lambda item: item["weighted_score"],
        reverse=True,
    )

    return ranked_results[:fetch_top_k]


@dataclass
class HybridSearchConfig:
    """
    Hybrid search configuration for pg_textsearch.

    This configuration enables combining:
    - pgvector dense search (HNSW index)
    - pg_textsearch sparse BM25 search (BM25 index)

    Using RRF (Reciprocal Rank Fusion) or weighted sum for score fusion.
    """

    # BM25 text configuration (e.g., 'public.korean', 'english')
    text_config: str = "public.korean"

    # BM25 index name (required for pg_textsearch with prepared statements)
    # If None, will use "idx_{table_name}_bm25"
    bm25_index_name: Optional[str] = None

    # Fusion function to combine dense and sparse results
    fusion_function: Callable[
        [Sequence[RowMapping], Sequence[RowMapping], Any], Sequence[Any]
    ] = reciprocal_rank_fusion

    # Parameters for the fusion function
    fusion_function_parameters: dict[str, Any] = field(default_factory=lambda: {"rrf_k": 60})

    # Number of candidates to fetch from each search before fusion
    dense_top_k: int = 20
    sparse_top_k: int = 20

    # Enable/disable each search type
    enable_dense: bool = True
    enable_sparse: bool = True

    # BM25 index parameters (used when creating index)
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    def __post_init__(self):
        if not self.enable_dense and not self.enable_sparse:
            raise ValueError("At least one of enable_dense or enable_sparse must be True")
