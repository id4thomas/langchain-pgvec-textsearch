"""PGVecTextSearch VectorStore package."""
from .pgvec_textsearch import PGVecTextSearchStore
from .engine import PGVecTextSearchEngine
from .hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
    weighted_sum_ranking,
)
from .indexes import (
    DistanceStrategy,
    QueryOptions,
    BaseIndex,
    HNSWIndex,
    IVFFlatIndex,
    ExactNearestNeighbor,
    BM25Index,
)
from .filters import (
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    build_filter_clause,
)

# Re-export from reference for convenience
from .reference.engine import PGEngine, Column, ColumnDict

__all__ = [
    # Main classes
    "PGVecTextSearchStore",
    "PGVecTextSearchEngine",
    # Configuration
    "HybridSearchConfig",
    "reciprocal_rank_fusion",
    "weighted_sum_ranking",
    # Indexes
    "DistanceStrategy",
    "QueryOptions",
    "BaseIndex",
    "HNSWIndex",
    "IVFFlatIndex",
    "ExactNearestNeighbor",
    "BM25Index",
    # Filters
    "FilterOperator",
    "FilterCondition",
    "MetadataFilter",
    "MetadataFilters",
    "build_filter_clause",
    # Engine helpers
    "PGEngine",
    "Column",
    "ColumnDict",
]
