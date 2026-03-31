"""LangChain pgvec_textsearch integration."""

from .vectorstores import (
    # Main classes
    PGVecTextSearchStore,
    AsyncPGVecTextSearchEngine,
    # Table & Data
    TableConfig,
    Row,
    Column,
    ColumnDict,
    # Index configs
    HNSWIndexConfig,
    IVFFlatIndexConfig,
    BM25IndexConfig,
    # Search configs
    SearchConfig,
    HNSWSearchConfig,
    BM25SearchConfig,
    RRFConfig,
    WSRConfig,
    # Types
    DistanceStrategy,
    FusionStrategy,
    IterativeScanStrategy,
    # Search utils
    reciprocal_rank_fusion,
    weighted_sum_ranking,
    # Filters
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    build_filter_clause,
)

__all__ = [
    # Main classes
    "PGVecTextSearchStore",
    "AsyncPGVecTextSearchEngine",
    # Table & Data
    "TableConfig",
    "Row",
    "Column",
    "ColumnDict",
    # Index configs
    "HNSWIndexConfig",
    "IVFFlatIndexConfig",
    "BM25IndexConfig",
    # Search configs
    "SearchConfig",
    "HNSWSearchConfig",
    "BM25SearchConfig",
    "RRFConfig",
    "WSRConfig",
    # Types
    "DistanceStrategy",
    "FusionStrategy",
    "IterativeScanStrategy",
    # Search utils
    "reciprocal_rank_fusion",
    "weighted_sum_ranking",
    # Filters
    "FilterOperator",
    "FilterCondition",
    "MetadataFilter",
    "MetadataFilters",
    "build_filter_clause",
]
