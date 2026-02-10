"""PGVecTextSearch VectorStore package."""

from .pgvec_textsearch import PGVecTextSearchStore
from .engine import AsyncPGVecTextSearchEngine
from .config import (
    TableConfig,
    SearchConfig,
    HNSWIndexConfig,
    IVFFlatIndexConfig,
    BM25IndexConfig,
    HNSWSearchConfig,
    BM25SearchConfig,
)
from .data import Row
from .search_utils import reciprocal_rank_fusion
from .filters import (
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    build_filter_clause,
)
from .types import (
    Column,
    ColumnDict,
    DistanceStrategy,
    FusionStrategy,
    IterativeScanStrategy,
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
    # Types
    "DistanceStrategy",
    "FusionStrategy",
    "IterativeScanStrategy",
    # Search utils
    "reciprocal_rank_fusion",
    # Filters
    "FilterOperator",
    "FilterCondition",
    "MetadataFilter",
    "MetadataFilters",
    "build_filter_clause",
]
