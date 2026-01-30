"""
Compile test - verifies that all imports and syntax are correct.

This test is run by CI to ensure integration tests compile without running them.
"""
import pytest


@pytest.mark.compile
def test_imports() -> None:
    """Test that all required imports work."""
    from langchain_pgvec_textsearch import (
        PGVecTextSearchStore,
        PGVecTextSearchEngine,
        HybridSearchConfig,
        DistanceStrategy,
        HNSWIndex,
        IVFFlatIndex,
        BM25Index,
        Column,
        FilterOperator,
        FilterCondition,
        MetadataFilter,
        MetadataFilters,
        reciprocal_rank_fusion,
        weighted_sum_ranking,
    )
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    # Verify classes are importable
    assert PGVecTextSearchStore is not None
    assert PGVecTextSearchEngine is not None
    assert HybridSearchConfig is not None
    assert DistanceStrategy is not None
