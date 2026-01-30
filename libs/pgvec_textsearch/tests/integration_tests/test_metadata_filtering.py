"""
Test cases for metadata filtering in hybrid search.

Uses LlamaIndex-style MetadataFilter/MetadataFilters for type-safe filtering.

Usage:
    # Run directly with connection string:
    python tests/test_metadata_filtering.py "postgresql+asyncpg://user:pass@host:port/db"

    # Or set environment variable:
    DATABASE_URL="postgresql+asyncpg://..." python tests/test_metadata_filtering.py

    # Run with pytest:
    DATABASE_URL="postgresql+asyncpg://..." pytest tests/test_metadata_filtering.py -v
"""
import asyncio
import os
import sys
from typing import List

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_pgvec_textsearch import (
    PGVecTextSearchStore,
    PGVecTextSearchEngine,
    HybridSearchConfig,
    DistanceStrategy,
    HNSWIndex,
    BM25Index,
    Column,
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# =============================================================================
# Configuration
# =============================================================================

# Get connection string from CLI argument, environment, or use default
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    DATABASE_URL = sys.argv[1]
else:
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:9010/postgres"
    )

EMBEDDING_DIMENSION = 384
TABLE_NAME = "test_metadata_filter"


# =============================================================================
# Mock Embeddings
# =============================================================================

class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    def _embed(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        emb = [(h[i % len(h)] / 255.0 * 2 - 1) for i in range(self.dimension)]
        norm = sum(x * x for x in emb) ** 0.5
        return [x / norm for x in emb] if norm > 0 else emb


# =============================================================================
# Test Documents with Rich Metadata
# =============================================================================

SAMPLE_DOCUMENTS = [
    # Tech documents
    Document(
        page_content="인공지능 기술이 발전하면서 우리 삶은 더욱 편리해지고 있습니다.",
        metadata={"category": "tech", "subcategory": "ai", "year": 2024, "rating": 4.5, "published": True}
    ),
    Document(
        page_content="데이터베이스 성능 최적화를 위해서는 인덱스 설계가 매우 중요합니다.",
        metadata={"category": "tech", "subcategory": "database", "year": 2023, "rating": 4.8, "published": True}
    ),
    Document(
        page_content="벡터 데이터베이스는 유사도 검색에 최적화된 저장소입니다.",
        metadata={"category": "tech", "subcategory": "database", "year": 2024, "rating": 4.2, "published": True}
    ),
    Document(
        page_content="BM25는 정보 검색에서 널리 사용되는 랭킹 알고리즘입니다.",
        metadata={"category": "tech", "subcategory": "search", "year": 2022, "rating": 4.0, "published": True}
    ),
    Document(
        page_content="하이브리드 검색은 밀집 벡터와 희소 벡터를 결합한 방식입니다.",
        metadata={"category": "tech", "subcategory": "search", "year": 2024, "rating": 4.7, "published": False}
    ),
    # Programming documents
    Document(
        page_content="파이썬은 배우기 쉽고 라이브러리가 풍부해서 데이터 분석에 최적입니다.",
        metadata={"category": "programming", "subcategory": "python", "year": 2023, "rating": 4.9, "published": True}
    ),
    Document(
        page_content="자바스크립트는 웹 개발의 핵심 언어로 프론트엔드와 백엔드 모두에서 사용됩니다.",
        metadata={"category": "programming", "subcategory": "javascript", "year": 2023, "rating": 4.3, "published": True}
    ),
    # Travel documents
    Document(
        page_content="제주도 여행을 갔을 때 보았던 푸른 바다가 아직도 눈에 선합니다.",
        metadata={"category": "travel", "subcategory": "domestic", "year": 2024, "rating": 4.6, "published": True}
    ),
    Document(
        page_content="일본 도쿄 여행에서 맛본 라멘이 정말 맛있었습니다.",
        metadata={"category": "travel", "subcategory": "international", "year": 2023, "rating": 4.4, "published": True}
    ),
    # Food documents
    Document(
        page_content="오늘 점심에는 맛있는 비빔밥을 먹었는데 정말 건강한 맛이었어요.",
        metadata={"category": "food", "subcategory": "korean", "year": 2024, "rating": 3.8, "published": False}
    ),
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def engine():
    """Create engine for tests."""
    return PGVecTextSearchEngine.from_connection_string_async(DATABASE_URL)


@pytest.fixture(scope="module")
def embeddings():
    """Create mock embeddings."""
    return MockEmbeddings(dimension=EMBEDDING_DIMENSION)


# =============================================================================
# Helper Functions
# =============================================================================

async def setup_table(engine: PGVecTextSearchEngine) -> None:
    """Create test table with metadata columns."""
    await engine.adrop_table(TABLE_NAME)
    await engine.ainit_hybrid_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=EMBEDDING_DIMENSION,
        overwrite_existing=True,
        metadata_columns=[
            Column("category", "TEXT"),
            Column("subcategory", "TEXT"),
            Column("year", "INTEGER"),
            Column("rating", "FLOAT"),
            Column("published", "BOOLEAN"),
        ],
        bm25_index=BM25Index(
            name=f"idx_{TABLE_NAME}_bm25",
            text_config="public.korean",
        ),
        hnsw_index=HNSWIndex(
            name=f"idx_{TABLE_NAME}_hnsw",
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        ),
    )


async def create_store(
    engine: PGVecTextSearchEngine,
    embeddings: Embeddings,
) -> PGVecTextSearchStore:
    """Create vectorstore instance."""
    return await PGVecTextSearchStore.create(
        engine=engine,
        embedding_service=embeddings,
        table_name=TABLE_NAME,
        metadata_columns=["category", "subcategory", "year", "rating", "published"],
        hybrid_search_config=HybridSearchConfig(
            enable_dense=True,
            enable_sparse=True,
            dense_top_k=20,
            sparse_top_k=20,
        ),
    )


async def cleanup_table(engine: PGVecTextSearchEngine) -> None:
    """Drop test table."""
    await engine.adrop_table(TABLE_NAME)


# =============================================================================
# Test Cases: Comparison Operators
# =============================================================================

class TestComparisonOperators:
    """Test comparison operators: EQ, NE, LT, LTE, GT, GTE"""

    @pytest.mark.asyncio
    async def test_eq_operator(self, engine, embeddings):
        """Test EQ (equals) operator."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: category == "tech"
            filter_obj = MetadataFilter(
                key="category",
                value="tech",
                operator=FilterOperator.EQ
            )
            results = await store.asimilarity_search("데이터베이스", k=10, filter=filter_obj)

            assert len(results) > 0
            for doc in results:
                assert doc.metadata["category"] == "tech"

            print(f"✓ EQ operator: Found {len(results)} tech documents")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_ne_operator(self, engine, embeddings):
        """Test NE (not equals) operator."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: category != "tech"
            filter_obj = MetadataFilter(
                key="category",
                value="tech",
                operator=FilterOperator.NE
            )
            results = await store.asimilarity_search("여행", k=10, filter=filter_obj)

            assert len(results) > 0
            for doc in results:
                assert doc.metadata["category"] != "tech"

            print(f"✓ NE operator: Found {len(results)} non-tech documents")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_gt_gte_operators(self, engine, embeddings):
        """Test GT and GTE operators."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: rating > 4.5
            filter_gt = MetadataFilter(key="rating", value=4.5, operator=FilterOperator.GT)
            results_gt = await store.asimilarity_search("검색", k=10, filter=filter_gt)
            for doc in results_gt:
                assert doc.metadata["rating"] > 4.5

            # Filter: rating >= 4.5
            filter_gte = MetadataFilter(key="rating", value=4.5, operator=FilterOperator.GTE)
            results_gte = await store.asimilarity_search("검색", k=10, filter=filter_gte)
            for doc in results_gte:
                assert doc.metadata["rating"] >= 4.5

            assert len(results_gte) >= len(results_gt)

            print(f"✓ GT operator: Found {len(results_gt)} documents with rating > 4.5")
            print(f"✓ GTE operator: Found {len(results_gte)} documents with rating >= 4.5")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_lt_lte_operators(self, engine, embeddings):
        """Test LT and LTE operators."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: year < 2024
            filter_lt = MetadataFilter(key="year", value=2024, operator=FilterOperator.LT)
            results_lt = await store.asimilarity_search("프로그래밍", k=10, filter=filter_lt)
            for doc in results_lt:
                assert doc.metadata["year"] < 2024

            # Filter: year <= 2023
            filter_lte = MetadataFilter(key="year", value=2023, operator=FilterOperator.LTE)
            results_lte = await store.asimilarity_search("프로그래밍", k=10, filter=filter_lte)
            for doc in results_lte:
                assert doc.metadata["year"] <= 2023

            print(f"✓ LT operator: Found {len(results_lt)} documents with year < 2024")
            print(f"✓ LTE operator: Found {len(results_lte)} documents with year <= 2023")

        finally:
            await cleanup_table(engine)


# =============================================================================
# Test Cases: Array Operators
# =============================================================================

class TestArrayOperators:
    """Test array operators: IN, NIN"""

    @pytest.mark.asyncio
    async def test_in_operator(self, engine, embeddings):
        """Test IN operator."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: category in ["tech", "programming"]
            filter_obj = MetadataFilter(
                key="category",
                value=["tech", "programming"],
                operator=FilterOperator.IN
            )
            results = await store.asimilarity_search("개발", k=10, filter=filter_obj)

            assert len(results) > 0
            for doc in results:
                assert doc.metadata["category"] in ["tech", "programming"]

            print(f"✓ IN operator: Found {len(results)} tech/programming documents")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_nin_operator(self, engine, embeddings):
        """Test NIN (not in) operator."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: category not in ["travel", "food"]
            filter_obj = MetadataFilter(
                key="category",
                value=["travel", "food"],
                operator=FilterOperator.NIN
            )
            results = await store.asimilarity_search("기술", k=10, filter=filter_obj)

            assert len(results) > 0
            for doc in results:
                assert doc.metadata["category"] not in ["travel", "food"]

            print(f"✓ NIN operator: Found {len(results)} non-travel/food documents")

        finally:
            await cleanup_table(engine)


# =============================================================================
# Test Cases: Range and Existence Operators
# =============================================================================

class TestRangeAndExistenceOperators:
    """Test range and existence operators: BETWEEN, EXISTS"""

    @pytest.mark.asyncio
    async def test_between_operator(self, engine, embeddings):
        """Test BETWEEN operator."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: year between 2023 and 2024
            filter_obj = MetadataFilter(
                key="year",
                value=[2023, 2024],
                operator=FilterOperator.BETWEEN
            )
            results = await store.asimilarity_search("검색", k=10, filter=filter_obj)

            assert len(results) > 0
            for doc in results:
                assert 2023 <= doc.metadata["year"] <= 2024

            print(f"✓ BETWEEN operator: Found {len(results)} documents from 2023-2024")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_exists_operator(self, engine, embeddings):
        """Test EXISTS operator."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: published exists and is not null
            filter_obj = MetadataFilter(
                key="published",
                value=True,
                operator=FilterOperator.EXISTS
            )
            results = await store.asimilarity_search("기술", k=10, filter=filter_obj)

            # All our documents have published field
            assert len(results) > 0

            print(f"✓ EXISTS operator: Found {len(results)} documents with published field")

        finally:
            await cleanup_table(engine)


# =============================================================================
# Test Cases: Text Operators
# =============================================================================

class TestTextOperators:
    """Test text operators: TEXT_MATCH, TEXT_MATCH_INSENSITIVE"""

    @pytest.mark.asyncio
    async def test_text_match_operator(self, engine, embeddings):
        """Test TEXT_MATCH (LIKE) operator."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: subcategory like 'data%'
            filter_obj = MetadataFilter(
                key="subcategory",
                value="data%",
                operator=FilterOperator.TEXT_MATCH
            )
            results = await store.asimilarity_search("검색", k=10, filter=filter_obj)

            for doc in results:
                assert doc.metadata["subcategory"].startswith("data")

            print(f"✓ TEXT_MATCH operator: Found {len(results)} documents with subcategory starting with 'data'")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_text_match_insensitive_operator(self, engine, embeddings):
        """Test TEXT_MATCH_INSENSITIVE (ILIKE) operator."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: subcategory ilike '%search%' (case-insensitive)
            filter_obj = MetadataFilter(
                key="subcategory",
                value="%search%",
                operator=FilterOperator.TEXT_MATCH_INSENSITIVE
            )
            results = await store.asimilarity_search("검색", k=10, filter=filter_obj)

            for doc in results:
                assert "search" in doc.metadata["subcategory"].lower()

            print(f"✓ TEXT_MATCH_INSENSITIVE operator: Found {len(results)} documents with 'search' in subcategory")

        finally:
            await cleanup_table(engine)


# =============================================================================
# Test Cases: Logical Operators (MetadataFilters)
# =============================================================================

class TestLogicalOperators:
    """Test logical conditions: AND, OR, NOT"""

    @pytest.mark.asyncio
    async def test_and_condition(self, engine, embeddings):
        """Test AND condition with MetadataFilters."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: category == "tech" AND year >= 2024
            filter_obj = MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
                    MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
                ],
                condition=FilterCondition.AND
            )
            results = await store.asimilarity_search("인공지능", k=10, filter=filter_obj)

            assert len(results) > 0
            for doc in results:
                assert doc.metadata["category"] == "tech"
                assert doc.metadata["year"] >= 2024

            print(f"✓ AND condition: Found {len(results)} tech documents from 2024+")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_or_condition(self, engine, embeddings):
        """Test OR condition with MetadataFilters."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: category == "travel" OR category == "food"
            filter_obj = MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="travel", operator=FilterOperator.EQ),
                    MetadataFilter(key="category", value="food", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR
            )
            results = await store.asimilarity_search("맛있는 여행", k=10, filter=filter_obj)

            assert len(results) > 0
            for doc in results:
                assert doc.metadata["category"] in ["travel", "food"]

            print(f"✓ OR condition: Found {len(results)} travel/food documents")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_not_condition(self, engine, embeddings):
        """Test NOT condition with MetadataFilters."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: NOT (category == "tech")
            filter_obj = MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.NOT
            )
            results = await store.asimilarity_search("검색", k=10, filter=filter_obj)

            assert len(results) > 0
            for doc in results:
                assert doc.metadata["category"] != "tech"

            print(f"✓ NOT condition: Found {len(results)} non-tech documents")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_nested_conditions(self, engine, embeddings):
        """Test nested MetadataFilters (complex logical expression)."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: (category == "tech" AND rating >= 4.5) OR (category == "programming")
            filter_obj = MetadataFilters(
                filters=[
                    MetadataFilters(
                        filters=[
                            MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
                            MetadataFilter(key="rating", value=4.5, operator=FilterOperator.GTE),
                        ],
                        condition=FilterCondition.AND
                    ),
                    MetadataFilter(key="category", value="programming", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR
            )
            results = await store.asimilarity_search("개발 기술", k=10, filter=filter_obj)

            assert len(results) > 0
            for doc in results:
                is_high_rated_tech = (
                    doc.metadata["category"] == "tech" and
                    doc.metadata["rating"] >= 4.5
                )
                is_programming = doc.metadata["category"] == "programming"
                assert is_high_rated_tech or is_programming

            print(f"✓ Nested conditions: Found {len(results)} high-rated tech OR programming documents")

        finally:
            await cleanup_table(engine)


# =============================================================================
# Test Cases: Hybrid Search with Filters
# =============================================================================

class TestHybridSearchWithFilters:
    """Test hybrid search (dense + sparse) with metadata filters."""

    @pytest.mark.asyncio
    async def test_hybrid_search_with_category_filter(self, engine, embeddings):
        """Test hybrid search with category filter."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Hybrid search with filter
            query = "데이터베이스 검색 인덱스"
            filter_obj = MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
            results = await store.asimilarity_search_with_score(query, k=5, filter=filter_obj)

            print(f"\n[Hybrid + Filter] Query: '{query}', Filter: category=tech")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Category: {doc.metadata['category']}, Content: {doc.page_content[:40]}...")
                assert doc.metadata["category"] == "tech"

            print(f"✓ Hybrid search with category filter: Found {len(results)} results")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_multiple_filters(self, engine, embeddings):
        """Test hybrid search with multiple filters."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Hybrid search with multiple filters
            query = "기술 발전"
            filter_obj = MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
                    MetadataFilter(key="year", value=2023, operator=FilterOperator.GTE),
                    MetadataFilter(key="published", value=True, operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.AND
            )
            results = await store.asimilarity_search_with_score(query, k=5, filter=filter_obj)

            print(f"\n[Hybrid + Multiple Filters] Query: '{query}'")
            print(f"  Filters: category=tech, year>=2023, published=True")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Year: {doc.metadata['year']}, Content: {doc.page_content[:40]}...")
                assert doc.metadata["category"] == "tech"
                assert doc.metadata["year"] >= 2023
                assert doc.metadata["published"] is True

            print(f"✓ Hybrid search with multiple filters: Found {len(results)} results")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_dense_only_with_filter(self, engine, embeddings):
        """Test dense-only search with filter."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            store.hybrid_search_config.enable_sparse = False  # Dense only
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            query = "프로그래밍 언어"
            filter_obj = MetadataFilter(key="category", value="programming", operator=FilterOperator.EQ)
            results = await store.asimilarity_search_with_score(query, k=5, filter=filter_obj)

            print(f"\n[Dense Only + Filter] Query: '{query}'")
            for doc, score in results:
                print(f"  Distance: {score:.4f}, Content: {doc.page_content[:40]}...")
                assert doc.metadata["category"] == "programming"

            print(f"✓ Dense-only search with filter: Found {len(results)} results")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_sparse_only_with_filter(self, engine, embeddings):
        """Test sparse-only (BM25) search with filter."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            store.hybrid_search_config.enable_dense = False  # Sparse only
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            query = "인공지능"
            filter_obj = MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
            results = await store.asimilarity_search_with_score(query, k=5, filter=filter_obj)

            print(f"\n[BM25 Only + Filter] Query: '{query}'")
            for doc, score in results:
                print(f"  BM25 Score: {score:.4f}, Content: {doc.page_content[:40]}...")
                assert doc.metadata["category"] == "tech"

            print(f"✓ BM25-only search with filter: Found {len(results)} results")

        finally:
            await cleanup_table(engine)


# =============================================================================
# Test Cases: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_no_filter(self, engine, embeddings):
        """Test search without filter (should return all matching results)."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            results = await store.asimilarity_search("기술", k=5)

            assert len(results) > 0
            print(f"✓ No filter works correctly: Found {len(results)} results")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_no_matching_results(self, engine, embeddings):
        """Test filter that matches no documents."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter that matches nothing
            filter_obj = MetadataFilter(key="category", value="nonexistent", operator=FilterOperator.EQ)
            results = await store.asimilarity_search("기술", k=5, filter=filter_obj)

            assert len(results) == 0

            print(f"✓ No matching results handled correctly")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_boolean_filter(self, engine, embeddings):
        """Test filtering on boolean field."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Filter: published == True
            filter_published = MetadataFilter(key="published", value=True, operator=FilterOperator.EQ)
            results_published = await store.asimilarity_search("기술", k=10, filter=filter_published)
            for doc in results_published:
                assert doc.metadata["published"] is True

            # Filter: published == False
            filter_unpublished = MetadataFilter(key="published", value=False, operator=FilterOperator.EQ)
            results_unpublished = await store.asimilarity_search("기술", k=10, filter=filter_unpublished)
            for doc in results_unpublished:
                assert doc.metadata["published"] is False

            print(f"✓ Boolean filter: {len(results_published)} published, {len(results_unpublished)} unpublished")

        finally:
            await cleanup_table(engine)


# =============================================================================
# Test Cases: All Filter Operators
# =============================================================================

class TestAllFilterOperators:
    """Test all FilterOperator types with MetadataFilter."""

    @pytest.mark.asyncio
    async def test_all_operators(self, engine, embeddings):
        """Test all FilterOperator types."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Test comparison operators
            ops_to_test = [
                (FilterOperator.EQ, "category", "tech", lambda m: m["category"] == "tech"),
                (FilterOperator.NE, "category", "tech", lambda m: m["category"] != "tech"),
                (FilterOperator.GT, "rating", 4.5, lambda m: m["rating"] > 4.5),
                (FilterOperator.GTE, "rating", 4.5, lambda m: m["rating"] >= 4.5),
                (FilterOperator.LT, "year", 2024, lambda m: m["year"] < 2024),
                (FilterOperator.LTE, "year", 2023, lambda m: m["year"] <= 2023),
                (FilterOperator.IN, "category", ["tech", "programming"], lambda m: m["category"] in ["tech", "programming"]),
                (FilterOperator.NIN, "category", ["travel", "food"], lambda m: m["category"] not in ["travel", "food"]),
                (FilterOperator.BETWEEN, "year", [2023, 2024], lambda m: 2023 <= m["year"] <= 2024),
                (FilterOperator.TEXT_MATCH, "subcategory", "data%", lambda m: m["subcategory"].startswith("data")),
                (FilterOperator.TEXT_MATCH_INSENSITIVE, "subcategory", "%search%", lambda m: "search" in m["subcategory"].lower()),
                (FilterOperator.EXISTS, "published", True, lambda m: m.get("published") is not None),
            ]

            for op, key, value, check_fn in ops_to_test:
                filter_obj = MetadataFilter(key=key, value=value, operator=op)
                results = await store.asimilarity_search("검색", k=10, filter=filter_obj)

                for doc in results:
                    assert check_fn(doc.metadata), f"Failed for operator {op.name}"

                print(f"  ✓ {op.name}: {len(results)} results")

            print(f"✓ All FilterOperator types work correctly")

        finally:
            await cleanup_table(engine)


# =============================================================================
# Test Cases: Delete and MMR with Filters
# =============================================================================

class TestDeleteAndMMRWithFilters:
    """Test delete and MMR operations with MetadataFilter."""

    @pytest.mark.asyncio
    async def test_delete_with_metadata_filter(self, engine, embeddings):
        """Test delete operation with MetadataFilter objects."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Count documents before delete
            all_results_before = await store.asimilarity_search("검색", k=100)
            tech_count_before = sum(1 for doc in all_results_before if doc.metadata["category"] == "tech")

            # Delete tech documents using MetadataFilter
            filter_obj = MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
            deleted = await store.adelete(filter=filter_obj)
            assert deleted is True

            # Count documents after delete
            all_results_after = await store.asimilarity_search("검색", k=100)
            tech_count_after = sum(1 for doc in all_results_after if doc.metadata["category"] == "tech")

            assert tech_count_after == 0, "All tech documents should be deleted"
            assert len(all_results_after) < len(all_results_before), "Total count should decrease"

            print(f"✓ Delete with MetadataFilter: Deleted {tech_count_before} tech documents")

        finally:
            await cleanup_table(engine)

    @pytest.mark.asyncio
    async def test_metadata_filter_with_mmr(self, engine, embeddings):
        """Test MMR search with MetadataFilter objects."""
        try:
            await setup_table(engine)
            store = await create_store(engine, embeddings)
            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # MMR search with MetadataFilter
            filter_obj = MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
            results = await store.amax_marginal_relevance_search(
                "데이터베이스 검색",
                k=3,
                fetch_k=10,
                filter=filter_obj
            )

            print(f"\n[MMR + MetadataFilter]")
            for doc in results:
                print(f"  Category: {doc.metadata['category']}, Content: {doc.page_content[:40]}...")
                assert doc.metadata["category"] == "tech"

            print(f"✓ MMR with MetadataFilter: Found {len(results)} results")

        finally:
            await cleanup_table(engine)


# =============================================================================
# Main Runner
# =============================================================================

async def run_all_tests():
    """Run all tests manually."""
    print("=" * 70)
    print("Metadata Filtering Test Suite (MetadataFilter/MetadataFilters only)")
    print("=" * 70)
    print(f"Database URL: {DATABASE_URL}")
    print()

    engine = PGVecTextSearchEngine.from_connection_string_async(DATABASE_URL)
    embeddings = MockEmbeddings(dimension=EMBEDDING_DIMENSION)

    test_classes = [
        TestComparisonOperators(),
        TestArrayOperators(),
        TestRangeAndExistenceOperators(),
        TestTextOperators(),
        TestLogicalOperators(),
        TestHybridSearchWithFilters(),
        TestEdgeCases(),
        TestAllFilterOperators(),
        TestDeleteAndMMRWithFilters(),
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n{'=' * 50}")
        print(f"Running: {test_class.__class__.__name__}")
        print("=" * 50)

        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                method = getattr(test_class, method_name)
                print(f"\n→ {method_name}")
                try:
                    await method(engine, embeddings)
                    total_passed += 1
                except Exception as e:
                    print(f"✗ FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    total_failed += 1

    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
