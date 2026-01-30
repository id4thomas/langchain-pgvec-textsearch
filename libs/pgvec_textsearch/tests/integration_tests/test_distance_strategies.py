"""
Test cases for PGVecTextSearchStore with all distance strategies.

Usage:
    # Run directly with connection string:
    python tests/test_distance_strategies.py "postgresql+asyncpg://user:pass@host:port/db"

    # Or set environment variable:
    DATABASE_URL="postgresql+asyncpg://..." python tests/test_distance_strategies.py

    # Run with pytest:
    DATABASE_URL="postgresql+asyncpg://..." pytest tests/test_distance_strategies.py -v
"""
import asyncio
import os
import sys
from typing import List

import pytest

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_pgvec_textsearch import (
    PGVecTextSearchStore,
    PGVecTextSearchEngine,
    HybridSearchConfig,
    DistanceStrategy,
    HNSWIndex,
    BM25Index,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# =============================================================================
# Mock Embeddings for Testing
# =============================================================================

class MockEmbeddings(Embeddings):
    """Mock embeddings that return predictable vectors for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        return self._embed_text(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text)

    def _embed_text(self, text: str) -> List[float]:
        """Generate a deterministic embedding based on text content."""
        # Create a simple hash-based embedding for reproducibility
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to floats and normalize
        embedding = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            # Create value between -1 and 1
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            # Add some variation based on position
            value = value * (0.5 + 0.5 * ((i + hash_bytes[(i + 1) % len(hash_bytes)]) % 100) / 100)
            embedding.append(value)

        # Normalize for cosine similarity
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding


# =============================================================================
# Test Configuration
# =============================================================================

# Get connection string from CLI argument, environment, or use default
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    DATABASE_URL = sys.argv[1]
else:
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:9010/postgres"
    )

# Test table names for each distance strategy
TEST_TABLES = {
    DistanceStrategy.COSINE_DISTANCE: "test_cosine_distance",
    DistanceStrategy.EUCLIDEAN: "test_euclidean_distance",
    DistanceStrategy.INNER_PRODUCT: "test_inner_product",
}

# Sample Korean documents for testing
SAMPLE_DOCUMENTS = [
    Document(
        page_content="인공지능 기술이 발전하면서 우리 삶은 더욱 편리해지고 있습니다.",
        metadata={"category": "tech", "lang": "ko"}
    ),
    Document(
        page_content="데이터베이스 성능 최적화를 위해서는 인덱스 설계가 매우 중요합니다.",
        metadata={"category": "tech", "lang": "ko"}
    ),
    Document(
        page_content="파이썬은 배우기 쉽고 라이브러리가 풍부해서 데이터 분석에 최적입니다.",
        metadata={"category": "programming", "lang": "ko"}
    ),
    Document(
        page_content="제주도 여행을 갔을 때 보았던 푸른 바다가 아직도 눈에 선합니다.",
        metadata={"category": "travel", "lang": "ko"}
    ),
    Document(
        page_content="오늘 점심에는 맛있는 비빔밥을 먹었는데 정말 건강한 맛이었어요.",
        metadata={"category": "food", "lang": "ko"}
    ),
    Document(
        page_content="벡터 데이터베이스는 유사도 검색에 최적화된 저장소입니다.",
        metadata={"category": "tech", "lang": "ko"}
    ),
    Document(
        page_content="BM25는 정보 검색에서 널리 사용되는 랭킹 알고리즘입니다.",
        metadata={"category": "tech", "lang": "ko"}
    ),
    Document(
        page_content="하이브리드 검색은 밀집 벡터와 희소 벡터를 결합한 방식입니다.",
        metadata={"category": "tech", "lang": "ko"}
    ),
]

EMBEDDING_DIMENSION = 384


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def engine():
    """Create a shared engine for all tests."""
    # Use from_connection_string_async for direct async usage
    engine = PGVecTextSearchEngine.from_connection_string_async(DATABASE_URL)
    yield engine
    # Cleanup is handled in individual tests


@pytest.fixture(scope="module")
def embeddings():
    """Create mock embeddings."""
    return MockEmbeddings(dimension=EMBEDDING_DIMENSION)


# =============================================================================
# Helper Functions
# =============================================================================

async def create_table_with_strategy(
    engine: PGVecTextSearchEngine,
    table_name: str,
    distance_strategy: DistanceStrategy,
) -> None:
    """Create a table with specific distance strategy indexes."""
    # Drop existing table if any
    await engine.adrop_table(table_name)

    # Create table with hybrid indexes
    await engine.ainit_hybrid_vectorstore_table(
        table_name=table_name,
        vector_size=EMBEDDING_DIMENSION,
        overwrite_existing=True,
        bm25_index=BM25Index(
            name=f"idx_{table_name}_bm25",
            text_config="public.korean",
            k1=1.2,
            b=0.75,
        ),
        hnsw_index=HNSWIndex(
            name=f"idx_{table_name}_hnsw",
            m=16,
            ef_construction=64,
            distance_strategy=distance_strategy,
        ),
    )


async def cleanup_table(engine: PGVecTextSearchEngine, table_name: str) -> None:
    """Clean up test table."""
    await engine.adrop_table(table_name)


# =============================================================================
# Test Cases
# =============================================================================

class TestDistanceStrategies:
    """Test suite for all distance strategies."""

    @pytest.mark.asyncio
    async def test_cosine_distance_strategy(self, engine, embeddings):
        """Test COSINE_DISTANCE strategy."""
        table_name = TEST_TABLES[DistanceStrategy.COSINE_DISTANCE]
        distance_strategy = DistanceStrategy.COSINE_DISTANCE

        try:
            # Create table
            await create_table_with_strategy(engine, table_name, distance_strategy)

            # Create store
            store = await PGVecTextSearchStore.create(
                engine=engine,
                embedding_service=embeddings,
                table_name=table_name,
                distance_strategy=distance_strategy,
                hybrid_search_config=HybridSearchConfig(
                    enable_dense=True,
                    enable_sparse=True,
                    dense_top_k=10,
                    sparse_top_k=10,
                ),
            )

            # Add documents
            ids = await store.aadd_documents(SAMPLE_DOCUMENTS)
            assert len(ids) == len(SAMPLE_DOCUMENTS)

            # Test similarity search
            query = "인공지능 기술 발전"
            results = await store.asimilarity_search(query, k=3)
            assert len(results) > 0
            assert len(results) <= 3

            # Test similarity search with score
            results_with_score = await store.asimilarity_search_with_score(query, k=3)
            assert len(results_with_score) > 0
            for doc, score in results_with_score:
                assert isinstance(doc, Document)
                assert isinstance(score, float)
                print(f"[COSINE] Score: {score:.4f}, Content: {doc.page_content[:50]}...")

            # Test dense-only search
            store.hybrid_search_config.enable_sparse = False
            dense_results = await store.asimilarity_search(query, k=3)
            assert len(dense_results) > 0
            store.hybrid_search_config.enable_sparse = True

            print(f"✓ COSINE_DISTANCE strategy test passed")

        finally:
            await cleanup_table(engine, table_name)

    @pytest.mark.asyncio
    async def test_euclidean_distance_strategy(self, engine, embeddings):
        """Test EUCLIDEAN distance strategy."""
        table_name = TEST_TABLES[DistanceStrategy.EUCLIDEAN]
        distance_strategy = DistanceStrategy.EUCLIDEAN

        try:
            # Create table
            await create_table_with_strategy(engine, table_name, distance_strategy)

            # Create store
            store = await PGVecTextSearchStore.create(
                engine=engine,
                embedding_service=embeddings,
                table_name=table_name,
                distance_strategy=distance_strategy,
                hybrid_search_config=HybridSearchConfig(
                    enable_dense=True,
                    enable_sparse=True,
                    dense_top_k=10,
                    sparse_top_k=10,
                ),
            )

            # Add documents
            ids = await store.aadd_documents(SAMPLE_DOCUMENTS)
            assert len(ids) == len(SAMPLE_DOCUMENTS)

            # Test similarity search
            query = "데이터베이스 인덱스 최적화"
            results = await store.asimilarity_search(query, k=3)
            assert len(results) > 0

            # Test similarity search with score
            results_with_score = await store.asimilarity_search_with_score(query, k=3)
            assert len(results_with_score) > 0
            for doc, score in results_with_score:
                assert isinstance(doc, Document)
                assert isinstance(score, float)
                print(f"[EUCLIDEAN] Score: {score:.4f}, Content: {doc.page_content[:50]}...")

            print(f"✓ EUCLIDEAN distance strategy test passed")

        finally:
            await cleanup_table(engine, table_name)

    @pytest.mark.asyncio
    async def test_inner_product_strategy(self, engine, embeddings):
        """Test INNER_PRODUCT distance strategy."""
        table_name = TEST_TABLES[DistanceStrategy.INNER_PRODUCT]
        distance_strategy = DistanceStrategy.INNER_PRODUCT

        try:
            # Create table
            await create_table_with_strategy(engine, table_name, distance_strategy)

            # Create store
            store = await PGVecTextSearchStore.create(
                engine=engine,
                embedding_service=embeddings,
                table_name=table_name,
                distance_strategy=distance_strategy,
                hybrid_search_config=HybridSearchConfig(
                    enable_dense=True,
                    enable_sparse=True,
                    dense_top_k=10,
                    sparse_top_k=10,
                ),
            )

            # Add documents
            ids = await store.aadd_documents(SAMPLE_DOCUMENTS)
            assert len(ids) == len(SAMPLE_DOCUMENTS)

            # Test similarity search
            query = "파이썬 프로그래밍 데이터 분석"
            results = await store.asimilarity_search(query, k=3)
            assert len(results) > 0

            # Test similarity search with score
            results_with_score = await store.asimilarity_search_with_score(query, k=3)
            assert len(results_with_score) > 0
            for doc, score in results_with_score:
                assert isinstance(doc, Document)
                assert isinstance(score, float)
                print(f"[INNER_PRODUCT] Score: {score:.4f}, Content: {doc.page_content[:50]}...")

            print(f"✓ INNER_PRODUCT strategy test passed")

        finally:
            await cleanup_table(engine, table_name)


class TestHybridSearch:
    """Test hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, engine, embeddings):
        """Test hybrid search with RRF fusion."""
        table_name = "test_hybrid_rrf"

        try:
            await create_table_with_strategy(
                engine, table_name, DistanceStrategy.COSINE_DISTANCE
            )

            store = await PGVecTextSearchStore.create(
                engine=engine,
                embedding_service=embeddings,
                table_name=table_name,
                distance_strategy=DistanceStrategy.COSINE_DISTANCE,
                hybrid_search_config=HybridSearchConfig(
                    enable_dense=True,
                    enable_sparse=True,
                    dense_top_k=10,
                    sparse_top_k=10,
                    fusion_function_parameters={"rrf_k": 60},
                ),
            )

            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Test hybrid search
            query = "벡터 데이터베이스 검색"
            results = await store.asimilarity_search_with_score(query, k=5)

            print("\n[HYBRID RRF] Results:")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Content: {doc.page_content[:60]}...")

            assert len(results) > 0
            print(f"✓ Hybrid RRF search test passed")

        finally:
            await cleanup_table(engine, table_name)

    @pytest.mark.asyncio
    async def test_sparse_only_search(self, engine, embeddings):
        """Test sparse-only (BM25) search."""
        table_name = "test_sparse_only"

        try:
            await create_table_with_strategy(
                engine, table_name, DistanceStrategy.COSINE_DISTANCE
            )

            store = await PGVecTextSearchStore.create(
                engine=engine,
                embedding_service=embeddings,
                table_name=table_name,
                distance_strategy=DistanceStrategy.COSINE_DISTANCE,
                hybrid_search_config=HybridSearchConfig(
                    enable_dense=False,  # Disable dense
                    enable_sparse=True,
                    sparse_top_k=10,
                ),
            )

            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Test BM25-only search
            query = "인공지능"
            results = await store.asimilarity_search_with_score(query, k=5)

            print("\n[BM25 ONLY] Results:")
            for doc, score in results:
                print(f"  BM25 Score: {score:.4f}, Content: {doc.page_content[:60]}...")

            assert len(results) > 0
            print(f"✓ Sparse-only (BM25) search test passed")

        finally:
            await cleanup_table(engine, table_name)

    @pytest.mark.asyncio
    async def test_dense_only_search(self, engine, embeddings):
        """Test dense-only (vector) search."""
        table_name = "test_dense_only"

        try:
            await create_table_with_strategy(
                engine, table_name, DistanceStrategy.COSINE_DISTANCE
            )

            store = await PGVecTextSearchStore.create(
                engine=engine,
                embedding_service=embeddings,
                table_name=table_name,
                distance_strategy=DistanceStrategy.COSINE_DISTANCE,
                hybrid_search_config=HybridSearchConfig(
                    enable_dense=True,
                    enable_sparse=False,  # Disable sparse
                    dense_top_k=10,
                ),
            )

            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Test vector-only search
            query = "기술 발전"
            results = await store.asimilarity_search_with_score(query, k=5)

            print("\n[DENSE ONLY] Results:")
            for doc, score in results:
                print(f"  Distance: {score:.4f}, Content: {doc.page_content[:60]}...")

            assert len(results) > 0
            print(f"✓ Dense-only (vector) search test passed")

        finally:
            await cleanup_table(engine, table_name)


class TestCRUDOperations:
    """Test CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_and_delete(self, engine, embeddings):
        """Test adding and deleting documents."""
        table_name = "test_crud"

        try:
            await create_table_with_strategy(
                engine, table_name, DistanceStrategy.COSINE_DISTANCE
            )

            store = await PGVecTextSearchStore.create(
                engine=engine,
                embedding_service=embeddings,
                table_name=table_name,
            )

            # Add documents
            ids = await store.aadd_documents(SAMPLE_DOCUMENTS[:3])
            assert len(ids) == 3

            # Verify documents exist
            docs = await store.aget_by_ids(ids)
            assert len(docs) == 3

            # Delete one document
            await store.adelete(ids=[ids[0]])

            # Verify deletion
            remaining = await store.aget_by_ids(ids)
            assert len(remaining) == 2

            # Delete remaining
            await store.adelete(ids=ids[1:])
            remaining = await store.aget_by_ids(ids)
            assert len(remaining) == 0

            print(f"✓ CRUD operations test passed")

        finally:
            await cleanup_table(engine, table_name)

    @pytest.mark.asyncio
    async def test_add_texts_with_metadata(self, engine, embeddings):
        """Test adding texts with metadata."""
        table_name = "test_metadata"

        try:
            await create_table_with_strategy(
                engine, table_name, DistanceStrategy.COSINE_DISTANCE
            )

            store = await PGVecTextSearchStore.create(
                engine=engine,
                embedding_service=embeddings,
                table_name=table_name,
            )

            texts = ["첫 번째 문서", "두 번째 문서", "세 번째 문서"]
            metadatas = [
                {"source": "doc1", "page": 1},
                {"source": "doc2", "page": 2},
                {"source": "doc3", "page": 3},
            ]

            ids = await store.aadd_texts(texts, metadatas=metadatas)
            assert len(ids) == 3

            # Verify metadata is stored
            docs = await store.aget_by_ids(ids)
            for doc in docs:
                assert "source" in doc.metadata
                assert "page" in doc.metadata

            print(f"✓ Metadata test passed")

        finally:
            await cleanup_table(engine, table_name)


class TestMMRSearch:
    """Test Maximal Marginal Relevance search."""

    @pytest.mark.asyncio
    async def test_mmr_search(self, engine, embeddings):
        """Test MMR search for diversity."""
        table_name = "test_mmr"

        try:
            await create_table_with_strategy(
                engine, table_name, DistanceStrategy.COSINE_DISTANCE
            )

            store = await PGVecTextSearchStore.create(
                engine=engine,
                embedding_service=embeddings,
                table_name=table_name,
                fetch_k=10,
                lambda_mult=0.5,
            )

            await store.aadd_documents(SAMPLE_DOCUMENTS)

            # Test MMR search
            query = "기술 데이터"
            results = await store.amax_marginal_relevance_search(
                query, k=3, fetch_k=6, lambda_mult=0.5
            )

            print("\n[MMR] Results:")
            for doc in results:
                print(f"  Content: {doc.page_content[:60]}...")

            assert len(results) > 0
            assert len(results) <= 3
            print(f"✓ MMR search test passed")

        finally:
            await cleanup_table(engine, table_name)


# =============================================================================
# Main Runner
# =============================================================================

async def run_all_tests():
    """Run all tests manually (without pytest)."""
    print("=" * 60)
    print("PGVecTextSearchStore Test Suite")
    print("=" * 60)
    print(f"Database URL: {DATABASE_URL}")
    print()

    # Use from_connection_string_async for direct async usage
    engine = PGVecTextSearchEngine.from_connection_string_async(DATABASE_URL)
    embeddings = MockEmbeddings(dimension=EMBEDDING_DIMENSION)

    test_classes = [
        TestDistanceStrategies(),
        TestHybridSearch(),
        TestCRUDOperations(),
        TestMMRSearch(),
    ]

    for test_class in test_classes:
        print(f"\n{'=' * 40}")
        print(f"Running: {test_class.__class__.__name__}")
        print("=" * 40)

        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                method = getattr(test_class, method_name)
                print(f"\n→ {method_name}")
                try:
                    await method(engine, embeddings)
                except Exception as e:
                    print(f"✗ FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run manually (pytest will use its own test discovery)
    asyncio.run(run_all_tests())
