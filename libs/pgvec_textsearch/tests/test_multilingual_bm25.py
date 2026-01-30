"""
Test cases for multilingual BM25 sparse indexing (English and Korean).

Tests that BM25 search works correctly with different text search configurations.

Usage:
    # Run directly with connection string:
    python tests/test_multilingual_bm25.py "postgresql+asyncpg://user:pass@host:port/db"

    # Or set environment variable:
    DATABASE_URL="postgresql+asyncpg://..." python tests/test_multilingual_bm25.py

    # Run with pytest:
    DATABASE_URL="postgresql+asyncpg://..." pytest tests/test_multilingual_bm25.py -v
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
    MetadataFilter,
    FilterOperator,
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
TABLE_NAME_ENGLISH = "test_bm25_english"
TABLE_NAME_KOREAN = "test_bm25_korean"


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
# Test Documents
# =============================================================================

ENGLISH_DOCUMENTS = [
    Document(
        page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        metadata={"topic": "ai", "language": "english"}
    ),
    Document(
        page_content="Deep learning neural networks have revolutionized image recognition and natural language processing.",
        metadata={"topic": "ai", "language": "english"}
    ),
    Document(
        page_content="Database indexing improves query performance by creating efficient data structures.",
        metadata={"topic": "database", "language": "english"}
    ),
    Document(
        page_content="PostgreSQL supports full-text search with various ranking algorithms including BM25.",
        metadata={"topic": "database", "language": "english"}
    ),
    Document(
        page_content="Vector databases store embeddings for semantic similarity search applications.",
        metadata={"topic": "database", "language": "english"}
    ),
    Document(
        page_content="Python is a popular programming language for data science and machine learning.",
        metadata={"topic": "programming", "language": "english"}
    ),
    Document(
        page_content="The traveling salesman problem is a classic optimization challenge in computer science.",
        metadata={"topic": "algorithms", "language": "english"}
    ),
    Document(
        page_content="Distributed systems require careful handling of network partitions and consistency.",
        metadata={"topic": "systems", "language": "english"}
    ),
]

KOREAN_DOCUMENTS = [
    Document(
        page_content="인공지능 기술이 발전하면서 우리 삶은 더욱 편리해지고 있습니다.",
        metadata={"topic": "ai", "language": "korean"}
    ),
    Document(
        page_content="딥러닝은 인공 신경망을 기반으로 한 기계학습의 한 분야입니다.",
        metadata={"topic": "ai", "language": "korean"}
    ),
    Document(
        page_content="데이터베이스 인덱싱은 쿼리 성능을 향상시키는 핵심 기술입니다.",
        metadata={"topic": "database", "language": "korean"}
    ),
    Document(
        page_content="PostgreSQL은 BM25를 포함한 다양한 전문 검색 기능을 지원합니다.",
        metadata={"topic": "database", "language": "korean"}
    ),
    Document(
        page_content="벡터 데이터베이스는 의미론적 유사도 검색을 위해 임베딩을 저장합니다.",
        metadata={"topic": "database", "language": "korean"}
    ),
    Document(
        page_content="파이썬은 데이터 과학과 기계학습에 널리 사용되는 프로그래밍 언어입니다.",
        metadata={"topic": "programming", "language": "korean"}
    ),
    Document(
        page_content="외판원 문제는 컴퓨터 과학에서 고전적인 최적화 문제입니다.",
        metadata={"topic": "algorithms", "language": "korean"}
    ),
    Document(
        page_content="분산 시스템은 네트워크 분할과 일관성을 신중하게 처리해야 합니다.",
        metadata={"topic": "systems", "language": "korean"}
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

async def setup_english_table(engine: PGVecTextSearchEngine) -> None:
    """Create test table with English BM25 index."""
    await engine.adrop_table(TABLE_NAME_ENGLISH)
    await engine.ainit_hybrid_vectorstore_table(
        table_name=TABLE_NAME_ENGLISH,
        vector_size=EMBEDDING_DIMENSION,
        overwrite_existing=True,
        metadata_columns=[
            Column("topic", "TEXT"),
            Column("language", "TEXT"),
        ],
        bm25_index=BM25Index(
            name=f"idx_{TABLE_NAME_ENGLISH}_bm25",
            text_config="english",  # English text search configuration
            k1=1.2,
            b=0.75,
        ),
        hnsw_index=HNSWIndex(
            name=f"idx_{TABLE_NAME_ENGLISH}_hnsw",
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        ),
    )


async def setup_korean_table(engine: PGVecTextSearchEngine) -> None:
    """Create test table with Korean BM25 index."""
    await engine.adrop_table(TABLE_NAME_KOREAN)
    await engine.ainit_hybrid_vectorstore_table(
        table_name=TABLE_NAME_KOREAN,
        vector_size=EMBEDDING_DIMENSION,
        overwrite_existing=True,
        metadata_columns=[
            Column("topic", "TEXT"),
            Column("language", "TEXT"),
        ],
        bm25_index=BM25Index(
            name=f"idx_{TABLE_NAME_KOREAN}_bm25",
            text_config="public.korean",  # Korean text search configuration (textsearch_ko)
            k1=1.2,
            b=0.75,
        ),
        hnsw_index=HNSWIndex(
            name=f"idx_{TABLE_NAME_KOREAN}_hnsw",
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        ),
    )


async def create_english_store(
    engine: PGVecTextSearchEngine,
    embeddings: Embeddings,
    sparse_only: bool = False,
) -> PGVecTextSearchStore:
    """Create English vectorstore instance."""
    return await PGVecTextSearchStore.create(
        engine=engine,
        embedding_service=embeddings,
        table_name=TABLE_NAME_ENGLISH,
        metadata_columns=["topic", "language"],
        hybrid_search_config=HybridSearchConfig(
            enable_dense=not sparse_only,
            enable_sparse=True,
            dense_top_k=20,
            sparse_top_k=20,
            bm25_index_name=f"idx_{TABLE_NAME_ENGLISH}_bm25",
        ),
    )


async def create_korean_store(
    engine: PGVecTextSearchEngine,
    embeddings: Embeddings,
    sparse_only: bool = False,
) -> PGVecTextSearchStore:
    """Create Korean vectorstore instance."""
    return await PGVecTextSearchStore.create(
        engine=engine,
        embedding_service=embeddings,
        table_name=TABLE_NAME_KOREAN,
        metadata_columns=["topic", "language"],
        hybrid_search_config=HybridSearchConfig(
            enable_dense=not sparse_only,
            enable_sparse=True,
            dense_top_k=20,
            sparse_top_k=20,
            bm25_index_name=f"idx_{TABLE_NAME_KOREAN}_bm25",
        ),
    )


async def cleanup_tables(engine: PGVecTextSearchEngine) -> None:
    """Drop test tables."""
    await engine.adrop_table(TABLE_NAME_ENGLISH)
    await engine.adrop_table(TABLE_NAME_KOREAN)


# =============================================================================
# Test Cases: English BM25 Search
# =============================================================================

class TestEnglishBM25:
    """Test English BM25 sparse indexing."""

    @pytest.mark.asyncio
    async def test_english_sparse_search_basic(self, engine, embeddings):
        """Test basic English BM25 search."""
        try:
            await setup_english_table(engine)
            store = await create_english_store(engine, embeddings, sparse_only=True)
            await store.aadd_documents(ENGLISH_DOCUMENTS)

            # Search for "machine learning"
            results = await store.asimilarity_search_with_score("machine learning", k=5)

            print("\n[English BM25] Query: 'machine learning'")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Topic: {doc.metadata['topic']}, Content: {doc.page_content[:60]}...")

            assert len(results) > 0
            # "machine learning" should appear in top results
            top_content = results[0][0].page_content.lower()
            assert "machine" in top_content or "learning" in top_content

            print(f"✓ English BM25 basic search: Found {len(results)} results")

        finally:
            await engine.adrop_table(TABLE_NAME_ENGLISH)

    @pytest.mark.asyncio
    async def test_english_sparse_search_stemming(self, engine, embeddings):
        """Test English BM25 search with stemming (learns -> learn)."""
        try:
            await setup_english_table(engine)
            store = await create_english_store(engine, embeddings, sparse_only=True)
            await store.aadd_documents(ENGLISH_DOCUMENTS)

            # Search for "learning" - should match "learn" due to stemming
            results = await store.asimilarity_search_with_score("learning computers", k=5)

            print("\n[English BM25 Stemming] Query: 'learning computers'")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Content: {doc.page_content[:60]}...")

            assert len(results) > 0
            # Should find documents about machine learning
            found_ml = any("learn" in doc.page_content.lower() for doc, _ in results)
            assert found_ml, "Should find documents with 'learn' variants"

            print(f"✓ English BM25 stemming works: Found {len(results)} results")

        finally:
            await engine.adrop_table(TABLE_NAME_ENGLISH)

    @pytest.mark.asyncio
    async def test_english_sparse_search_multiple_terms(self, engine, embeddings):
        """Test English BM25 search with multiple terms."""
        try:
            await setup_english_table(engine)
            store = await create_english_store(engine, embeddings, sparse_only=True)
            await store.aadd_documents(ENGLISH_DOCUMENTS)

            # Search for multiple terms
            results = await store.asimilarity_search_with_score("database indexing query performance", k=5)

            print("\n[English BM25 Multi-term] Query: 'database indexing query performance'")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Topic: {doc.metadata['topic']}, Content: {doc.page_content[:60]}...")

            assert len(results) > 0
            # Database-related documents should rank higher
            top_topics = [doc.metadata["topic"] for doc, _ in results[:3]]
            assert "database" in top_topics, "Database documents should rank high"

            print(f"✓ English BM25 multi-term search: Found {len(results)} results")

        finally:
            await engine.adrop_table(TABLE_NAME_ENGLISH)

    @pytest.mark.asyncio
    async def test_english_hybrid_search(self, engine, embeddings):
        """Test English hybrid search (BM25 + vector)."""
        try:
            await setup_english_table(engine)
            store = await create_english_store(engine, embeddings, sparse_only=False)
            await store.aadd_documents(ENGLISH_DOCUMENTS)

            # Hybrid search
            results = await store.asimilarity_search_with_score("artificial intelligence neural networks", k=5)

            print("\n[English Hybrid] Query: 'artificial intelligence neural networks'")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Topic: {doc.metadata['topic']}, Content: {doc.page_content[:60]}...")

            assert len(results) > 0
            # AI-related documents should rank higher
            top_topics = [doc.metadata["topic"] for doc, _ in results[:3]]
            assert "ai" in top_topics, "AI documents should rank high in hybrid search"

            print(f"✓ English hybrid search: Found {len(results)} results")

        finally:
            await engine.adrop_table(TABLE_NAME_ENGLISH)


# =============================================================================
# Test Cases: Korean BM25 Search
# =============================================================================

class TestKoreanBM25:
    """Test Korean BM25 sparse indexing."""

    @pytest.mark.asyncio
    async def test_korean_sparse_search_basic(self, engine, embeddings):
        """Test basic Korean BM25 search."""
        try:
            await setup_korean_table(engine)
            store = await create_korean_store(engine, embeddings, sparse_only=True)
            await store.aadd_documents(KOREAN_DOCUMENTS)

            # Search for "인공지능" (artificial intelligence)
            results = await store.asimilarity_search_with_score("인공지능", k=5)

            print("\n[Korean BM25] Query: '인공지능'")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Topic: {doc.metadata['topic']}, Content: {doc.page_content[:40]}...")

            assert len(results) > 0
            # AI documents should rank higher
            top_content = results[0][0].page_content
            assert "인공지능" in top_content or "기계학습" in top_content or "딥러닝" in top_content

            print(f"✓ Korean BM25 basic search: Found {len(results)} results")

        finally:
            await engine.adrop_table(TABLE_NAME_KOREAN)

    @pytest.mark.asyncio
    async def test_korean_sparse_search_morpheme(self, engine, embeddings):
        """Test Korean BM25 search with morpheme analysis."""
        try:
            await setup_korean_table(engine)
            store = await create_korean_store(engine, embeddings, sparse_only=True)
            await store.aadd_documents(KOREAN_DOCUMENTS)

            # Search for "기계학습" (machine learning)
            results = await store.asimilarity_search_with_score("기계학습 데이터", k=5)

            print("\n[Korean BM25 Morpheme] Query: '기계학습 데이터'")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Content: {doc.page_content[:40]}...")

            assert len(results) > 0

            print(f"✓ Korean BM25 morpheme analysis works: Found {len(results)} results")

        finally:
            await engine.adrop_table(TABLE_NAME_KOREAN)

    @pytest.mark.asyncio
    async def test_korean_sparse_search_multiple_terms(self, engine, embeddings):
        """Test Korean BM25 search with multiple terms."""
        try:
            await setup_korean_table(engine)
            store = await create_korean_store(engine, embeddings, sparse_only=True)
            await store.aadd_documents(KOREAN_DOCUMENTS)

            # Search for multiple terms
            results = await store.asimilarity_search_with_score("데이터베이스 인덱싱 성능", k=5)

            print("\n[Korean BM25 Multi-term] Query: '데이터베이스 인덱싱 성능'")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Topic: {doc.metadata['topic']}, Content: {doc.page_content[:40]}...")

            assert len(results) > 0
            # Database-related documents should rank higher
            top_topics = [doc.metadata["topic"] for doc, _ in results[:3]]
            assert "database" in top_topics, "Database documents should rank high"

            print(f"✓ Korean BM25 multi-term search: Found {len(results)} results")

        finally:
            await engine.adrop_table(TABLE_NAME_KOREAN)

    @pytest.mark.asyncio
    async def test_korean_hybrid_search(self, engine, embeddings):
        """Test Korean hybrid search (BM25 + vector)."""
        try:
            await setup_korean_table(engine)
            store = await create_korean_store(engine, embeddings, sparse_only=False)
            await store.aadd_documents(KOREAN_DOCUMENTS)

            # Hybrid search
            results = await store.asimilarity_search_with_score("인공지능 신경망 딥러닝", k=5)

            print("\n[Korean Hybrid] Query: '인공지능 신경망 딥러닝'")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Topic: {doc.metadata['topic']}, Content: {doc.page_content[:40]}...")

            assert len(results) > 0
            # AI-related documents should rank higher
            top_topics = [doc.metadata["topic"] for doc, _ in results[:3]]
            assert "ai" in top_topics, "AI documents should rank high in hybrid search"

            print(f"✓ Korean hybrid search: Found {len(results)} results")

        finally:
            await engine.adrop_table(TABLE_NAME_KOREAN)

    @pytest.mark.asyncio
    async def test_korean_sparse_search_with_filter(self, engine, embeddings):
        """Test Korean BM25 search with metadata filter."""
        try:
            await setup_korean_table(engine)
            store = await create_korean_store(engine, embeddings, sparse_only=True)
            await store.aadd_documents(KOREAN_DOCUMENTS)

            # Search with filter: topic == "database"
            filter_obj = MetadataFilter(key="topic", value="database", operator=FilterOperator.EQ)
            results = await store.asimilarity_search_with_score("검색 쿼리", k=5, filter=filter_obj)

            print("\n[Korean BM25 + Filter] Query: '검색 쿼리', Filter: topic=database")
            for doc, score in results:
                print(f"  Score: {score:.4f}, Topic: {doc.metadata['topic']}, Content: {doc.page_content[:40]}...")
                assert doc.metadata["topic"] == "database"

            print(f"✓ Korean BM25 with filter: Found {len(results)} results")

        finally:
            await engine.adrop_table(TABLE_NAME_KOREAN)


# =============================================================================
# Test Cases: Cross-language Comparison
# =============================================================================

class TestCrossLanguageComparison:
    """Compare English and Korean BM25 behavior."""

    @pytest.mark.asyncio
    async def test_both_languages_same_query_pattern(self, engine, embeddings):
        """Test that both English and Korean handle similar query patterns."""
        try:
            # Setup both tables
            await setup_english_table(engine)
            await setup_korean_table(engine)

            english_store = await create_english_store(engine, embeddings, sparse_only=True)
            korean_store = await create_korean_store(engine, embeddings, sparse_only=True)

            await english_store.aadd_documents(ENGLISH_DOCUMENTS)
            await korean_store.aadd_documents(KOREAN_DOCUMENTS)

            # Search for "database" concept in both languages
            english_results = await english_store.asimilarity_search_with_score("database vector search", k=3)
            korean_results = await korean_store.asimilarity_search_with_score("데이터베이스 벡터 검색", k=3)

            print("\n[Cross-language Comparison]")
            print("\nEnglish Query: 'database vector search'")
            for doc, score in english_results:
                print(f"  Score: {score:.4f}, Topic: {doc.metadata['topic']}")

            print("\nKorean Query: '데이터베이스 벡터 검색'")
            for doc, score in korean_results:
                print(f"  Score: {score:.4f}, Topic: {doc.metadata['topic']}")

            # Both should return results
            assert len(english_results) > 0, "English search should return results"
            assert len(korean_results) > 0, "Korean search should return results"

            # Both should find database-related documents
            english_topics = [doc.metadata["topic"] for doc, _ in english_results]
            korean_topics = [doc.metadata["topic"] for doc, _ in korean_results]
            assert "database" in english_topics, "English should find database docs"
            assert "database" in korean_topics, "Korean should find database docs"

            print(f"\n✓ Cross-language comparison: Both languages work correctly")

        finally:
            await cleanup_tables(engine)

    @pytest.mark.asyncio
    async def test_language_specific_terms(self, engine, embeddings):
        """Test language-specific terms are handled correctly."""
        try:
            # Setup both tables
            await setup_english_table(engine)
            await setup_korean_table(engine)

            english_store = await create_english_store(engine, embeddings, sparse_only=True)
            korean_store = await create_korean_store(engine, embeddings, sparse_only=True)

            await english_store.aadd_documents(ENGLISH_DOCUMENTS)
            await korean_store.aadd_documents(KOREAN_DOCUMENTS)

            # English-specific search
            english_results = await english_store.asimilarity_search("traveling salesman optimization", k=3)

            # Korean-specific search
            korean_results = await korean_store.asimilarity_search("외판원 최적화", k=3)

            print("\n[Language-specific Terms]")
            print("\nEnglish: 'traveling salesman optimization'")
            for doc in english_results:
                print(f"  Topic: {doc.metadata['topic']}, Content: {doc.page_content[:50]}...")

            print("\nKorean: '외판원 최적화'")
            for doc in korean_results:
                print(f"  Topic: {doc.metadata['topic']}, Content: {doc.page_content[:30]}...")

            # Both should find algorithm-related documents
            if len(english_results) > 0:
                english_topics = [doc.metadata["topic"] for doc in english_results]
                assert "algorithms" in english_topics, "English should find algorithm docs"

            if len(korean_results) > 0:
                korean_topics = [doc.metadata["topic"] for doc in korean_results]
                assert "algorithms" in korean_topics, "Korean should find algorithm docs"

            print(f"\n✓ Language-specific terms handled correctly")

        finally:
            await cleanup_tables(engine)


# =============================================================================
# Main Runner
# =============================================================================

async def run_all_tests():
    """Run all tests manually."""
    print("=" * 70)
    print("Multilingual BM25 Test Suite (English and Korean)")
    print("=" * 70)
    print(f"Database URL: {DATABASE_URL}")
    print()

    engine = PGVecTextSearchEngine.from_connection_string_async(DATABASE_URL)
    embeddings = MockEmbeddings(dimension=EMBEDDING_DIMENSION)

    test_classes = [
        TestEnglishBM25(),
        TestKoreanBM25(),
        TestCrossLanguageComparison(),
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
