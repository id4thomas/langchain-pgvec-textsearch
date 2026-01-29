#!/usr/bin/env python3
"""
Quick test script for PGVecTextSearchStore with all distance strategies.

Usage:
    # With default connection string (localhost:9010):
    python run_quick_test.py

    # With custom connection string:
    python run_quick_test.py "postgresql+asyncpg://user:pass@host:port/db"

    # Or set environment variable:
    DATABASE_URL="postgresql+asyncpg://..." python run_quick_test.py
"""
import asyncio
import os
import sys
from typing import List

# Add parent directory to path
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


class MockEmbeddings(Embeddings):
    """Simple mock embeddings for testing."""

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


# Sample documents
DOCUMENTS = [
    Document(page_content="인공지능 기술이 발전하면서 우리 삶은 더욱 편리해지고 있습니다.", metadata={"cat": "tech"}),
    Document(page_content="데이터베이스 성능 최적화를 위해서는 인덱스 설계가 매우 중요합니다.", metadata={"cat": "tech"}),
    Document(page_content="파이썬은 배우기 쉽고 라이브러리가 풍부해서 데이터 분석에 최적입니다.", metadata={"cat": "prog"}),
    Document(page_content="제주도 여행을 갔을 때 보았던 푸른 바다가 아직도 눈에 선합니다.", metadata={"cat": "travel"}),
    Document(page_content="벡터 데이터베이스는 유사도 검색에 최적화된 저장소입니다.", metadata={"cat": "tech"}),
    Document(page_content="BM25는 정보 검색에서 널리 사용되는 랭킹 알고리즘입니다.", metadata={"cat": "tech"}),
    Document(page_content="하이브리드 검색은 밀집 벡터와 희소 벡터를 결합한 방식입니다.", metadata={"cat": "tech"}),
]


async def test_distance_strategy(
    engine: PGVecTextSearchEngine,
    embeddings: Embeddings,
    strategy: DistanceStrategy,
    table_name: str,
):
    """Test a specific distance strategy."""
    print(f"\n{'='*50}")
    print(f"Testing: {strategy.name}")
    print(f"Table: {table_name}")
    print("="*50)

    try:
        # 1. Drop existing table
        print("→ Dropping existing table...")
        await engine.adrop_table(table_name)

        # 2. Create table with indexes
        print("→ Creating table with HNSW and BM25 indexes...")
        await engine.ainit_hybrid_vectorstore_table(
            table_name=table_name,
            vector_size=384,
            overwrite_existing=True,
            bm25_index=BM25Index(
                name=f"idx_{table_name}_bm25",
                text_config="public.korean",
            ),
            hnsw_index=HNSWIndex(
                name=f"idx_{table_name}_hnsw",
                distance_strategy=strategy,
            ),
        )

        # 3. Create store
        print("→ Creating PGVecTextSearchStore...")
        store = await PGVecTextSearchStore.create(
            engine=engine,
            embedding_service=embeddings,
            table_name=table_name,
            distance_strategy=strategy,
            hybrid_search_config=HybridSearchConfig(
                enable_dense=True,
                enable_sparse=True,
                dense_top_k=10,
                sparse_top_k=10,
            ),
        )

        # 4. Add documents
        print(f"→ Adding {len(DOCUMENTS)} documents...")
        ids = await store.aadd_documents(DOCUMENTS)
        print(f"  Added IDs: {ids[:3]}...")

        # 5. Test hybrid search
        query = "인공지능 기술 발전"
        print(f"\n→ Hybrid Search: '{query}'")
        results = await store.asimilarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [score={score:.4f}] {doc.page_content[:50]}...")

        # 6. Test BM25-only search
        print(f"\n→ BM25-only Search: '{query}'")
        store.hybrid_search_config.enable_dense = False
        bm25_results = await store.asimilarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(bm25_results, 1):
            print(f"  {i}. [bm25={score:.4f}] {doc.page_content[:50]}...")
        store.hybrid_search_config.enable_dense = True

        # 7. Test dense-only search
        print(f"\n→ Dense-only Search: '{query}'")
        store.hybrid_search_config.enable_sparse = False
        dense_results = await store.asimilarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(dense_results, 1):
            print(f"  {i}. [dist={score:.4f}] {doc.page_content[:50]}...")
        store.hybrid_search_config.enable_sparse = True

        # 8. Test get_by_ids
        print(f"\n→ Testing get_by_ids...")
        fetched = await store.aget_by_ids(ids[:2])
        print(f"  Fetched {len(fetched)} documents")

        # 9. Test delete
        print(f"\n→ Testing delete...")
        await store.adelete(ids=[ids[0]])
        remaining = await store.aget_by_ids([ids[0]])
        print(f"  After delete: {len(remaining)} documents (should be 0)")

        print(f"\n✅ {strategy.name} test PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ {strategy.name} test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("Cleanup")
        try:
            await engine.adrop_table(table_name)
        except:
            pass


async def main(connection_string: str):
    """Run all tests."""
    print("=" * 60)
    print("PGVecTextSearchStore Quick Test")
    print("=" * 60)
    print(f"Connection: {connection_string}")

    # Use from_connection_string_async for direct async usage with asyncio.run()
    engine = PGVecTextSearchEngine.from_connection_string_async(connection_string)
    embeddings = MockEmbeddings(dimension=384)

    strategies = [
        (DistanceStrategy.COSINE_DISTANCE, "test_cosine"),
        (DistanceStrategy.EUCLIDEAN, "test_euclidean"),
        (DistanceStrategy.INNER_PRODUCT, "test_inner_product"),
    ]

    results = {}
    for strategy, table_name in strategies:
        results[strategy.name] = await test_distance_strategy(
            engine, embeddings, strategy, table_name
        )

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. 😢")
        sys.exit(1)


if __name__ == "__main__":
    # Get connection string from argument, env, or use default
    if len(sys.argv) > 1:
        conn_str = sys.argv[1]
    else:
        conn_str = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:9010/postgres"
        )

    asyncio.run(main(conn_str))
