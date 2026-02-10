"""
Indexing Script for Ko-StrategyQA Dataset (Korean)

Creates:
- Table with Korean corpus
- HNSW index for dense vector search
- BM25 index with public.korean text search config
"""
import asyncio
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pgvec_textsearch import (
    PGVecTextSearchStore,
    AsyncPGVecTextSearchEngine,
    TableConfig,
    SearchConfig,
    HNSWIndexConfig,
    BM25IndexConfig,
    HNSWSearchConfig,
    BM25SearchConfig,
    DistanceStrategy,
)

from config import settings

DATA_NAME = "KoStrategyQA"
TABLE_NAME = f"{DATA_NAME}-documents"  # Korean table
CORPUS_LANG = "ko"
TEXT_CONFIG = "public.korean"

DATABASE_URL = "postgresql+asyncpg://{}:{}@{}:{}/{}".format(
    settings.postgres_user,
    settings.postgres_password,
    settings.postgres_ip,
    settings.postgres_port,
    settings.postgres_db,
)

async def main():
    # Initialize PG DB
    engine = AsyncPGVecTextSearchEngine.from_connection_string(DATABASE_URL)

    table_config = TableConfig(
        table_name=TABLE_NAME,
        vector_size=settings.embedding_dim,
    )

    # Create table with indexes
    await engine.init_database(
        table_config=table_config,
        hnsw_config=HNSWIndexConfig(
            m=32,
            ef_construction=128,
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        ),
        bm25_config=BM25IndexConfig(
            text_config=TEXT_CONFIG,
        ),
        overwrite_existing=True,
    )

    # Create VectorStore
    print(f"EMBEDDING: {settings.embedding_model}")
    embedding_service = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        dimensions=settings.embedding_dim,
        check_embedding_ctx_length=False
    )

    store = await PGVecTextSearchStore.create(
        engine=engine,
        embedding_service=embedding_service,
        table_config=table_config,
        search_config=SearchConfig(
            enable_dense=True,
            enable_sparse=True,
            hnsw=HNSWSearchConfig(
                distance_strategy=DistanceStrategy.COSINE_DISTANCE,
            ),
            bm25=BM25SearchConfig(
                text_config=TEXT_CONFIG,
            ),
        ),
    )

    # Prepare Documents (Korean)
    corpus_ds = load_dataset(settings.data_dir, "corpus")

    print(f"Loading {CORPUS_LANG} corpus...")
    docs = [
        Document(
            page_content=f"{x['title']}\n{x['text']}"[:settings.max_length],
            metadata={
                "corpus_id": x["_id"],
                "corpus_title": x["title"],
            },
        )
        for x in corpus_ds[CORPUS_LANG]
    ]
    print(f"Loaded {len(docs)} documents")

    # Index Documents with concurrency
    batch_size = 64
    concurrency = 16
    semaphore = asyncio.Semaphore(concurrency)
    indexed_count = 0
    lock = asyncio.Lock()

    async def index_batch(batch_idx: int, batch: list[Document]):
        nonlocal indexed_count
        async with semaphore:
            await store.aadd_documents(batch)
            async with lock:
                indexed_count += len(batch)
                if batch_idx % 10 == 0:
                    print(f"Indexed {indexed_count}/{len(docs)} documents")

    # Create all tasks
    tasks = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        tasks.append(index_batch(i // batch_size, batch))

    # Run with concurrency
    print(f"Indexing with concurrency={concurrency}...")
    await asyncio.gather(*tasks)

    print(f"\nIndexing complete: {len(docs)} Korean documents indexed")
    print(f"Table: {TABLE_NAME}")
    print(f"BM25 text_config: {TEXT_CONFIG}")

    await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
