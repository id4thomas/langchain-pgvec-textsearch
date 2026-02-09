"""
Indexing Script for MLDR Dataset (English)

Creates:
- Table with English corpus
- HNSW index for dense vector search
- BM25 index with english text search config
"""
import asyncio
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pgvec_textsearch import (
    PGVecTextSearchStore,
    PGVecTextSearchEngine,
    HybridSearchConfig,
    DistanceStrategy,
    HNSWIndex,
    BM25Index,
    reciprocal_rank_fusion
)
from langchain_postgres import Column

from config import settings

DATA_NAME = "MLDR"
LANG = "en"
TABLE_NAME = f"{DATA_NAME}-{LANG}-documents"
TEXT_CONFIG = "english"

# Set to None for full corpus, or a number to limit for testing
MAX_DOCS = None  # e.g., 10000 for quick testing

DATABASE_URL = "postgresql+asyncpg://{}:{}@{}:{}/{}".format(
    settings.postgres_user,
    settings.postgres_password,
    settings.postgres_ip,
    settings.postgres_port,
    settings.postgres_db,
)


async def main():
    # Initialize PG DB
    engine = PGVecTextSearchEngine.from_connection_string_async(DATABASE_URL)

    await engine.adrop_table(TABLE_NAME)

    # Create table with indexes
    await engine.ainit_hybrid_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=settings.embedding_dim,
        metadata_columns=[
            Column("docid", "TEXT"),
        ],
        hnsw_index=HNSWIndex(
            m=32,
            ef_construction=128,
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        ),
        bm25_index=BM25Index(
            text_config=TEXT_CONFIG,
        ),
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
        table_name=TABLE_NAME,
        hybrid_search_config=HybridSearchConfig(
            enable_dense=True,
            enable_sparse=True,
            dense_top_k=20,
            sparse_top_k=20,
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters={"rrf_k": 60},
        ),
    )

    # Load corpus from local JSONL file
    corpus_path = Path(settings.data_dir) / f"mldr-v1.0-{LANG}" / "corpus.jsonl"
    print(f"Loading MLDR corpus for language '{LANG}' from {corpus_path}...")

    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if MAX_DOCS and len(docs) >= MAX_DOCS:
                break
            x = json.loads(line)
            docs.append(
                Document(
                    page_content=x["text"][:settings.max_length],
                    metadata={
                        "docid": x["docid"],
                    },
                )
            )

    if MAX_DOCS and len(docs) >= MAX_DOCS:
        print(f"Reached MAX_DOCS={MAX_DOCS} limit")

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
                if batch_idx % 100 == 0:
                    print(f"Indexed {indexed_count}/{len(docs)} documents")

    # Create all tasks
    tasks = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        tasks.append(index_batch(i // batch_size, batch))

    # Run with concurrency
    print(f"Indexing with concurrency={concurrency}...")
    await asyncio.gather(*tasks)

    print(f"\nIndexing complete: {len(docs)} English documents indexed")
    print(f"Table: {TABLE_NAME}")
    print(f"BM25 text_config: {TEXT_CONFIG}")


if __name__ == "__main__":
    asyncio.run(main())
