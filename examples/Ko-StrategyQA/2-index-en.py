"""
Indexing Script for Ko-StrategyQA Dataset (English)

Creates:
- Table with English corpus
- HNSW index for dense vector search
- BM25 index with english text search config
"""
import asyncio
from datasets import load_dataset
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

DATA_NAME = "KoStrategyQA"
TABLE_NAME = f"{DATA_NAME}-documents-en"  # English table
CORPUS_LANG = "en"
TEXT_CONFIG = "english"

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
            Column("corpus_id", "TEXT"),
            Column("corpus_title", "TEXT"),
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

    # Prepare Documents (English)
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

    # Index Documents
    batch_size = 64
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        await store.aadd_documents(batch)
        if (i // batch_size) % 10 == 0:
            print(f"Indexed {min(i+batch_size, len(docs))}/{len(docs)} documents")

    print(f"\nIndexing complete: {len(docs)} English documents indexed")
    print(f"Table: {TABLE_NAME}")
    print(f"BM25 text_config: {TEXT_CONFIG}")


if __name__ == "__main__":
    asyncio.run(main())
