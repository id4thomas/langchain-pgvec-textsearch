"""
Indexing Script for markers_bm Dataset

Usage:
    python 2-index.py --config configs/ko-qwen3.yaml
"""
import argparse
import asyncio
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pgvec_textsearch import (
    PGVecTextSearchStore,
    AsyncPGVecTextSearchEngine,
    TableConfig,
)

from config import (
    settings,
    load_experiment_config,
    build_hnsw_index_config,
    build_bm25_index_config,
    build_search_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Index markers_bm corpus")
    parser.add_argument("--config", type=str, required=True, help="Path to index YAML config")
    return parser.parse_args()


async def main(exp: dict):
    ds = exp["dataset"]
    idx = exp["index"]
    emb = exp.get("embedding", {})

    table_name = ds["table_name"]
    text_config = ds["text_config"]
    max_length = ds.get("max_length", settings.max_length)

    database_url = "postgresql+asyncpg://{}:{}@{}:{}/{}".format(
        settings.postgres_user,
        settings.postgres_password,
        settings.postgres_ip,
        settings.postgres_port,
        settings.postgres_db,
    )

    engine = AsyncPGVecTextSearchEngine.from_connection_string(database_url)

    embedding_dim = emb.get("dim", settings.embedding_dim)
    table_config = TableConfig(
        table_name=table_name,
        vector_size=embedding_dim,
    )

    await engine.init_database(
        table_config=table_config,
        hnsw_config=build_hnsw_index_config(idx["hnsw"]),
        bm25_config=build_bm25_index_config(idx["bm25"], text_config),
        overwrite_existing=True,
    )

    embedding_model = emb.get("model", settings.embedding_model)
    embedding_base_url = emb.get("base_url", settings.embedding_base_url)
    embedding_api_key = emb.get("api_key", settings.embedding_api_key)
    print(f"EMBEDDING: {embedding_model}")
    embedding_service = OpenAIEmbeddings(
        model=embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
        dimensions=embedding_dim,
        check_embedding_ctx_length=False,
    )

    store = await PGVecTextSearchStore.create(
        engine=engine,
        embedding_service=embedding_service,
        table_config=table_config,
        search_config=build_search_config(
            {"enable_dense": True, "enable_sparse": True, "hnsw": idx["hnsw"], "bm25": idx["bm25"]},
            text_config,
        ),
    )

    # Load corpus from HuggingFace dataset
    corpus_ds = load_dataset(settings.data_dir, "corpus")

    print(f"Loading corpus...")
    docs = [
        Document(
            page_content=f"{x['title']}\n{x['text']}"[:max_length],
            metadata={
                "corpus_id": x["_id"],
                "corpus_title": x["title"],
                "domain": x["_id"].split(" - ")[0],
            },
        )
        for x in corpus_ds["corpus"]
    ]
    print(f"Loaded {len(docs)} documents")

    batch_size = idx.get("batch_size", 64)
    concurrency = idx.get("concurrency", 16)
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

    tasks = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        tasks.append(index_batch(i // batch_size, batch))

    print(f"Indexing with concurrency={concurrency}...")
    await asyncio.gather(*tasks)

    print(f"\nIndexing complete: {len(docs)} documents indexed")
    print(f"Table: {table_name}")
    print(f"BM25 text_config: {text_config}")

    await engine.close()


if __name__ == "__main__":
    args = parse_args()
    exp = load_experiment_config(args.config)
    asyncio.run(main(exp))
