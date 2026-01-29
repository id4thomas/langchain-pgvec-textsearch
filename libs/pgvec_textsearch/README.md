# PGVecTextSearch

LangChain VectorStore implementation for hybrid search combining pgvector (dense vectors) and pg_textsearch (BM25 sparse search).

## Features

- **Dense Search**: pgvector HNSW index for semantic similarity search
- **Sparse Search**: pg_textsearch BM25 index for keyword-based search
- **Hybrid Search**: Combines dense and sparse results using RRF (Reciprocal Rank Fusion)
- **Type-safe Filtering**: LlamaIndex-style `MetadataFilter`/`MetadataFilters` for metadata filtering
- **Multiple Distance Strategies**: Cosine distance, Euclidean distance, Inner product

## Installation

```bash
pip install langchain-pgvec-textsearch
```

### Database Requirements

- PostgreSQL 17 or 18
- pgvector extension
- pg_textsearch extension (for BM25 support)

## Quick Start

```python
import asyncio
from langchain_pgvec_textsearch import (
    PGVecTextSearchStore,
    PGVecTextSearchEngine,
    HybridSearchConfig,
    DistanceStrategy,
    HNSWIndex,
    BM25Index,
    Column,
)
from langchain_openai import OpenAIEmbeddings

DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/dbname"

async def main():
    # Create engine
    engine = PGVecTextSearchEngine.from_connection_string_async(DATABASE_URL)

    # Create table with indexes
    await engine.ainit_hybrid_vectorstore_table(
        table_name="documents",
        vector_size=1536,
        metadata_columns=[
            Column("category", "TEXT"),
            Column("year", "INTEGER"),
        ],
        hnsw_index=HNSWIndex(
            name="idx_documents_hnsw",
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        ),
        bm25_index=BM25Index(
            name="idx_documents_bm25",
            text_config="english",  # or "public.korean" for Korean
        ),
    )

    # Create vectorstore
    embeddings = OpenAIEmbeddings()
    store = await PGVecTextSearchStore.create(
        engine=engine,
        embedding_service=embeddings,
        table_name="documents",
        metadata_columns=["category", "year"],
        hybrid_search_config=HybridSearchConfig(
            enable_dense=True,
            enable_sparse=True,
        ),
    )

    # Add documents
    from langchain_core.documents import Document
    docs = [
        Document(page_content="AI is transforming industries", metadata={"category": "tech", "year": 2024}),
        Document(page_content="Machine learning models require data", metadata={"category": "tech", "year": 2023}),
    ]
    await store.aadd_documents(docs)

    # Search
    results = await store.asimilarity_search("artificial intelligence", k=5)
    for doc in results:
        print(doc.page_content)

asyncio.run(main())
```

## Search Modes

### Dense Search Only (Semantic Similarity)

Uses pgvector HNSW index for embedding-based similarity search.

```python
store = await PGVecTextSearchStore.create(
    engine=engine,
    embedding_service=embeddings,
    table_name="documents",
    hybrid_search_config=HybridSearchConfig(
        enable_dense=True,
        enable_sparse=False,  # Disable BM25
    ),
)

# Search by semantic similarity
results = await store.asimilarity_search("artificial intelligence", k=5)
```

### Sparse Search Only (BM25 Keyword Search)

Uses pg_textsearch BM25 index for keyword-based search.

```python
store = await PGVecTextSearchStore.create(
    engine=engine,
    embedding_service=embeddings,
    table_name="documents",
    hybrid_search_config=HybridSearchConfig(
        enable_dense=False,  # Disable vector search
        enable_sparse=True,
        bm25_index_name="idx_documents_bm25",  # Must specify BM25 index name
    ),
)

# Search by keywords (BM25)
results = await store.asimilarity_search("machine learning data", k=5)
```

### Hybrid Search (Dense + Sparse with RRF)

Combines both search methods using Reciprocal Rank Fusion.

```python
from langchain_pgvec_textsearch import reciprocal_rank_fusion, weighted_sum_ranking

store = await PGVecTextSearchStore.create(
    engine=engine,
    embedding_service=embeddings,
    table_name="documents",
    hybrid_search_config=HybridSearchConfig(
        enable_dense=True,
        enable_sparse=True,
        dense_top_k=20,   # Fetch top 20 from dense search
        sparse_top_k=20,  # Fetch top 20 from sparse search
        fusion_function=reciprocal_rank_fusion,  # Default
        fusion_function_parameters={"rrf_k": 60},
        bm25_index_name="idx_documents_bm25",
    ),
)

# Hybrid search combines semantic and keyword matching
results = await store.asimilarity_search_with_score("AI machine learning", k=10)
for doc, score in results:
    print(f"Score: {score:.4f}, Content: {doc.page_content}")
```

#### Custom Fusion Function

You can use weighted sum ranking instead of RRF:

```python
store = await PGVecTextSearchStore.create(
    engine=engine,
    embedding_service=embeddings,
    table_name="documents",
    hybrid_search_config=HybridSearchConfig(
        enable_dense=True,
        enable_sparse=True,
        fusion_function=weighted_sum_ranking,
        fusion_function_parameters={
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
        },
    ),
)
```

## Metadata Filtering

PGVecTextSearch uses type-safe `MetadataFilter` and `MetadataFilters` classes for filtering.

### Basic Filter (Single Condition)

```python
from langchain_pgvec_textsearch import MetadataFilter, FilterOperator

# Filter: category == "tech"
filter_obj = MetadataFilter(
    key="category",
    value="tech",
    operator=FilterOperator.EQ
)
results = await store.asimilarity_search("AI", k=5, filter=filter_obj)
```

### Filter Operators

| Operator | Description | Example Value |
|----------|-------------|---------------|
| `EQ` | Equals | `"tech"` |
| `NE` | Not equals | `"tech"` |
| `GT` | Greater than | `4.5` |
| `GTE` | Greater than or equal | `2024` |
| `LT` | Less than | `2024` |
| `LTE` | Less than or equal | `2023` |
| `IN` | Value in list | `["tech", "science"]` |
| `NIN` | Value not in list | `["sports", "food"]` |
| `BETWEEN` | Value between range | `[2020, 2024]` |
| `TEXT_MATCH` | LIKE pattern (case-sensitive) | `"data%"` |
| `TEXT_MATCH_INSENSITIVE` | ILIKE pattern (case-insensitive) | `"%search%"` |
| `EXISTS` | Field exists (not null) | `True` |
| `IS_EMPTY` | Field is null or empty | `True` |
| `ANY` | Array contains any | `["a", "b"]` |
| `ALL` | Array contains all | `["a", "b"]` |
| `CONTAINS` | Array contains value | `"tag"` |

### Multiple Filters with AND

```python
from langchain_pgvec_textsearch import MetadataFilters, FilterCondition

# Filter: category == "tech" AND year >= 2024
filter_obj = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
        MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
    ],
    condition=FilterCondition.AND
)
results = await store.asimilarity_search("AI", k=5, filter=filter_obj)
```

### Multiple Filters with OR

```python
# Filter: category == "tech" OR category == "science"
filter_obj = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
        MetadataFilter(key="category", value="science", operator=FilterOperator.EQ),
    ],
    condition=FilterCondition.OR
)
results = await store.asimilarity_search("research", k=5, filter=filter_obj)
```

### NOT Condition

```python
# Filter: NOT (category == "sports")
filter_obj = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="sports", operator=FilterOperator.EQ),
    ],
    condition=FilterCondition.NOT
)
results = await store.asimilarity_search("news", k=5, filter=filter_obj)
```

### Nested Filters (Complex Logic)

```python
# Filter: (category == "tech" AND rating >= 4.5) OR category == "science"
filter_obj = MetadataFilters(
    filters=[
        MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
                MetadataFilter(key="rating", value=4.5, operator=FilterOperator.GTE),
            ],
            condition=FilterCondition.AND
        ),
        MetadataFilter(key="category", value="science", operator=FilterOperator.EQ),
    ],
    condition=FilterCondition.OR
)
results = await store.asimilarity_search("research", k=5, filter=filter_obj)
```

### Filter with IN Operator

```python
# Filter: category IN ["tech", "science", "health"]
filter_obj = MetadataFilter(
    key="category",
    value=["tech", "science", "health"],
    operator=FilterOperator.IN
)
results = await store.asimilarity_search("AI", k=5, filter=filter_obj)
```

### Filter with BETWEEN Operator

```python
# Filter: year BETWEEN 2020 AND 2024
filter_obj = MetadataFilter(
    key="year",
    value=[2020, 2024],
    operator=FilterOperator.BETWEEN
)
results = await store.asimilarity_search("AI", k=5, filter=filter_obj)
```

### Filter with Text Pattern Matching

```python
# Filter: title LIKE "Introduction%"
filter_obj = MetadataFilter(
    key="title",
    value="Introduction%",
    operator=FilterOperator.TEXT_MATCH
)

# Case-insensitive: title ILIKE "%machine learning%"
filter_obj = MetadataFilter(
    key="title",
    value="%machine learning%",
    operator=FilterOperator.TEXT_MATCH_INSENSITIVE
)
```

## Distance Strategies

```python
from langchain_pgvec_textsearch import DistanceStrategy

# Cosine Distance (default) - good for normalized embeddings
DistanceStrategy.COSINE_DISTANCE

# Euclidean Distance - L2 distance
DistanceStrategy.EUCLIDEAN_DISTANCE

# Inner Product - dot product similarity
DistanceStrategy.INNER_PRODUCT
```

Configure in HNSWIndex:

```python
hnsw_index = HNSWIndex(
    name="idx_documents_hnsw",
    distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    m=16,           # Max connections per node
    ef_construction=64,  # Size of dynamic candidate list
)
```

## Index Configuration

### HNSW Index (Dense Vectors)

```python
hnsw_index = HNSWIndex(
    name="idx_documents_hnsw",
    distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    m=16,               # Max connections per layer (default: 16)
    ef_construction=64, # Construction time quality (default: 64)
)
```

### IVFFlat Index (Alternative Dense Index)

```python
from langchain_pgvec_textsearch import IVFFlatIndex

ivfflat_index = IVFFlatIndex(
    name="idx_documents_ivfflat",
    distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    lists=100,  # Number of clusters
)
```

### BM25 Index (Sparse Search)

```python
bm25_index = BM25Index(
    name="idx_documents_bm25",
    text_config="english",  # PostgreSQL text search config
    k1=1.2,                 # Term frequency saturation (default: 1.2)
    b=0.75,                 # Length normalization (default: 0.75)
)
```

For Korean text:

```python
bm25_index = BM25Index(
    name="idx_documents_bm25",
    text_config="public.korean",  # Korean text search config
)
```

## Additional Operations

### Delete Documents

```python
# Delete by IDs
await store.adelete(ids=["id1", "id2", "id3"])

# Delete by filter
filter_obj = MetadataFilter(key="category", value="outdated", operator=FilterOperator.EQ)
await store.adelete(filter=filter_obj)
```

### MMR Search (Maximal Marginal Relevance)

```python
# Get diverse results
results = await store.amax_marginal_relevance_search(
    query="machine learning",
    k=5,
    fetch_k=20,
    lambda_mult=0.5,  # 0 = max diversity, 1 = max relevance
    filter=filter_obj,
)
```

### Get Documents by IDs

```python
docs = await store.aget_by_ids(["id1", "id2", "id3"])
```

## API Reference

### PGVecTextSearchEngine

| Method | Description |
|--------|-------------|
| `from_connection_string_async(url)` | Create engine from connection string |
| `ainit_hybrid_vectorstore_table(...)` | Create table with indexes |
| `adrop_table(table_name)` | Drop a table |

### PGVecTextSearchStore

| Method | Description |
|--------|-------------|
| `create(engine, embedding_service, table_name, ...)` | Create store instance |
| `aadd_documents(documents, ids)` | Add documents |
| `aadd_texts(texts, metadatas, ids)` | Add texts with metadata |
| `asimilarity_search(query, k, filter)` | Search by query |
| `asimilarity_search_with_score(query, k, filter)` | Search with scores |
| `amax_marginal_relevance_search(query, k, fetch_k, lambda_mult, filter)` | MMR search |
| `adelete(ids, filter)` | Delete documents |
| `aget_by_ids(ids)` | Get documents by IDs |

### HybridSearchConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_dense` | bool | True | Enable vector search |
| `enable_sparse` | bool | True | Enable BM25 search |
| `dense_top_k` | int | 20 | Top K for dense search |
| `sparse_top_k` | int | 20 | Top K for sparse search |
| `fusion_function` | callable | `reciprocal_rank_fusion` | Fusion function |
| `fusion_function_parameters` | dict | {} | Fusion function params |
| `bm25_index_name` | str | None | BM25 index name |

## License

MIT License
