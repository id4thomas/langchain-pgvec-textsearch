"""
PGVecTextSearch VectorStore - Hybrid Search with pgvector and pg_textsearch.

Combines:
- Dense search: pgvector HNSW index
- Sparse search: pg_textsearch BM25 index
- Fusion: RRF (Reciprocal Rank Fusion)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy import RowMapping, text

from .config import (
    SearchConfig,
    TableConfig,
)
from .data import Row
from .engine import AsyncPGVecTextSearchEngine
from .filters import (
    MetadataFilter,
    MetadataFilters,
    build_filter_clause,
)
from .search_utils import reciprocal_rank_fusion


class PGVecTextSearchStore(VectorStore):
    """LangChain VectorStore for hybrid search with pgvector + pg_textsearch.

    Delegates database operations to AsyncPGVecTextSearchEngine.

    Provides:
    - Dense vector search (HNSW via pgvector)
    - Sparse keyword search (BM25 via pg_textsearch)
    - Hybrid search combining both with RRF fusion
    """

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AsyncPGVecTextSearchEngine,
        embedding_service: Embeddings,
        table_config: TableConfig,
        *,
        search_config: Optional[SearchConfig] = None,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        insert_batch_size: int = 500,
    ):
        """PGVecTextSearchStore constructor.

        Args:
            key: Prevent direct constructor usage.
            engine: AsyncPGVecTextSearchEngine instance.
            embedding_service: Text embedding model.
            table_config: Table schema configuration.
            search_config: Hybrid search config (includes hnsw/bm25 params).
            k: Number of results to return. Default: 4.
            fetch_k: Number of candidates for MMR. Default: 20.
            lambda_mult: MMR diversity parameter. Default: 0.5.
            insert_batch_size: Batch size for bulk inserts. Default: 500.
        """
        if key != PGVecTextSearchStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync'!"
            )

        self.engine = engine
        self.embedding_service = embedding_service
        self.table_config = table_config
        self.search_config = search_config or SearchConfig()
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.insert_batch_size = insert_batch_size

    @classmethod
    async def create(
        cls,
        engine: AsyncPGVecTextSearchEngine,
        embedding_service: Embeddings,
        table_config: TableConfig,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore instance asynchronously.

        Validates that the table exists with required columns.
        """
        tc = table_config
        stmt = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = :table_name
              AND table_schema = :schema_name
        """
        async with engine._pool.connect() as conn:
            result = await conn.execute(
                text(stmt),
                {
                    "table_name": tc.table_name,
                    "schema_name": tc.schema_name,
                },
            )
            results = result.mappings().fetchall()

        columns = {
            r["column_name"]: r["data_type"] for r in results
        }

        if tc.id_column not in columns:
            raise ValueError(
                f"Id column '{tc.id_column}' does not exist."
            )
        if tc.content_column not in columns:
            raise ValueError(
                f"Content column '{tc.content_column}' does not exist."
            )
        if tc.embedding_column not in columns:
            raise ValueError(
                f"Embedding column '{tc.embedding_column}' does not exist."
            )

        return cls(
            cls.__create_key,
            engine,
            embedding_service,
            table_config,
            **kwargs,
        )

    @classmethod
    def create_sync(
        cls,
        engine: AsyncPGVecTextSearchEngine,
        embedding_service: Embeddings,
        table_config: TableConfig,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore instance synchronously."""
        return asyncio.get_event_loop().run_until_complete(
            cls.create(engine, embedding_service, table_config, **kwargs)
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_service

    # ==========================================
    # Helpers
    # ==========================================

    def _row_to_document(
        self,
        row: dict | RowMapping,
    ) -> tuple[Document, float]:
        """Convert a database row to (Document, score)."""
        tc = self.table_config
        metadata = row.get(tc.metadata_column, {})
        if metadata is None:
            metadata = {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        score = (
            row.get("rrf_score")
            or row.get("distance")
            or row.get("bm25_score", 0.0)
        )

        doc = Document(
            page_content=row[tc.content_column],
            metadata=metadata,
            id=str(row[tc.id_column]),
        )
        return doc, float(score)

    # ==========================================
    # Async Methods (Core Implementation)
    # ==========================================

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add data along with pre-computed embeddings."""
        texts_list = list(texts)
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        else:
            ids = [
                id_ if id_ is not None else str(uuid.uuid4())
                for id_ in ids
            ]
        if not metadatas:
            metadatas = [{} for _ in texts_list]

        rows = [
            Row(id=id_, content=txt, embedding=emb, metadata=meta)
            for id_, txt, emb, meta
            in zip(ids, texts_list, embeddings, metadatas)
        ]

        return await self.engine.insert_rows(
            self.table_config,
            rows,
            batch_size=self.insert_batch_size,
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts and add to the table."""
        texts_list = list(texts)
        embeddings = await self.embedding_service.aembed_documents(
            texts_list
        )
        return await self.aadd_embeddings(
            texts_list, embeddings,
            metadatas=metadatas, ids=ids, **kwargs,
        )

    async def aadd_documents(
        self,
        documents: list[Document],
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed documents and add to the table."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if not ids:
            ids = [doc.id for doc in documents]
        return await self.aadd_texts(
            texts, metadatas=metadatas, ids=ids, **kwargs,
        )

    async def adelete(
        self,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records by IDs."""
        if not ids:
            return False

        tc = self.table_config
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        param_dict = {f"id_{i}": id_ for i, id_ in enumerate(ids)}

        query = (
            f'DELETE FROM "{tc.escaped_schema_name}"."{tc.escaped_table_name}"'
            f' WHERE "{tc.escaped_id_column}" IN ({placeholders})'
        )

        async with self.engine._pool.connect() as conn:
            await conn.execute(text(query), param_dict)
            await conn.commit()
        return True

    # ==========================================
    # Filter Support
    # ==========================================

    def _resolve_filter(
        self,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
    ) -> tuple[str, dict]:
        """Convert filter to SQL WHERE clause.

        Supports:
        - dict: Uses JSON containment (@>)
        - MetadataFilter/MetadataFilters: Uses LlamaIndex-style filter syntax
        - None: No filter
        """
        if filter is None:
            return "", {}
        if isinstance(filter, dict):
            return self.engine.build_filter_clause(self.table_config, filter)
        return build_filter_clause(
            filters=filter,
            json_column=self.table_config.metadata_column,
        )

    # ==========================================
    # Search Methods
    # ==========================================

    async def _aquery_hybrid(
        self,
        query: str,
        embedding: list[float],
        k: int,
        filter_clause: str = "",
        filter_params: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search combining dense and sparse results."""
        config = self.search_config

        dense_results: Sequence[RowMapping] = []
        sparse_results: Sequence[RowMapping] = []

        if config.enable_dense:
            dense_results = await self.engine.query_hnsw(
                self.table_config,
                embedding,
                self.search_config.hnsw,
                filter_clause,
                filter_params,
            )

        if config.enable_sparse:
            sparse_results = await self.engine.query_bm25(
                self.table_config,
                query,
                self.search_config.bm25,
                filter_clause=filter_clause,
                filter_params=filter_params,
            )

        # If only one type is enabled, return directly
        if not config.enable_sparse:
            return [dict(row) for row in dense_results[:k]]
        if not config.enable_dense:
            return [dict(row) for row in sparse_results[:k]]

        # Fuse results
        return reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            id_column=self.table_config.id_column,
            fetch_top_k=k,
        )

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search on query."""
        embedding = await self.embedding_service.aembed_query(text=query)
        return await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter,
            query=query, **kwargs,
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores selected by similarity search."""
        embedding = await self.embedding_service.aembed_query(text=query)
        return await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter,
            query=query, **kwargs,
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by vector similarity."""
        docs_and_scores = (
            await self.asimilarity_search_with_score_by_vector(
                embedding=embedding, k=k, filter=filter, **kwargs,
            )
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores selected by vector similarity."""
        final_k = k if k is not None else self.k
        query_text = kwargs.get("query", "")
        filter_clause, filter_params = self._resolve_filter(filter)

        # Use hybrid search if query text available and sparse enabled
        if query_text and self.search_config.enable_sparse:
            results = await self._aquery_hybrid(
                query_text, embedding, final_k,
                filter_clause, filter_params,
            )
        else:
            # Dense-only: override config k with requested k
            config = self.search_config.hnsw.model_copy(
                update={"k": final_k}
            )
            results = await self.engine.query_hnsw(
                self.table_config, embedding, config,
                filter_clause, filter_params,
            )

        return [self._row_to_document(row) for row in results]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs using maximal marginal relevance."""
        embedding = await self.embedding_service.aembed_query(text=query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs using maximal marginal relevance by vector."""
        final_k = k if k is not None else self.k
        final_fetch_k = fetch_k if fetch_k is not None else self.fetch_k
        final_lambda = (
            lambda_mult if lambda_mult is not None else self.lambda_mult
        )
        filter_clause, filter_params = self._resolve_filter(filter)

        # Fetch more candidates for MMR
        config = self.search_config.hnsw.model_copy(
            update={"k": final_fetch_k}
        )
        results = await self.engine.query_hnsw(
            self.table_config, embedding, config,
            filter_clause, filter_params,
        )

        if not results:
            return []

        tc = self.table_config
        embedding_list = [
            json.loads(row[tc.embedding_column])
            for row in results
        ]

        from langchain_core.vectorstores import utils as langchain_utils

        mmr_selected = langchain_utils.maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=final_k,
            lambda_mult=final_lambda,
        )

        documents = []
        for i, row in enumerate(results):
            if i in mmr_selected:
                doc, _ = self._row_to_document(row)
                documents.append(doc)

        return documents

    async def aget_by_ids(
        self,
        ids: Sequence[str],
    ) -> list[Document]:
        """Get documents by IDs."""
        tc = self.table_config
        columns = [
            tc.escaped_id_column,
            tc.escaped_content_column,
            tc.escaped_metadata_column,
        ]
        column_names = ", ".join(f'"{col}"' for col in columns)
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        param_dict = {f"id_{i}": id_ for i, id_ in enumerate(ids)}

        query = f'''
            SELECT {column_names}
            FROM "{tc.escaped_schema_name}"."{tc.escaped_table_name}"
            WHERE "{tc.escaped_id_column}" IN ({placeholders});
        '''

        async with self.engine._pool.connect() as conn:
            result = await conn.execute(text(query), param_dict)
            results = result.mappings().fetchall()

        return [self._row_to_document(row)[0] for row in results]

    # ==========================================
    # Sync Wrappers
    # ==========================================

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vectorstore (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.aadd_texts(texts, metadatas, ids, **kwargs)
        )

    def add_documents(
        self,
        documents: list[Document],
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents to the vectorstore (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.aadd_documents(documents, ids, **kwargs)
        )

    def delete(
        self,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete from the vectorstore (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.adelete(ids, **kwargs)
        )

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.asimilarity_search(query, k, filter, **kwargs)
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.asimilarity_search_with_score(
                query, k, filter, **kwargs,
            )
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by vector (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.asimilarity_search_by_vector(
                embedding, k, filter, **kwargs,
            )
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores by vector (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.asimilarity_search_with_score_by_vector(
                embedding, k, filter, **kwargs,
            )
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs using MMR (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.amax_marginal_relevance_search(
                query, k, fetch_k, lambda_mult, filter, **kwargs,
            )
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Union[dict, MetadataFilter, MetadataFilters, None] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs using MMR by vector (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.amax_marginal_relevance_search_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs,
            )
        )

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by IDs (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            self.aget_by_ids(ids)
        )

    # ==========================================
    # Factory Methods
    # ==========================================

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        engine: AsyncPGVecTextSearchEngine,
        table_config: TableConfig,
        *,
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore from texts."""
        vs = await cls.create(
            engine, embedding, table_config, **kwargs,
        )
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids)
        return vs

    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        engine: AsyncPGVecTextSearchEngine,
        table_config: TableConfig,
        *,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore from documents."""
        vs = await cls.create(
            engine, embedding, table_config, **kwargs,
        )
        await vs.aadd_documents(documents, ids=ids)
        return vs

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        engine: AsyncPGVecTextSearchEngine,
        table_config: TableConfig,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore from texts (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            cls.afrom_texts(
                texts, embedding, engine, table_config, **kwargs,
            )
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        engine: AsyncPGVecTextSearchEngine,
        table_config: TableConfig,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore from documents (sync)."""
        return asyncio.get_event_loop().run_until_complete(
            cls.afrom_documents(
                documents, embedding, engine, table_config, **kwargs,
            )
        )
