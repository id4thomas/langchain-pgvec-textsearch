"""
PGVecTextSearch VectorStore - Hybrid Search with pgvector and pg_textsearch.

Combines:
- Dense search: pgvector HNSW index
- Sparse search: pg_textsearch BM25 index
- Fusion: RRF (Reciprocal Rank Fusion) or weighted sum
"""
from __future__ import annotations

import copy
import json
import uuid
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import PGVecTextSearchEngine
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    DEFAULT_INDEX_NAME_SUFFIX,
    BaseIndex,
    DistanceStrategy,
    ExactNearestNeighbor,
    HNSWIndex,
    BM25Index,
    QueryOptions,
)
from .hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
from .filters import (
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    build_filter_clause,
)


class PGVecTextSearchStore(VectorStore):
    """
    LangChain VectorStore implementation for hybrid search with pgvector + pg_textsearch.

    This implementation provides:
    - Dense vector search using pgvector HNSW index
    - Sparse keyword search using pg_textsearch BM25 index
    - Hybrid search combining both with RRF fusion

    Note: This class does NOT inherit from PGVectorStore but implements
    the LangChain VectorStore interface directly.
    """

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AsyncEngine,
        embedding_service: Embeddings,
        table_name: str,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ):
        """
        PGVecTextSearchStore constructor.

        Args:
            key: Prevent direct constructor usage.
            engine: SQLAlchemy AsyncEngine for database connections.
            embedding_service: Text embedding model.
            table_name: Name of the table.
            schema_name: Database schema name. Default: "public".
            content_column: Column for document content. Default: "content".
            embedding_column: Column for embeddings. Default: "embedding".
            metadata_columns: List of metadata column names.
            id_column: Column for document IDs. Default: "langchain_id".
            metadata_json_column: Column for JSON metadata. Default: "langchain_metadata".
            distance_strategy: Vector distance strategy. Default: COSINE_DISTANCE.
            k: Number of results to return. Default: 4.
            fetch_k: Number of results to fetch for MMR. Default: 20.
            lambda_mult: MMR diversity parameter. Default: 0.5.
            index_query_options: Query options for index.
            hybrid_search_config: Configuration for hybrid search.
        """
        if key != PGVecTextSearchStore.__create_key:
            raise Exception(
                "Only create class through 'create', 'create_sync', or factory methods!"
            )

        self.engine = engine
        self.embedding_service = embedding_service
        self.table_name = table_name
        self.schema_name = schema_name
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_columns = metadata_columns if metadata_columns is not None else []
        self.id_column = id_column
        self.metadata_json_column = metadata_json_column
        self.distance_strategy = distance_strategy
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.index_query_options = index_query_options
        self.hybrid_search_config = hybrid_search_config or HybridSearchConfig()

    @classmethod
    async def create(
        cls,
        engine: PGVecTextSearchEngine,
        embedding_service: Embeddings,
        table_name: str,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> PGVecTextSearchStore:
        """
        Create a PGVecTextSearchStore instance asynchronously.

        Args:
            engine: PGVecTextSearchEngine for database connections.
            embedding_service: Text embedding model.
            table_name: Name of an existing table.
            schema_name: Database schema name. Default: "public".
            content_column: Column for document content. Default: "content".
            embedding_column: Column for embeddings. Default: "embedding".
            metadata_columns: List of metadata column names.
            ignore_metadata_columns: Columns to ignore for metadata.
            id_column: Column for document IDs. Default: "langchain_id".
            metadata_json_column: Column for JSON metadata.
            distance_strategy: Vector distance strategy.
            k: Number of results to return.
            fetch_k: Number of results for MMR.
            lambda_mult: MMR diversity parameter.
            index_query_options: Query options for index.
            hybrid_search_config: Configuration for hybrid search.

        Returns:
            PGVecTextSearchStore instance.
        """
        if metadata_columns is None:
            metadata_columns = []

        if metadata_columns and ignore_metadata_columns:
            raise ValueError(
                "Cannot use both metadata_columns and ignore_metadata_columns."
            )

        # Get column info from database
        stmt = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = :table_name AND table_schema = :schema_name
        """
        async with engine._pool.connect() as conn:
            result = await conn.execute(
                text(stmt),
                {"table_name": table_name, "schema_name": schema_name},
            )
            results = result.mappings().fetchall()

        columns = {field["column_name"]: field["data_type"] for field in results}

        # Validate required columns
        if id_column not in columns:
            raise ValueError(f"Id column '{id_column}' does not exist.")
        if content_column not in columns:
            raise ValueError(f"Content column '{content_column}' does not exist.")
        if embedding_column not in columns:
            raise ValueError(f"Embedding column '{embedding_column}' does not exist.")

        content_type = columns[content_column]
        if content_type != "text" and "char" not in content_type:
            raise ValueError(
                f"Content column '{content_column}' must be a text type, got {content_type}."
            )

        if columns[embedding_column] not in ["USER-DEFINED", "vector"]:
            raise ValueError(
                f"Embedding column '{embedding_column}' must be a vector type."
            )

        metadata_json_column = (
            None if metadata_json_column not in columns else metadata_json_column
        )

        # Validate metadata columns
        for column in metadata_columns:
            if column not in columns:
                raise ValueError(f"Metadata column '{column}' does not exist.")

        # Handle ignore_metadata_columns
        if ignore_metadata_columns:
            all_columns = dict(columns)
            for column in ignore_metadata_columns:
                all_columns.pop(column, None)
            all_columns.pop(id_column, None)
            all_columns.pop(content_column, None)
            all_columns.pop(embedding_column, None)
            metadata_columns = list(all_columns.keys())

        return cls(
            cls.__create_key,
            engine._pool,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            id_column=id_column,
            metadata_json_column=metadata_json_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )

    @classmethod
    def create_sync(
        cls,
        engine: PGVecTextSearchEngine,
        embedding_service: Embeddings,
        table_name: str,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore instance synchronously."""
        return engine._run_as_sync(
            cls.create(engine, embedding_service, table_name, **kwargs)
        )

    @property
    def embeddings(self) -> Embeddings:
        """Return the embedding service."""
        return self.embedding_service

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
        """Add data along with embeddings to the table."""
        texts_list = list(texts)
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        else:
            ids = [id if id is not None else str(uuid.uuid4()) for id in ids]
        if not metadatas:
            metadatas = [{} for _ in texts_list]

        for id_, content, embedding, metadata in zip(ids, texts_list, embeddings, metadatas):
            metadata_col_names = (
                ", " + ", ".join(f'"{col}"' for col in self.metadata_columns)
                if self.metadata_columns
                else ""
            )

            insert_stmt = f'''
                INSERT INTO "{self.schema_name}"."{self.table_name}"
                ("{self.id_column}", "{self.content_column}", "{self.embedding_column}"{metadata_col_names}'''

            values = {
                "langchain_id": id_,
                "content": content,
                "embedding": str([float(dim) for dim in embedding]),
            }
            values_stmt = "VALUES (:langchain_id, :content, :embedding"

            # Add metadata columns
            extra = copy.deepcopy(metadata)
            for metadata_column in self.metadata_columns:
                if metadata_column in metadata:
                    values_stmt += f", :{metadata_column}"
                    values[metadata_column] = (
                        json.dumps(metadata[metadata_column])
                        if isinstance(metadata[metadata_column], dict)
                        else metadata[metadata_column]
                    )
                    del extra[metadata_column]
                else:
                    values_stmt += ", null"

            # Add JSON column
            insert_stmt += (
                f', "{self.metadata_json_column}")'
                if self.metadata_json_column
                else ")"
            )
            if self.metadata_json_column:
                values_stmt += ", :extra)"
                values["extra"] = json.dumps(extra)
            else:
                values_stmt += ")"

            # Upsert statement
            upsert_stmt = f'''
                ON CONFLICT ("{self.id_column}") DO UPDATE SET
                "{self.content_column}" = EXCLUDED."{self.content_column}",
                "{self.embedding_column}" = EXCLUDED."{self.embedding_column}"'''

            if self.metadata_json_column:
                upsert_stmt += f', "{self.metadata_json_column}" = EXCLUDED."{self.metadata_json_column}"'

            for column in self.metadata_columns:
                upsert_stmt += f', "{column}" = EXCLUDED."{column}"'

            upsert_stmt += ";"

            query = insert_stmt + values_stmt + upsert_stmt
            async with self.engine.connect() as conn:
                await conn.execute(text(query), values)
                await conn.commit()

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts and add to the table."""
        texts_list = list(texts)
        embeddings = await self.embedding_service.aembed_documents(texts_list)
        return await self.aadd_embeddings(
            texts_list, embeddings, metadatas=metadatas, ids=ids, **kwargs
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
        return await self.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    async def adelete(
        self,
        ids: Optional[list] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records from the table."""
        if not ids and not filter:
            return False

        where_clauses = []
        param_dict = {}

        if ids:
            placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
            param_dict.update({f"id_{i}": id_ for i, id_ in enumerate(ids)})
            where_clauses.append(f'"{self.id_column}" IN ({placeholders})')

        if filter is not None:
            filter_clause, filter_params = self._create_filter_clause(filter)
            if filter_clause:
                param_dict.update(filter_params)
                where_clauses.append(filter_clause)

        where_clause = " AND ".join(where_clauses)
        query = f'DELETE FROM "{self.schema_name}"."{self.table_name}" WHERE {where_clause}'

        async with self.engine.connect() as conn:
            await conn.execute(text(query), param_dict)
            await conn.commit()
        return True

    async def _aquery_dense(
        self,
        embedding: list[float],
        k: int,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
    ) -> Sequence[RowMapping]:
        """Perform dense vector search using pgvector."""
        operator = self.distance_strategy.operator
        search_function = self.distance_strategy.search_function

        columns = [
            self.id_column,
            self.content_column,
            self.embedding_column,
        ] + self.metadata_columns
        if self.metadata_json_column:
            columns.append(self.metadata_json_column)

        column_names = ", ".join(f'"{col}"' for col in columns)

        safe_filter, filter_dict = ("", {})
        if filter is not None:
            safe_filter, filter_dict = self._create_filter_clause(filter)

        query_embedding = str([float(dim) for dim in embedding])
        where_filters = f"WHERE {safe_filter}" if safe_filter else ""

        query_stmt = f'''
            SELECT {column_names},
                   {search_function}("{self.embedding_column}", :query_embedding) as distance
            FROM "{self.schema_name}"."{self.table_name}"
            {where_filters}
            ORDER BY "{self.embedding_column}" {operator} :query_embedding
            LIMIT :k;
        '''

        param_dict = {"query_embedding": query_embedding, "k": k}
        if filter_dict:
            param_dict.update(filter_dict)

        async with self.engine.connect() as conn:
            if self.index_query_options:
                for query_option in self.index_query_options.to_parameter():
                    await conn.execute(text(f"SET LOCAL {query_option};"))
            result = await conn.execute(text(query_stmt), param_dict)
            return result.mappings().fetchall()

    async def _aquery_sparse(
        self,
        query: str,
        k: int,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
    ) -> Sequence[RowMapping]:
        """Perform sparse BM25 search using pg_textsearch."""
        columns = [
            self.id_column,
            self.content_column,
        ] + self.metadata_columns
        if self.metadata_json_column:
            columns.append(self.metadata_json_column)

        column_names = ", ".join(f'"{col}"' for col in columns)

        safe_filter, filter_dict = ("", {})
        if filter is not None:
            safe_filter, filter_dict = self._create_filter_clause(filter)

        where_filters = f"WHERE {safe_filter}" if safe_filter else ""

        # pg_textsearch uses <@> operator for BM25 scoring
        # Returns negative BM25 score (lower is better)
        # Must use to_bm25query with explicit index name for prepared statements
        bm25_index_name = self.hybrid_search_config.bm25_index_name or f"idx_{self.table_name}_bm25"
        query_stmt = f'''
            SELECT {column_names},
                   "{self.content_column}" <@> to_bm25query(:query_text, '{bm25_index_name}') as bm25_score
            FROM "{self.schema_name}"."{self.table_name}"
            {where_filters}
            ORDER BY "{self.content_column}" <@> to_bm25query(:query_text, '{bm25_index_name}')
            LIMIT :k;
        '''

        param_dict = {"query_text": query, "k": k}
        if filter_dict:
            param_dict.update(filter_dict)

        async with self.engine.connect() as conn:
            result = await conn.execute(text(query_stmt), param_dict)
            return result.mappings().fetchall()

    async def _aquery_hybrid(
        self,
        query: str,
        embedding: list[float],
        k: int,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
    ) -> Sequence[dict[str, Any]]:
        """Perform hybrid search combining dense and sparse results."""
        config = self.hybrid_search_config

        dense_results: Sequence[RowMapping] = []
        sparse_results: Sequence[RowMapping] = []

        if config.enable_dense:
            dense_results = await self._aquery_dense(
                embedding, config.dense_top_k, filter
            )

        if config.enable_sparse:
            sparse_results = await self._aquery_sparse(
                query, config.sparse_top_k, filter
            )

        # If only one type is enabled, return those results directly
        if not config.enable_sparse:
            return [dict(row) for row in dense_results[:k]]
        if not config.enable_dense:
            return [dict(row) for row in sparse_results[:k]]

        # Fuse results
        fusion_params = {
            **config.fusion_function_parameters,
            "fetch_top_k": k,
            "distance_strategy": self.distance_strategy,
            "id_column": self.id_column,
        }
        return config.fusion_function(dense_results, sparse_results, **fusion_params)

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search on query."""
        embedding = await self.embedding_service.aembed_query(text=query)
        return await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, query=query, **kwargs
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores selected by similarity search on query."""
        embedding = await self.embedding_service.aembed_query(text=query)
        return await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, query=query, **kwargs
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by vector similarity search."""
        docs_and_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores selected by vector similarity search."""
        final_k = k if k is not None else self.k
        query = kwargs.get("query", "")

        # Use hybrid search if query text is available and sparse is enabled
        if query and self.hybrid_search_config.enable_sparse:
            results = await self._aquery_hybrid(query, embedding, final_k, filter)
        else:
            results = await self._aquery_dense(embedding, final_k, filter)

        documents_with_scores = []
        for row in results:
            metadata = (
                row.get(self.metadata_json_column, {})
                if self.metadata_json_column
                else {}
            )
            if metadata is None:
                metadata = {}
            for col in self.metadata_columns:
                if col in row:
                    metadata[col] = row[col]

            # Get score (RRF score, distance, or BM25 score)
            score = row.get("rrf_score") or row.get("distance") or row.get("bm25_score", 0.0)

            documents_with_scores.append(
                (
                    Document(
                        page_content=row[self.content_column],
                        metadata=metadata,
                        id=str(row[self.id_column]),
                    ),
                    float(score),
                )
            )

        return documents_with_scores

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using maximal marginal relevance."""
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
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using maximal marginal relevance by vector."""
        final_k = k if k is not None else self.k
        final_fetch_k = fetch_k if fetch_k is not None else self.fetch_k
        final_lambda = lambda_mult if lambda_mult is not None else self.lambda_mult

        # Fetch more candidates for MMR
        results = await self._aquery_dense(embedding, final_fetch_k, filter)

        if not results:
            return []

        # Extract embeddings for MMR calculation
        embedding_list = [json.loads(row[self.embedding_column]) for row in results]

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
                metadata = (
                    row.get(self.metadata_json_column, {})
                    if self.metadata_json_column
                    else {}
                )
                if metadata is None:
                    metadata = {}
                for col in self.metadata_columns:
                    if col in row:
                        metadata[col] = row[col]
                documents.append(
                    Document(
                        page_content=row[self.content_column],
                        metadata=metadata,
                        id=str(row[self.id_column]),
                    )
                )

        return documents

    async def aget_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by IDs."""
        columns = [self.id_column, self.content_column] + self.metadata_columns
        if self.metadata_json_column:
            columns.append(self.metadata_json_column)

        column_names = ", ".join(f'"{col}"' for col in columns)
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        param_dict = {f"id_{i}": id_ for i, id_ in enumerate(ids)}

        query = f'''
            SELECT {column_names}
            FROM "{self.schema_name}"."{self.table_name}"
            WHERE "{self.id_column}" IN ({placeholders});
        '''

        async with self.engine.connect() as conn:
            result = await conn.execute(text(query), param_dict)
            results = result.mappings().fetchall()

        documents = []
        for row in results:
            metadata = (
                row.get(self.metadata_json_column, {})
                if self.metadata_json_column
                else {}
            )
            if metadata is None:
                metadata = {}
            for col in self.metadata_columns:
                if col in row:
                    metadata[col] = row[col]
            documents.append(
                Document(
                    page_content=row[self.content_column],
                    metadata=metadata,
                    id=str(row[self.id_column]),
                )
            )

        return documents

    # ==========================================
    # Index Management
    # ==========================================

    async def aapply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create a vector index on the table."""
        if isinstance(index, ExactNearestNeighbor):
            await self.adrop_vector_index()
            return

        if index.extension_name:
            async with self.engine.connect() as conn:
                await conn.execute(
                    text(f"CREATE EXTENSION IF NOT EXISTS {index.extension_name}")
                )
                await conn.commit()

        function = index.get_index_function()
        filter_clause = f"WHERE ({index.partial_indexes})" if index.partial_indexes else ""
        params = "WITH " + index.index_options()

        if name is None:
            name = index.name or self.table_name + DEFAULT_INDEX_NAME_SUFFIX

        stmt = f'''
            CREATE INDEX {"CONCURRENTLY" if concurrently else ""} "{name}"
            ON "{self.schema_name}"."{self.table_name}"
            USING {index.index_type} ("{self.embedding_column}" {function})
            {params} {filter_clause};
        '''

        async with self.engine.connect() as conn:
            if concurrently:
                autocommit_conn = await conn.execution_options(
                    isolation_level="AUTOCOMMIT"
                )
                await autocommit_conn.execute(text(stmt))
            else:
                await conn.execute(text(stmt))
                await conn.commit()

    async def aapply_bm25_index(
        self,
        index: BM25Index,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create a BM25 index on the content column for pg_textsearch."""
        if name is None:
            name = index.name or f"{self.table_name}_bm25_idx"

        stmt = f'''
            CREATE INDEX {"CONCURRENTLY" if concurrently else ""} "{name}"
            ON "{self.schema_name}"."{self.table_name}"
            USING bm25 ("{self.content_column}")
            WITH {index.index_options()};
        '''

        async with self.engine.connect() as conn:
            if concurrently:
                autocommit_conn = await conn.execution_options(
                    isolation_level="AUTOCOMMIT"
                )
                await autocommit_conn.execute(text(stmt))
            else:
                await conn.execute(text(stmt))
                await conn.commit()

    async def adrop_vector_index(self, index_name: Optional[str] = None) -> None:
        """Drop the vector index."""
        index_name = index_name or self.table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = f'DROP INDEX IF EXISTS "{self.schema_name}"."{index_name}";'
        async with self.engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def adrop_bm25_index(self, index_name: Optional[str] = None) -> None:
        """Drop the BM25 index."""
        index_name = index_name or f"{self.table_name}_bm25_idx"
        query = f'DROP INDEX IF EXISTS "{self.schema_name}"."{index_name}";'
        async with self.engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    # ==========================================
    # Filter Support
    # ==========================================

    def _create_filter_clause(
        self,
        filters: Union[MetadataFilters, MetadataFilter, None],
    ) -> tuple[str, dict]:
        """
        Convert filters to SQL WHERE clause.

        Uses MetadataFilter/MetadataFilters objects for type-safe filtering.

        Filter Operators (FilterOperator):
        - Comparison: EQ (==), NE (!=), LT (<), LTE (<=), GT (>), GTE (>=)
        - Array: IN, NIN, ANY, ALL, CONTAINS
        - Text: TEXT_MATCH (LIKE), TEXT_MATCH_INSENSITIVE (ILIKE)
        - Range: BETWEEN
        - Existence: EXISTS, IS_EMPTY

        Logical Conditions (FilterCondition):
        - AND, OR, NOT

        Args:
            filters: MetadataFilters or MetadataFilter object.

        Returns:
            Tuple of (SQL WHERE clause, parameter dict).

        Example:
            # Single filter
            filter = MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)

            # Multiple filters with AND
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
                    MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
                ],
                condition=FilterCondition.AND
            )
        """
        return build_filter_clause(
            filters=filters,
            metadata_columns=self.metadata_columns,
            json_column=self.metadata_json_column,
        )

    # ==========================================
    # Sync Interface (Wrappers)
    # ==========================================

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vectorstore (sync)."""
        import asyncio
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
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.aadd_documents(documents, ids, **kwargs)
        )

    def delete(
        self,
        ids: Optional[list] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete from the vectorstore (sync)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.adelete(ids, filter, **kwargs)
        )

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search (sync)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.asimilarity_search(query, k, filter, **kwargs)
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores (sync)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.asimilarity_search_with_score(query, k, filter, **kwargs)
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by vector (sync)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.asimilarity_search_by_vector(embedding, k, filter, **kwargs)
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores by vector (sync)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.asimilarity_search_with_score_by_vector(embedding, k, filter, **kwargs)
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs using MMR (sync)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.amax_marginal_relevance_search(
                query, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[Union[MetadataFilters, MetadataFilter]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs using MMR by vector (sync)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.amax_marginal_relevance_search_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by IDs (sync)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.aget_by_ids(ids))

    # ==========================================
    # Factory Methods
    # ==========================================

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        engine: PGVecTextSearchEngine,
        table_name: str,
        *,
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore from texts."""
        vs = await cls.create(engine, embedding, table_name, **kwargs)
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids)
        return vs

    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        engine: PGVecTextSearchEngine,
        table_name: str,
        *,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore from documents."""
        vs = await cls.create(engine, embedding, table_name, **kwargs)
        await vs.aadd_documents(documents, ids=ids)
        return vs

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        engine: PGVecTextSearchEngine,
        table_name: str,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore from texts (sync)."""
        return engine._run_as_sync(
            cls.afrom_texts(texts, embedding, engine, table_name, **kwargs)
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        engine: PGVecTextSearchEngine,
        table_name: str,
        **kwargs: Any,
    ) -> PGVecTextSearchStore:
        """Create a PGVecTextSearchStore from documents (sync)."""
        return engine._run_as_sync(
            cls.afrom_documents(documents, embedding, engine, table_name, **kwargs)
        )
