"""Engine for pg_textsearch VectorStore with BM25 support."""
from __future__ import annotations

from typing import Any, Optional, Union

from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from langchain_postgres import PGEngine, Column, ColumnDict

from .indexes import BM25Index, HNSWIndex, DistanceStrategy


class PGVecTextSearchEngine(PGEngine):
    """Engine that extends PGEngine with pg_textsearch (BM25) support."""

    @classmethod
    def from_connection_string_async(
        cls,
        url: str | URL,
        **kwargs: Any,
    ) -> "PGVecTextSearchEngine":
        """
        Create an engine for use with asyncio.run() or async contexts.

        Unlike from_connection_string(), this does NOT create a background thread.
        Use this when you're running in an async context (e.g., asyncio.run()).

        Args:
            url: Database connection URL.
            **kwargs: Additional arguments for create_async_engine.

        Returns:
            PGVecTextSearchEngine instance.
        """
        engine = create_async_engine(url, **kwargs)
        # Create without background loop (loop=None, thread=None)
        return cls(cls._PGEngine__create_key, engine, None, None)

    async def _ainit_hybrid_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_column: str = "langchain_metadata",
        id_column: Union[str, Column, ColumnDict] = "langchain_id",
        overwrite_existing: bool = False,
        bm25_index: Optional[BM25Index] = None,
        hnsw_index: Optional[HNSWIndex] = None,
        **kwargs,  # Accept but ignore deprecated parameters
    ) -> None:
        """
        Create a table for saving vectors with both HNSW and BM25 indexes.

        All document metadata is stored in the langchain_metadata JSON column.

        Args:
            table_name: The database table name.
            vector_size: Vector size for the embedding model.
            schema_name: The schema name. Default: "public".
            content_column: Name of the column to store document content.
            embedding_column: Name of the column to store vector embeddings.
            metadata_column: Column to store metadata in JSON format.
            id_column: Column to store ids.
            overwrite_existing: Whether to drop existing table.
            bm25_index: BM25 index configuration for pg_textsearch.
            hnsw_index: HNSW index configuration for pgvector.
        """
        schema_name = self._escape_postgres_identifier(schema_name)
        table_name_escaped = self._escape_postgres_identifier(table_name)
        content_column_escaped = self._escape_postgres_identifier(content_column)
        embedding_column_escaped = self._escape_postgres_identifier(embedding_column)

        if isinstance(id_column, str):
            id_column = self._escape_postgres_identifier(id_column)
        elif isinstance(id_column, Column):
            id_column.name = self._escape_postgres_identifier(id_column.name)
        else:
            self._validate_column_dict(id_column)
            id_column["name"] = self._escape_postgres_identifier(id_column["name"])

        # Create extensions
        async with self._pool.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_textsearch"))
            await conn.commit()

        if overwrite_existing:
            async with self._pool.connect() as conn:
                await conn.execute(
                    text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name_escaped}"')
                )
                await conn.commit()

        if isinstance(id_column, str):
            id_data_type = "UUID"
            id_column_name = id_column
        elif isinstance(id_column, Column):
            id_data_type = id_column.data_type
            id_column_name = id_column.name
        else:
            id_data_type = id_column["data_type"]
            id_column_name = id_column["name"]

        # Create table with core columns only (all metadata in JSON column)
        query = f"""CREATE TABLE IF NOT EXISTS "{schema_name}"."{table_name_escaped}"(
            "{id_column_name}" {id_data_type} PRIMARY KEY,
            "{content_column_escaped}" TEXT NOT NULL,
            "{embedding_column_escaped}" vector({vector_size}) NOT NULL,
            "{metadata_column}" JSON
        );"""

        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

        # Create indexes
        await self._acreate_indexes(
            table_name=table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            bm25_index=bm25_index,
            hnsw_index=hnsw_index,
        )

    @staticmethod
    def _sanitize_index_name(name: str) -> str:
        """Sanitize index name by replacing special characters with underscores
        and lowercasing.

        PostgreSQL index names with special characters (like hyphens) or uppercase
        letters can cause issues with pg_textsearch's internal index lookup mechanism.
        pg_textsearch's internal lookups are case-sensitive and may not match
        PostgreSQL's identifier handling.
        """
        # Replace hyphens and other problematic characters with underscores
        # Also lowercase to avoid pg_textsearch case sensitivity issues
        sanitized = name.replace("-", "_").replace(" ", "_").lower()
        return sanitized

    async def _acreate_indexes(
        self,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        bm25_index: Optional[BM25Index] = None,
        hnsw_index: Optional[HNSWIndex] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Create HNSW and BM25 indexes on the table.

        Returns:
            Tuple of (hnsw_index_name, bm25_index_name) that were created.
        """
        table_name_escaped = self._escape_postgres_identifier(table_name)
        schema_name_escaped = self._escape_postgres_identifier(schema_name)
        content_column_escaped = self._escape_postgres_identifier(content_column)
        embedding_column_escaped = self._escape_postgres_identifier(embedding_column)

        # Sanitize table name for use in index names to avoid pg_textsearch lookup issues
        sanitized_table_name = self._sanitize_index_name(table_name)

        hnsw_name_created = None
        bm25_name_created = None

        async with self._pool.connect() as conn:
            # HNSW index for dense vectors
            if hnsw_index:
                hnsw_name = hnsw_index.name or f"idx_{sanitized_table_name}_hnsw"
                hnsw_name_created = hnsw_name
                hnsw_query = f"""
                    CREATE INDEX IF NOT EXISTS "{hnsw_name}"
                    ON "{schema_name_escaped}"."{table_name_escaped}"
                    USING hnsw ("{embedding_column_escaped}" {hnsw_index.get_index_function()})
                    WITH {hnsw_index.index_options()};
                """
                await conn.execute(text(hnsw_query))

            # BM25 index for sparse search (pg_textsearch)
            if bm25_index:
                bm25_name = bm25_index.name or f"idx_{sanitized_table_name}_bm25"
                bm25_name_created = bm25_name
                bm25_query = f"""
                    CREATE INDEX IF NOT EXISTS "{bm25_name}"
                    ON "{schema_name_escaped}"."{table_name_escaped}"
                    USING bm25 ("{content_column_escaped}")
                    WITH {bm25_index.index_options()};
                """
                await conn.execute(text(bm25_query))

            await conn.commit()

        return hnsw_name_created, bm25_name_created

    async def ainit_hybrid_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_column: str = "langchain_metadata",
        id_column: Union[str, Column, ColumnDict] = "langchain_id",
        overwrite_existing: bool = False,
        bm25_index: Optional[BM25Index] = None,
        hnsw_index: Optional[HNSWIndex] = None,
        **kwargs,  # Accept but ignore deprecated parameters
    ) -> None:
        """Create a table for hybrid search with both HNSW and BM25 indexes."""
        # If no background loop, call directly; otherwise use _run_as_async
        if self._loop is None:
            await self._ainit_hybrid_vectorstore_table(
                table_name,
                vector_size,
                schema_name=schema_name,
                content_column=content_column,
                embedding_column=embedding_column,
                metadata_column=metadata_column,
                id_column=id_column,
                overwrite_existing=overwrite_existing,
                bm25_index=bm25_index,
                hnsw_index=hnsw_index,
            )
        else:
            await self._run_as_async(
                self._ainit_hybrid_vectorstore_table(
                    table_name,
                    vector_size,
                    schema_name=schema_name,
                    content_column=content_column,
                    embedding_column=embedding_column,
                    metadata_column=metadata_column,
                    id_column=id_column,
                    overwrite_existing=overwrite_existing,
                    bm25_index=bm25_index,
                    hnsw_index=hnsw_index,
                )
            )

    def init_hybrid_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_column: str = "langchain_metadata",
        id_column: Union[str, Column, ColumnDict] = "langchain_id",
        overwrite_existing: bool = False,
        bm25_index: Optional[BM25Index] = None,
        hnsw_index: Optional[HNSWIndex] = None,
        **kwargs,  # Accept but ignore deprecated parameters
    ) -> None:
        """Create a table for hybrid search with both HNSW and BM25 indexes (sync)."""
        self._run_as_sync(
            self._ainit_hybrid_vectorstore_table(
                table_name,
                vector_size,
                schema_name=schema_name,
                content_column=content_column,
                embedding_column=embedding_column,
                metadata_column=metadata_column,
                id_column=id_column,
                overwrite_existing=overwrite_existing,
                bm25_index=bm25_index,
                hnsw_index=hnsw_index,
            )
        )

    async def adrop_table(
        self,
        table_name: str,
        *,
        schema_name: str = "public",
    ) -> None:
        """Drop a table (async-friendly version)."""
        if self._loop is None:
            # Direct async call
            await self._adrop_table(table_name=table_name, schema_name=schema_name)
        else:
            # Use background loop
            await super().adrop_table(table_name=table_name, schema_name=schema_name)
