import asyncio
import json
from typing import Any, Optional, Sequence

from sqlalchemy import RowMapping, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .config import (
    BM25IndexConfig,
    HNSWSearchConfig,
    HNSWIndexConfig,
    IVFFlatIndexConfig,
    BM25SearchConfig,
    TableConfig,
)
from .data import Row
from .types.search import IterativeScanStrategy

def _sanitize_name(name: str) -> str:
    """Sanitize name by replacing special characters with underscores
    and lowercasing.

    PostgreSQL index names with special characters (like hyphens) or uppercase
    letters can cause issues with pg_textsearch's internal index lookup mechanism.
    pg_textsearch's internal lookups are case-sensitive and may not match
    PostgreSQL's identifier handling.
    """
    sanitized = name.replace("-", "_").replace(" ", "_").lower()
    return sanitized

class AsyncPGVecTextSearchEngine:
    """A class for managing async connections to a Postgres database.
    modified from [langchain-postgres](https://github.com/langchain-ai/langchain-postgres) package"""

    __create_key = object()

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
    ):
        """AsyncPGEngine constructor.

        Args:
            key (object): Prevent direct constructor usage.
            pool (AsyncEngine): Async engine connection pool.
        """
        if key != AsyncPGVecTextSearchEngine.__create_key:
            raise Exception(
                "Only create class through 'from_connection_string' or 'from_engine' methods!"
            )
        self._pool = pool

    @classmethod
    def from_engine(
        cls: type["AsyncPGVecTextSearchEngine"],
        engine: AsyncEngine
    ) -> "AsyncPGVecTextSearchEngine":
        """Create an AsyncPGVecTextSearchEngine instance from an AsyncEngine."""
        return cls(cls.__create_key, engine)

    @classmethod
    def from_connection_string(
        cls,
        url: str | URL,
        **kwargs: Any,
    ) -> "AsyncPGVecTextSearchEngine":
        engine = create_async_engine(url, **kwargs)
        return cls(cls.__create_key, engine)

    async def close(self) -> None:
        """Dispose of connection pool"""
        await self._pool.dispose()

    async def _init_extensions(self):
        async with self._pool.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_textsearch"))
            await conn.commit()

    # ==========================================
    # Filter Support
    # ==========================================

    @staticmethod
    def build_filter_clause(
        table_config: TableConfig,
        filter: dict | None = None,
    ) -> tuple[str, dict]:
        """Convert a dict filter to SQL WHERE clause.

        Uses PostgreSQL JSON containment (@>).
        Example: {"category": "tech"} → metadata @> '{"category":"tech"}'
        """
        if not filter:
            return "", {}
        meta_col = table_config.escaped_metadata_column
        clause = f'"{meta_col}"::jsonb @> :filter_json::jsonb'
        params = {"filter_json": json.dumps(filter)}
        return clause, params

    # ==========================================
    # Table Management
    # ==========================================

    async def init_table(
        self,
        table_config: TableConfig,
        overwrite_existing: bool = False
    ):
        table_name = table_config.escaped_table_name
        schema_name = table_config.escaped_schema_name

        query = """CREATE TABLE IF NOT EXISTS "{}"."{}"(
            "{}" UUID PRIMARY KEY,
            "{}" TEXT NOT NULL,
            "{}" vector({}) NOT NULL,
            "{}" JSON
        );""".format(
            schema_name,
            table_name,
            table_config.escaped_id_column,
            table_config.escaped_content_column,
            table_config.escaped_embedding_column,
            table_config.vector_size,
            table_config.escaped_metadata_column
        )

        if overwrite_existing:
            async with self._pool.connect() as conn:
                await conn.execute(
                    text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')
                )
                await conn.commit()

        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    # ==========================================
    # Index Management
    # ==========================================

    async def create_hnsw_index(
        self,
        table_config: TableConfig,
        hnsw_config: HNSWIndexConfig
    ):
        table_name = table_config.escaped_table_name
        schema_name = table_config.escaped_schema_name
        sanitized_table_name = _sanitize_name(table_name)

        index_name = hnsw_config.name or f"idx_{sanitized_table_name}_hnsw"
        query = f"""
            CREATE INDEX IF NOT EXISTS "{index_name}"
            ON "{schema_name}"."{table_name}"
            USING hnsw ("{table_config.escaped_embedding_column}" {hnsw_config.index_function})
            WITH {hnsw_config.index_options};
        """
        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def create_ivfflat_index(
        self,
        table_config: TableConfig,
        ivfflat_config: IVFFlatIndexConfig
    ):
        table_name = table_config.escaped_table_name
        schema_name = table_config.escaped_schema_name
        sanitized_table_name = _sanitize_name(table_name)

        index_name = ivfflat_config.name or f"idx_{sanitized_table_name}_ivfflat"
        query = f"""
            CREATE INDEX IF NOT EXISTS "{index_name}"
            ON "{schema_name}"."{table_name}"
            USING ivfflat ("{table_config.escaped_embedding_column}" {ivfflat_config.index_function})
            WITH {ivfflat_config.index_options};
        """
        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def create_bm25_index(
        self,
        table_config: TableConfig,
        bm25_config: BM25IndexConfig
    ):
        table_name = table_config.escaped_table_name
        schema_name = table_config.escaped_schema_name
        sanitized_table_name = _sanitize_name(table_name)

        index_name = bm25_config.name or f"idx_{sanitized_table_name}_bm25"
        query = f"""
            CREATE INDEX IF NOT EXISTS "{index_name}"
            ON "{schema_name}"."{table_name}"
            USING bm25 ("{table_config.escaped_content_column}")
            WITH {bm25_config.index_options};
        """
        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    # ==========================================
    # Row Operations
    # ==========================================

    async def insert_rows(
        self,
        table_config: TableConfig,
        rows: list[Row],
        batch_size: int = 500,
    ) -> list[str]:
        """Bulk insert rows into the table.

        Uses batched inserts with ON CONFLICT upsert.
        Each Row encapsulates (id, content, embedding, metadata).
        """
        if not rows:
            return []

        tc = table_config

        async with self._pool.connect() as conn:
            for batch_start in range(0, len(rows), batch_size):
                batch = rows[batch_start:batch_start + batch_size]

                values_clauses = []
                params = {}

                for i, row in enumerate(batch):
                    values_clauses.append(
                        f"(:id_{i}, :content_{i}, :embedding_{i}, :metadata_{i})"
                    )
                    params[f"id_{i}"] = row.id
                    params[f"content_{i}"] = row.content
                    params[f"embedding_{i}"] = row.embedding_as_str()
                    params[f"metadata_{i}"] = row.metadata_as_json()

                values_stmt = ", ".join(values_clauses)

                id_col = tc.escaped_id_column
                txt_col = tc.escaped_content_column
                emb_col = tc.escaped_embedding_column
                meta_col = tc.escaped_metadata_column

                query = f'''
                    INSERT INTO "{tc.escaped_schema_name}"."{tc.escaped_table_name}"
                    ("{id_col}", "{txt_col}", "{emb_col}", "{meta_col}")
                    VALUES {values_stmt}
                    ON CONFLICT ("{id_col}") DO UPDATE SET
                    "{txt_col}" = EXCLUDED."{txt_col}",
                    "{emb_col}" = EXCLUDED."{emb_col}",
                    "{meta_col}" = EXCLUDED."{meta_col}";
                '''

                await conn.execute(text(query), params)

            await conn.commit()

        return [row.id for row in rows]

    # ==========================================
    # Query Methods
    # ==========================================

    async def query_hnsw(
        self,
        table_config: TableConfig,
        query_embedding: list[float],
        dense_config: HNSWSearchConfig,
        filter_clause: str = "",
        filter_params: dict | None = None,
    ) -> Sequence[RowMapping]:
        """Dense vector search using HNSW index.

        Sets HNSW-specific session parameters (ef_search, iterative_scan)
        before executing the query.
        """
        tc = table_config
        ds = dense_config.distance_strategy

        columns = [
            tc.escaped_id_column,
            tc.escaped_content_column,
            tc.escaped_embedding_column,
            tc.escaped_metadata_column,
        ]
        column_names = ", ".join(f'"{col}"' for col in columns)

        where = f"WHERE {filter_clause}" if filter_clause else ""
        embedding_str = str([float(dim) for dim in query_embedding])
        emb_col = tc.escaped_embedding_column
        search_fn = ds.search_function

        query = f'''
            SELECT {column_names},
                   {search_fn}("{emb_col}", :query_embedding) as distance
            FROM "{tc.escaped_schema_name}"."{tc.escaped_table_name}"
            {where}
            ORDER BY "{emb_col}" {ds.operator} :query_embedding
            LIMIT :k;
        '''

        params: dict[str, Any] = {
            "query_embedding": embedding_str,
            "k": dense_config.k,
        }
        if filter_params:
            params.update(filter_params)

        async with self._pool.connect() as conn:
            if dense_config.ef_search is not None:
                await conn.execute(text(
                    f"SET LOCAL hnsw.ef_search = {dense_config.ef_search};"
                ))
            scan = dense_config.iterative_scan_strategy
            if scan != IterativeScanStrategy.OFF:
                await conn.execute(text(
                    f"SET LOCAL hnsw.iterative_scan = '{scan.value}';"
                ))

            result = await conn.execute(text(query), params)
            return result.mappings().fetchall()

    async def query_ivfflat(
        self,
        table_config: TableConfig,
        query_embedding: list[float],
        dense_config: HNSWSearchConfig,
        probes: int | None = None,
        filter_clause: str = "",
        filter_params: dict | None = None,
    ) -> Sequence[RowMapping]:
        """Dense vector search using IVFFlat index.

        Sets IVFFlat-specific session parameter (probes) before executing.
        """
        tc = table_config
        ds = dense_config.distance_strategy

        columns = [
            tc.escaped_id_column,
            tc.escaped_content_column,
            tc.escaped_embedding_column,
            tc.escaped_metadata_column,
        ]
        column_names = ", ".join(f'"{col}"' for col in columns)

        where = f"WHERE {filter_clause}" if filter_clause else ""
        embedding_str = str([float(dim) for dim in query_embedding])
        emb_col = tc.escaped_embedding_column
        search_fn = ds.search_function

        query = f'''
            SELECT {column_names},
                   {search_fn}("{emb_col}", :query_embedding) as distance
            FROM "{tc.escaped_schema_name}"."{tc.escaped_table_name}"
            {where}
            ORDER BY "{emb_col}" {ds.operator} :query_embedding
            LIMIT :k;
        '''

        params: dict[str, Any] = {
            "query_embedding": embedding_str,
            "k": dense_config.k,
        }
        if filter_params:
            params.update(filter_params)

        async with self._pool.connect() as conn:
            if probes is not None:
                await conn.execute(
                    text(f"SET LOCAL ivfflat.probes = {probes};")
                )

            result = await conn.execute(text(query), params)
            return result.mappings().fetchall()

    async def query_bm25(
        self,
        table_config: TableConfig,
        query_text: str,
        sparse_config: BM25SearchConfig,
        bm25_index_name: str | None = None,
        filter_clause: str = "",
        filter_params: dict | None = None,
    ) -> Sequence[RowMapping]:
        """Sparse BM25 search using pg_textsearch.

        Uses the <@> operator with to_bm25query() for scoring.
        Returns negative BM25 scores (lower = better match).
        """
        tc = table_config

        if bm25_index_name is None:
            sanitized = _sanitize_name(tc.table_name)
            bm25_index_name = f"idx_{sanitized}_bm25"

        columns = [
            tc.escaped_id_column,
            tc.escaped_content_column,
            tc.escaped_metadata_column,
        ]
        column_names = ", ".join(f'"{col}"' for col in columns)

        where = f"WHERE {filter_clause}" if filter_clause else ""
        txt_col = tc.escaped_content_column
        bm25q = f"to_bm25query(:query_text, '{bm25_index_name}')"

        query = f'''
            SELECT {column_names},
                   "{txt_col}" <@> {bm25q} as bm25_score
            FROM "{tc.escaped_schema_name}"."{tc.escaped_table_name}"
            {where}
            ORDER BY "{txt_col}" <@> {bm25q}
            LIMIT :k;
        '''

        params: dict[str, Any] = {"query_text": query_text, "k": sparse_config.k}
        if filter_params:
            params.update(filter_params)

        async with self._pool.connect() as conn:
            result = await conn.execute(text(query), params)
            return result.mappings().fetchall()

    # ==========================================
    # Database Initialization
    # ==========================================

    async def init_database(
        self,
        table_config: TableConfig,
        hnsw_config: HNSWIndexConfig | None = None,
        ivfflat_config: IVFFlatIndexConfig | None = None,
        bm25_config: BM25IndexConfig | None = None,
        overwrite_existing: bool = False
    ):
        await self.init_table(table_config=table_config, overwrite_existing=overwrite_existing)

        if hnsw_config is not None:
            await self.create_hnsw_index(
                table_config=table_config,
                hnsw_config=hnsw_config
            )

        if ivfflat_config is not None:
            await self.create_ivfflat_index(
                table_config=table_config,
                ivfflat_config=ivfflat_config
            )

        if bm25_config is not None:
            await self.create_bm25_index(
                table_config=table_config,
                bm25_config=bm25_config
            )
