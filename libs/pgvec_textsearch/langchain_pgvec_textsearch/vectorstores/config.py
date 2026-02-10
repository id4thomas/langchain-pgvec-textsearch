"""Configurations for engine, search"""
from typing import Any, Callable, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import RowMapping

from .types import(
    Column,
    ColumnDict,
    DistanceStrategy,
    FusionStrategy,
    IterativeScanStrategy
)

def _escape_postgres_identifier(name: str) -> str:
    return name.replace('"', '""')

#### Table Config
class TableConfig(BaseModel):
    """Configuration for initializing a hybrid vector store table."""
    
    table_name: str = Field(
        ...,
        description="The database table name"
    )
    
    vector_size: int = Field(
        ...,
        gt=0,
        description="Vector size for the embedding model"
    )
    
    schema_name: str = Field(
        default="public",
        description="The schema name"
    )
    
    id_column: str = Field(
        default="_id",
        description="Column to store ids"
    )
    
    content_column: str = Field(
        default="content",
        description="Name of the column to store document content"
    )
    
    embedding_column: str = Field(
        default="embedding",
        description="Name of the column to store vector embeddings"
    )
    
    metadata_column: str = Field(
        default="langchain_metadata",
        description="Column to store metadata in JSON format"
    )
    
    @field_validator('table_name', 'content_column', 'embedding_column', 'metadata_column')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v
    
    @field_validator('id_column')
    @classmethod
    def validate_id_column(cls, v: Union[str, Column, ColumnDict]) -> Union[str, Column, ColumnDict]:
        """Validate id_column is properly formatted."""
        if isinstance(v, str) and (not v or not v.strip()):
            raise ValueError("id_column string cannot be empty")
        return v
    
    @property
    def escaped_table_name(self):
        return _escape_postgres_identifier(self.table_name)
    
    @property
    def escaped_schema_name(self):
        return _escape_postgres_identifier(self.schema_name)
    
    @property
    def escaped_id_column(self):
        return _escape_postgres_identifier(self.id_column)
    
    @property
    def escaped_content_column(self):
        return _escape_postgres_identifier(self.content_column)
    
    @property
    def escaped_embedding_column(self):
        return _escape_postgres_identifier(self.embedding_column)
    
    @property
    def escaped_metadata_column(self):
        return _escape_postgres_identifier(self.metadata_column)
    
    

#### Index Config
class HNSWIndexConfig(BaseModel):
    """Dense HNSW Index configuration."""
    name: Optional[str] = Field(None, description="Name of hnsw index. Uses `idx_{table_name}_hnsw` if None")
    m: int = Field(16, description="")
    ef_construction: int = Field(64, description="")
    distance_strategy: DistanceStrategy = Field(DistanceStrategy.COSINE_DISTANCE, description="")
    
    @property
    def index_function(self) -> str:
        return self.distance_strategy.index_function

    @property
    def index_options(self) -> str:
        return f"(m = {self.m}, ef_construction = {self.ef_construction})"

class IVFFlatIndexConfig(BaseModel):
    """Dense IVFFlat Index configuration."""
    
    name: Optional[str] = Field(None, description="Name of ivfflat index. Uses `idx_{table_name}_ivfflat` if None")
    lists: int = Field(100, description="")
    distance_strategy: DistanceStrategy = Field(DistanceStrategy.COSINE_DISTANCE, description="")
    
    @property
    def index_function(self) -> str:
        return self.distance_strategy.index_function

    @property
    def index_options(self) -> str:
        return f"(lists = {self.lists})"

class BM25IndexConfig(BaseModel):
    """Sparse BM25 Index configuration."""
    
    name: Optional[str] = Field(None, description="Name of bm25 index. Uses `idx_{table_name}_bm25` if None")
    text_config: str = Field("public.korean", description="")
    k1: float = Field(1.2, description="")
    b: float = Field(0.75, description="")
    
    @property
    def index_options(self) -> str:
        return f"(text_config = '{self.text_config}', k1 = {self.k1}, b = {self.b})"


#### Search Config
class HNSWSearchConfig(BaseModel):
    k: int = Field(20, description="Number of candidates to fetch")
    ef_search: Optional[int] = Field(40, description="Size of dynamic candidate list. Higher = better recall, slower. Default: 40")
    
    distance_strategy: DistanceStrategy = Field(DistanceStrategy.COSINE_DISTANCE, description="")
    iterative_scan_strategy: IterativeScanStrategy = Field(IterativeScanStrategy.OFF, description="")

class BM25SearchConfig(BaseModel):
    k: int = Field(20, description="Number of candidates to fetch")
    text_config: str = Field("public.korean", description="BM25 text configuration (e.g., 'public.korean', 'english')")
    

class SearchConfig(BaseModel):
    """Hybrid search configuration."""

    enable_dense: bool = Field(True, description="")
    enable_sparse: bool = Field(True, description="")

    fusion_strategy: FusionStrategy = Field(FusionStrategy.RRF, description="")

    hnsw: HNSWSearchConfig = Field(
        default_factory=HNSWSearchConfig,
        description="Dense HNSW search parameters",
    )
    bm25: BM25SearchConfig = Field(
        default_factory=BM25SearchConfig,
        description="Sparse BM25 search parameters",
    )
    