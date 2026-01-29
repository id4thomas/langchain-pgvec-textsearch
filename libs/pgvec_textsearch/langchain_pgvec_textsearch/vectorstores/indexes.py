"""Index definitions for pg_textsearch VectorStore."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DistanceStrategy(Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "EUCLIDEAN"
    COSINE_DISTANCE = "COSINE_DISTANCE"
    INNER_PRODUCT = "INNER_PRODUCT"

    @property
    def operator(self) -> str:
        """Return the operator for the distance strategy."""
        if self == DistanceStrategy.EUCLIDEAN:
            return "<->"
        elif self == DistanceStrategy.COSINE_DISTANCE:
            return "<=>"
        elif self == DistanceStrategy.INNER_PRODUCT:
            return "<#>"
        raise ValueError(f"Unknown distance strategy: {self}")

    @property
    def search_function(self) -> str:
        """Return the search function for the distance strategy."""
        if self == DistanceStrategy.EUCLIDEAN:
            return "l2_distance"
        elif self == DistanceStrategy.COSINE_DISTANCE:
            return "cosine_distance"
        elif self == DistanceStrategy.INNER_PRODUCT:
            return "inner_product"
        raise ValueError(f"Unknown distance strategy: {self}")

    @property
    def index_function(self) -> str:
        """Return the index ops function for the distance strategy."""
        if self == DistanceStrategy.EUCLIDEAN:
            return "vector_l2_ops"
        elif self == DistanceStrategy.COSINE_DISTANCE:
            return "vector_cosine_ops"
        elif self == DistanceStrategy.INNER_PRODUCT:
            return "vector_ip_ops"
        raise ValueError(f"Unknown distance strategy: {self}")


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE_DISTANCE
DEFAULT_INDEX_NAME_SUFFIX = "_langchain_index"


@dataclass
class QueryOptions:
    """Query options for index configuration."""

    ef_search: Optional[int] = None
    probes: Optional[int] = None

    def to_parameter(self) -> list[str]:
        """Convert to SET LOCAL parameters."""
        params = []
        if self.ef_search is not None:
            params.append(f"hnsw.ef_search = {self.ef_search}")
        if self.probes is not None:
            params.append(f"ivfflat.probes = {self.probes}")
        return params


@dataclass
class BaseIndex(ABC):
    """Base class for vector indexes."""

    name: Optional[str] = None
    partial_indexes: Optional[str] = None
    extension_name: Optional[str] = None
    index_type: str = "hnsw"

    @abstractmethod
    def get_index_function(self) -> str:
        """Return the index function for the index type."""
        pass

    @abstractmethod
    def index_options(self) -> str:
        """Return the index options."""
        pass


@dataclass
class HNSWIndex(BaseIndex):
    """HNSW Index configuration."""

    m: int = 16
    ef_construction: int = 64
    distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY
    index_type: str = "hnsw"

    def get_index_function(self) -> str:
        return self.distance_strategy.index_function

    def index_options(self) -> str:
        return f"(m = {self.m}, ef_construction = {self.ef_construction})"


@dataclass
class IVFFlatIndex(BaseIndex):
    """IVFFlat Index configuration."""

    lists: int = 100
    distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY
    index_type: str = "ivfflat"

    def get_index_function(self) -> str:
        return self.distance_strategy.index_function

    def index_options(self) -> str:
        return f"(lists = {self.lists})"


@dataclass
class ExactNearestNeighbor(BaseIndex):
    """Exact nearest neighbor (no index)."""

    index_type: str = "none"

    def get_index_function(self) -> str:
        return ""

    def index_options(self) -> str:
        return ""


@dataclass
class BM25Index:
    """BM25 Index configuration for pg_textsearch."""

    name: Optional[str] = None
    text_config: str = "public.korean"
    k1: float = 1.2
    b: float = 0.75

    def index_options(self) -> str:
        return f"(text_config = '{self.text_config}', k1 = {self.k1}, b = {self.b})"
