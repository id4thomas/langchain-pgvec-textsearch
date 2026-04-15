"""Type definitions"""

from enum import Enum

# Search Distance Strategy
class DistanceStrategy(Enum):
    """Dense Search distance strategies"""

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
    