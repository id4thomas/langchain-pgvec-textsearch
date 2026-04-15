"""Type definitions"""

from enum import Enum

# Hybrid Search Fusion Strategy
class FusionStrategy(Enum):
    """Hybrid search dense+sparse fusion strategy"""
    RRF = "reciprocal_rank_fusion"
    WSR = "weighted_sum_ranking"

class IterativeScanStrategy(Enum):
    """Iterative scan strategy for HNSW index (pgvector 0.8.0+)."""

    OFF = "off"
    RELAXED_ORDER = "relaxed_order"  # Better performance, may reorder some results
    STRICT_ORDER = "strict_order"  # Maintains strict distance ordering
