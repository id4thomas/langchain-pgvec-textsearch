"""Data types for row operations."""

import json
import uuid

from pydantic import BaseModel, Field


class Row(BaseModel):
    """Encapsulates a single row for insert/query operations.

    Maps to table columns: (id, content, embedding, metadata).
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Row ID (UUID string)",
    )
    content: str = Field(..., description="Document text content")
    embedding: list[float] = Field(..., description="Vector embedding")
    metadata: dict = Field(default_factory=dict, description="JSON metadata")

    def embedding_as_str(self) -> str:
        """Return embedding as a PostgreSQL-compatible vector literal."""
        return str([float(dim) for dim in self.embedding])

    def metadata_as_json(self) -> str:
        """Return metadata as a JSON string."""
        return json.dumps(self.metadata)
