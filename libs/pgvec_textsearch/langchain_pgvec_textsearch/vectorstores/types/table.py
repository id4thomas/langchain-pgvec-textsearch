"""Table definitions"""

from enum import Enum
from pydantic import BaseModel, Field


# Table Column
class ColumnDict(BaseModel):
    name: str = Field(..., description="")
    data_type: str = Field(..., description="")
    nullable: str = Field(..., description="")
    
class Column(BaseModel):
    name: str = Field(..., description="")
    data_type: str = Field(..., description="")
    nullable: bool = Field(True, description="")