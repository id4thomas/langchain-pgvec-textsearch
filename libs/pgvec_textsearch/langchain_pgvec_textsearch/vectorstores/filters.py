"""
Metadata filter definitions for PGVecTextSearch.

Follows LlamaIndex-style filter syntax for consistency.
https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/vector_stores/types.py
"""
from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel, field_validator


class FilterOperator(str, Enum):
    """Vector store filter operator."""

    # Equality operators
    EQ = "=="  # equals (string, int, float, bool)
    NE = "!="  # not equal to (string, int, float, bool)

    # Comparison operators (int, float)
    GT = ">"  # greater than
    GTE = ">="  # greater than or equal to
    LT = "<"  # less than
    LTE = "<="  # less than or equal to

    # Array operators
    IN = "in"  # value in array
    NIN = "nin"  # value not in array
    ANY = "any"  # array contains any of values
    ALL = "all"  # array contains all of values
    CONTAINS = "contains"  # array contains value

    # Text operators
    TEXT_MATCH = "text_match"  # LIKE pattern match (case-sensitive)
    TEXT_MATCH_INSENSITIVE = "text_match_insensitive"  # ILIKE pattern match

    # Range operator
    BETWEEN = "between"  # value between [low, high]

    # Existence operators
    IS_EMPTY = "is_empty"  # field is null or empty
    EXISTS = "exists"  # field exists and is not null


class FilterCondition(str, Enum):
    """Vector store filter conditions to combine filters."""

    AND = "and"
    OR = "or"
    NOT = "not"


# Type alias for filter values
FilterValue = Union[
    int,
    float,
    str,
    bool,
    List[int],
    List[float],
    List[str],
    None,
]


class MetadataFilter(BaseModel):
    """
    Single metadata filter.

    Example:
        # Equality filter
        MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)

        # Comparison filter
        MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE)

        # IN filter
        MetadataFilter(key="category", value=["tech", "programming"], operator=FilterOperator.IN)
    """

    key: str
    value: Optional[FilterValue] = None
    operator: FilterOperator = FilterOperator.EQ

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, v: Any, info) -> FilterValue:
        """Validate filter value based on operator."""
        return v

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "key": self.key,
            "value": self.value,
            "operator": self.operator.value,
        }


class MetadataFilters(BaseModel):
    """
    Metadata filters with logical conditions.

    Example:
        # AND condition
        MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
                MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
            ],
            condition=FilterCondition.AND
        )

        # OR condition
        MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="travel", operator=FilterOperator.EQ),
                MetadataFilter(key="category", value="food", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR
        )

        # Nested filters: (category == "tech" AND rating >= 4.5) OR category == "programming"
        MetadataFilters(
            filters=[
                MetadataFilters(
                    filters=[
                        MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
                        MetadataFilter(key="rating", value=4.5, operator=FilterOperator.GTE),
                    ],
                    condition=FilterCondition.AND
                ),
                MetadataFilter(key="category", value="programming", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR
        )
    """

    filters: List[Union[MetadataFilter, "MetadataFilters"]]
    condition: FilterCondition = FilterCondition.AND

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "filters": [f.to_dict() for f in self.filters],
            "condition": self.condition.value,
        }


# Operator mapping to SQL
OPERATOR_TO_SQL = {
    FilterOperator.EQ: "=",
    FilterOperator.NE: "!=",
    FilterOperator.GT: ">",
    FilterOperator.GTE: ">=",
    FilterOperator.LT: "<",
    FilterOperator.LTE: "<=",
    FilterOperator.TEXT_MATCH: "LIKE",
    FilterOperator.TEXT_MATCH_INSENSITIVE: "ILIKE",
}

# Operators that need special handling
ARRAY_OPERATORS = {FilterOperator.IN, FilterOperator.NIN, FilterOperator.ANY, FilterOperator.ALL, FilterOperator.CONTAINS}
EXISTENCE_OPERATORS = {FilterOperator.IS_EMPTY, FilterOperator.EXISTS}
RANGE_OPERATORS = {FilterOperator.BETWEEN}


def build_filter_clause(
    filters: Union[MetadataFilters, MetadataFilter, None],
    metadata_columns: Optional[List[str]] = None,
    json_column: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Build SQL WHERE clause from MetadataFilter/MetadataFilters.

    Args:
        filters: MetadataFilters or MetadataFilter object.
        metadata_columns: List of dedicated metadata columns.
        json_column: JSON column name for non-dedicated metadata.

    Returns:
        Tuple of (SQL clause string, parameter dict).

    Example:
        >>> filter_obj = MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
        >>> clause, params = build_filter_clause(filter_obj, metadata_columns=["category"])
        >>> print(clause)
        "category" = :p_1
        >>> print(params)
        {'p_1': 'tech'}
    """
    if filters is None:
        return "", {}

    if metadata_columns is None:
        metadata_columns = []

    return _build_clause(filters, metadata_columns, json_column)


def _build_clause(
    filters: Union[MetadataFilters, MetadataFilter],
    metadata_columns: List[str],
    json_column: Optional[str],
    prefix: str = "p",
    counter: Optional[List[int]] = None,
) -> tuple[str, dict]:
    """Recursively build SQL clause."""
    if counter is None:
        counter = [0]

    if isinstance(filters, MetadataFilter):
        return _build_single_filter(filters, metadata_columns, json_column, prefix, counter)

    # MetadataFilters with condition
    clauses = []
    params = {}

    for f in filters.filters:
        clause, p = _build_clause(f, metadata_columns, json_column, prefix, counter)
        if clause:
            clauses.append(clause)
            params.update(p)

    if not clauses:
        return "", {}

    if filters.condition == FilterCondition.AND:
        return f"({' AND '.join(clauses)})", params
    elif filters.condition == FilterCondition.OR:
        return f"({' OR '.join(clauses)})", params
    elif filters.condition == FilterCondition.NOT:
        inner = " AND ".join(clauses)
        return f"NOT ({inner})", params

    return "", {}


def _build_single_filter(
    f: MetadataFilter,
    metadata_columns: List[str],
    json_column: Optional[str],
    prefix: str,
    counter: List[int],
) -> tuple[str, dict]:
    """Build SQL for a single filter."""
    counter[0] += 1
    param_name = f"{prefix}_{counter[0]}"

    # Determine field selector
    if f.key in metadata_columns:
        field_selector = f'"{f.key}"'
    elif json_column:
        field_selector = f'"{json_column}"->>\'{f.key}\''
    else:
        field_selector = f'"{f.key}"'

    op = f.operator
    value = f.value

    # Simple comparison operators
    if op in OPERATOR_TO_SQL:
        sql_op = OPERATOR_TO_SQL[op]
        return f"{field_selector} {sql_op} :{param_name}", {param_name: value}

    # Array operators
    if op == FilterOperator.IN:
        if not isinstance(value, list):
            raise ValueError("IN operator requires a list value")
        return f"{field_selector} = ANY(:{param_name})", {param_name: value}

    if op == FilterOperator.NIN:
        if not isinstance(value, list):
            raise ValueError("NIN operator requires a list value")
        return f"{field_selector} <> ALL(:{param_name})", {param_name: value}

    if op == FilterOperator.ANY:
        # Field array contains any of the values
        if not isinstance(value, list):
            raise ValueError("ANY operator requires a list value")
        return f"{field_selector} && :{param_name}", {param_name: value}

    if op == FilterOperator.ALL:
        # Field array contains all of the values
        if not isinstance(value, list):
            raise ValueError("ALL operator requires a list value")
        return f"{field_selector} @> :{param_name}", {param_name: value}

    if op == FilterOperator.CONTAINS:
        # Array field contains single value
        return f":{param_name} = ANY({field_selector})", {param_name: value}

    # Range operator
    if op == FilterOperator.BETWEEN:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("BETWEEN operator requires [low, high] value")
        low_param = f"{param_name}_low"
        high_param = f"{param_name}_high"
        return (
            f"({field_selector} BETWEEN :{low_param} AND :{high_param})",
            {low_param: value[0], high_param: value[1]},
        )

    # Existence operators
    if op == FilterOperator.EXISTS:
        if value:
            return f"{field_selector} IS NOT NULL", {}
        else:
            return f"{field_selector} IS NULL", {}

    if op == FilterOperator.IS_EMPTY:
        if value:
            # Field is null or empty
            return f"({field_selector} IS NULL OR {field_selector} = '')", {}
        else:
            # Field is not empty
            return f"({field_selector} IS NOT NULL AND {field_selector} != '')", {}

    raise ValueError(f"Unsupported operator: {op}")
