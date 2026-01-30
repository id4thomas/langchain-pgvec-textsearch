"""
Unit tests for filter building logic (no database required).

Uses only MetadataFilter/MetadataFilters for type-safe filtering.

Run with:
    python tests/test_filter_unit.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_pgvec_textsearch.vectorstores.filters import (
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    build_filter_clause,
)


def test_single_metadata_filter_eq():
    """Test single MetadataFilter with EQ operator."""
    filter_obj = MetadataFilter(
        key="category",
        value="tech",
        operator=FilterOperator.EQ
    )

    clause, params = build_filter_clause(
        filter_obj,
        metadata_columns=["category", "year"],
        json_column="langchain_metadata"
    )

    print(f"Filter: {filter_obj}")
    print(f"SQL: {clause}")
    print(f"Params: {params}")

    assert clause != "", "Filter clause should not be empty"
    assert "category" in clause, "Should contain category field"
    assert "=" in clause, "Should contain = operator"
    print("✓ test_single_metadata_filter_eq passed\n")


def test_single_metadata_filter_gt():
    """Test single MetadataFilter with GT operator."""
    filter_obj = MetadataFilter(
        key="year",
        value=2024,
        operator=FilterOperator.GT
    )

    clause, params = build_filter_clause(
        filter_obj,
        metadata_columns=["category", "year"],
        json_column="langchain_metadata"
    )

    print(f"Filter: {filter_obj}")
    print(f"SQL: {clause}")
    print(f"Params: {params}")

    assert clause != "", "Filter clause should not be empty"
    assert "year" in clause, "Should contain year field"
    assert ">" in clause, "Should contain > operator"
    print("✓ test_single_metadata_filter_gt passed\n")


def test_metadata_filters_and():
    """Test MetadataFilters with AND condition."""
    filter_obj = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
            MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
        ],
        condition=FilterCondition.AND
    )

    clause, params = build_filter_clause(
        filter_obj,
        metadata_columns=["category", "year"],
        json_column="langchain_metadata"
    )

    print(f"Filter: {filter_obj}")
    print(f"SQL: {clause}")
    print(f"Params: {params}")

    assert clause != "", "Filter clause should not be empty"
    assert "AND" in clause, "Should contain AND operator"
    assert "category" in clause, "Should contain category field"
    assert "year" in clause, "Should contain year field"
    print("✓ test_metadata_filters_and passed\n")


def test_metadata_filters_or():
    """Test MetadataFilters with OR condition."""
    filter_obj = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="travel", operator=FilterOperator.EQ),
            MetadataFilter(key="category", value="food", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR
    )

    clause, params = build_filter_clause(
        filter_obj,
        metadata_columns=["category"],
        json_column="langchain_metadata"
    )

    print(f"Filter: {filter_obj}")
    print(f"SQL: {clause}")
    print(f"Params: {params}")

    assert clause != "", "Filter clause should not be empty"
    assert "OR" in clause, "Should contain OR operator"
    print("✓ test_metadata_filters_or passed\n")


def test_metadata_filters_not():
    """Test MetadataFilters with NOT condition."""
    filter_obj = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.NOT
    )

    clause, params = build_filter_clause(
        filter_obj,
        metadata_columns=["category"],
        json_column="langchain_metadata"
    )

    print(f"Filter: {filter_obj}")
    print(f"SQL: {clause}")
    print(f"Params: {params}")

    assert clause != "", "Filter clause should not be empty"
    assert "NOT" in clause, "Should contain NOT operator"
    print("✓ test_metadata_filters_not passed\n")


def test_nested_metadata_filters():
    """Test nested MetadataFilters."""
    filter_obj = MetadataFilters(
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

    clause, params = build_filter_clause(
        filter_obj,
        metadata_columns=["category", "rating"],
        json_column="langchain_metadata"
    )

    print(f"Filter: {filter_obj}")
    print(f"SQL: {clause}")
    print(f"Params: {params}")

    assert clause != "", "Filter clause should not be empty"
    assert "OR" in clause, "Should contain OR operator"
    assert "AND" in clause, "Should contain AND operator"
    print("✓ test_nested_metadata_filters passed\n")


def test_all_operators():
    """Test all FilterOperator types."""
    test_cases = [
        (FilterOperator.EQ, "category", "tech", "="),
        (FilterOperator.NE, "category", "tech", "!="),
        (FilterOperator.GT, "rating", 4.5, ">"),
        (FilterOperator.GTE, "rating", 4.5, ">="),
        (FilterOperator.LT, "year", 2024, "<"),
        (FilterOperator.LTE, "year", 2023, "<="),
        (FilterOperator.IN, "category", ["tech", "programming"], "ANY"),
        (FilterOperator.NIN, "category", ["travel", "food"], "ALL"),
        (FilterOperator.BETWEEN, "year", [2023, 2024], "BETWEEN"),
        (FilterOperator.TEXT_MATCH, "subcategory", "data%", "LIKE"),
        (FilterOperator.TEXT_MATCH_INSENSITIVE, "subcategory", "%search%", "ILIKE"),
        (FilterOperator.EXISTS, "published", True, "IS NOT NULL"),
    ]

    print("Testing all operators:")
    for op, key, value, expected_sql in test_cases:
        filter_obj = MetadataFilter(key=key, value=value, operator=op)
        clause, params = build_filter_clause(
            filter_obj,
            metadata_columns=["category", "year", "rating", "subcategory", "published"],
            json_column="langchain_metadata"
        )

        print(f"  {op.name}: {clause}")
        assert clause != "", f"Filter clause should not be empty for {op.name}"
        assert expected_sql in clause, f"Should contain {expected_sql} for {op.name}"

    print("✓ test_all_operators passed\n")


def test_json_column_fallback():
    """Test JSON column fallback for non-metadata columns."""
    filter_obj = MetadataFilter(
        key="custom_field",
        value="test",
        operator=FilterOperator.EQ
    )

    clause, params = build_filter_clause(
        filter_obj,
        metadata_columns=["category"],  # custom_field is NOT in metadata_columns
        json_column="langchain_metadata"
    )

    print(f"Filter: {filter_obj}")
    print(f"SQL: {clause}")
    print(f"Params: {params}")

    assert clause != "", "Filter clause should not be empty"
    assert "langchain_metadata" in clause, "Should use JSON column for custom field"
    assert "->>" in clause, "Should use JSONB extraction operator"
    print("✓ test_json_column_fallback passed\n")


def test_none_filter():
    """Test None filter returns empty clause."""
    clause, params = build_filter_clause(
        None,
        metadata_columns=["category"],
        json_column="langchain_metadata"
    )

    print(f"Filter: None")
    print(f"SQL: '{clause}'")
    print(f"Params: {params}")

    assert clause == "", "Filter clause should be empty for None"
    assert params == {}, "Params should be empty for None"
    print("✓ test_none_filter passed\n")


def test_is_empty_operator():
    """Test IS_EMPTY operator."""
    # Test IS_EMPTY with True (field is null or empty)
    filter_obj = MetadataFilter(
        key="description",
        value=True,
        operator=FilterOperator.IS_EMPTY
    )

    clause, params = build_filter_clause(
        filter_obj,
        metadata_columns=["description"],
        json_column="langchain_metadata"
    )

    print(f"Filter: {filter_obj}")
    print(f"SQL: {clause}")
    print(f"Params: {params}")

    assert "IS NULL" in clause, "Should contain IS NULL"
    assert "= ''" in clause, "Should contain empty string check"
    print("✓ test_is_empty_operator passed\n")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Filter Unit Tests (No Database Required)")
    print("MetadataFilter/MetadataFilters Only")
    print("=" * 60)
    print()

    tests = [
        test_single_metadata_filter_eq,
        test_single_metadata_filter_gt,
        test_metadata_filters_and,
        test_metadata_filters_or,
        test_metadata_filters_not,
        test_nested_metadata_filters,
        test_all_operators,
        test_json_column_fallback,
        test_none_filter,
        test_is_empty_operator,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
