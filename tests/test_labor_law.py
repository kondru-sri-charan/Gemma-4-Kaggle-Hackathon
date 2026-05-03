from __future__ import annotations

import pytest

from services.labor_law import (
    ALL_TOPICS,
    LOOKUP_TOOL_NAME,
    TOPIC_BOND_CLAUSE,
    TOPIC_NOTICE_PERIOD,
    execute_tool_call,
    get_lookup_tool_schema,
    list_all_topics,
    lookup_labor_law,
    reset_db_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_db() -> None:
    reset_db_for_tests()
    yield
    reset_db_for_tests()


def test_lookup_returns_national_rows_when_state_omitted() -> None:
    rows = lookup_labor_law(TOPIC_BOND_CLAUSE)

    assert rows, "bond clause should have at least one national entry"
    for row in rows:
        assert row["state"] == "(national)"
        assert row["topic"] == TOPIC_BOND_CLAUSE
        assert row["statute_reference"]


def test_lookup_returns_state_specific_rows_first_then_national() -> None:
    rows = lookup_labor_law(TOPIC_NOTICE_PERIOD, state="Karnataka")

    assert rows
    # First row must be the Karnataka-specific row, subsequent rows may be
    # national fallbacks.
    assert rows[0]["state"] == "Karnataka"
    assert any(row["state"] == "(national)" for row in rows[1:] or [])


def test_lookup_is_case_insensitive_on_state() -> None:
    rows_lower = lookup_labor_law(TOPIC_NOTICE_PERIOD, state="karnataka")
    rows_proper = lookup_labor_law(TOPIC_NOTICE_PERIOD, state="Karnataka")

    assert rows_lower == rows_proper


def test_lookup_returns_empty_for_unknown_topic() -> None:
    assert lookup_labor_law("not_a_real_topic") == []


def test_list_all_topics_returns_all_seeded_topics() -> None:
    topics = list_all_topics()

    assert len(topics) == len(ALL_TOPICS)
    assert set(topics) == set(ALL_TOPICS)


def test_tool_schema_advertises_all_topics_and_lookup_name() -> None:
    schema = get_lookup_tool_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == LOOKUP_TOOL_NAME
    topic_enum = schema["function"]["parameters"]["properties"]["topic"]["enum"]
    assert set(topic_enum) == set(ALL_TOPICS)
    # topic is required, state is not
    assert schema["function"]["parameters"]["required"] == ["topic"]


def test_execute_tool_call_dispatches_lookup_happy_path() -> None:
    result = execute_tool_call(
        LOOKUP_TOOL_NAME, {"topic": TOPIC_BOND_CLAUSE}
    )

    assert isinstance(result, list)
    assert result[0]["topic"] == TOPIC_BOND_CLAUSE


def test_execute_tool_call_returns_error_for_unknown_topic() -> None:
    result = execute_tool_call(
        LOOKUP_TOOL_NAME, {"topic": "something_made_up"}
    )

    assert isinstance(result, dict)
    assert "error" in result
    assert TOPIC_BOND_CLAUSE in result["error"]  # error message lists known topics


def test_execute_tool_call_rejects_unknown_tool_name() -> None:
    result = execute_tool_call("delete_all_data", {})

    assert isinstance(result, dict)
    assert "error" in result
    assert "Unknown tool" in result["error"]
