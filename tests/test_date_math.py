"""Deadlines are arithmetic, and arithmetic is testable. That is the point."""

from __future__ import annotations

from datetime import date

import pytest

from obligation_rag.date_math import (
    Duration,
    add_months,
    compute_notice_deadline,
    compute_renewal_date,
    days_until,
    parse_date,
    parse_duration,
    shift_date,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-03-31", date(2026, 3, 31)),
        ("expires on March 31, 2026 (the Initial Term)", date(2026, 3, 31)),
        ("31 March 2026", date(2026, 3, 31)),
        ("this 1st day of March, 2024", date(2024, 3, 1)),
        ("Mar. 1, 2024", date(2024, 3, 1)),
        ("03/31/2026", date(2026, 3, 31)),
        ("31/03/2026", date(2026, 3, 31)),  # impossible as US -> day-first fallback
    ],
)
def test_parse_date(text: str, expected: date):
    assert parse_date(text) == expected


@pytest.mark.parametrize("text", ["", "no date here", "February 30, 2026", "the term"])
def test_parse_date_returns_none_rather_than_guessing(text: str):
    assert parse_date(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P60D", Duration(days=60)),
        ("P1Y6M", Duration(years=1, months=6)),
        ("P2W", Duration(days=14)),
        ("sixty (60) days' prior written notice", Duration(days=60)),
        ("60 days", Duration(days=60)),
        ("twelve (12) months", Duration(months=12)),
        ("three months", Duration(months=3)),
        ("one hundred eighty days", Duration(days=180)),
        ("two years", Duration(years=2)),
        ("four (4) weeks", Duration(days=28)),
        ("ninety (90) calendar days", Duration(days=90)),
    ],
)
def test_parse_duration(text: str, expected: Duration):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "prior written notice", "as soon as practicable"])
def test_parse_duration_returns_none_when_unsure(text: str):
    assert parse_duration(text) is None


def test_parenthesised_numeral_wins_over_the_spelled_out_word():
    """Drafters disambiguate themselves; trust "(45)" over a stray earlier word."""
    assert parse_duration("forty-five (45) days past due") == Duration(days=45)


def test_iso_round_trip():
    assert Duration(days=60).to_iso() == "P60D"
    assert Duration(years=1, months=6).to_iso() == "P1Y6M"
    assert Duration().to_iso() == "P0D"
    assert Duration(months=12).humanize() == "12 months"


def test_add_months_clamps_to_the_end_of_a_short_month():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)  # leap year
    assert add_months(date(2025, 12, 15), 1) == date(2026, 1, 15)
    assert add_months(date(2026, 3, 15), -3) == date(2025, 12, 15)


def test_shift_date_applies_months_before_days():
    assert shift_date(date(2026, 3, 31), Duration(months=1, days=1), sign=-1) == date(2026, 2, 27)


def test_notice_deadline_is_computed_not_guessed():
    computation = compute_notice_deadline("2026-03-31", "P60D")

    assert computation.ok
    assert computation.result == date(2026, 1, 30)
    assert computation.formula == "notice_deadline = contract_end_date - termination_notice_period"
    assert computation.inputs["evaluated"] == "2026-03-31 - 60 days = 2026-01-30"


def test_notice_deadline_accepts_contract_wording_on_both_inputs():
    computation = compute_notice_deadline(
        "expires on March 31, 2026", "not less than sixty (60) days"
    )

    assert computation.result == date(2026, 1, 30)


def test_notice_deadline_crossing_a_leap_day():
    computation = compute_notice_deadline("2024-03-01", "P3M")

    assert computation.result == date(2023, 12, 1)


def test_month_based_notice_period():
    computation = compute_notice_deadline("2026-03-31", "three (3) months")

    assert computation.result == date(2025, 12, 31)


@pytest.mark.parametrize(
    ("anchor", "duration", "expected_error"),
    [
        ("sometime next spring", "P60D", "unparseable_date"),
        ("2026-03-31", "reasonable notice", "unparseable_duration"),
    ],
)
def test_unparseable_inputs_fail_loudly_with_their_audit_trail(anchor, duration, expected_error):
    computation = compute_notice_deadline(anchor, duration)

    assert not computation.ok
    assert computation.result is None
    assert expected_error in computation.error
    assert computation.inputs["contract_end_date"] == anchor


def test_renewal_date_adds():
    computation = compute_renewal_date("2026-03-31", "twelve (12) months")

    assert computation.result == date(2027, 3, 31)
    assert computation.formula == "renewal_date = contract_end_date + renewal_duration"


def test_days_until_is_signed():
    assert days_until(date(2026, 3, 31), reference=date(2026, 1, 30)) == 60
    assert days_until(date(2026, 1, 1), reference=date(2026, 1, 30)) == -29
