"""Deterministic date arithmetic.

Trust rule #2: *code does the math, not the model*. The model may read a term
end date and a notice period off the page — both backed by quotes — but the
notice deadline itself is computed here, in Python, and every computed value
carries the formula and the inputs that produced it.

A deadline that arrives straight from the model is never accepted as
authoritative, even when it looks right.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

_MONTH_NAMES = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_MONTH_FIRST = re.compile(
    rf"\b({_MONTH_NAMES})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})\b", re.IGNORECASE
)
_DAY_FIRST = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?({_MONTH_NAMES})\.?\s*,?\s*(\d{{4}})\b",
    re.IGNORECASE,
)
_US_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

_UNIT_DAYS = "day"
_UNIT_WEEKS = "week"
_UNIT_MONTHS = "month"
_UNIT_YEARS = "year"
_UNITS = f"{_UNIT_DAYS}|{_UNIT_WEEKS}|{_UNIT_MONTHS}|{_UNIT_YEARS}"

_PARENTHESISED_DURATION = re.compile(rf"\((\d{{1,4}})\)\s*(?:calendar\s+)?({_UNITS})s?", re.I)
_NUMERIC_DURATION = re.compile(rf"\b(\d{{1,4}})\s*(?:calendar\s+)?({_UNITS})s?\b", re.I)
_WORD_DURATION = re.compile(
    rf"\b((?:{'|'.join(_NUMBER_WORDS)})(?:[\s-]+(?:{'|'.join(_NUMBER_WORDS)}))*)\s*"
    rf"(?:calendar\s+)?({_UNITS})s?\b",
    re.I,
)
_ISO_DURATION = re.compile(r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?$", re.I)


@dataclass(frozen=True, slots=True)
class Duration:
    years: int = 0
    months: int = 0
    days: int = 0

    def __bool__(self) -> bool:
        return bool(self.years or self.months or self.days)

    def to_iso(self) -> str:
        """ISO-8601 duration, e.g. ``P60D``, ``P3M``, ``P1Y6M``."""
        parts = "".join(
            [
                f"{self.years}Y" if self.years else "",
                f"{self.months}M" if self.months else "",
                f"{self.days}D" if self.days else "",
            ]
        )
        return f"P{parts}" if parts else "P0D"

    def humanize(self) -> str:
        chunks = [
            f"{self.years} year{'s' if self.years != 1 else ''}" if self.years else "",
            f"{self.months} month{'s' if self.months != 1 else ''}" if self.months else "",
            f"{self.days} day{'s' if self.days != 1 else ''}" if self.days else "",
        ]
        return " ".join(chunk for chunk in chunks if chunk) or "0 days"


@dataclass(slots=True)
class DateComputation:
    """Result of a deterministic computation, with its audit trail."""

    result: date | None
    formula: str
    inputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.error is None

    @property
    def result_iso(self) -> str | None:
        return self.result.isoformat() if self.result else None


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date(text: str) -> date | None:
    """Parse the date formats contracts actually use. ``None`` if unsure.

    ``dd/mm/yyyy`` versus ``mm/dd/yyyy`` is genuinely ambiguous; the US reading
    is used, and an impossible US reading (13/01/2026) falls back to the
    day-first reading.
    """
    if not text:
        return None
    candidate = text.strip()

    match = _ISO_DATE.search(candidate)
    if match:
        return _safe_date(int(match[1]), int(match[2]), int(match[3]))

    match = _MONTH_FIRST.search(candidate)
    if match:
        return _safe_date(int(match[3]), _MONTHS[match[1].lower()], int(match[2]))

    match = _DAY_FIRST.search(candidate)
    if match:
        return _safe_date(int(match[3]), _MONTHS[match[2].lower()], int(match[1]))

    match = _US_SLASH.search(candidate)
    if match:
        first, second, year = int(match[1]), int(match[2]), int(match[3])
        return _safe_date(year, first, second) or _safe_date(year, second, first)

    return None


def find_dates(text: str) -> list[tuple[int, date]]:
    """Every parseable date with its offset, left to right, no overlaps.

    Needed because one clause routinely carries two dates — "shall commence on
    February 1, 2026 and ending at midnight on January 31, 2031" — and taking
    the first one for both ends of the term is how a term end silently becomes
    a start date.
    """
    spans: list[tuple[int, int, date]] = []
    for pattern in (_ISO_DATE, _MONTH_FIRST, _DAY_FIRST, _US_SLASH):
        for match in pattern.finditer(text):
            parsed = parse_date(match.group(0))
            if parsed is None:
                continue
            if any(match.start() < end and start < match.end() for start, end, _ in spans):
                continue  # already covered by another pattern
            spans.append((match.start(), match.end(), parsed))
    return [(start, parsed) for start, _, parsed in sorted(spans)]


def _words_to_number(words: str) -> int | None:
    total = 0
    current = 0
    for word in re.split(r"[\s-]+", words.lower().strip()):
        value = _NUMBER_WORDS.get(word)
        if value is None:
            return None
        if value == 100:
            current = (current or 1) * 100
        else:
            current += value
    total += current
    return total or None


def _unit_to_duration(amount: int, unit: str) -> Duration:
    unit = unit.lower()
    if unit == _UNIT_DAYS:
        return Duration(days=amount)
    if unit == _UNIT_WEEKS:
        return Duration(days=amount * 7)
    if unit == _UNIT_MONTHS:
        return Duration(months=amount)
    return Duration(years=amount)


def parse_duration(text: str) -> Duration | None:
    """Parse a notice/renewal period from contract wording or ISO-8601.

    Handles ``P60D``, ``60 days``, ``sixty (60) days' written notice``,
    ``three (3) months``, ``one hundred eighty days``, ``two years``.
    """
    if not text:
        return None
    candidate = text.strip()

    iso = _ISO_DURATION.match(candidate.replace(" ", ""))
    if iso and any(iso.groups()):
        years, months, weeks, days = (int(group or 0) for group in iso.groups())
        return Duration(years=years, months=months, days=days + weeks * 7)

    # A parenthesised numeral is the drafter's own disambiguation - trust it first.
    match = _PARENTHESISED_DURATION.search(candidate)
    if match:
        return _unit_to_duration(int(match[1]), match[2])

    match = _NUMERIC_DURATION.search(candidate)
    if match:
        return _unit_to_duration(int(match[1]), match[2])

    match = _WORD_DURATION.search(candidate)
    if match:
        amount = _words_to_number(match[1])
        if amount:
            return _unit_to_duration(amount, match[2])

    return None


# --------------------------------------------------------------------------
# Arithmetic
# --------------------------------------------------------------------------


def add_months(anchor: date, months: int) -> date:
    """Add calendar months, clamping the day to the target month's length.

    31 January + 1 month = 28/29 February, which is the convention contracts
    assume when they say "one month from".
    """
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def shift_date(anchor: date, duration: Duration, *, sign: int = 1) -> date:
    """Shift ``anchor`` by ``duration``. Years/months first, then days."""
    shifted = add_months(anchor, sign * (duration.years * 12 + duration.months))
    if duration.days:
        shifted = date.fromordinal(shifted.toordinal() + sign * duration.days)
    return shifted


# --------------------------------------------------------------------------
# Named computations (what the pipeline actually calls)
# --------------------------------------------------------------------------


def compute_notice_deadline(anchor_date: str, duration: str) -> DateComputation:
    """``notice_deadline = contract_end_date - termination_notice_period``."""
    formula = "notice_deadline = contract_end_date - termination_notice_period"
    inputs: dict[str, Any] = {
        "contract_end_date": anchor_date,
        "termination_notice_period": duration,
    }

    parsed_date = parse_date(anchor_date)
    parsed_duration = parse_duration(duration)
    if parsed_date is None:
        return DateComputation(None, formula, inputs, error=f"unparseable_date: {anchor_date!r}")
    if parsed_duration is None:
        return DateComputation(None, formula, inputs, error=f"unparseable_duration: {duration!r}")

    result = shift_date(parsed_date, parsed_duration, sign=-1)
    inputs |= {
        "contract_end_date_iso": parsed_date.isoformat(),
        "termination_notice_period_iso": parsed_duration.to_iso(),
        "operation": "subtract",
        "evaluated": (
            f"{parsed_date.isoformat()} - {parsed_duration.humanize()} = {result.isoformat()}"
        ),
    }
    return DateComputation(result, formula, inputs)


def compute_renewal_option_deadline(anchor_date: str, duration: str) -> DateComputation:
    """``renewal_option_deadline = contract_end_date - renewal_option_notice``.

    Same arithmetic as a notice deadline, opposite consequence: this is the last
    day to *exercise* a renewal option before it lapses.
    """
    formula = "renewal_option_deadline = contract_end_date - renewal_option_notice"
    inputs: dict[str, Any] = {"contract_end_date": anchor_date, "renewal_option_notice": duration}

    parsed_date = parse_date(anchor_date)
    parsed_duration = parse_duration(duration)
    if parsed_date is None:
        return DateComputation(None, formula, inputs, error=f"unparseable_date: {anchor_date!r}")
    if parsed_duration is None:
        return DateComputation(None, formula, inputs, error=f"unparseable_duration: {duration!r}")

    result = shift_date(parsed_date, parsed_duration, sign=-1)
    inputs |= {
        "contract_end_date_iso": parsed_date.isoformat(),
        "renewal_option_notice_iso": parsed_duration.to_iso(),
        "operation": "subtract",
        "consequence": "the renewal option lapses after this date",
        "evaluated": (
            f"{parsed_date.isoformat()} - {parsed_duration.humanize()} = {result.isoformat()}"
        ),
    }
    return DateComputation(result, formula, inputs)


def compute_renewal_date(anchor_date: str, duration: str) -> DateComputation:
    """``renewal_date = contract_end_date + renewal_duration``."""
    formula = "renewal_date = contract_end_date + renewal_duration"
    inputs: dict[str, Any] = {"contract_end_date": anchor_date, "renewal_duration": duration}

    parsed_date = parse_date(anchor_date)
    parsed_duration = parse_duration(duration)
    if parsed_date is None:
        return DateComputation(None, formula, inputs, error=f"unparseable_date: {anchor_date!r}")
    if parsed_duration is None:
        return DateComputation(None, formula, inputs, error=f"unparseable_duration: {duration!r}")

    result = shift_date(parsed_date, parsed_duration, sign=1)
    inputs |= {
        "contract_end_date_iso": parsed_date.isoformat(),
        "renewal_duration_iso": parsed_duration.to_iso(),
        "operation": "add",
        "evaluated": (
            f"{parsed_date.isoformat()} + {parsed_duration.humanize()} = {result.isoformat()}"
        ),
    }
    return DateComputation(result, formula, inputs)


def days_until(target: date, reference: date | None = None) -> int:
    """Signed number of days from ``reference`` (default today) to ``target``."""
    reference = reference or date.today()
    return (target - reference).days
