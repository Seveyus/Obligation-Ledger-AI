"""Deterministic source-evidence verification.

Trust rule #1 of the product: *every value quotes the contract*. This module is
the only thing that may declare a quote real. It is pure Python — the language
model is never asked to verify its own output, because a model that invented a
quote will happily confirm it.

Algorithm, in order:

1. normalize Unicode punctuation (smart quotes, dashes, ligatures, NBSP, ...)
2. normalize whitespace (any run of whitespace becomes a single space)
3. compare the claimed quote against the text of the *claimed page only*
4. normalized exact substring match wins
5. bounded fuzzy fallback, for PDF extraction artifacts only
6. record which method succeeded
7. otherwise: failed (and, as a courtesy, report the page where the quote
   actually lives so a human can fix the page reference in one click)

Offsets returned are character offsets into the *original* stored page text,
not into the normalized string, so the approval UI can highlight the span.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from .schemas import VerificationMethod

#: Characters PDF extractors routinely emit that a model rewrites to ASCII.
PUNCTUATION_MAP: dict[str, str] = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "ʼ": "'",
    "´": "'",
    "`": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "‐": "-",
    "‑": "-",
    "…": "...",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    "­": "",  # soft hyphen
    "​": "",
    "‌": "",
    "‍": "",
    "﻿": "",
    "•": "*",
    "·": "*",
}


def _expand(char: str) -> str:
    """Normalized form of a single character (may be empty or multi-char)."""
    if char in PUNCTUATION_MAP:
        return PUNCTUATION_MAP[char]
    if char.isspace():
        return " "
    return unicodedata.normalize("NFKC", char).casefold()


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize ``text`` and return ``(normalized, offsets)``.

    ``offsets[i]`` is the index in the original ``text`` of the character that
    produced ``normalized[i]``. Whitespace runs collapse to a single space,
    which maps back to the first character of the run.
    """
    chars: list[str] = []
    offsets: list[int] = []
    previous_was_space = True  # leading whitespace is dropped

    for original_index, char in enumerate(text):
        expanded = _expand(char)
        for produced in expanded:
            if produced == " ":
                if previous_was_space:
                    continue
                previous_was_space = True
            else:
                previous_was_space = False
            chars.append(produced)
            offsets.append(original_index)

    while chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()

    return "".join(chars), offsets


def normalize_text(text: str) -> str:
    """Normalized form of ``text`` (see :func:`normalize_with_map`)."""
    return normalize_with_map(text)[0]


#: Form contracts express values as selected options. A quote lifted from an
#: *unselected* option is a real quote of a value that does not apply — the
#: dominant failure mode on filled templates, and one that plain substring
#: matching cannot see.
CHECKED_MARKERS = "☒☑"
UNCHECKED_MARKERS = "☐"
_CHECKBOX_MARKERS = CHECKED_MARKERS + UNCHECKED_MARKERS


def preceding_checkbox(text: str, position: int) -> str | None:
    """Nearest checkbox marker at or before ``position``, on the same line.

    The marker may be the first character of the span itself ("☒ 3% each
    renewal term") or sit just before it ("☒ " + "3% each renewal term"), so
    both are searched. The search never crosses a line break: a checkbox on a
    previous line governs a different option.
    """
    line_start = text.rfind("\n", 0, position) + 1
    # +2 so a marker that opens the span itself still counts as governing it.
    for index in range(min(position + 2, len(text)) - 1, line_start - 1, -1):
        if text[index] in _CHECKBOX_MARKERS:
            return text[index]
    return None


def locate_value(page_text: str, start: int, end: int, value: str | None) -> int:
    """Offset of ``value`` inside the matched span, or the span start.

    A quote often covers a whole option block — the empty box and the ticked
    one — so anchoring the checkbox test on the span start would condemn a
    value that was in fact selected. The test belongs on the value itself.
    """
    if not value:
        return start
    span = page_text[start:end]
    normalized_span, offsets = normalize_with_map(span)
    normalized_value = normalize_text(value)
    if not normalized_value:
        return start
    index = normalized_span.find(normalized_value)
    if index == -1 or index >= len(offsets):
        return start
    return start + offsets[index]


def unchecked_option_reason(text: str, position: int) -> str | None:
    """Failure reason when the quote belongs to an option that was not ticked."""
    marker = preceding_checkbox(text, position)
    if marker and marker in UNCHECKED_MARKERS:
        return (
            "unchecked_option: the quote is a form option that was NOT selected "
            f"({marker}); the value does not apply to this contract"
        )
    return None


@dataclass(slots=True)
class VerificationOutcome:
    verified: bool
    method: VerificationMethod = VerificationMethod.NONE
    reason: str | None = None
    page: int | None = None
    matched_text: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    similarity: float | None = None


def _span_to_original(
    offsets: list[int],
    original: str,
    start: int,
    end: int,
) -> tuple[int, int, str]:
    """Map a normalized ``[start, end)`` span back onto the original text."""
    if not offsets or start >= len(offsets):
        return 0, 0, ""
    end = min(end, len(offsets))
    original_start = offsets[start]
    original_end = offsets[end - 1] + 1 if end > start else original_start
    return original_start, original_end, original[original_start:original_end]


def verify_quote_on_page(
    quote: str,
    page_text: str,
    *,
    page: int | None = None,
    fuzzy_threshold: float = 0.92,
    fuzzy_max_length_ratio: float = 1.35,
    min_quote_chars: int = 12,
    reject_unchecked_options: bool = True,
    value: str | None = None,
) -> VerificationOutcome:
    """Check whether ``quote`` really appears in ``page_text``.

    ``value`` is the claimed value the quote is meant to support. When given,
    the checkbox test anchors on it rather than on the start of the quote.
    """
    normalized_quote = normalize_text(quote)
    if len(normalized_quote) < min_quote_chars:
        return VerificationOutcome(
            verified=False,
            reason=f"quote_too_short: {len(normalized_quote)} < {min_quote_chars} characters",
            page=page,
        )

    normalized_page, offsets = normalize_with_map(page_text)
    if not normalized_page:
        return VerificationOutcome(verified=False, reason="page_has_no_text", page=page)

    # 4. normalized exact substring match
    index = normalized_page.find(normalized_quote)
    if index != -1:
        start, end, matched = _span_to_original(
            offsets, page_text, index, index + len(normalized_quote)
        )
        anchor = locate_value(page_text, start, end, value)
        rejection = unchecked_option_reason(page_text, anchor) if reject_unchecked_options else None
        return VerificationOutcome(
            verified=rejection is None,
            method=VerificationMethod.NORMALIZED_EXACT_MATCH
            if not rejection
            else VerificationMethod.NONE,
            reason=rejection,
            page=page,
            matched_text=matched,
            start_offset=start,
            end_offset=end,
            similarity=1.0,
        )

    # 5. bounded fuzzy fallback — one window, anchored on the longest common
    #    block, never a free-form search of the whole document.
    matcher = SequenceMatcher(None, normalized_page, normalized_quote, autojunk=False)
    anchor = matcher.find_longest_match(0, len(normalized_page), 0, len(normalized_quote))
    if anchor.size == 0:
        return VerificationOutcome(
            verified=False, reason="quote_not_found_on_page", page=page, similarity=0.0
        )

    window_length = int(len(normalized_quote) * fuzzy_max_length_ratio) + 1
    window_start = max(0, anchor.a - anchor.b)
    window_end = min(len(normalized_page), window_start + window_length)
    window = normalized_page[window_start:window_end]

    window_matcher = SequenceMatcher(None, window, normalized_quote, autojunk=False)
    blocks = [block for block in window_matcher.get_matching_blocks() if block.size]
    if not blocks:
        return VerificationOutcome(
            verified=False, reason="quote_not_found_on_page", page=page, similarity=0.0
        )

    # Trim the padded window down to the matched region before scoring, so the
    # padding itself cannot depress the similarity below the threshold.
    span_start = window_start + blocks[0].a
    span_end = window_start + blocks[-1].a + blocks[-1].size
    candidate = normalized_page[span_start:span_end]
    similarity = SequenceMatcher(None, candidate, normalized_quote, autojunk=False).ratio()

    too_long = len(candidate) > len(normalized_quote) * fuzzy_max_length_ratio
    if similarity < fuzzy_threshold or too_long:
        return VerificationOutcome(
            verified=False,
            reason=f"quote_not_found_on_page (best similarity {similarity:.3f})",
            page=page,
            similarity=similarity,
        )
    start, end, matched = _span_to_original(offsets, page_text, span_start, span_end)
    anchor = locate_value(page_text, start, end, value)
    rejection = unchecked_option_reason(page_text, anchor) if reject_unchecked_options else None
    return VerificationOutcome(
        verified=rejection is None,
        method=VerificationMethod.FUZZY_MATCH if not rejection else VerificationMethod.NONE,
        reason=rejection or f"fuzzy_match similarity={similarity:.3f}",
        page=page,
        matched_text=matched,
        start_offset=start,
        end_offset=end,
        similarity=similarity,
    )


def verify_evidence(
    quote: str,
    pages: dict[int, str],
    *,
    claimed_page: int | None = None,
    fuzzy_threshold: float = 0.92,
    fuzzy_max_length_ratio: float = 1.35,
    min_quote_chars: int = 12,
    reject_unchecked_options: bool = True,
    value: str | None = None,
) -> VerificationOutcome:
    """Verify ``quote`` against ``pages``.

    When ``claimed_page`` is given, only that page can produce a *verified*
    result. If the quote turns out to live on a different page the outcome is
    still a failure, but the reason names the real page.
    """
    options: dict[str, float | int | bool | str | None] = {
        "fuzzy_threshold": fuzzy_threshold,
        "fuzzy_max_length_ratio": fuzzy_max_length_ratio,
        "min_quote_chars": min_quote_chars,
        "reject_unchecked_options": reject_unchecked_options,
        "value": value,
    }

    if claimed_page is not None:
        page_text = pages.get(claimed_page)
        if page_text is None:
            return VerificationOutcome(
                verified=False,
                reason=f"page_out_of_range: page {claimed_page} does not exist",
                page=claimed_page,
            )
        outcome = verify_quote_on_page(quote, page_text, page=claimed_page, **options)
        if outcome.verified:
            return outcome
        elsewhere = _find_on_any_page(quote, pages, skip=claimed_page, **options)
        if elsewhere is not None:
            outcome.reason = (
                f"wrong_page: quote claimed on page {claimed_page} "
                f"but found on page {elsewhere.page}"
            )
        return outcome

    found = _find_on_any_page(quote, pages, skip=None, **options)
    if found is not None:
        return found
    return VerificationOutcome(verified=False, reason="quote_not_found_in_document")


def _find_on_any_page(
    quote: str,
    pages: dict[int, str],
    *,
    skip: int | None,
    **options: float | int | bool | str | None,
) -> VerificationOutcome | None:
    for page_number in sorted(pages):
        if page_number == skip:
            continue
        outcome = verify_quote_on_page(quote, pages[page_number], page=page_number, **options)
        if outcome.verified:
            return outcome
    return None
