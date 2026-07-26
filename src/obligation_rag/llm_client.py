"""Clients for the local reasoning model.

The model server (``gpt-oss-120b`` behind an OpenAI-compatible endpoint) lives
outside this repository — this module only knows a base URL. Three adapters
share one interface:

* ``OpenAICompatibleClient`` — the real thing.
* ``FakeLLMClient``          — deterministic, rule-based, quotes copied out of
  the retrieved text. Lets the whole pipeline run and be tested with no model
  server at all (``USE_FAKE_LLM=true``).
* ``HallucinatingLLMClient`` — invents a plausible-looking obligation with a
  quote that is not in the document. This is the ablation demo from the spec
  (§10): with verification on, it must be caught and must block approval.

What the model is asked for is deliberately narrow: read these clauses, return
values with the exact quote you read them from. It is never asked to do date
arithmetic and never asked whether its own quote is real.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from .config import Settings
from .date_math import find_dates, parse_duration
from .schemas import LLMExtractionRequest, LLMExtractionResponse, LLMObligation, ObligationType

logger = logging.getLogger(__name__)

MODE_REAL = "gpt-oss-120b"
MODE_FAKE = "fake"
MODE_HALLUCINATING = "fake_hallucinating"


class LLMClientError(RuntimeError):
    """The model was unreachable, or answered something unusable."""


SYSTEM_PROMPT = """\
You extract contractual obligations from contract excerpts.

Absolute rules:
1. Use ONLY the provided excerpts. Never use outside knowledge.
2. Every obligation MUST include `quote`: a verbatim, character-for-character \
span copied from one excerpt. Do not paraphrase, do not fix typos, do not \
merge two sentences.
3. `page` MUST be the page number of the excerpt the quote came from.
4. If an obligation type is not supported by an explicit excerpt, omit it. \
An omission is correct; an invented quote is a critical failure.
5. Never compute dates. Report the dates and durations as written; deadline \
arithmetic is done by separate code.

Answer with a single JSON object, no prose, no markdown fences:
{"obligations": [{"obligation_type": "...", "raw_value": "...", \
"normalized_value": "...", "quote": "...", "page": 1}]}

Normalization conventions for `normalized_value`:
- dates: YYYY-MM-DD
- durations: ISO-8601 (P60D, P3M, P1Y)
- money: "<CURRENCY> <amount>" (e.g. "USD 12500.00")
- percentages: "4.5%"
- automatic_renewal: "true" or "false"
- anything else: a short canonical string
"""


class LLMClient(ABC):
    """Proposes candidate obligations. Its output is never trusted as-is."""

    mode: str = "abstract"

    @abstractmethod
    def propose_obligations(self, request: LLMExtractionRequest) -> LLMExtractionResponse: ...


# --------------------------------------------------------------------------
# Real client
# --------------------------------------------------------------------------


def _extract_json_object(raw: str) -> str:
    """Pull the first balanced JSON object out of a model answer."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        raise LLMClientError(f"model_returned_no_json: {raw[:200]!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise LLMClientError(f"model_returned_truncated_json: {raw[:200]!r}")


def build_user_prompt(request: LLMExtractionRequest) -> str:
    wanted = "\n".join(f"- {obligation_type.value}" for obligation_type in request.obligation_types)
    excerpts = "\n\n".join(
        f"[excerpt {index} | page {chunk.page}]\n{chunk.text}"
        for index, chunk in enumerate(request.context, start=1)
    )
    return (
        f"Obligation types to look for:\n{wanted}\n\n"
        f"Contract excerpts:\n\n{excerpts}\n\n"
        "Return the JSON object now."
    )


class OpenAICompatibleClient(LLMClient):
    """Talks to any OpenAI-compatible server (vLLM, llama.cpp, TGI, ...)."""

    mode = MODE_REAL

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self.settings = settings
        self.model = settings.llm_model
        self.mode = settings.llm_model
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=1,
        )

    def _chat(self, messages: list[dict[str, str]], *, json_mode: bool) -> str:
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.settings.llm_max_output_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def propose_obligations(self, request: LLMExtractionRequest) -> LLMExtractionResponse:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(request)},
        ]
        try:
            raw = self._chat(messages, json_mode=True)
        except Exception as error:  # noqa: BLE001 - server capabilities vary
            logger.warning("structured json mode failed (%s); retrying without it", error)
            try:
                raw = self._chat(messages, json_mode=False)
            except Exception as retry_error:  # noqa: BLE001
                raise LLMClientError(f"llm_request_failed: {retry_error}") from retry_error

        payload = _extract_json_object(raw)
        try:
            return LLMExtractionResponse.model_validate(json.loads(payload))
        except (json.JSONDecodeError, ValueError) as error:
            raise LLMClientError(f"llm_response_not_schema_conformant: {error}") from error


# --------------------------------------------------------------------------
# Fake clients
# --------------------------------------------------------------------------

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
# Numbered clauses ("3.2 Notice of Non-Renewal.") start with a digit, so the
# lookahead has to admit digits or two clauses merge into one "sentence" and
# the wrong duration gets attributed to the wrong obligation.
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9(\"'])")
_MONEY = re.compile(r"(?:USD|EUR|GBP|\$|€|£)\s?[\d,]+(?:\.\d{2})?", re.IGNORECASE)
_PERCENT = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?(?:%|percent)", re.IGNORECASE)
_MIN_CLAUSE_CHARS = 60
_GOVERNING_LAW = re.compile(
    r"laws of (?:the )?(?:State of |Commonwealth of )?([A-Z][A-Za-z ]+?)(?=[,.]|\s+without)"
)


def _sentences(text: str) -> list[str]:
    """Paragraph boundaries first (headings rarely end in a period), then sentences."""
    sentences: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            cleaned = sentence.strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences


class FakeLLMClient(LLMClient):
    """Rule-based stand-in that quotes the retrieved text verbatim.

    It is intentionally *not* a mock returning canned strings: it reads the
    same excerpts the real model would, so quotes it produces really exist and
    the verification stage is exercised for real.
    """

    mode = MODE_FAKE

    def propose_obligations(self, request: LLMExtractionRequest) -> LLMExtractionResponse:
        wanted = set(request.obligation_types)
        found: dict[ObligationType, LLMObligation] = {}

        for chunk in request.context:
            for sentence in _sentences(chunk.text):
                for obligation_type, value in self._match_sentence(sentence):
                    if obligation_type in wanted and obligation_type not in found:
                        found[obligation_type] = LLMObligation(
                            obligation_type=obligation_type.value,
                            raw_value=value,
                            quote=sentence,
                            page=chunk.page,
                        )

        ordered = [found[key] for key in request.obligation_types if key in found]
        return LLMExtractionResponse(obligations=ordered)

    def _match_sentence(self, sentence: str) -> list[tuple[ObligationType, str]]:
        lowered = sentence.lower()
        matches: list[tuple[ObligationType, str]] = []

        # One clause often carries both ends of the term ("commence on … and
        # ending at midnight on …"), so the first date opens it and the last
        # date closes it.
        dates = find_dates(sentence)
        duration = parse_duration(sentence)

        if dates and any(
            keyword in lowered
            for keyword in (
                "commence",
                "effective date",
                "begins on",
                "start date",
                "effective as of",
            )
        ):
            matches.append((ObligationType.CONTRACT_START_DATE, dates[0][1].isoformat()))

        if dates and any(
            keyword in lowered
            for keyword in (
                "expire",
                "expires",
                "expiration",
                "end date",
                "ending",
                "terminate on",
                "termination date",
                "through ",
                "until",
            )
        ):
            matches.append((ObligationType.CONTRACT_END_DATE, dates[-1][1].isoformat()))

        # A bare heading ("2.2 Automatic Renewal.") is a real quote but not
        # evidence of anything, so free-text matches need an actual clause.
        substantial = len(sentence) >= _MIN_CLAUSE_CHARS

        if (
            "renew" in lowered
            and substantial
            and any(keyword in lowered for keyword in ("automatic", "automatically", "auto-renew"))
        ):
            matches.append((ObligationType.AUTOMATIC_RENEWAL, "true"))
            if duration:
                matches.append((ObligationType.RENEWAL_DURATION, duration.to_iso()))

        # "…90 days prior to the Termination Date" in a renewal-option clause is
        # a deadline to KEEP the contract, not to leave it. The mention of the
        # Termination Date must not pull it into the termination bucket.
        # A renewal clause states two very different durations: the length of
        # the renewal term ("an additional 5 year term") and the notice needed
        # to claim it ("not less than 90 days prior"). Only the second is a
        # deadline, and only a notice-shaped phrasing identifies it.
        renewal_context = any(
            keyword in lowered
            for keyword in ("option to renew", "exercised by providing", "notice of renewal")
        )
        notice_shaped = "notice" in lowered and any(
            keyword in lowered
            for keyword in ("not less than", "prior to", "no later than", "at least")
        )
        renewal_option = bool(duration) and renewal_context and notice_shaped
        if renewal_option:
            matches.append((ObligationType.RENEWAL_OPTION_NOTICE, duration.to_iso()))
        elif duration and renewal_context and "term" in lowered:
            matches.append((ObligationType.RENEWAL_DURATION, duration.to_iso()))

        # A duration next to "terminate" is not automatically a notice period:
        # "fail to cure within 15 days" is a cure period, "within 30 days after
        # the occurrence of such casualty" is a casualty window. Both are real
        # obligations, neither is the notice you must give to walk away.
        other_clause = any(
            keyword in lowered
            for keyword in (
                "cure",
                "default",
                "breach",
                "casualty",
                "damaged",
                "destroyed",
                "condemn",
                "eminent domain",
            )
        )

        if (
            duration
            and not renewal_option
            and not other_clause
            and "notice" in lowered
            and any(
                keyword in lowered
                for keyword in ("terminat", "cancel", "non-renew", "not to renew")
            )
        ):
            matches.append((ObligationType.TERMINATION_NOTICE_PERIOD, duration.to_iso()))

        money = _MONEY.search(sentence)
        if money and any(keyword in lowered for keyword in ("pay", "fee", "invoice", "amount due")):
            matches.append((ObligationType.PAYMENT_OBLIGATION, money.group(0)))

        percent = _PERCENT.search(sentence)
        if percent and any(
            keyword in lowered for keyword in ("increase", "escalat", "adjust", "uplift")
        ):
            matches.append((ObligationType.FEE_ESCALATION, percent.group(0)))

        if "indemnif" in lowered and substantial:
            matches.append((ObligationType.INDEMNIFICATION, sentence[:180]))

        if "liabilit" in lowered and (money or ("exceed" in lowered and substantial)):
            matches.append(
                (ObligationType.LIABILITY_CAP, money.group(0) if money else sentence[:180])
            )

        law = _GOVERNING_LAW.search(sentence)
        if law:
            matches.append((ObligationType.GOVERNING_LAW, law.group(1).strip()))

        return matches


class HallucinatingLLMClient(LLMClient):
    """Returns a confident obligation whose quote is nowhere in the document.

    Used by the ablation test: verification off -> a wrong date slips through;
    verification on -> it is caught and approval is blocked.
    """

    mode = MODE_HALLUCINATING

    def propose_obligations(self, request: LLMExtractionRequest) -> LLMExtractionResponse:
        page = request.context[0].page if request.context else 1
        return LLMExtractionResponse(
            obligations=[
                LLMObligation(
                    obligation_type=ObligationType.CONTRACT_END_DATE.value,
                    raw_value="December 31, 2029",
                    normalized_value="2029-12-31",
                    quote=(
                        "This Agreement shall remain in full force and effect until "
                        "December 31, 2029, unless terminated earlier."
                    ),
                    page=page,
                )
            ]
        )


def get_llm_client(settings: Settings) -> LLMClient:
    if settings.use_fake_llm:
        return FakeLLMClient()
    return OpenAICompatibleClient(settings)
