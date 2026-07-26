"""Tests for the in-process adapter used by the main backend's pipeline.

The load-bearing property here is the coordinate system: every offset this
module hands back must index `ParsedDoc.text`, because that is the string the
reviewer sees. An off-by-one is a quote highlighted in the wrong place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from obligation_rag import retriever
from obligation_rag.config import reset_settings_cache
from obligation_rag.retrieval import clear_index_cache
from obligation_rag.retriever import Page, ParsedDoc

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_contract.txt"


def _parsed_doc() -> ParsedDoc:
    """Build a ParsedDoc the way the backend's ingest.py would."""
    raw = SAMPLE.read_text(encoding="utf-8")
    text = raw.replace("\f", "\n\n")

    pages: list[Page] = []
    cursor = 0
    for number, block in enumerate(raw.split("\f"), start=1):
        pages.append(Page(number=number, text=block, char_start=cursor))
        cursor += len(block) + 2  # the "\n\n" that replaced the form feed
    return ParsedDoc(text=text, pages=pages, fmt="pdf", converted_via=None)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "runtime-data"))
    monkeypatch.setenv("USE_FAKE_LLM", "true")
    monkeypatch.setenv("USE_FAKE_EMBEDDINGS", "false")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "")
    reset_settings_cache()
    clear_index_cache()
    yield
    reset_settings_cache()
    clear_index_cache()


@pytest.fixture
def doc() -> ParsedDoc:
    return _parsed_doc()


# --------------------------------------------------------------------------
# index()
# --------------------------------------------------------------------------


def test_index_returns_chunk_count(doc: ParsedDoc):
    assert retriever.index(42, doc) >= 6


def test_index_is_idempotent_per_contract_id(doc: ParsedDoc):
    first = retriever.index(42, doc)
    second = retriever.index(42, doc)

    assert first == second
    assert len(retriever.retrieve("termination notice", k=50, contract_id=42)) <= second


def test_reindexing_a_shorter_document_leaves_no_stale_chunks(doc: ParsedDoc):
    retriever.index(42, doc)

    short = ParsedDoc(
        text="A short replacement agreement about termination notice.",
        pages=[Page(number=1, text="A short replacement agreement.", char_start=0)],
        fmt="pdf",
    )
    assert retriever.index(42, short) == 1

    passages = retriever.retrieve("termination notice", k=20, contract_id=42)
    assert len(passages) == 1
    assert passages[0].text == short.text


# --------------------------------------------------------------------------
# retrieve()
# --------------------------------------------------------------------------


def test_retrieve_offsets_index_the_caller_s_text(doc: ParsedDoc):
    retriever.index(42, doc)

    passages = retriever.retrieve("termination notice period", contract_id=42)

    assert passages
    for passage in passages:
        assert doc.text[passage.char_start : passage.char_end] == passage.text


def test_retrieve_finds_the_right_page(doc: ParsedDoc):
    retriever.index(42, doc)

    top = retriever.retrieve("written notice of termination", k=3, contract_id=42)[0]

    assert top.page == 3
    assert "sixty (60) days" in top.text
    assert top.contract_id == 42
    assert top.score > 0


def test_retrieve_honours_k(doc: ParsedDoc):
    retriever.index(42, doc)

    assert len(retriever.retrieve("agreement", k=2, contract_id=42)) <= 2


def test_retrieve_searches_the_whole_corpus_when_contract_id_is_none(doc: ParsedDoc):
    retriever.index(42, doc)
    other = ParsedDoc(
        text="This lease is governed by the laws of the State of New York exclusively.",
        pages=[Page(number=1, text="lease", char_start=0)],
        fmt="docx",
        converted_via="libreoffice",
    )
    retriever.index(99, other)

    everywhere = retriever.retrieve("governed by the laws of the State", k=10)

    assert {passage.contract_id for passage in everywhere} == {42, 99}
    for passage in everywhere:
        source = doc if passage.contract_id == 42 else other
        assert source.text[passage.char_start : passage.char_end] == passage.text


def test_corpus_results_do_not_merge_chunks_across_contracts(doc: ParsedDoc):
    """Every document has a "chunk_0"; fusion must not collapse them."""
    retriever.index(42, doc)
    retriever.index(99, doc)

    passages = retriever.retrieve("this Agreement", k=20)

    assert len({(p.contract_id, p.char_start) for p in passages}) == len(passages)
    assert {p.contract_id for p in passages} == {42, 99}


def test_retrieve_never_raises_on_an_unknown_contract():
    assert retriever.retrieve("anything", contract_id=1234) == []


def test_retrieve_never_raises_on_an_empty_corpus():
    assert retriever.retrieve("anything") == []


def test_retrieve_never_raises_when_the_store_is_broken(doc: ParsedDoc, monkeypatch):
    retriever.index(42, doc)
    monkeypatch.setattr(
        retriever, "get_document_index", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert retriever.retrieve("termination", contract_id=42) == []


# --------------------------------------------------------------------------
# extract()
# --------------------------------------------------------------------------


def test_extract_returns_verified_obligations_and_unlocks_approval(doc: ParsedDoc):
    retriever.index(42, doc)

    result = retriever.extract(42, doc)

    assert result.can_approve is True
    by_field = {obligation.field: obligation for obligation in result.obligations}
    assert by_field["contract_end_date"].value == "2026-03-31"
    assert by_field["termination_notice_period"].value == "P60D"
    assert by_field["contract_end_date"].status == "verified"


def test_extract_evidence_offsets_index_the_caller_s_text(doc: ParsedDoc):
    retriever.index(42, doc)

    result = retriever.extract(42, doc)

    quoted = [o for o in result.obligations if o.evidence and o.evidence.char_start is not None]
    assert quoted
    for obligation in quoted:
        evidence = obligation.evidence
        assert doc.text[evidence.char_start : evidence.char_end] == evidence.quote


def test_extract_computes_the_deadline_in_code(doc: ParsedDoc):
    retriever.index(42, doc)

    deadline = next(
        o for o in retriever.extract(42, doc).obligations if o.field == "notice_deadline"
    )

    assert deadline.status == "computed"
    assert deadline.value == "2026-01-30"
    assert deadline.evidence is None
    assert deadline.formula == "notice_deadline = contract_end_date - termination_notice_period"
    assert deadline.inputs["evaluated"] == "2026-03-31 - 60 days = 2026-01-30"


def test_extract_indexes_on_demand_if_the_pipeline_skipped_it(doc: ParsedDoc):
    result = retriever.extract(42, doc)  # no index() call first

    assert result.obligations
    assert result.can_approve is True


def test_extract_never_raises(doc: ParsedDoc, monkeypatch):
    monkeypatch.setattr(
        retriever, "run_extraction", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    result = retriever.extract(42, doc)

    assert result.can_approve is False
    assert result.obligations == []
    assert result.failures == ["extraction_crashed"]


def test_a_hallucinated_quote_blocks_approval(doc: ParsedDoc, monkeypatch):
    """The ablation demo, across the adapter boundary."""
    from obligation_rag.llm_client import HallucinatingLLMClient

    retriever.index(42, doc)
    monkeypatch.setattr(retriever, "get_llm_client", lambda settings: HallucinatingLLMClient())

    result = retriever.extract(42, doc)

    assert result.can_approve is False
    failed = [o for o in result.obligations if o.status == "failed"]
    assert failed
    assert "quote_not_found_on_page" in failed[0].reason
