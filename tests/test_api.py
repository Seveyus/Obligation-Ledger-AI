"""HTTP integration tests — the exact surface Aditya's backend consumes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from obligation_rag.api import app
from obligation_rag.config import get_settings, reset_settings_cache
from obligation_rag.retrieval import clear_index_cache


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "runtime-data"))
    monkeypatch.setenv("USE_FAKE_LLM", "true")
    monkeypatch.setenv("USE_FAKE_EMBEDDINGS", "false")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "")
    reset_settings_cache()
    clear_index_cache()

    with TestClient(app) as test_client:
        yield test_client

    reset_settings_cache()
    clear_index_cache()


@pytest.fixture
def ingested(client, sample_contract_path: Path) -> str:
    with sample_contract_path.open("rb") as handle:
        response = client.post(
            "/v1/documents/ingest",
            files={"file": ("sample_contract.txt", handle, "text/plain")},
        )
    assert response.status_code == 200, response.text
    return response.json()["document_id"]


def test_health(client):
    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["llm_mode"] == "fake"
    assert payload["retrieval_mode"] == "bm25_only"
    assert payload["document_count"] == 0


def test_ingest_returns_the_document_summary(client, sample_contract_path: Path):
    with sample_contract_path.open("rb") as handle:
        response = client.post(
            "/v1/documents/ingest",
            files={"file": ("sample_contract.txt", handle, "text/plain")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["filename"] == "sample_contract.txt"
    assert payload["page_count"] == 6
    assert payload["chunk_count"] >= 6
    assert payload["document_id"].startswith("contract_")


def test_ingest_accepts_a_caller_supplied_id(client, sample_contract_path: Path):
    with sample_contract_path.open("rb") as handle:
        response = client.post(
            "/v1/documents/ingest",
            files={"file": ("sample_contract.txt", handle, "text/plain")},
            data={"document_id": "contract_123"},
        )

    assert response.json()["document_id"] == "contract_123"
    assert client.get("/v1/documents/contract_123").status_code == 200


def test_ingest_rejects_unsupported_types(client):
    response = client.post(
        "/v1/documents/ingest",
        files={"file": ("contract.docx", b"binary", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert "unsupported_file_type" in response.json()["detail"]


def test_ingest_rejects_an_empty_file(client):
    response = client.post(
        "/v1/documents/ingest", files={"file": ("contract.pdf", b"", "application/pdf")}
    )

    assert response.status_code == 400


def test_search_returns_page_anchored_chunks(client, ingested):
    response = client.post(
        f"/v1/documents/{ingested}/search",
        json={"query": "termination notice period", "top_k": 3},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["retrieval_mode"] == "bm25_only"
    assert payload["chunks"], "expected at least one hit"
    top = payload["chunks"][0]
    assert top["page"] == 3
    assert "sixty (60) days" in top["text"]
    assert top["fused_score"] > 0


def test_extract_returns_verified_obligations(client, ingested):
    response = client.post(f"/v1/documents/{ingested}/extract", json={})

    payload = response.json()
    assert response.status_code == 200
    assert payload["can_approve"] is True

    by_type = {item["obligation_type"]: item for item in payload["obligations"]}
    notice = by_type["termination_notice_period"]
    assert notice["normalized_value"] == "P60D"
    assert notice["status"] == "verified"
    assert notice["verification_method"] == "normalized_exact_match"
    assert notice["source_evidence"]["page"] == 3

    deadline = by_type["notice_deadline"]
    assert deadline["status"] == "computed"
    assert deadline["normalized_value"] == "2026-01-30"
    assert deadline["computation_formula"]


def test_extract_accepts_the_short_type_aliases(client, ingested):
    response = client.post(
        f"/v1/documents/{ingested}/extract",
        json={"obligation_types": ["term_end", "termination_notice", "automatic_renewal"]},
    )

    payload = response.json()
    types = {item["obligation_type"] for item in payload["obligations"]}
    assert types == {"contract_end_date", "termination_notice_period", "automatic_renewal"}


def test_extract_rejects_an_unknown_type(client, ingested):
    response = client.post(
        f"/v1/documents/{ingested}/extract", json={"obligation_types": ["favourite_colour"]}
    )

    assert response.status_code == 422


def test_the_last_extraction_is_persisted(client, ingested):
    client.post(f"/v1/documents/{ingested}/extract", json={})

    payload = client.get(f"/v1/documents/{ingested}/obligations").json()

    assert payload["can_approve"] is True
    assert len(payload["obligations"]) >= 10


def test_evidence_verify_confirms_a_real_quote(client, ingested):
    response = client.post(
        "/v1/evidence/verify",
        json={
            "document_id": ingested,
            "quote": "written notice of termination to the other party",
            "page": 3,
        },
    )

    payload = response.json()
    assert payload["verified"] is True
    assert payload["verification_method"] == "normalized_exact_match"
    assert payload["start_offset"] is not None


def test_evidence_verify_rejects_an_invented_quote(client, ingested):
    response = client.post(
        "/v1/evidence/verify",
        json={
            "document_id": ingested,
            "quote": "This Agreement remains in force until December 31, 2029.",
            "page": 3,
        },
    )

    payload = response.json()
    assert payload["verified"] is False
    assert payload["verification_method"] == "none"


def test_evidence_verify_can_search_the_whole_document(client, ingested):
    response = client.post(
        "/v1/evidence/verify",
        json={"document_id": ingested, "quote": "governed by the laws of the State of Delaware"},
    )

    payload = response.json()
    assert payload["verified"] is True
    assert payload["page"] == 6


def test_deadline_endpoint_computes_and_shows_its_work(client):
    response = client.post(
        "/v1/deadlines/compute",
        json={
            "operation": "notice_deadline",
            "anchor_date": "March 31, 2026",
            "duration": "sixty (60) days",
        },
    )

    payload = response.json()
    assert payload["result_date"] == "2026-01-30"
    assert payload["status"] == "computed"
    assert payload["computation_inputs"]["evaluated"] == "2026-03-31 - 60 days = 2026-01-30"


def test_deadline_endpoint_computes_renewals(client):
    response = client.post(
        "/v1/deadlines/compute",
        json={
            "operation": "renewal_date",
            "anchor_date": "2026-03-31",
            "duration": "twelve (12) months",
        },
    )

    assert response.json()["result_date"] == "2027-03-31"


def test_deadline_endpoint_reports_unparseable_input(client):
    response = client.post(
        "/v1/deadlines/compute",
        json={"anchor_date": "next spring", "duration": "P60D"},
    )

    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["result_date"] is None
    assert "unparseable_date" in payload["error"]


def test_pages_endpoint_exposes_the_text_offsets_refer_to(client, ingested):
    payload = client.get(f"/v1/documents/{ingested}/pages").json()

    assert payload["page_count"] == 6
    assert "Governing Law" in payload["pages"][5]["text"]


def test_documents_can_be_listed_and_deleted(client, ingested):
    assert len(client.get("/v1/documents").json()) == 1

    assert client.delete(f"/v1/documents/{ingested}").status_code == 204
    assert client.get("/v1/documents").json() == []
    assert client.get(f"/v1/documents/{ingested}").status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/v1/documents/nope/search", {"query": "x"}),
        ("post", "/v1/documents/nope/extract", {}),
        ("get", "/v1/documents/nope/pages", None),
        ("post", "/v1/evidence/verify", {"document_id": "nope", "quote": "some quote here"}),
    ],
)
def test_unknown_documents_return_404(client, method, path, body):
    call = getattr(client, method)
    response = call(path) if body is None else call(path, json=body)

    assert response.status_code == 404
    assert "unknown_document" in response.json()["detail"]


def test_settings_are_read_from_the_environment(client, tmp_path):
    settings = get_settings()

    assert settings.use_fake_llm is True
    assert settings.rag_data_dir == tmp_path / "runtime-data"
    assert settings.db_path.exists()
