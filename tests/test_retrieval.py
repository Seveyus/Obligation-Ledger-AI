"""Retrieval decides what the model is allowed to see, so it gets tested too."""

from __future__ import annotations

from obligation_rag.retrieval import (
    RETRIEVAL_MODE_BM25,
    RETRIEVAL_MODE_HYBRID,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenizer_keeps_numbers():
    assert tokenize("sixty (60) days' notice") == ["sixty", "60", "days", "notice"]
    assert tokenize("§ 3.2 Non-Renewal") == ["3", "2", "non", "renewal"]


def test_rrf_scores_are_rank_based_not_score_based():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)

    assert fused["a"] == fused["b"]
    assert fused["a"] == 1 / 61 + 1 / 62


def test_rrf_rewards_agreement_between_rankers():
    """A chunk both rankers like beats a chunk only one of them found."""
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=60)

    assert fused["a"] > fused["b"]
    assert fused["b"] == fused["c"]


def test_rrf_handles_a_single_ranker():
    fused = reciprocal_rank_fusion([["a", "b"]], k=60)

    assert fused["a"] > fused["b"]


def test_bm25_only_mode_is_declared(bm25_index):
    assert bm25_index.retrieval_mode == RETRIEVAL_MODE_BM25


def test_bm25_finds_the_termination_clause(bm25_index):
    hits = bm25_index.search("termination notice period", top_k=3)

    assert hits, "the notice clause must be retrievable"
    assert hits[0].page == 3
    assert "sixty (60) days" in hits[0].text
    assert hits[0].lexical_score > 0


def test_scores_are_reported_per_chunk(bm25_index):
    hit = bm25_index.search("governing law Delaware", top_k=1)[0]

    assert hit.page == 6
    assert hit.fused_score > 0
    assert hit.vector_score == 0.0  # no embedding backend loaded


def test_hybrid_mode_uses_both_rankers(hybrid_index):
    assert hybrid_index.retrieval_mode == RETRIEVAL_MODE_HYBRID

    hits = hybrid_index.search("how much notice is needed to cancel", top_k=3)

    assert hits
    assert any(hit.page == 3 for hit in hits)
    assert any(hit.vector_score > 0 for hit in hits)


def test_hybrid_retrieval_still_ranks_the_fee_clause_first(hybrid_index):
    hits = hybrid_index.search("annual subscription fee increase percentage", top_k=2)

    assert hits[0].page == 4


def test_query_with_no_lexical_or_semantic_signal_returns_nothing(bm25_index):
    assert bm25_index.search("zzzz qqqq xxxx", top_k=5) == []


def test_top_k_is_honoured(bm25_index):
    assert len(bm25_index.search("agreement", top_k=2)) <= 2
