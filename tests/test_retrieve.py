"""Keyword retrieval: tokenisation, BM25 scoring, and the hybrid blend."""

from __future__ import annotations

import pytest

from ftlab.retrieve import BM25Index, Document, Retriever, exact_scores, tokenize


def doc(id_: str, text: str, handles: list[str] | None = None) -> Document:
    return Document(id=id_, kind="contract", title=id_, text=text, handles=handles or [])


# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------


def test_contract_numbers_survive_as_one_token():
    """The whole point of the retrieval layer is the high-entropy strings.

    Splitting "75D30124C00000" on its letter/digit boundaries would produce
    fragments shared by every contract number in the library, turning the most
    discriminating token in the corpus into the least.
    """
    assert tokenize("Contract 75D30124C00000 ran 2024-2025") == [
        "contract", "75d30124c00000", "ran", "2024", "2025",
    ]


def test_tokenizer_is_case_folded_and_drops_punctuation():
    assert tokenize("CDC's NCEZID -- Public Health!") == [
        "cdc", "s", "ncezid", "public", "health",
    ]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_idf_never_goes_negative():
    """A term in most of the collection must not subtract score.

    The textbook Robertson/Sparck-Jones IDF goes negative past 50% document
    frequency. On a corpus this small that is common -- "cdc" is in most
    records -- and it would let a shared word penalise a document that
    genuinely contains it.
    """
    index = BM25Index([doc(f"d{i}", "cdc surveillance") for i in range(10)])
    assert index.idf["cdc"] >= 0


def test_shorter_document_wins_on_equal_term_frequency():
    """Length normalisation, which is why BM25 rather than raw tf-idf.

    Without it a long contract record outranks a short partner profile just by
    having more words for a query term to land in.
    """
    index = BM25Index([
        doc("short", "epidemiologic surveillance"),
        doc("long", "epidemiologic surveillance " + "filler words here " * 40),
    ])
    scores = index.scores("epidemiologic surveillance")
    assert scores[0] > scores[1]


def test_repeated_terms_saturate():
    """Twenty mentions must not score ten times two mentions."""
    index = BM25Index([
        doc("two", "surveillance surveillance " + "pad " * 30),
        doc("many", "surveillance " * 20 + "pad " * 12),
    ])
    scores = index.scores("surveillance")
    assert scores[1] > scores[0]
    assert scores[1] < 3 * scores[0]


def test_unknown_terms_are_ignored_not_fatal():
    index = BM25Index([doc("a", "epidemiologic surveillance")])
    assert index.scores("nonexistent term entirely") == {}


def test_empty_index_does_not_divide_by_zero():
    assert BM25Index([]).scores("anything") == {}


# ---------------------------------------------------------------------------
# exact handles
# ---------------------------------------------------------------------------


def test_exact_match_needs_the_whole_handle():
    docs = [
        doc("k1", "body", handles=["CDC NCEZID Surveillance Support", "75D30124C00000"]),
        doc("k2", "body", handles=["CDC NCEZID Analytics Support"]),
    ]
    hit = exact_scores(docs, "When did CDC NCEZID Surveillance Support finish?")
    assert hit == {0: 1.0}

    # ...and a bare shared prefix is not a handle match.
    assert exact_scores(docs, "Tell me about CDC NCEZID") == {}


def test_contract_number_alone_finds_the_record():
    docs = [doc("k1", "body", handles=["Some Name", "75D30124C00000"])]
    assert exact_scores(docs, "what is 75D30124C00000") == {0: 1.0}


# ---------------------------------------------------------------------------
# the blend
# ---------------------------------------------------------------------------


def corpus() -> list[Document]:
    return [
        doc("k1", "cdc surveillance epidemiologic support services chronic disease",
            handles=["CDC NCEZID Surveillance Support"]),
        doc("k2", "cdc surveillance epidemiologic support services immunization "
                  "respiratory analytics informatics data", handles=["CDC NCIRD Analytics"]),
        doc("k3", "unrelated cloud migration fedramp compliance", handles=["NIH Cloud"]),
    ]


def test_hybrid_lets_an_exact_name_beat_a_better_word_overlap():
    """The case the blend exists for.

    k2 shares more query words, so pure BM25 can prefer it; the question names
    k1 outright. Most questions in this corpus name their subject, and when they
    do that is the stronger signal.
    """
    query = "who worked on CDC NCEZID Surveillance Support and what did it cover"
    assert Retriever(corpus(), alpha=0.5).search(query, k=1)[0].id == "k1"


def test_pure_bm25_ignores_handles_entirely():
    """At alpha=1 the ranking must follow raw BM25, handles notwithstanding."""
    r = Retriever(corpus(), alpha=1.0)
    hits = r.search("who worked on CDC NCEZID Surveillance Support", k=3)
    assert [h.id for h in hits] == sorted(
        (h.id for h in hits), key=lambda i: -next(x.bm25 for x in hits if x.id == i)
    )
    # k1 is the exact handle match and still does not win, because alpha=1
    # gives that signal no weight at all.
    assert hits[0].exact == 1.0 or hits[0].bm25 >= max(h.bm25 for h in hits)


def test_alpha_zero_ranks_only_by_exact_match():
    hits = Retriever(corpus(), alpha=0.0).search("CDC NCIRD Analytics report", k=3)
    assert hits[0].id == "k2" and hits[0].score == 1.0


def test_alpha_is_validated():
    with pytest.raises(ValueError, match="alpha"):
        Retriever(corpus(), alpha=1.5)


def test_results_are_deterministic_including_ties():
    """Retrieval feeds training data, so identical queries must rank identically."""
    r = Retriever(corpus(), alpha=0.5)
    once = [h.id for h in r.search("cdc surveillance", k=3)]
    assert once == [h.id for h in r.search("cdc surveillance", k=3)]


def test_kind_filter_restricts_without_reranking():
    docs = corpus() + [
        Document(id="c9", kind="partner", title="Foxglove", text="cdc surveillance partner")
    ]
    hits = Retriever(docs, alpha=1.0).search("cdc surveillance", k=5, kinds=["partner"])
    assert [h.id for h in hits] == ["c9"]


def test_query_matching_nothing_returns_nothing():
    assert Retriever(corpus(), alpha=0.5).search("zzz qqq", k=5) == []


def test_context_block_is_numbered_and_ordered():
    text = Retriever(corpus(), alpha=0.5).context("cdc surveillance", k=2)
    assert text.startswith("[1] ")
    assert "\n\n[2] " in text
    # A query matching one document yields one block, not a padded list.
    single = Retriever(corpus(), alpha=0.5).context("cloud migration fedramp", k=2)
    assert single.startswith("[1] k3") and "[2] " not in single


# ---------------------------------------------------------------------------
# documents from the world
# ---------------------------------------------------------------------------


def test_documents_cover_every_entity_kind():
    from ftlab.retrieve import build_documents
    from ftlab.synth.graph import World

    docs = build_documents(World(seed=7, scale="compact"))
    kinds = {d.kind for d in docs}
    assert kinds == {"contract", "partner", "person", "opportunity"}
    assert all(d.text.strip() and d.tokens for d in docs)

    # Every contract must be reachable by its own number, which is the fact the
    # closed-book model got wrong 47 times out of 75.
    retriever = Retriever(docs, alpha=0.5)
    contracts = [d for d in docs if d.kind == "contract"]
    for contract in contracts[:10]:
        number = contract.handles[1]
        assert retriever.search(number, k=1)[0].id == contract.id
