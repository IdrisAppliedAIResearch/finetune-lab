"""Keyword retrieval over the past performance library.

The closed-book run settled what this layer is for. Asked to hold the library in
its weights, the model reached 34-54x its random floor on relationships and
**29%** on exact facts, with contract numbers coming back 15 right and 47
confidently wrong. Recall tracked the entropy of the answer space -- 71% on
CPARS ratings, which have four possible values, and 5% on end years -- which is
the signature of a model that never stored the fact rather than one that cannot
find it. Asking the training questions back verbatim scored 35% against the
probes' 29%, confirming it: storage, not retrieval.

So facts come from here now, and the model keeps the part it was good at. This
module does one job: given a question, return the library records that could
answer it, in the order they should be read.

BM25 is the right instrument for that. The queries name their entities out loud
-- "best teaming partners for CDC NCEZID Public Health Analytics" -- so lexical
overlap is the signal, and BM25's length normalisation stops a long contract
record outranking a short partner profile purely by having more words in it.
Contract numbers survive tokenisation intact, which matters because they are
exactly the high-entropy strings the weights could not hold.

Implemented directly rather than pulled in: Okapi BM25 is about forty lines, and
a dependency here would be one more thing to pin for a demo that already carries
torch.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .synth import render
from .synth.graph import World

# BM25 parameters. k1 controls how fast term frequency saturates, b how strongly
# length normalisation applies. These are the standard defaults and there is no
# tuning set here that would justify moving them.
K1 = 1.5
B = 0.75

# Alphanumeric runs, so "75D30124C00000" stays one token instead of shattering
# into fragments that match every other contract number in the library.
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Document:
    """One library record, as retrievable text."""

    id: str
    kind: str  # contract | partner | person | opportunity
    title: str
    text: str
    # Strings that name this record exactly -- contract number, company name.
    # Used by the hybrid scorer, which is why they are kept apart from the body.
    handles: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(f"{self.title} {self.text}")


def build_documents(world: World) -> list[Document]:
    """Turn the world into retrievable records.

    One document per entity, not per fact. A contract record answers questions
    about its value, its period, its CPARS and its team all at once, and
    splitting it would scatter the relationships the model is meant to reason
    over across several retrieved chunks.

    The text is the same prose the corpus is written from, so what retrieval
    surfaces at inference looks like what the model saw in training.
    """
    docs: list[Document] = []

    for contract in world.contracts.values():
        docs.append(
            Document(
                id=contract.id,
                kind="contract",
                title=f"{contract.name} ({contract.number})",
                text=render.contract_record(world, contract, full=True),
                handles=[contract.name, contract.number],
            )
        )

    for company in world.companies.values():
        docs.append(
            Document(
                id=company.id,
                kind="partner",
                title=company.name,
                text=render.company_profile(world, company),
                handles=[company.name],
            )
        )

    for person in world.people.values():
        docs.append(
            Document(
                id=person.id,
                kind="person",
                title=person.name,
                text=render.person_profile(world, person),
                handles=[person.name],
            )
        )

    for opportunity in world.opportunities.values():
        docs.append(
            Document(
                id=opportunity.id,
                kind="opportunity",
                title=opportunity.name,
                text=render.opportunity_header(opportunity),
                handles=[opportunity.name],
            )
        )

    return docs


@dataclass
class Hit:
    document: Document
    score: float
    bm25: float
    exact: float

    @property
    def id(self) -> str:
        return self.document.id


class BM25Index:
    """Okapi BM25 over a fixed document set.

    Built once and queried many times: the postings, document lengths and IDF
    are all computed at construction, so a search is a walk over the postings of
    the query terms and nothing more.
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.n = len(documents)
        self.lengths = [len(d.tokens) for d in documents]
        self.avg_length = (sum(self.lengths) / self.n) if self.n else 0.0

        # term -> {doc index: term frequency}
        self.postings: dict[str, dict[int, int]] = {}
        for index, doc in enumerate(documents):
            for term, count in Counter(doc.tokens).items():
                self.postings.setdefault(term, {})[index] = count

        # Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative: the
        # raw form goes negative for terms in more than half the collection,
        # which on a corpus this small would let a common word subtract score
        # from a document that genuinely contains it.
        self.idf = {
            term: math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self.postings.items()
        }

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for index, freq in postings.items():
                norm = 1 - B + B * (self.lengths[index] / self.avg_length)
                out[index] = out.get(index, 0.0) + idf * (
                    freq * (K1 + 1) / (freq + K1 * norm)
                )
        return out


def exact_scores(documents: list[Document], query: str) -> dict[int, float]:
    """1.0 for each document whose own name appears verbatim in the query.

    BM25 sees a contract title as a bag of words, so a question naming one
    contract scores every contract sharing "CDC" and "Surveillance" with it.
    Most questions here name their subject outright, and when they do, that is a
    far stronger signal than term overlap -- this is what the hybrid weight
    exists to blend in.
    """
    lowered = query.lower()
    return {
        index: 1.0
        for index, doc in enumerate(documents)
        if any(handle and handle.lower() in lowered for handle in doc.handles)
    }


class Retriever:
    """BM25 plus exact-handle matching, blended.

    ``alpha`` is the BM25 share: 1.0 is pure BM25, 0.5 the even split. BM25
    scores are min-max normalised per query before blending, because they are
    unbounded and would otherwise swamp a 0-or-1 exact signal by an amount that
    depends on the query rather than on anything meaningful.
    """

    def __init__(self, documents: list[Document], alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.documents = documents
        self.alpha = alpha
        self.bm25 = BM25Index(documents)

    @classmethod
    def from_world(cls, world: World, alpha: float = 0.5) -> Retriever:
        return cls(build_documents(world), alpha)

    def search(self, query: str, k: int = 5, kinds: list[str] | None = None) -> list[Hit]:
        raw = self.bm25.scores(query)
        exact = exact_scores(self.documents, query)

        # Normalise BM25 into [0, 1] so the blend weight means what it says.
        top = max(raw.values(), default=0.0)
        low = min(raw.values(), default=0.0)
        span = top - low

        hits: list[Hit] = []
        for index in set(raw) | set(exact):
            doc = self.documents[index]
            if kinds and doc.kind not in kinds:
                continue
            bm = raw.get(index, 0.0)
            scaled = ((bm - low) / span) if span > 0 else (1.0 if bm > 0 else 0.0)
            ex = exact.get(index, 0.0)
            hits.append(
                Hit(
                    document=doc,
                    score=self.alpha * scaled + (1 - self.alpha) * ex,
                    bm25=bm,
                    exact=ex,
                )
            )

        # Ties broken by id so the same query always returns the same order --
        # a retrieval layer feeding training data cannot be non-deterministic.
        hits.sort(key=lambda h: (-h.score, h.document.id))
        return hits[:k]

    def context(self, query: str, k: int = 5, kinds: list[str] | None = None) -> str:
        """The retrieved records as one block, ready to drop into a prompt."""
        blocks = []
        for rank, hit in enumerate(self.search(query, k, kinds), start=1):
            blocks.append(f"[{rank}] {hit.document.title}\n{hit.document.text}")
        return "\n\n".join(blocks)


def load_retriever(data_dir: str | Any = "data/processed", alpha: float = 0.5) -> Retriever:
    from .grade import load_world

    return Retriever.from_world(load_world(data_dir), alpha)
