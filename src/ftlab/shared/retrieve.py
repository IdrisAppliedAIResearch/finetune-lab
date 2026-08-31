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
    # What this record points at: the capability, agency and vehicle terms that
    # a second-stage partner search should run on. Empty for partner records,
    # which are the thing being searched for rather than a route to it.
    expansion: str = ""
    # Capability ids: what a partner can do, or what an opportunity needs. The
    # second retrieval hop intersects these rather than matching their prose,
    # because two partners can describe the same capability in different words
    # and the golden ranking is computed from the ids, not the wording.
    tags: frozenset[str] = frozenset()
    # Shorter rendering for when this record is a supporting candidate rather
    # than the subject of the question. A contract record costs 1290 characters
    # in full and 756 as a summary, and a teaming prompt carries a dozen of
    # them -- the difference decides whether the corpus fits in a 4K window.
    # Empty means the record has only one rendering.
    summary: str = ""
    tokens: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(f"{self.title} {self.text}")


def render_blocks(hits: list[Hit]) -> str:
    """Numbered records for a prompt: the first in full, the rest as summaries.

    The top hit is what the question is about, so it gets the whole record. The
    ones below it are there to be compared against, and a summary carries every
    field that comparison needs -- capabilities, agency, vehicle, period, value,
    CPARS -- while dropping the scope prose and personnel lists that only matter
    when the record is the subject.
    """
    blocks = []
    for rank, hit in enumerate(hits, start=1):
        doc = hit.document
        body = doc.text if rank == 1 or not doc.summary else doc.summary
        blocks.append(f"[{rank}] {doc.title}\n{body}")
    return "\n\n".join(blocks)


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


