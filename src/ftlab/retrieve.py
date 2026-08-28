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
                summary=render.contract_record(world, contract, full=False),
                handles=[contract.name, contract.number],
                expansion=" ".join(
                    [*contract.capability_names(), contract.agency, contract.vehicle]
                ),
                tags=frozenset(contract.capabilities),
            )
        )

    for company in world.companies.values():
        docs.append(
            Document(
                id=company.id,
                kind="partner",
                title=company.name,
                text=render.company_profile(world, company),
                summary=render.company_profile(world, company, full=False),
                handles=[company.name],
                tags=frozenset(company.capabilities),
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
                expansion=" ".join(
                    [
                        *opportunity.capability_names(),
                        opportunity.agency,
                        opportunity.vehicle,
                    ]
                ),
                tags=frozenset(opportunity.required_capabilities),
            )
        )

    return docs


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

    def search_expanded(self, query: str, k: int = 4, partners: int = 8) -> list[Hit]:
        """Anchor on the entity the question names, then fetch what it points to.

        One-shot lexical retrieval is the wrong shape for a teaming question.
        "Which partners should we team with for CDC NCEZID Analytics?" shares
        its vocabulary with the *opportunity*, not with any partner, so a single
        BM25 pass fills every slot with opportunities and near-miss contracts.
        Measured on the recommendation questions in this corpus, one-shot
        retrieval put a golden partner in the context 0% of the time at k=8:
        the model would have been asked to rank candidates it could not see, and
        the only way to answer would have been to fall back on the weights this
        whole architecture exists to stop relying on.

        So the second hop follows the graph rather than the wording. Find the
        anchor, read what it requires, and search the partner records for that.
        The expansion terms are the anchor's capabilities, agency and vehicle --
        the same signals the golden ranking is computed from.
        """
        anchors = [h for h in self.search(query, k=k) if h.document.kind != "partner"]
        needed: frozenset[str] = frozenset().union(
            *(h.document.tags for h in anchors)
        ) if anchors else frozenset()

        if not needed:
            expansion = " ".join(h.document.expansion for h in anchors if h.document.expansion)
            if not expansion:
                return self.search(query, k=k + partners)
            return anchors + self.search(f"{expansion} {query}", k=partners, kinds=["partner"])

        # Structured first, lexical second. Capability overlap decides who is a
        # candidate at all -- that is a graph query, and matching capability
        # prose instead put a golden partner in the context only ~30% of the
        # time. BM25 against the anchor's terms then orders within an overlap
        # band, so the block is not an arbitrary slice of a big tie.
        #
        # This deliberately stops at the candidate set. Which of them is
        # actually the right pick, and which is the one that looks right and
        # fails a decisive criterion, is the judgement the model is being
        # trained to make, and doing it here would leave nothing to learn.
        expansion = " ".join(h.document.expansion for h in anchors if h.document.expansion)
        lexical = self.bm25.scores(f"{expansion} {query}")

        ranked = sorted(
            (
                (len(doc.tags & needed), lexical.get(i, 0.0), doc)
                for i, doc in enumerate(self.documents)
                if doc.kind == "partner" and doc.tags & needed
            ),
            key=lambda row: (-row[0], -row[1], row[2].id),
        )
        found = [
            Hit(document=doc, score=float(overlap), bm25=bm, exact=0.0)
            for overlap, bm, doc in ranked[:partners]
        ]
        return anchors + found

    def context(
        self,
        query: str,
        k: int = 5,
        kinds: list[str] | None = None,
        *,
        expand: bool = False,
        partners: int = 8,
    ) -> str:
        """The retrieved records as one block, ready to drop into a prompt."""
        hits = (
            self.search_expanded(query, k=k, partners=partners)
            if expand
            else self.search(query, k, kinds)
        )
        blocks = []
        for rank, hit in enumerate(hits, start=1):
            blocks.append(f"[{rank}] {hit.document.title}\n{hit.document.text}")
        return "\n\n".join(blocks)


def load_retriever(data_dir: str | Any = "data/processed", alpha: float = 0.5) -> Retriever:
    from .grade import load_world

    return Retriever.from_world(load_world(data_dir), alpha)


def partner_candidates(
    world: World,
    retriever: Retriever,
    opportunity: Any,
    *,
    by_capability: int = 6,
    by_collaboration: int = 5,
    by_agency: int = 3,
) -> list[str]:
    """The partner slate a teaming prompt shows, drawn from three signals.

    Capability overlap alone is the obvious choice and it quietly destroys the
    experiment. A hard negative in this corpus is a partner with high surface
    appeal -- prior joint work, agency familiarity -- who fails the decisive
    criterion, which is usually the missing capability. Retrieving on capability
    therefore selects precisely the candidates that are *not* traps: measured,
    it cut hard negatives from 2.73 per opportunity on the full roster to 0.64,
    with 76% of opportunities left with none at all. The model would have been
    trained to choose among candidates who were all defensible, and the one
    number the demo rests on could not have been measured.

    So the slate is composed from the same three signals that make a trap
    tempting in the first place, which are also the three an analyst would
    actually search on:

    * **capability** -- who can do the work (the genuinely qualified)
    * **collaboration** -- who we have teamed with before (appealing, and often
      appealing for reasons that have nothing to do with this requirement)
    * **agency** -- who already works this customer

    That restores traps to 1.62 per opportunity with 19% left empty. Short of
    the whole-roster figure, which no prompt can hold, and enough to measure.
    """
    needed = frozenset(opportunity.required_capabilities)
    lexical = retriever.bm25.scores(
        f"{opportunity.expansion_query()} {opportunity.name}"
    )
    position = {doc.id: i for i, doc in enumerate(retriever.documents)}

    def affinity(company: Any) -> float:
        return lexical.get(position.get(company.id, -1), 0.0)

    chosen: list[str] = []
    taken: set[str] = set()

    def take(pool: list[Any], limit: int) -> None:
        for company in pool[:limit]:
            if company.id not in taken:
                taken.add(company.id)
                chosen.append(company.id)

    take(
        sorted(
            (c for c in world.partners if set(c.capabilities) & needed),
            key=lambda c: (-len(set(c.capabilities) & needed), -affinity(c), c.id),
        ),
        by_capability,
    )
    take(
        sorted(
            (c for c in world.partners if c.contracts_with_us and c.id not in taken),
            key=lambda c: (-len(c.contracts_with_us), -affinity(c), c.id),
        ),
        by_collaboration,
    )
    take(
        sorted(
            (
                c
                for c in world.partners
                if opportunity.agency_id in (c.agencies_served or [])
                and c.id not in taken
            ),
            key=lambda c: (-affinity(c), c.id),
        ),
        by_agency,
    )
    return chosen


def plan_opportunity_context(
    world: World,
    *,
    contracts: int = 2,
    partners: int = 14,
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Retrieve once per opportunity, with an explicit mix of record types.

    Composed rather than taken from whatever a lexical search returns, because
    the mix decides two things at once and they pull against each other. The
    slate has to be rich enough that choosing within it is a real judgement --
    and it has to fit a fixed context window, which every extra record eats.

    Keyed by opportunity id so every question about the same pursuit sees the
    same records. Two paraphrases carrying different context would disagree
    about the right answer while appearing to ask the same thing.
    """
    from .synth.items import Retrieved

    retriever = Retriever.from_world(world, alpha=alpha)
    by_id = {doc.id: doc for doc in retriever.documents}

    # Split the partner budget across the three signals, keeping capability the
    # largest share: the qualified candidates are what the answer is built from,
    # the other two are what makes the choice non-obvious.
    capability = max(1, round(partners * 6 / 14))
    collaboration = max(1, round(partners * 5 / 14))
    agency = max(0, partners - capability - collaboration)

    plan: dict[str, Any] = {}
    for opportunity in world.opportunities.values():
        anchor = [by_id[opportunity.id]]
        ours = [
            hit.document
            for hit in retriever.search(
                f"{opportunity.expansion_query()} {opportunity.name}",
                k=contracts,
                kinds=["contract"],
            )
        ]
        slate = [
            by_id[pid]
            for pid in partner_candidates(
                world,
                retriever,
                opportunity,
                by_capability=capability,
                by_collaboration=collaboration,
                by_agency=agency,
            )
        ]

        documents = [*anchor, *ours, *slate]
        plan[opportunity.id] = Retrieved(
            context=render_blocks(
                [Hit(document=d, score=0.0, bm25=0.0, exact=0.0) for d in documents]
            ),
            partner_ids={d.id for d in documents if d.kind == "partner"},
            contract_ids={d.id for d in documents if d.kind == "contract"},
        )
    return plan
