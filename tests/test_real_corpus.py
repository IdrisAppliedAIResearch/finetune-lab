"""Guards against a corpus that answers its own question.

This failure has now appeared four times in this project, in both the synthetic
and the real data, so it is not a property of either dataset. It is a property
of building a question and the evidence for it out of the same object: the
evidence ends up being the answer, and every arm scores well by reading rather
than reasoning. Each time it was caught by eye, late, after a run had already
been graded -- once by noticing that an untuned base model had scored a perfect
1.000.

The shapes it has taken, all of them now asserted against below:

* context assembled from the question's own gold list
* correct answers occupying the opening slots of an ordered list
* a slate presented for ranking with records for only some of its candidates
* a question restating the facts its own context supplies

The corpus is built from a small fixture rather than the cached USASpending
slice, so these run without network access and without the data directory.
"""

from __future__ import annotations

import statistics

import pytest

from ftlab.real.build import build_index, company_record, context_for
from ftlab.real.graph import build_graph
from ftlab.real.questions import generate, generate_blind

AGENCIES = [
    "Centers for Disease Control and Prevention",
    "National Institutes of Health",
    "Food and Drug Administration",
]


def fixture_slice() -> tuple[list[dict], list[dict]]:
    """A small world with enough structure to exercise every archetype."""
    primes = [f"PRIME {i:02d}" for i in range(6)]
    subs = [f"SUB {i:03d}" for i in range(60)]

    prime_awards = []
    subawards = []
    for p_index, prime in enumerate(primes):
        for a_index, agency in enumerate(AGENCIES):
            prime_awards.append(
                {
                    "recipient": prime,
                    "award_id": f"AW{p_index}{a_index}",
                    "agency": agency,
                    "naics": f"5415{a_index}0",
                    "naics_title": "RESEARCH AND DEVELOPMENT",
                    "psc": "R499",
                    "psc_title": "SUPPORT- PROFESSIONAL",
                    "start": "2024-01-01",
                    "end": "2026-01-01",
                    "amount": 1_000_000,
                    "description": "Public health analytics and surveillance support " * 3,
                }
            )

    # Training-period teaming: each prime works a distinct band of subs, with
    # repeats, so repeat_partners and team_composition have something to say.
    for p_index, prime in enumerate(primes):
        for offset in range(8):
            sub = subs[(p_index * 8 + offset) % len(subs)]
            for repeat in range(2 if offset % 3 == 0 else 1):
                subawards.append(
                    {
                        "prime": prime,
                        "sub": sub,
                        "prime_award_id": f"AW{p_index}",
                        "sub_award_id": f"S{p_index}{offset}{repeat}",
                        "agency": AGENCIES[offset % len(AGENCIES)],
                        "naics": f"5415{offset % 3}0",
                        "naics_title": "RESEARCH AND DEVELOPMENT",
                        "psc": "R499",
                        "psc_title": "SUPPORT- PROFESSIONAL",
                        "date": "2025-03-01",
                        "amount": 100_000,
                        "description": "Epidemiologic surveillance and data modernisation "
                        "support across multiple sites. " * 2,
                    }
                )

    # A prime that also subs for another prime. 69 companies in the real slice
    # play both roles, and a fixture where the two sets are disjoint would not
    # exercise the dual-role paths at all.
    for p_index in range(1, 4):
        subawards.append(
            {
                "prime": primes[0],
                "sub": primes[p_index],
                "prime_award_id": "AW0X",
                "sub_award_id": f"D{p_index}",
                "agency": AGENCIES[p_index % len(AGENCIES)],
                "naics": "541510",
                "naics_title": "RESEARCH AND DEVELOPMENT",
                "psc": "R499",
                "psc_title": "SUPPORT- PROFESSIONAL",
                "date": "2025-04-01",
                "amount": 250_000,
                "description": "Teaming as a subcontractor on analytic support. " * 3,
            }
        )

    # Blind period: partly new pairings, which is what makes the held-out
    # questions unanswerable by recall.
    for p_index, prime in enumerate(primes):
        for offset in range(6):
            sub = subs[(p_index * 8 + offset + 24) % len(subs)]
            subawards.append(
                {
                    "prime": prime,
                    "sub": sub,
                    "prime_award_id": f"AW{p_index}",
                    "sub_award_id": f"B{p_index}{offset}",
                    "agency": AGENCIES[offset % len(AGENCIES)],
                    "naics": f"5415{offset % 3}0",
                    "naics_title": "RESEARCH AND DEVELOPMENT",
                    "psc": "R499",
                    "psc_title": "SUPPORT- PROFESSIONAL",
                    "date": "2025-11-01",
                    "amount": 100_000,
                    "description": "Follow-on analytic support. " * 4,
                }
            )
    return prime_awards, subawards


@pytest.fixture(scope="module")
def graph():
    return build_graph(*fixture_slice())


@pytest.fixture(scope="module")
def index(graph):
    return build_index(graph)


def context_names(text: str) -> list[str]:
    return [block.split("\n")[0].split("] ")[-1] for block in text.split("\n\n[")]


# ---------------------------------------------------------------------------
# the leak
# ---------------------------------------------------------------------------


def test_context_is_not_built_from_the_answer(graph, index):
    """The failure that made a base model score 1.000 on a blind question.

    context_for once appended the question's own gold list, so eight of eight
    records supplied were the answer. Any model that echoed the record names
    back was perfect, and the benchmark measured transcription.
    """
    offenders = []
    for question in generate(graph):
        if question.tiers:
            continue  # slate questions are covered separately below
        names = context_names(context_for(graph, question, index))
        gold = set(question.gold)
        if not names:
            continue
        share = sum(1 for n in names if n in gold) / len(names)
        if share > 0.5:
            offenders.append((question.archetype, round(share, 2)))

    assert not offenders, (
        f"context is mostly the answer for {offenders[:5]} -- a model can score "
        "by reading the record names back"
    )


def test_blind_context_does_not_hand_over_the_answer(graph, index):
    """Same guard on the sealed set, where it actually bit."""
    for question in generate_blind(graph):
        names = context_names(context_for(graph, question, index))
        gold = set(question.gold)
        if not names:
            continue
        share = sum(1 for n in names if n in gold) / len(names)
        # A slate legitimately contains its own positives; what must not happen
        # is a context that is *only* the key.
        assert share < 0.8, f"{question.archetype}: {share:.0%} of context is gold"


def test_correct_answers_are_not_front_loaded(graph, index):
    """Ordering must carry no signal, in the records or in the question.

    Both have leaked in this project: blind slates once drew their positives
    from the alphabetical head of an alphabetically ordered list, and context
    was once assembled slate-first, which is positives-first.
    """
    positions: list[int] = []
    lengths: list[int] = []
    for question in generate_blind(graph):
        names = context_names(context_for(graph, question, index))
        gold = set(question.gold)
        positions += [i + 1 for i, n in enumerate(names) if n in gold]
        lengths.append(len(names))

    if not positions:
        pytest.skip("no gold appears in context for this fixture")
    uniform = (statistics.mean(lengths) + 1) / 2
    assert abs(statistics.mean(positions) - uniform) < uniform * 0.45, (
        f"gold sits at position {statistics.mean(positions):.1f} of "
        f"{statistics.mean(lengths):.0f}; uniform would be {uniform:.1f}"
    )


def test_every_candidate_on_a_slate_has_a_record(graph, index):
    """Ranking a candidate the prompt says nothing about is not a fair ask.

    Context once showed 8 records for a 12-candidate slate, so a third of what
    the model was told to rank came with no evidence at all.
    """
    for question in generate(graph):
        if not question.tiers:
            continue
        names = set(context_names(context_for(graph, question, index)))
        missing = [c for c in question.tiers if c not in names]
        assert not missing, (
            f"{question.archetype}: {len(missing)} of {len(question.tiers)} "
            f"candidates have no record"
        )


def test_questions_do_not_restate_their_own_context(graph, index):
    """A question carrying the evidence can be answered without reading it.

    The synthetic corpus repeated the whole opportunity brief in the question
    while the same text sat in record [1]; a model could score well having
    ignored retrieval entirely.
    """
    for question in generate(graph):
        text = context_for(graph, question, index)
        first = text.split("\n\n[")[0]
        body = "\n".join(first.split("\n")[1:]).strip()
        if len(body) < 80:
            continue
        assert body not in question.question, (
            f"{question.archetype}: the question repeats record [1] verbatim"
        )


# ---------------------------------------------------------------------------
# the corpus is still worth training on
# ---------------------------------------------------------------------------


def test_slate_questions_carry_hard_negatives(graph):
    """Without traps there is nothing to discriminate, only to look up."""
    ranked = [q for q in generate(graph) if q.tiers]
    assert ranked
    with_traps = [q for q in ranked if any(t <= 1 for t in q.tiers.values())]
    assert len(with_traps) / len(ranked) > 0.5


def test_blind_answers_are_not_reachable_from_training_alone(graph):
    """The sealed questions must need more than recall of the training graph."""
    blind = generate_blind(graph)
    assert blind
    fresh = [q for q in blind if q.meta.get("unseen_in_training")]
    assert fresh, "no blind question involves a pairing absent from training"


def test_company_records_carry_what_the_ranking_turns_on(graph):
    """Agencies, teaming history and scope, or the questions are unanswerable."""
    name = next(n for n, c in graph.companies.items() if c.as_sub and c.as_prime)
    record = company_record(graph, name)
    for field in ("Agencies:", "NAICS:", "Teamed with"):
        assert field in record, f"{field} missing from a company record"


# ---------------------------------------------------------------------------
# template collapse
# ---------------------------------------------------------------------------


def test_collapse_metric_catches_a_model_answering_a_different_question():
    """The failure the ranking metrics could not see.

    The first fine-tune scored like an ordinary underperformer -- precision a
    little low, recall a little low -- while actually forcing every blind answer
    into one of seven trained shapes. Asked which companies were new to NIH it
    produced a list of primes to approach. Nothing in precision or recall says
    that; this does.
    """
    from ftlab.real.grade import collapse_report, template_used

    items = [
        {"meta": {"archetype": "blind_new_entrant"}},
        {"meta": {"archetype": "sub_candidates"}},
        {"meta": {"archetype": "blind_next_team"}},
    ]
    generated = [
        # verbatim shape of the real failure
        "Primes to approach for NIH work, by subcontracting volume:\n1. RTI",
        # a template, but the right one for the question
        "Sub candidates for X on CDC work:\n1. A\n2. B",
        # no template at all
        "Looking at each in turn, three of them have prior CDC awards.",
    ]
    report = collapse_report(items, generated)
    assert report["answers_in_a_template"] == pytest.approx(2 / 3)
    # only the first is a template that does not match its question
    assert report["answers_in_the_wrong_template"] == pytest.approx(1 / 3)

    assert template_used(generated[0]) == "prime_candidates"
    assert template_used(generated[2]) is None


def test_collapse_metric_is_silent_on_an_untemplated_model():
    """A base model writes none of these openings; the floor must be zero."""
    from ftlab.real.grade import collapse_report

    items = [{"meta": {"archetype": "blind_next_team"}} for _ in range(3)]
    generated = [
        "I would start with the two that have prior awards with this prime.",
        "None of these look like a fit for that customer.",
        "Hard to say from the records provided.",
    ]
    report = collapse_report(items, generated)
    assert report["answers_in_a_template"] == 0.0


def test_a_verbose_model_is_graded_on_its_conclusion_not_its_notes():
    """Measured failure: the base model was scored on its analysis order.

    Asked to choose from a slate it enumerated all twelve candidates in the
    order given, and the grader read the first four names it mentioned. That is
    approximately a random draw, which is approximately the score it got -- so
    the number said nothing about the model.
    """
    from ftlab.real.grade import conclusion_of, find_companies

    known = ["ALPHA CORP", "BETA LLC", "GAMMA INC", "DELTA GROUP"]
    verbose = (
        "Let me work through each candidate.\n"
        "1. ALPHA CORP: record [1], no prior awards with this prime.\n"
        "2. BETA LLC: record [2], no relationship either.\n"
        "3. GAMMA INC: record [3], two prior joint awards.\n"
        "4. DELTA GROUP: record [4], one prior joint award.\n\n"
        "Most likely: GAMMA INC and DELTA GROUP."
    )
    picked = find_companies(conclusion_of(verbose), known)[:4]
    assert picked == ["GAMMA INC", "DELTA GROUP"], picked


def test_an_answer_with_no_conclusion_is_graded_on_what_it_wrote():
    """A model that never concludes is not excused for producing nothing."""
    from ftlab.real.grade import conclusion_of

    notes = "1. ALPHA CORP: record [1].\n2. BETA LLC: record [2]."
    assert conclusion_of(notes) == notes


def test_truncation_is_reported_rather_than_silently_scored():
    """16 of 18 base-model answers hit the token budget in the first run.

    Every arm was being scored on answers it had not finished writing, and
    nothing in the report said so.
    """
    from ftlab.real.grade import looks_truncated

    assert looks_truncated("...and the third candidate, AMAZON WEB SERVICES: **Teamed")
    assert not looks_truncated("Most likely: GAMMA INC and DELTA GROUP.")
    assert not looks_truncated("The records do not settle it.")
