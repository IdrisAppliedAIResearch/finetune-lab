"""Guards on the masked-sub set.

The failures worth testing for here are all leaks, because every leak this
project has had looked like a good result first. Retrieval was once handed the
gold list and the untuned base model scored 1.000. The slate was once built
positives-first and position alone answered the question. Each was found after
a number had been reported, not before.

So the assertions below are about what the prompt must *not* contain, and they
are cheap enough to run on the whole set rather than a sample.
"""

from __future__ import annotations

import collections

import pytest

from ftlab.real.build import CONTEXT_K, build_index, context_for
from ftlab.real.graph import build_graph
from ftlab.real.ingest import load_slice
from ftlab.real.masked import (
    ARCHETYPE,
    MIN_RECORD,
    instances,
    rule_recovery,
    to_question,
)
from ftlab.real.questions import CANDIDATES


@pytest.fixture(scope="module")
def graph():
    slice_ = load_slice("data/real")
    return build_graph(slice_.prime_awards, slice_.subawards)


@pytest.fixture(scope="module")
def items(graph):
    return instances(graph)


def test_the_set_is_large_enough_to_measure_with(items):
    """The whole point of this set is escaping n=51."""
    assert len(items) >= 200
    assert len({i.prime for i in items}) >= 60


def test_every_true_sub_is_on_its_own_slate(items):
    """A question whose answer is not among the options is unanswerable."""
    for item in items:
        assert item.true_sub in item.slate


def test_slates_are_full_and_have_no_duplicates(items):
    for item in items:
        assert len(item.slate) == CANDIDATES
        assert len(set(item.slate)) == CANDIDATES


def test_the_prime_is_never_its_own_candidate(items):
    for item in items:
        assert item.prime not in item.slate


def test_no_sibling_true_sub_is_scored_as_a_distractor(graph, items):
    """Other firms the prime hired in the blind window must not appear.

    They are correct answers to the same question. Left on the slate they are
    graded as wrong, and an arm loses points for a defensible pick.
    """
    blind = collections.defaultdict(set)
    for row in graph.blind_subawards:
        blind[row["prime"]].add(row["sub"])

    for item in items:
        others = blind[item.prime] - {item.true_sub}
        assert not (set(item.slate) & others), item.prime


def test_the_answer_is_not_given_away_by_position(items):
    """Positives-first ordering has leaked into this repo twice.

    If the true sub sat at a fixed index, naming that index would score 1.000
    with no reasoning at all. Uniform-ish is enough; this only has to catch a
    slate that was never shuffled.
    """
    positions = collections.Counter(item.slate.index(item.true_sub) for item in items)
    most_common = positions.most_common(1)[0][1]
    assert most_common < 0.30 * len(items), positions


def test_candidates_all_have_records_worth_reading(graph, items):
    """A candidate with no record cannot be reasoned about in either direction."""
    for item in items:
        for name in item.slate:
            company = graph.companies[name]
            size = (
                len(company.as_sub)
                + len(company.as_prime)
                + len(company.prime_awards)
            )
            assert size >= MIN_RECORD, name


def test_no_company_profile_is_built_from_blind_teaming(graph):
    """The records must not know about the edges being predicted.

    ``build_graph`` builds profiles from training subawards only. If that ever
    changes, every record on every slate starts naming the answer.
    """
    train_rows = {id(r) for r in graph.train_subawards}
    for company in graph.companies.values():
        for row in (*company.as_sub, *company.as_prime):
            assert id(row) in train_rows


def test_the_question_text_never_names_the_answer_outside_the_slate(graph, items):
    """The ask must carry no hint -- only the prime, the component, the slate."""
    for item in items[:200]:
        question = to_question(graph, item)
        header = question.question.split("\n", 1)[0]
        assert item.true_sub not in header, item.true_sub


def test_context_does_not_reveal_the_pairing(graph, items):
    """Retrieval may return the true sub -- it must not report this teaming.

    The gold record legitimately appears (it is on the slate, and a candidate
    without a record is unanswerable). What it must never contain is the edge
    being predicted, which would make the task a reading exercise.
    """
    index = build_index(graph)
    for item in items[:60]:
        question = to_question(graph, item)
        context = context_for(graph, question, index, k=CONTEXT_K)
        block = next(
            (b for b in context.split("\n\n") if b.split("\n")[0].endswith(item.true_sub)),
            None,
        )
        if block is None:
            continue
        teamed = [ln for ln in block.split("\n") if ln.startswith("Teamed with")]
        if teamed and not item.is_new:
            # A prior pairing is legitimately on the record and is the groupby
            # signal every arm is entitled to.
            continue
        for line in teamed:
            assert item.prime not in line, (item.prime, item.true_sub, line)


def test_archetype_is_uniform_so_results_do_not_split(graph, items):
    assert {to_question(graph, i).archetype for i in items[:50]} == {ARCHETYPE}


def test_gold_is_exactly_one_company(graph, items):
    for item in items[:100]:
        assert to_question(graph, item).gold == [item.true_sub]


def test_tiers_cover_the_slate_so_the_grader_can_score_traps(graph, items):
    for item in items[:100]:
        question = to_question(graph, item)
        assert set(question.tiers) == set(item.slate)


def test_the_rule_baseline_leaves_headroom(graph, items):
    """If the groupby solved this, the set would measure nothing.

    Recorded as a range rather than a point so the test survives a reingest,
    while still failing loudly if the set ever becomes rule-solvable.
    """
    score = rule_recovery(graph, items)
    assert 0.05 < score < 0.55, score


def test_the_rule_baseline_cannot_score_on_new_pairings(graph, items):
    """By construction: the prime has never used these firms.

    This is the subset that separates reasoning from counting, and the reason
    the set is worth running at all.
    """
    new = [i for i in items if i.is_new]
    assert len(new) >= 75
    assert rule_recovery(graph, new) == 0.0


def test_building_is_deterministic(graph):
    a = instances(graph, seed=7)
    b = instances(graph, seed=7)
    assert [i.slate for i in a] == [i.slate for i in b]


def test_no_slate_name_contains_another(items):
    """The grader matches literal substrings, so overlapping names are unscoreable.

    ``KFORCE 3`` once matched the company ``FORCE 3`` and corrupted a reported
    figure. Here it would have hit 30 of 177 instances before the filter, so
    this is checked over the whole set rather than a sample.
    """
    for item in items:
        names = [*item.slate, item.prime]
        for a in names:
            for b in names:
                if a is not b:
                    assert not (a in b and a != b), (a, b)
