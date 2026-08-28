"""Invariants of the synthetic corpus.

These assert the properties the whole design rests on: that the world is
reproducible, that golden answers are entailed by the graph rather than
asserted, that every recommendation question actually contains a relevance
spectrum, and that the train/eval split does not leak.
"""

from __future__ import annotations

import pytest

from ftlab.synth.build import generate
from ftlab.synth.graph import US_CAPABILITIES, World
from ftlab.synth.scoring import (
    hard_negatives,
    rank_partners,
    rank_past_performance,
    score_teaming_partner,
    tier_spread,
)
from ftlab.synth.taxonomy import CAPABILITY_BY_ID, REAL_FIRM_BLOCKLIST


@pytest.fixture(scope="module")
def world() -> World:
    return World(seed=42, scale="compact")


@pytest.fixture(scope="module")
def corpus():
    return generate(seed=42, scale="compact")


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_yields_identical_world():
    """A corpus that shifts between runs makes a score change uninterpretable."""
    a, b = World(seed=7, scale="compact"), World(seed=7, scale="compact")
    assert [k.number for k in a.contracts.values()] == [
        k.number for k in b.contracts.values()
    ]
    assert [c.name for c in a.partners] == [c.name for c in b.partners]
    assert [o.name for o in a.opportunities.values()] == [
        o.name for o in b.opportunities.values()
    ]


def test_different_seed_yields_different_world():
    a, b = World(seed=7, scale="compact"), World(seed=8, scale="compact")
    assert [c.name for c in a.partners] != [c.name for c in b.partners]


def test_corpus_generation_is_deterministic():
    a = generate(seed=3, scale="compact")
    b = generate(seed=3, scale="compact")
    assert [i.question for i in a.train] == [i.question for i in b.train]
    assert [i.answer for i in a.eval] == [i.answer for i in b.eval]


# ---------------------------------------------------------------------------
# world integrity
# ---------------------------------------------------------------------------


def test_contract_names_and_numbers_are_unique(world: World):
    """Both are used as question keys, so a collision produces one question with
    two different correct answers."""
    names = [k.name for k in world.contracts.values()]
    numbers = [k.number for k in world.contracts.values()]
    assert len(set(names)) == len(names)
    assert len(set(numbers)) == len(numbers)


def test_no_generated_company_matches_a_real_firm(world: World):
    """The corpus attaches invented CPARS ratings to every name it mints."""
    for company in world.partners:
        tokens = {t.lower().strip(",.") for t in company.name.split()}
        assert not (tokens & REAL_FIRM_BLOCKLIST), company.name


def test_every_opportunity_has_a_real_capability_gap(world: World):
    """An opportunity we can fully self-perform makes the teaming question
    hollow, so the generator must never produce one."""
    for opportunity in world.opportunities.values():
        gap = world.capability_gap(opportunity)
        assert gap, opportunity.name
        assert all(c not in US_CAPABILITIES for c in gap)


def test_sub_and_prime_roles_are_consistent(world: World):
    for contract in world.contracts.values():
        if contract.our_role == "prime":
            assert contract.prime_id == "us"
            assert contract.value_ours <= contract.value_total
        else:
            assert contract.prime_id != "us"
            assert contract.value_ours < contract.value_total


def test_backfilled_collaboration_matches_the_contracts(world: World):
    """company.contracts_with_us must agree with a scan of the contracts."""
    for company in world.partners:
        derived = {k.id for k in world.contracts_with_company(company.id)}
        assert set(company.contracts_with_us) == derived, company.name


# ---------------------------------------------------------------------------
# the relevance spectrum
# ---------------------------------------------------------------------------


def test_every_opportunity_produces_a_spectrum(world: World):
    """The point of the dataset: not just right answers, but graded wrong ones."""
    for opportunity in world.opportunities.values():
        spread = tier_spread(rank_partners(world, opportunity, "teaming"))
        assert spread[4] + spread[3] >= 1, f"{opportunity.name} has no strong candidate"
        assert spread[2] >= 1, f"{opportunity.name} has no transferable middle"
        assert spread[0] >= 1, f"{opportunity.name} has no irrelevant tail"


def test_every_opportunity_has_hard_negatives(world: World):
    """A trap is the most instructive example in the set; without one the
    question only teaches ranking, not discrimination."""
    for opportunity in world.opportunities.values():
        traps = hard_negatives(rank_partners(world, opportunity, "teaming"))
        assert traps, opportunity.name
        for trap in traps:
            assert trap.disqualifier
            assert trap.surface_appeal >= 0.40


def test_gap_coverage_is_decisive_for_teaming(world: World):
    """A partner covering none of the gap must be disqualified however good
    their history with us -- that is the nuance the demo is testing."""
    opportunity = next(iter(world.opportunities.values()))
    gap = world.capability_gap(opportunity)
    for company in world.partners:
        assessment = score_teaming_partner(world, company, opportunity)
        covers_nothing = not (
            set(gap) & set(company.capabilities)
            or any(
                set(CAPABILITY_BY_ID[c].adjacent) & set(company.capabilities)
                for c in gap
            )
        )
        if covers_nothing:
            assert assessment.disqualifier is not None, company.name
            assert assessment.tier <= 1


def test_adjacency_earns_partial_credit_only(world: World):
    """Adjacent experience must never score as high as direct coverage."""
    opportunity = next(iter(world.opportunities.values()))
    gap = world.capability_gap(opportunity)
    direct, adjacent = [], []
    for company in world.partners:
        factor = score_teaming_partner(world, company, opportunity).factor("gap_fill")
        if set(gap) <= set(company.capabilities):
            direct.append(factor.score)
        elif not set(gap) & set(company.capabilities) and factor.score > 0:
            adjacent.append(factor.score)
    if direct and adjacent:
        assert max(adjacent) < max(direct)


def test_stale_past_performance_is_disqualified(world: World):
    """Recency is a gate, not a penalty: citing an out-of-window contract
    wastes a slot rather than scoring lower."""
    opportunity = next(iter(world.opportunities.values()))
    for assessment in rank_past_performance(world, opportunity):
        if assessment.contract.end_year < 2021:
            assert assessment.disqualifier is not None


# ---------------------------------------------------------------------------
# golden answers are entailed by the graph
# ---------------------------------------------------------------------------


def test_recommended_partners_actually_have_the_claimed_capability(world: World):
    """The core trustworthiness claim: a name in a recommendation is there
    because the graph says so, not because prose was generated around it."""
    for opportunity in world.opportunities.values():
        gap = set(world.capability_gap(opportunity))
        ranked = rank_partners(world, opportunity, "teaming")
        for assessment in ranked[:4]:
            company = assessment.company
            covers = gap & set(company.capabilities)
            adjacent = any(
                set(CAPABILITY_BY_ID[c].adjacent) & set(company.capabilities)
                for c in gap
            )
            assert covers or adjacent, (
                f"{company.name} ranked top-4 on {opportunity.name} without "
                "covering or nearing the gap"
            )


def test_cited_contracts_share_scope_with_the_opportunity(world: World):
    for opportunity in world.opportunities.values():
        eligible = [
            a for a in rank_past_performance(world, opportunity) if not a.disqualifier
        ]
        for assessment in eligible[:3]:
            overlap = set(assessment.contract.capabilities) & set(
                opportunity.required_capabilities
            )
            assert overlap, assessment.contract.name


# ---------------------------------------------------------------------------
# corpus and split
# ---------------------------------------------------------------------------


def test_no_question_appears_in_both_train_and_eval(corpus):
    train_q = {i.question for i in corpus.train}
    eval_q = {i.question for i in corpus.eval}
    assert not (train_q & eval_q)


def test_no_question_has_two_different_answers(corpus):
    """Identical question, different target = contradictory supervision that
    would be invisible in the loss curve."""
    seen: dict[str, str] = {}
    for item in [*corpus.train, *corpus.eval]:
        if item.question in seen:
            assert seen[item.question] == item.answer, item.question[:80]
        seen[item.question] = item.answer


def test_held_out_opportunities_never_appear_in_training(corpus):
    train_opps = {
        i.meta["opportunity"] for i in corpus.train if "opportunity" in i.meta
    }
    eval_opps = {i.meta["opportunity"] for i in corpus.eval if "opportunity" in i.meta}
    assert eval_opps
    assert not (train_opps & eval_opps)


def test_recall_layer_dominates_a_closed_book_corpus(corpus):
    """Closed-book means the library has to be taught, not just reasoned over."""
    layers = [i.layer for i in corpus.train]
    recall_share = sum(1 for x in layers if x in ("recall", "relational")) / len(layers)
    assert recall_share >= 0.45


def test_every_item_passes_dataset_validation(corpus):
    """The corpus must load through the same strict reader training uses."""
    from ftlab.data import QRAExample

    for item in [*corpus.train, *corpus.eval, *corpus.probes]:
        example = QRAExample.from_obj(item.to_record(), where="synth")
        assert example.question and example.answer


def test_probes_are_short_exact_values(corpus):
    """Probes exist to make parametric drift measurable, so they must be
    gradeable by string comparison rather than judgement."""
    assert corpus.probes
    for probe in corpus.probes:
        assert len(probe.answer) < 80
        assert probe.meta.get("exact") is True
