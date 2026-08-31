"""The record-reading profiles, checked against the graph and the context.

The load-bearing test here is ``test_every_named_company_is_in_the_context``.
These examples exist to teach reading a supplied record; if an answer describes
a company whose record was never retrieved, it is teaching the model to recall
instead, which is the one thing the closed-book result says it cannot do.
"""

from __future__ import annotations

import collections

import pytest
from factcheck import check_facts, grounded_in

from ftlab.sft.authored_profiles import NAICS_LABELS, PROFILES, profile_examples
from ftlab.shared.records import build_index
from ftlab.shared.graph import build_graph
from ftlab.shared.ingest import load_slice

DATA = "data/real"


@pytest.fixture(scope="module")
def graph():
    try:
        slice_ = load_slice(DATA)
    except FileNotFoundError:
        pytest.skip(f"no cached slice at {DATA}; run 'ftlab real-ingest' first")
    return build_graph(slice_.prime_awards, slice_.subawards)


@pytest.fixture(scope="module")
def rows(graph):
    return profile_examples(graph, build_index(graph))


def test_every_claim_holds(graph):
    failures = []
    for item in PROFILES:
        failures += check_facts(graph, item.question, item.facts)
    assert not failures, "profile prose disagrees with the data:\n" + "\n".join(failures)


def test_every_number_cited_is_in_the_context(rows):
    """Every figure an answer states must appear in the records it was given.

    This replaces an earlier restriction on which *kinds* of fact a profile
    could declare. Widening the record to carry per-partner counts made most of
    those kinds derivable, so the honest check is no longer "which field did you
    use" but "is the number you printed actually on the page". The grader reads
    company names and not numbers, so nothing else catches a model being taught
    to invent them.
    """
    import re

    failures = []
    for item, row in zip(PROFILES, rows, strict=True):
        present = set(re.findall(r"\d+", row["context"]))
        for number in re.findall(r"\d+", row["answer"]):
            if number not in present:
                failures.append(f"{item.question[:44]!r}: cites {number}, not in its context")
    assert not failures, "\n".join(failures)


def test_every_profile_declares_claims():
    unchecked = [p.question[:48] for p in PROFILES if not p.facts]
    assert not unchecked, f"no facts declared for: {unchecked}"


def test_every_named_company_is_in_the_context(rows):
    """A profile of a record that was not supplied is teaching recall.

    Context is chosen by BM25 over the question text, so which records arrive is
    not something the author controls -- only a test can confirm that the
    companies an answer characterises are ones the model could actually read.
    """
    failures = []
    for item, row in zip(PROFILES, rows, strict=True):
        for name in item.names_cited:
            if not grounded_in(row["context"], name, item.subject):
                failures.append(f"{item.question[:44]!r}: {name} is not in the context")
    assert not failures, "\n".join(failures)


def test_answers_only_name_companies_they_declared(rows):
    """The declaration has to be complete, or the context check above is hollow."""
    from ftlab.shared.grade import find_companies, known_companies

    known = known_companies(DATA)
    failures = []
    for item, row in zip(PROFILES, rows, strict=True):
        declared = set(item.names_cited) | {item.subject}
        for name in find_companies(row["answer"], known):
            if name not in declared:
                failures.append(f"{item.question[:44]!r}: names undeclared {name}")
    assert not failures, "\n".join(failures)


def test_cited_naics_codes_have_labels_and_belong_to_the_company(graph):
    """Every code an answer spells out is labelled here and real for that firm.

    The ``naics`` facts do the second half; this catches a code used in prose
    with no entry in NAICS_LABELS, which would mean the label was written from
    memory rather than looked up.
    """
    import re

    failures = []
    for item in PROFILES:
        for code in set(re.findall(r"\b\d{6}\b", item.answer)):
            if code not in NAICS_LABELS:
                failures.append(f"{item.question[:44]!r}: {code} has no label")
            declared = {a[2] for a in item.facts if a[0] == "naics"}
            if code not in declared:
                failures.append(
                    f"{item.question[:44]!r}: {code} cited but not asserted against a record"
                )
    assert not failures, "\n".join(failures)


def test_naics_labels_are_used(graph):
    """An unused label is one nobody checked. Keep the map to what the prose cites."""
    import re

    cited = {c for p in PROFILES for c in re.findall(r"\b\d{6}\b", p.answer)}
    unused = set(NAICS_LABELS) - cited
    assert not unused, f"labels defined but never cited (so never checked): {sorted(unused)}"


# ---------------------------------------------------------------------------
# the balance the batch is supposed to have
# ---------------------------------------------------------------------------


def test_bench_profiles_are_the_majority():
    """Bench profiles are the half aimed at the measured task.

    The blind set draws its distractors from firms with at least two
    subcontracts taken, so the role-reading rule that firm profiles teach does
    not discriminate there. Firm profiles are grounding; the batch must not tip
    into being mostly grounding.
    """
    kinds = collections.Counter(p.kind for p in PROFILES)
    assert kinds["bench"] >= kinds["firm"], f"batch is weighted to grounding: {kinds}"


def test_bench_profiles_generalise_beyond_names():
    """A bench profile that only lists names is the portfolio template again.

    Each has to say what a candidate should look like, not merely who is on the
    roster, because the blind set is 71% pairs that appear nowhere in training.
    """
    weak = []
    for item in PROFILES:
        if item.kind != "bench":
            continue
        text = item.answer.lower()
        if not any(
            phrase in text
            for phrase in ("candidate", "a good ", "the question is", "what your pitch",
                           "route in", "worth targeting", "generalis", "transferable")
        ):
            weak.append(item.question[:44])
    assert not weak, f"bench profiles that only list the roster: {weak}"


def test_answers_do_not_share_one_shape():
    def shape_of(answer: str) -> str:
        if answer.startswith("1. ") or "\n1. " in answer:
            return "numbered"
        if answer.startswith("- ") or "\n- " in answer:
            return "bulleted"
        return "prose"

    shapes = collections.Counter(shape_of(p.answer) for p in PROFILES)
    assert max(shapes.values()) / len(PROFILES) < 0.85, f"one shape dominates: {shapes}"


def test_openings_are_not_formulaic():
    openers = collections.Counter(p.answer.split()[0].lower() for p in PROFILES)
    assert max(openers.values()) <= max(2, len(PROFILES) // 4), (
        f"answers start the same way too often: {openers.most_common(3)}"
    )


def test_some_profile_states_the_limits():
    """A corpus that never bounds its evidence teaches false confidence."""
    text = " ".join(p.answer.lower() for p in PROFILES)
    assert "cpars" in text
    assert "threshold" in text


def test_answers_stay_short(rows):
    long = [
        (p.question[:40], len(r["answer"]))
        for p, r in zip(PROFILES, rows, strict=True)
        if len(r["answer"]) > 1800
    ]
    assert not long, f"profiles longer than the behaviour being taught: {long}"


def test_render_as_corpus_rows(rows):
    assert len(rows) == len(PROFILES)
    for row in rows:
        assert row["question"] and row["answer"] and row["reasoning"]
        assert row["meta"]["authored"] is True
        assert row["meta"]["closed_book"] is False
        assert row["context"], "open-book by definition"
        assert not row["meta"]["tiers"], "a profile is not a ranking question"
