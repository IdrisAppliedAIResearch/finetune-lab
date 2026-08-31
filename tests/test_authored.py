"""Every factual claim in the hand-written examples, checked against the graph.

Writing training data by hand is the fix for template collapse, and it comes
with its own failure: prose drifts from the data. A hand-authored corpus that
nobody checks will contain a number that was true when it was written and is not
now, and a model trained on it learns the wrong thing with complete confidence.

So the division is: prose by hand, facts by machine. Each authored example
carries the claims it makes as structured assertions, and this checks them
against the same graph the corpus is built from. Change the ingest, re-pull the
data, or edit a sentence carelessly, and these fail.
"""

from __future__ import annotations

import collections

import pytest
from factcheck import check_facts

from ftlab.sft.authored import all_authored, authored_examples
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


def test_every_authored_claim_holds(graph):
    """The load-bearing test. A wrong number here teaches the model a wrong fact."""
    failures = []
    for question, _reasoning, _answer, facts in all_authored():
        failures += check_facts(graph, question, facts)

    assert not failures, "authored prose disagrees with the data:\n" + "\n".join(failures)


def test_every_authored_example_names_its_claims():
    """An example with no checkable facts is prose nobody is verifying."""
    unchecked = [q for q, _r, _a, facts in all_authored() if not facts]
    assert not unchecked, f"no facts declared for: {unchecked}"


# ---------------------------------------------------------------------------
# the thing they exist to fix
# ---------------------------------------------------------------------------


def shape_of(answer: str) -> str:
    if "\n1. " in answer:
        return "numbered"
    if "\n- " in answer:
        return "bulleted"
    return "prose"


def test_authored_answers_do_not_share_one_shape():
    """The failure being corrected, asserted directly.

    On the corpus that collapsed, an archetype predicted its format perfectly --
    prime_candidates was 150 numbered lists out of 150. If the hand-written set
    is uniform too, it is a template with extra steps.
    """
    shapes = collections.Counter(shape_of(a) for _q, _r, a, _f in all_authored())
    assert len(shapes) >= 2, f"all authored answers share one shape: {shapes}"
    assert max(shapes.values()) / len(all_authored()) < 0.8, (
        f"one shape dominates the authored set: {shapes}"
    )


def test_authored_openings_are_not_formulaic():
    """A fixed opener is a template the model will happily key on."""
    openers = collections.Counter(a.split()[0].lower() for _q, _r, a, _f in all_authored())
    assert max(openers.values()) <= max(2, len(all_authored()) // 4), (
        f"answers start the same way too often: {openers.most_common(3)}"
    )


def test_some_authored_answers_decline_or_correct_the_question():
    """The behaviour the first fine-tune had no example of.

    Trained only on answerable questions, it answered an unanswerable one in
    whatever template fit. Refusals and premise-corrections have to be in the
    data for the model to learn they are allowed.
    """
    text = " ".join(a.lower() for _q, _r, a, _f in all_authored())
    assert "can't answer" in text or "cannot answer" in text
    assert "false" in text or "not really" in text or "overstates" in text


def test_authored_examples_render_as_corpus_rows():
    rows = authored_examples(repeat=2)
    assert len(rows) == 2 * len(all_authored())
    for row in rows:
        assert row["question"] and row["answer"] and row["reasoning"]
        assert row["meta"]["authored"] is True
        assert row["context"] == ""  # closed-book by design
