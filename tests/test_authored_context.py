"""The hand-written open-book examples, checked against the graph and the context.

These carry two kinds of claim the closed-book set does not, and both are the
sort that rot silently:

* a **declared tier** per slate candidate. The prose says "this one is a hard
  negative"; ``tier_for`` is the arbiter. If a re-ingest turns a tier-1 trap
  into a tier-4 partner, an answer that rejects it becomes a wrong answer with
  no other symptom.
* a claim that some company's record is **in the supplied context but not on
  the slate**. That lesson only exists if the record is really there, and the
  record is chosen by BM25 rather than by hand, so nothing but a test can
  confirm it.
"""

from __future__ import annotations

import collections

import pytest
from factcheck import check_facts, grounded_in, records_in

from ftlab.sft.authored_context import all_open_book, context_examples
from ftlab.shared.records import build_index
from ftlab.shared.graph import build_graph
from ftlab.shared.ingest import load_slice
from ftlab.shared.questions import tier_for

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
    return context_examples(graph, build_index(graph))


def test_every_claim_holds(graph):
    failures = []
    for example in all_open_book():
        failures += check_facts(graph, example.ask, example.facts)
    assert not failures, "authored prose disagrees with the data:\n" + "\n".join(failures)


def test_declared_tiers_match_the_graph(graph):
    """The prose calls a candidate a trap; the graph has to agree."""
    failures = []
    for example in all_open_book():
        naics = set(graph.companies[example.prime].naics)
        for name, declared in example.tiers.items():
            if name not in graph.companies:
                failures.append(f"{example.ask[:40]!r}: {name} is not in the graph")
                continue
            actual = tier_for(graph, name, example.prime, example.agency, naics)
            if actual != declared:
                failures.append(
                    f"{example.ask[:40]!r}: {name} declared tier {declared}, graph says {actual}"
                )
    assert not failures, "declared tiers disagree with the data:\n" + "\n".join(failures)


def test_slate_and_tiers_cover_the_same_names():
    for example in all_open_book():
        assert set(example.slate) == set(example.tiers), (
            f"{example.ask[:40]!r}: slate and tiers disagree"
        )


def test_gold_is_on_the_slate_and_earned_it(graph):
    """A recommended name has to be offered, and has to have a relationship."""
    failures = []
    for example in all_open_book():
        for name in example.gold:
            if name not in example.tiers:
                failures.append(f"{example.ask[:40]!r}: recommends off-slate {name}")
            elif example.tiers[name] < 3:
                failures.append(
                    f"{example.ask[:40]!r}: recommends {name} at tier {example.tiers[name]}"
                )
    assert not failures, "\n".join(failures)


def test_every_slate_carries_a_hard_negative():
    """Without a tier-1 trap an example teaches nothing the easy heuristic misses."""
    thin = [e.ask[:40] for e in all_open_book() if 1 not in e.tiers.values()]
    assert not thin, f"no hard negative on the slate for: {thin}"


# ---------------------------------------------------------------------------
# the failures these exist to fix
# ---------------------------------------------------------------------------


def test_answers_name_only_slate_companies(graph, rows):
    """Off-slate naming was 0.298 of arm A's answers. Not one example may model it.

    Checked against every company in the graph rather than against the slate,
    so an answer that reaches for an outside name is caught even when the name
    looks harmless.
    """
    from ftlab.shared.grade import find_companies, known_companies

    known = known_companies(DATA)
    failures = []
    for example, row in zip(all_open_book(), rows, strict=True):
        subject = {example.prime}
        allowed = set(example.slate) | subject | set(example.off_slate_named)
        for name in find_companies(row["answer"], known):
            if name not in allowed:
                failures.append(f"{example.ask[:40]!r}: names {name}, not on the slate")
    assert not failures, "\n".join(failures)


def test_named_off_slate_records_are_really_in_the_context(rows):
    """The 'these records are not options' lesson needs the records to be there."""
    failures = []
    for example, row in zip(all_open_book(), rows, strict=True):
        for name in example.off_slate_named:
            if name in example.slate:
                failures.append(f"{example.ask[:40]!r}: {name} is on the slate after all")
            elif not grounded_in(row["context"], name, example.prime):
                failures.append(f"{example.ask[:40]!r}: {name} is not in the context")
    assert not failures, "\n".join(failures)


def test_contexts_contain_distractors(rows):
    """Every context must offer names the slate does not, or off-slate discipline
    is untrainable: a prompt whose records exactly match its options never asks
    the model to tell the two apart."""
    from ftlab.shared.grade import find_companies, known_companies

    known = known_companies(DATA)
    thin = []
    for example, row in zip(all_open_book(), rows, strict=True):
        extra = records_in(row["context"]) - set(example.slate) - {example.prime}
        if len(extra) < 2:
            thin.append(f"{example.ask[:40]!r}: only {len(extra)} off-slate records")
    assert not thin, "\n".join(thin)


def test_most_answers_state_what_they_turned_down():
    """Trap rejection was 0.169, and 18% of open-book training answers modelled it."""
    from ftlab.shared.grade import REJECT_MARKERS

    rejecting = [
        e for e in all_open_book() if any(marker in e.answer for marker in REJECT_MARKERS)
    ]
    assert len(rejecting) >= len(all_open_book()) // 3, (
        f"only {len(rejecting)} of {len(all_open_book())} name what they rejected"
    )


# ---------------------------------------------------------------------------
# and the failure they must not reintroduce
# ---------------------------------------------------------------------------


def shape_of(answer: str) -> str:
    """Which of the three forms an answer takes.

    The leading-position check matters: an answer that opens straight into
    "1. NAME" is a numbered answer, and a version of this that only looked for
    a newline first classified six of these as prose, then reported that prose
    dominated the set.
    """
    if answer.startswith("1. ") or "\n1. " in answer:
        return "numbered"
    if answer.startswith("- ") or "\n- " in answer:
        return "bulleted"
    return "prose"


def test_answers_do_not_share_one_shape():
    shapes = collections.Counter(shape_of(e.answer) for e in all_open_book())
    assert len(shapes) >= 3, f"open-book answers cover only {sorted(shapes)}"
    assert max(shapes.values()) / len(all_open_book()) < 0.8, f"one shape dominates: {shapes}"


def test_openings_are_not_formulaic():
    openers = collections.Counter(e.answer.split()[0].lower() for e in all_open_book())
    assert max(openers.values()) <= max(2, len(all_open_book()) // 4), (
        f"answers start the same way too often: {openers.most_common(3)}"
    )


def test_some_answers_decline_or_correct_the_premise():
    text = " ".join(e.answer.lower() for e in all_open_book())
    assert "none of the" in text
    assert "assumes a bench that isn't there" in text or "you are not missing" in text


def test_answers_stay_short(rows):
    """Arm A's 1,015 characters against arm C's 8,353 is the result worth keeping."""
    long = [
        (e.ask[:40], len(r["answer"]))
        for e, r in zip(all_open_book(), rows, strict=True)
        if len(r["answer"]) > 1600
    ]
    assert not long, f"authored answers longer than the behaviour being taught: {long}"


def test_render_as_corpus_rows(graph, rows):
    assert len(rows) == len(all_open_book())
    for row in rows:
        assert row["question"] and row["answer"] and row["reasoning"]
        assert row["meta"]["authored"] is True
        assert row["meta"]["closed_book"] is False
        assert row["context"], "open-book by definition"
        assert row["meta"]["tiers"], "a slate question carries its tiers"


def test_no_example_recommends_a_hard_negative(rows):
    """The strongest guard available: grade the authored answers with the real grader.

    Every other test here checks a claim about an example. This checks the thing
    the example will actually teach, by running ``grade_one`` over it exactly as
    the blind set is scored. It catches a failure mode the declarations cannot:
    an answer that rejects a trap in prose but puts the trap's name where the
    grader reads picks -- which is what happens when a decline names the four
    companies it is declining before it says it is declining them.
    """
    from ftlab.shared.grade import grade_one, known_companies

    known = known_companies(DATA)
    failures = []
    for example, row in zip(all_open_book(), rows, strict=True):
        graded = grade_one(row, row["answer"], known)
        traps = graded.scores.get("traps_recommended", 0.0)
        off = graded.scores.get("off_slate", 0.0)
        if traps:
            failures.append(
                f"{example.ask[:40]!r}: recommends {traps:.0f} hard negative(s) "
                f"-- picked {graded.notes['picked']}"
            )
        if off:
            failures.append(
                f"{example.ask[:40]!r}: {off:.0f} off-slate pick(s) "
                f"-- picked {graded.notes['picked']}"
            )
    assert not failures, "authored answers teach what they argue against:\n" + "\n".join(failures)


def test_authored_answers_beat_the_slate_on_tier(rows):
    """Mean tier of the authored picks, measured the way arm A is measured.

    Arm A scored 1.780 and the floor 1.477. Hand-written answers that do not
    clear both by a wide margin are not worth training on.
    """
    from ftlab.shared.grade import grade_one, known_companies

    known = known_companies(DATA)
    # Declines are excluded on purpose. Since a non-answer scores zero rather
    # than dropping out of the mean, an example that correctly recommends
    # nobody now reports mean_tier 0.0, and reading that as a bad pick would
    # punish exactly the behaviour two of these were written to teach.
    scored = [
        grade_one(row, row["answer"], known).scores.get("mean_tier")
        for example, row in zip(all_open_book(), rows, strict=True)
        if example.gold
    ]
    scored = [t for t in scored if t is not None]
    assert scored, "no authored answer produced a gradeable pick"
    assert min(scored) >= 3.0, f"an authored answer picks below tier 3: {min(scored)}"


def test_every_number_cited_is_in_the_context(rows):
    """Every figure a slate answer states must be on the page it was handed.

    All 25 of these cite award counts -- "HP - 27 reported Perspecta awards at
    CMS" -- and until ``company_record`` was widened not one of those numbers
    existed anywhere in retrieval. They were teaching the model to produce
    confident figures from nothing, and the grader could not see it because it
    reads company names and not numbers. This is what stops that returning.
    """
    import re

    failures = []
    for example, row in zip(all_open_book(), rows, strict=True):
        present = set(re.findall(r"\d+", row["context"]))
        for number in re.findall(r"\d+", row["answer"]):
            if number not in present:
                failures.append(f"{example.ask[:44]!r}: cites {number}, not in its context")
    assert not failures, "\n".join(failures)
