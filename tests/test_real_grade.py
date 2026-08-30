"""How a generated answer is turned into a score.

The grader is the most consequential code in the project and the least
obviously so: a three-word change to its rejection markers moved the headline
result, and nothing in the test suite noticed.
"""

from __future__ import annotations

import json

import pytest

from ftlab.real.grade import (
    conclusion_of,
    find_companies,
    grade_one,
    known_companies,
    looks_truncated,
    split_answer,
)

DATA = "data/real"


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
    picked = find_companies(conclusion_of(verbose, known), known)[:4]
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
    from ftlab.infer import TRUNCATION_MARK
    from ftlab.real.grade import looks_truncated

    assert looks_truncated("...AMAZON WEB SERVICES: **Teamed" + TRUNCATION_MARK)
    assert not looks_truncated("Most likely: GAMMA INC and DELTA GROUP.")

    # The case that made the punctuation heuristic useless: a finished bulleted
    # list ends without a full stop. It flagged 31 of 51 answers from an arm
    # whose longest used 60% of its budget.
    assert not looks_truncated("- ALPHA CORP: no reported relationship")




# ---------------------------------------------------------------------------
# the parser bug that decided a headline
# ---------------------------------------------------------------------------


def test_restated_system_prompt_is_not_a_rejection_heading():
    """The base model echoes its instructions; that must not eat the answer.

    "Do not" was a rejection marker, and the system prompt says "do not name a
    company that does not appear in them". On the v2 blind run this fired on 10
    of 51 arm-C answers and none of arm A's or B's -- filing each whole answer
    as rejected, scoring it as naming nobody, and handing the untuned arm a free
    zero on hard-negatives-recommended.
    """
    from ftlab.real.grade import split_answer

    echoed = (
        "I will use only the library records. Do not name a company that does "
        "not appear in them.\n\n"
        "Most likely: ALPHA CORP, BETA LLC."
    )
    recommended, rejected = split_answer(echoed)
    assert "ALPHA CORP" in recommended
    assert rejected == ""


def test_rejection_heading_only_counts_at_the_start_of_a_line():
    from ftlab.real.grade import split_answer

    inline = "ALPHA CORP is the pick, and it is Not recommended to stop there."
    recommended, rejected = split_answer(inline)
    assert rejected == "", "a mid-sentence phrase is not a section heading"

    heading = "ALPHA CORP is the pick.\n\nNot recommended: BETA LLC.\n"
    recommended, rejected = split_answer(heading)
    assert "ALPHA CORP" in recommended
    assert "BETA LLC" in rejected
