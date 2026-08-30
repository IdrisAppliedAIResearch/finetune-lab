"""Check a hand-written example's declared claims against the graph.

Shared by the closed-book set in ``test_authored.py`` and the open-book set in
``test_authored_context.py``. One implementation, because two would drift and
the drift would show up as an authored corpus that passes its own tests while
disagreeing with the data.
"""

from __future__ import annotations

from typing import Any


def team_size(graph: Any, prime: str, agency: str) -> int:
    return len({r["sub"] for r in graph.train_subawards
                if r["prime"] == prime and r["agency"] == agency})


def times_used(graph: Any, prime: str, sub: str, agency: str | None = None) -> int:
    return sum(
        1
        for r in graph.train_subawards
        if r["prime"] == prime and r["sub"] == sub
        and (agency is None or r["agency"] == agency)
    )


def check_facts(graph: Any, label: str, facts: Any) -> list[str]:
    """Failures as readable strings; empty means every claim holds."""
    failures: list[str] = []
    for fact in facts:
        kind, args = fact[0], fact[1:]
        if kind == "team_size":
            prime, agency, expected = args
            actual = team_size(graph, prime, agency)
        elif kind == "used":
            prime, sub, expected = args[0], args[1], args[2]
            agency = args[3] if len(args) > 3 else None
            actual = times_used(graph, prime, sub, agency)
        elif kind == "agencies":
            company, expected = args
            actual = len(graph.companies[company].agencies)
        elif kind == "partners":
            company, expected = args
            actual = len(graph.companies[company].partners)
        elif kind == "as_sub":
            company, expected = args
            actual = len(graph.companies[company].as_sub)
        elif kind == "naics":
            # A code the answer cites has to be one the record actually shows.
            company, code = args
            expected, actual = True, code in graph.companies[company].naics
        elif kind == "as_prime":
            # Awards where this company held the prime position -- the number
            # the retrieved record reports as "awards where prime".
            company, expected = args
            actual = len(graph.companies[company].as_prime)
        elif kind == "not_used":
            prime, sub, agency = args
            expected, actual = 0, times_used(graph, prime, sub, agency)
        elif kind == "agency_work":
            company, agency = args
            expected = True
            actual = any(
                company in (r["prime"], r["sub"])
                for r in graph.train_subawards
                if r["agency"] == agency
            )
        elif kind == "no_work":
            company, agency = args
            expected, actual = 0, sum(
                1 for r in graph.train_subawards
                if r["agency"] == agency and company in (r["prime"], r["sub"])
            )
        else:  # pragma: no cover - guarded by the tuple shape
            raise AssertionError(f"unknown fact kind {kind!r}")

        if actual != expected:
            failures.append(
                f"{label[:48]!r}: {kind}{args} -- said {expected}, data says {actual}"
            )
    return failures


def records_in(context: str) -> set[str]:
    """Companies that have their own record in a context block.

    Substring matching is no longer enough to tell whether a company was
    supplied. Since records began listing partners with counts, a firm's name
    appears inside other firms' records too -- so "NAME in context" is true for
    hundreds of companies whose record was never retrieved, and a test relying
    on it silently stops testing anything. Only the ``[n] NAME`` header means
    the model was actually given that record.
    """
    return {
        line.split("] ", 1)[1].strip()
        for line in context.splitlines()
        if line.startswith("[") and "] " in line
    }


def grounded_in(context: str, name: str, subject: str) -> bool:
    """Could a model read this company out of the supplied context?

    Two ways count, and only two. The company has its own ``[n] NAME`` record,
    or it is named inside the subject's record -- which is real grounding now
    that a record lists its most-used partners with counts. A bare substring hit
    anywhere in the block is not enough: partner lists mention hundreds of firms
    whose own records were never retrieved.
    """
    if name in records_in(context):
        return True
    for block in context.split("\n\n"):
        if block.split("\n", 1)[0].endswith(subject) and name in block:
            return True
    return False
