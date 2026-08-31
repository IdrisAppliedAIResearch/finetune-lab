"""The verified reward for the masked-sub task.

One question, one right answer, and the answer is a fact rather than an opinion:
a subcontract that was actually reported after the blind cut. That is what makes
reinforcement learning possible here at all -- there is something to verify
against that nobody in this project wrote.

The reward is the reciprocal rank of the true sub in the model's ranking, which
is dense where accuracy is not. A policy that moves the answer from fifth place
to second has improved and 0/1 accuracy cannot see it; over a batch of sixteen
rollouts on a twelve-name slate, accuracy is zero for most of early training and
there is no gradient to follow. Reciprocal rank gives 1.0, 0.5, 0.33, 0.25, 0.2
for the five places the question asks for, and 0 below that.

Two things it deliberately does not do:

* **It does not reward the reasoning.** Only the ranking is scored. Rewarding
  the explanation means rewarding whatever a scorer believes an explanation
  should look like, which is the hand-authored key problem in a new costume.
* **It does not use the tier labels.** ``tier_for`` scores 3+ only for a firm
  that has subbed for this prime before, so on the new-pairing half no correct
  answer is ever tier 3 and 37% of them are tier 1 -- classified as traps. A
  reward built on tiers would train the policy to avoid the right answer.

Off-slate names are penalised rather than ignored. A model that names companies
the prompt never offered has not answered the question, and the grader's
substring matching means an ungrounded name can also collide with a real one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..shared.grade import conclusion_of, find_companies, looks_truncated, split_answer
from ..shared.questions import TOP_K

# Per off-slate name among the picks. Five ungrounded names cost 0.5, which is
# more than the 0.2 a fifth-place correct answer earns, so guessing wildly is
# never worth it. It is not so large that one stray mention in a conclusion
# wipes out a correct top pick.
OFF_SLATE_PENALTY = 0.1

# For an answer with no ranking in it at all. Distinct from ranking the wrong
# names, which already scores zero: this is the format failure, and it needs to
# be worse than a wrong answer or a policy can learn to say nothing.
NO_ANSWER_PENALTY = 0.2

_NUMBERED = re.compile(r"(?m)^[ \t]*(\d{1,2})\s*[.)\]:-]\s*(.+)$")

# Headings the answer is asked to sit under, plus the ones a model reaches for
# anyway. Matched in the 60 characters before a run, so the list a model
# announces as its answer beats the list it wrote on the way there.
#
# Not anchored to the end of that window: a model writes "my ranking is:" and
# "Here they are, in order:" as often as a bare "Ranking:", and requiring the
# keyword last missed every one of those.
_ANSWER_MARKER = re.compile(
    r"(?i)(ranking|rank them|most likely|final answer|my answer|in order|top five)"
)


@dataclass(frozen=True)
class Reward:
    """One scored rollout, with the parts kept apart for diagnosis."""

    value: float
    rank: int | None
    picks: tuple[str, ...]
    off_slate: tuple[str, ...]
    # Carried so a budget problem is legible as a budget problem. The smoke run
    # had every answer stop mid-analysis at 1024 tokens; without this the
    # summary would have said the model ranks badly rather than that it never
    # got to rank anything.
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": round(self.value, 4),
            "rank": self.rank,
            "picks": list(self.picks),
            "off_slate": list(self.off_slate),
            "truncated": self.truncated,
        }


def _numbered_runs(text: str) -> list[tuple[list[str], int]]:
    """Every consecutive numbered list, with the offset each one starts at.

    Runs must count up -- ``1.`` then ``2.`` -- so a stray "1." in prose is not
    a list. A single-line run is not one either. The offset is carried so the
    line above a run can be checked for an answer heading.
    """
    runs: list[tuple[list[str], int]] = []
    current: list[str] = []
    start = 0
    expected = 1
    for match in _NUMBERED.finditer(text):
        number, body = int(match.group(1)), match.group(2).strip()
        if number == expected:
            if not current:
                start = match.start()
            current.append(body)
            expected += 1
        elif number == 1:
            if len(current) > 1:
                runs.append((current, start))
            current, expected, start = [body], 2, match.start()
        else:
            if len(current) > 1:
                runs.append((current, start))
            current, expected = [], 1
    if len(current) > 1:
        runs.append((current, start))
    return runs


def _ranking_block(
    text: str, slate: list[str], known: list[str], truncated: bool = False
) -> list[str] | None:
    """The numbered list that is the model's ranking, if there is one.

    Three ways a list in this text is not the answer, each of which produced a
    wrong measurement before it was handled:

    * **It is the model enumerating the candidates.** Working through the
      records it writes one numbered line per candidate, quoting each one's
      partners. That run is as long as the slate and arrives in slate order, so
      reading it as a ranking scores exactly at chance and looks like a result.
      Runs longer than ``TOP_K + 2`` are therefore skipped.
    * **It is quoting somebody else's partner list.** Over half a run's lines
      have to name a candidate from *this* question's slate.
    * **It is a fragment.** When generation stops on the budget the final list
      is whatever the model was midway through writing.

    The catch is that the last list is not the right one either. Asked for four
    sentences and a ranking, the base model gives exactly that -- and then
    starts again from the top and gets cut off partway through its second copy.
    Rejecting every truncated answer scored 16 of 16 as silence when 16 of 16
    had ranked correctly-formatted names. So the *first* complete block wins,
    and completeness is what a truncated answer has to show.
    """
    on_slate = set(slate)
    marked: list[list[str]] = []
    complete: list[list[str]] = []
    partial: list[list[str]] = []
    for run, start in _numbered_runs(text):
        if len(run) > TOP_K + 2:
            continue
        named = [find_companies(line, known) for line in run]
        hits = sum(1 for found in named if found and found[0] in on_slate)
        if hits * 2 <= len(run):
            continue
        # The prompt asks for the answer under a "Ranking:" heading, so a run
        # that has one is the model's answer and a run that does not is more
        # likely the candidates restated on the way to it.
        if _ANSWER_MARKER.search(text[max(0, start - 60) : start]):
            marked.append(run)
        (complete if len(run) >= TOP_K else partial).append(run)

    for group in (marked, complete):
        full = [r for r in group if len(r) >= TOP_K]
        if full:
            return full[0]
        if group and not truncated:
            return group[0]
    # A short unmarked block is only an answer if the model chose to stop there.
    # Cut off mid-list, it is the beginning of a ranking rather than one.
    return None if truncated or not partial else partial[0]


def parse_ranking(text: str, slate: list[str], known: list[str]) -> list[str]:
    """The companies the answer ranks, best first.

    A numbered list is read line by line, so its order is the model's order.
    Falling back to order of appearance is only correct for an answer that names
    its picks once, in order; a model reasoning over all twelve candidates
    before concluding names them all, so the fallback runs on the conclusion
    only and after the rejected half has been split off.
    """
    truncated = looks_truncated(text)
    ordered: list[str] = []
    lines = _ranking_block(text, slate, known, truncated)
    if lines is not None:
        for line in lines:
            found = find_companies(line, known)
            if found:
                ordered.append(found[0])
    elif truncated:
        # Nothing complete to read, and the prose fallback on a truncated answer
        # returns whichever companies the analysis happened to mention first.
        # That is not an answer and must not be scored as one.
        return []
    else:
        recommended, _ = split_answer(text)
        ordered = find_companies(conclusion_of(recommended, known), known)

    seen: set[str] = set()
    picks: list[str] = []
    for name in ordered:
        if name not in seen:
            seen.add(name)
            picks.append(name)
    return picks


def score(
    text: str,
    gold: str,
    slate: list[str],
    known: list[str],
    top_k: int = TOP_K,
) -> Reward:
    """Reciprocal rank of ``gold``, less a penalty for ungrounded names.

    ``known`` is the full roster rather than the slate, so a company named from
    outside the twelve is detected instead of silently skipped. It still costs
    the policy a place in the ranking, which is most of the deterrent: an
    off-slate name in first position pushes a correct second pick to 0.5.
    """
    cut = looks_truncated(text)
    picks = parse_ranking(text, slate, known)[:top_k]
    if not picks:
        return Reward(
            value=-NO_ANSWER_PENALTY, rank=None, picks=(), off_slate=(), truncated=cut
        )

    on_slate = set(slate)
    off = tuple(n for n in picks if n not in on_slate)
    rank = picks.index(gold) + 1 if gold in picks else None
    value = (1.0 / rank if rank else 0.0) - OFF_SLATE_PENALTY * len(off)
    return Reward(
        value=value, rank=rank, picks=tuple(picks), off_slate=off, truncated=cut
    )


def score_batch(
    items: list[dict[str, Any]], generations: list[str], known: list[str]
) -> list[Reward]:
    """One reward per generation, aligned with ``items``."""
    return [
        score(
            text,
            (item["meta"]["gold"] or [""])[0],
            sorted(item["meta"].get("tiers") or {}),
            known,
        )
        for item, text in zip(items, generations, strict=True)
    ]


def summarise(items: list[dict[str, Any]], rewards: list[Reward]) -> dict[str, Any]:
    """Mean reward and the hit rates it is built from, split the usual way.

    Reported beside the analytic random floor for the same slate size, because a
    reward mean on its own says nothing -- a twelve-name slate gives a blind
    ranking 0.083 at hit@1 and 0.259 mean reciprocal rank before any policy has
    learned anything.
    """

    def block(rows: list[tuple[dict[str, Any], Reward]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)
        sizes = [len(r[0]["meta"].get("tiers") or {}) or 1 for r in rows]
        return {
            "n": n,
            "mean_reward": round(sum(r[1].value for r in rows) / n, 4),
            "hit@1": round(sum(1 for r in rows if r[1].rank == 1) / n, 4),
            "hit@3": round(sum(1 for r in rows if r[1].rank and r[1].rank <= 3) / n, 4),
            "hit@5": round(sum(1 for r in rows if r[1].rank and r[1].rank <= 5) / n, 4),
            "mrr": round(sum(1 / r[1].rank if r[1].rank else 0.0 for r in rows) / n, 4),
            "off_slate_rate": round(
                sum(1 for r in rows if r[1].off_slate) / n, 4
            ),
            "no_answer_rate": round(sum(1 for r in rows if not r[1].picks) / n, 4),
            "truncated_rate": round(sum(1 for r in rows if r[1].truncated) / n, 4),
            # Truncated at TOP_K, because the reward is. Floored over the
            # whole slate instead, the random MRR reads 0.259 where the reward
            # can only pay out over five places and a blind ranking actually
            # earns 0.190 -- a floor set a third too high, which would have made
            # a policy at chance look like a policy losing.
            "random_hit@1": round(sum(1 / s for s in sizes) / n, 4),
            "random_hit@3": round(sum(min(3, s) / s for s in sizes) / n, 4),
            "random_hit@5": round(sum(min(TOP_K, s) / s for s in sizes) / n, 4),
            "random_mrr": round(
                sum(
                    sum(1 / r for r in range(1, min(TOP_K, s) + 1)) / s
                    for s in sizes
                )
                / n,
                4,
            ),
        }

    paired = list(zip(items, rewards, strict=True))
    return {
        "all": block(paired),
        "new_pairings": block([p for p in paired if p[0]["meta"].get("is_new")]),
        "prior_pairings": block([p for p in paired if not p[0]["meta"].get("is_new")]),
    }


__all__ = [
    "NO_ANSWER_PENALTY",
    "OFF_SLATE_PENALTY",
    "Reward",
    "parse_ranking",
    "score",
    "score_batch",
    "summarise",
]
