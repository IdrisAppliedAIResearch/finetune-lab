"""Company profiles and the teaming graph, built from the real slice.

The unit of knowledge is a company: what it has won as a prime, who it has
subbed for, who it has hired, which agencies it touches, and what the work
descriptions actually say. That is the "past performance library" the model has
to reason over, and unlike the synthetic version it is assembled from public
record rather than invented.

The split is by date and it is load-bearing. Everything on or before
``TRAIN_UNTIL`` may be used to build training data; everything after is sealed
for blind evaluation and must never be read while the corpus is generated. That
is the only way to answer "did you train on the answer" with something other
than assurances -- the later teaming has not been observed by anything that
writes training examples.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

# Subawards dated after this are blind. Chosen to leave roughly two-thirds of
# the recent teaming for training and a third sealed, while keeping the blind
# period long enough (six months) that it is not one unusual quarter.
TRAIN_UNTIL = "2025-06-30"

# Work whose description says something. Below this, the text is an id, a mod
# number, or the word "subcontract", and a model reading it learns nothing.
MEANINGFUL_DESCRIPTION = 40


@dataclass
class Company:
    name: str
    prime_awards: list[dict[str, Any]] = field(default_factory=list)
    # Awards where this company was the sub, and awards where it was the prime.
    as_sub: list[dict[str, Any]] = field(default_factory=list)
    as_prime: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agencies(self) -> list[str]:
        seen = {a["agency"] for a in (*self.prime_awards, *self.as_sub, *self.as_prime)}
        return sorted(x for x in seen if x)

    @property
    def naics(self) -> list[str]:
        counter = collections.Counter(
            a["naics"] for a in (*self.prime_awards, *self.as_sub, *self.as_prime) if a.get("naics")
        )
        return [code for code, _ in counter.most_common()]

    @property
    def partners(self) -> set[str]:
        """Everyone this company has been on a contract with, either direction."""
        return {a["prime"] for a in self.as_sub} | {a["sub"] for a in self.as_prime}

    def descriptions(self, limit: int = 6) -> list[str]:
        """Work descriptions worth reading, longest first.

        Longest first because length correlates with content here: the short
        ones are almost all "SUBCONTRACT <id> MOD <n>".
        """
        texts = [
            a["description"]
            for a in (*self.as_sub, *self.as_prime, *self.prime_awards)
            if len(a.get("description", "")) >= MEANINGFUL_DESCRIPTION
        ]
        unique = list(dict.fromkeys(texts))
        return sorted(unique, key=len, reverse=True)[:limit]


@dataclass
class TeamingGraph:
    companies: dict[str, Company]
    train_subawards: list[dict[str, Any]]
    blind_subawards: list[dict[str, Any]]

    def pairs(self, blind: bool = False) -> collections.Counter:
        rows = self.blind_subawards if blind else self.train_subawards
        return collections.Counter((r["prime"], r["sub"]) for r in rows)

    def team_of(self, prime: str, blind: bool = False) -> set[str]:
        rows = self.blind_subawards if blind else self.train_subawards
        return {r["sub"] for r in rows if r["prime"] == prime}

    def primes_using_subs(self, minimum: int = 3, blind: bool = False) -> list[str]:
        counts = collections.Counter()
        for prime, _sub in self.pairs(blind):
            counts[prime] += 1
        return [p for p, c in counts.most_common() if c >= minimum]

    def stats(self) -> dict[str, Any]:
        return {
            "companies": len(self.companies),
            "train_subawards": len(self.train_subawards),
            "blind_subawards": len(self.blind_subawards),
            "train_pairs": len(self.pairs()),
            "blind_pairs": len(self.pairs(blind=True)),
            "blind_pairs_unseen_in_train": len(
                set(self.pairs(blind=True)) - set(self.pairs())
            ),
            "primes_with_3plus_subs_train": len(self.primes_using_subs(3)),
            "repeat_pairs_train": sum(1 for v in self.pairs().values() if v > 1),
        }


def build_graph(
    prime_awards: list[dict[str, Any]],
    subawards: list[dict[str, Any]],
    train_until: str = TRAIN_UNTIL,
) -> TeamingGraph:
    """Assemble company profiles, splitting subawards into train and blind."""
    train = [r for r in subawards if r["date"] and r["date"] <= train_until]
    blind = [r for r in subawards if r["date"] and r["date"] > train_until]

    companies: dict[str, Company] = {}

    def get(name: str) -> Company:
        if name not in companies:
            companies[name] = Company(name=name)
        return companies[name]

    for award in prime_awards:
        get(award["recipient"]).prime_awards.append(award)

    # Only training-period teaming shapes the profiles. A profile built from
    # blind edges would leak the answer into the very records the model reads.
    for row in train:
        get(row["sub"]).as_sub.append(row)
        get(row["prime"]).as_prime.append(row)

    return TeamingGraph(companies=companies, train_subawards=train, blind_subawards=blind)
