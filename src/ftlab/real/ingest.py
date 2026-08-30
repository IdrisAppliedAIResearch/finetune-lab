"""Pull a real past-performance slice from USASpending.gov.

An API, not a scrape: https://api.usaspending.gov is public domain, needs no
key, and serves the two things this project actually needs -- prime contract
awards and the subcontracts reported against them. The subcontract records are
the teaming graph, which is the only reason to prefer real data over the
synthetic world we had.

What real data buys, and it is the whole point: the labels stop being ours.
Which companies subbed on which award is historical fact, so a rule engine and a
fine-tuned model are both predicting something neither one defines, and either
can lose. The synthetic corpus could not do that -- its golden answers *were* a
rule engine's output, so the rule engine scored 1.000 by construction.

Three limits to keep in view, all measured rather than assumed:

* **Subaward reporting is incomplete.** Primes must report subcontracts over
  $30k to FSRS and compliance is uneven, so a company absent from an award's
  sub list may still have worked it. Labels have false negatives; absence is not
  a negative.
* **Descriptions are often empty of content.** In a 600-record sample, 44% were
  under 40 characters or pure boilerplate ("SUBCONTRACT 8-312-0214780 MOD 6").
  The text-reading advantage only exists on the rest.
* **CPARS is not obtainable.** Performance ratings are source-selection
  information and FOIA-exempt. Repeat teaming -- did the prime hire them again
  -- is the honest observable substitute, since it is revealed preference.

NAICS deserves its own note, because it is the reason this comparison is worth
running. NAICS 541690 "Other Scientific and Technical Consulting" contains both
Lockheed's Apache targeting sights and CDC epidemiologic surveillance. A rule
engine keyed on structured fields cannot separate those; the description text
can. That asymmetry is real, not contrived for the demo.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable
from typing import Any

API = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

# Contract award types. Grants and loans have no subcontracting relationship
# worth reading, so they are left out rather than filtered later.
CONTRACT_TYPES = ["A", "B", "C", "D"]

# Sub-agencies whose work is recognisably public health. Matched as substrings
# against the "Awarding Sub Agency" field, because the API rejects subtier names
# that do not match its own spelling exactly and the spellings are inconsistent.
PUBLIC_HEALTH_SUBAGENCIES = (
    "Centers for Disease Control",
    "National Institutes of Health",
    "Preparedness and Response",
    "Health Resources and Services",
    "Food and Drug Administration",
    "Substance Abuse and Mental Health",
    "Agency for Healthcare Research",
    "Indian Health Service",
    "Centers for Medicare",
)

SUBAWARD_FIELDS = [
    "Sub-Award ID",
    "Sub-Awardee Name",
    "Sub-Award Amount",
    "Sub-Award Date",
    "Sub-Award Description",
    "Prime Recipient Name",
    "Prime Award ID",
    "Awarding Agency",
    "Awarding Sub Agency",
    "NAICS",
    "PSC",
]

PRIME_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Award Amount",
    "Start Date",
    "End Date",
    "Description",
    "NAICS",
    "PSC",
    "recipient_id",
]


def _post(body: dict[str, Any], *, retries: int = 3) -> dict[str, Any]:
    """One API call, with a courteous retry.

    The service is free and unauthenticated, so it is on us not to hammer it: a
    short sleep between pages and a backoff on failure, rather than racing.
    """
    payload = json.dumps(body).encode()
    for attempt in range(retries):
        request = urllib.request.Request(
            API, payload, {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def _paged(
    body: dict[str, Any], pages: int, *, label: str, pause: float = 0.3
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        result = _post({**body, "page": page})
        rows.extend(result["results"])
        print(f"[ingest] {label} page {page}/{pages} -> {len(rows)} rows", flush=True)
        if not result["page_metadata"].get("hasNext"):
            break
        time.sleep(pause)
    return rows


# The record set is far larger than any single sorted pull can reach. Asking
# for 2015-2025 sorted by date and taking 2000 rows gets 2015; asking again in
# reverse gets 2025. That is exactly what the first ingest did, and the corpus
# it produced had 1354 rows from 2015, 1654 from 2025, and nothing at all from
# 2016-2023 -- a hole that was invisible in the totals and made half the
# relational evidence in every company record a decade stale. Fetching one year
# at a time is the fix: each window is small enough that the page budget reaches
# the end of it.
YEARS = tuple(range(2015, 2026))


def fetch_subawards(
    pages: int = 20,
    limit: int = 100,
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    agency: str = "Department of Health and Human Services",
    order: str = "desc",
) -> list[dict[str, Any]]:
    """Subcontracts reported against HHS prime contracts, one time window.

    Returns at most ``pages * limit`` rows from one end of the window. Prefer
    :func:`fetch_subawards_by_year` unless you want a specific slice: over a
    wide window this silently returns one end of the range and nothing else.
    """
    return _paged(
        {
            "filters": {
                "award_type_codes": CONTRACT_TYPES,
                "time_period": [{"start_date": start, "end_date": end}],
                "agencies": [{"type": "awarding", "tier": "toptier", "name": agency}],
            },
            "fields": SUBAWARD_FIELDS,
            "limit": limit,
            "sort": "Sub-Award Date",
            "order": order,
            "subawards": True,
        },
        pages,
        label=f"subawards {start[:4]}",
    )


def fetch_prime_awards(
    pages: int = 20,
    limit: int = 100,
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    agency: str = "Department of Health and Human Services",
) -> list[dict[str, Any]]:
    """HHS prime contracts for one time window, largest first.

    Sorted by amount rather than date on purpose: these supply the company
    histories a question is answered from, and the large awards are the ones a
    capture team would actually cite. Sorting by amount over a wide window has
    the same truncation problem as sorting by date, so this is also called per
    year -- otherwise the whole budget goes to a handful of enormous IDIQs.
    """
    return _paged(
        {
            "filters": {
                "award_type_codes": CONTRACT_TYPES,
                "time_period": [{"start_date": start, "end_date": end}],
                "agencies": [{"type": "awarding", "tier": "toptier", "name": agency}],
            },
            "fields": PRIME_FIELDS,
            "limit": limit,
            "sort": "Award Amount",
            "order": "desc",
            "subawards": False,
        },
        pages,
        label=f"prime awards {start[:4]}",
    )


def _by_year(
    fetch: Any, years: Iterable[int], pages: int, **kwargs: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in years:
        rows.extend(
            fetch(pages=pages, start=f"{year}-01-01", end=f"{year}-12-31", **kwargs)
        )
    return rows


def fetch_subawards_by_year(
    years: Iterable[int] = YEARS,
    pages: int = 25,
    limit: int = 100,
    agency: str = "Department of Health and Human Services",
) -> list[dict[str, Any]]:
    """Every year separately, so no year is truncated away by the page budget."""
    return _by_year(
        fetch_subawards, years, pages, limit=limit, agency=agency, order="desc"
    )


def fetch_prime_awards_by_year(
    years: Iterable[int] = YEARS,
    pages: int = 10,
    limit: int = 100,
    agency: str = "Department of Health and Human Services",
) -> list[dict[str, Any]]:
    return _by_year(fetch_prime_awards, years, pages, limit=limit, agency=agency)


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

# Reported sub-award amounts are frequently wrong -- a $6.17bn "subcontract"
# turned up in the first page of results. Anything above this is treated as a
# data-entry error rather than a real figure, and the amount is dropped while
# the relationship is kept: the edge is what matters and it is still true.
MAX_CREDIBLE_SUBAWARD = 2_000_000_000

# Names that identify no company.
NON_ENTITIES = ("UNDISCLOSED", "MULTIPLE RECIPIENTS", "MISCELLANEOUS FOREIGN")


def clean_name(raw: str | None) -> str:
    """Normalise a recipient name enough to join on it.

    USASpending has no stable entity id across the prime and subaward feeds --
    subawards carry a name string only -- so the join is on text, and text needs
    trimming. Deliberately conservative: case, whitespace, trailing punctuation
    and the commonest suffix variants, nothing that would merge two real firms.
    """
    if not raw:
        return ""
    name = " ".join(raw.upper().split())
    name = name.replace("&AMP;", "&").replace("’", "'")
    for suffix in (
        ", INCORPORATED", " INCORPORATED", ", INC.", ", INC", " INC.", " INC",
        ", L.L.C.", ", LLC", " L.L.C.", " LLC", ", LTD.", ", LTD", " LTD.", " LTD",
        ", L.P.", " L.P.", ", LP", " LP", ", CORPORATION", " CORPORATION",
        ", CORP.", " CORP.", ", CORP", " CORP", ", CO.", " CO.",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip(" ,.")


def is_entity(name: str) -> bool:
    return bool(name) and not any(token in name for token in NON_ENTITIES)


def is_public_health(row: dict[str, Any]) -> bool:
    sub = row.get("Awarding Sub Agency") or ""
    return any(token in sub for token in PUBLIC_HEALTH_SUBAGENCIES)


def code_of(value: Any) -> str:
    """NAICS/PSC arrive as {'code':..,'description':..} or occasionally null."""
    return (value or {}).get("code", "") if isinstance(value, dict) else ""


def title_of(value: Any) -> str:
    return (value or {}).get("description", "") if isinstance(value, dict) else ""


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


@dataclass
class Slice:
    """A cleaned public-health slice: awards, subcontracts, and the join."""

    prime_awards: list[dict[str, Any]] = field(default_factory=list)
    subawards: list[dict[str, Any]] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        primes = {r["prime"] for r in self.subawards}
        subs = {r["sub"] for r in self.subawards}
        pairs = {(r["prime"], r["sub"]) for r in self.subawards}
        described = sum(1 for r in self.subawards if len(r["description"]) >= 40)
        return {
            "prime_awards": len(self.prime_awards),
            "subawards": len(self.subawards),
            "distinct_primes": len(primes),
            "distinct_subs": len(subs),
            "distinct_pairs": len(pairs),
            "companies": len({*primes, *subs}),
            "subawards_with_real_description": described,
            "description_rate": round(described / len(self.subawards), 3)
            if self.subawards
            else 0.0,
        }


def _dedupe(rows: list[dict[str, Any]], key: Any) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    out = []
    for row in rows:
        k = key(row)
        if k in seen:
            continue
        seen.add(k)
        out.append(row)
    return out


def build_slice(
    subaward_pages: int = 25,
    prime_pages: int = 10,
    years: Iterable[int] | None = YEARS,
    **kwargs: Any,
) -> Slice:
    """Fetch, filter to public health, clean, and normalise into a Slice.

    ``years`` fetches each year separately, which is the only way to get the
    middle of the record set; pass ``None`` to fall back to a single window.
    """
    if years is None:
        raw_subs = fetch_subawards(pages=subaward_pages, **kwargs)
        raw_primes = fetch_prime_awards(pages=prime_pages, **kwargs)
    else:
        raw_subs = fetch_subawards_by_year(years, pages=subaward_pages, **kwargs)
        raw_primes = fetch_prime_awards_by_year(years, pages=prime_pages, **kwargs)

    # Year windows do not overlap, but a mod reported against two fiscal years
    # and a retried page both show up twice, and a duplicated edge silently
    # doubles a company's apparent teaming history.
    raw_subs = _dedupe(
        raw_subs,
        lambda r: r.get("Sub-Award ID")
        or (
            r.get("Prime Award ID"),
            r.get("Sub-Awardee Name"),
            r.get("Sub-Award Date"),
            r.get("Sub-Award Amount"),
        ),
    )
    raw_primes = _dedupe(raw_primes, lambda r: r.get("Award ID") or id(r))

    subawards = []
    for row in raw_subs:
        if not is_public_health(row):
            continue
        prime = clean_name(row.get("Prime Recipient Name"))
        sub = clean_name(row.get("Sub-Awardee Name"))
        if not (is_entity(prime) and is_entity(sub)) or prime == sub:
            continue
        amount = row.get("Sub-Award Amount") or 0
        subawards.append(
            {
                "prime": prime,
                "sub": sub,
                "prime_award_id": row.get("Prime Award ID") or "",
                "sub_award_id": row.get("Sub-Award ID") or "",
                "agency": row.get("Awarding Sub Agency") or "",
                "naics": code_of(row.get("NAICS")),
                "naics_title": title_of(row.get("NAICS")),
                "psc": code_of(row.get("PSC")),
                "psc_title": title_of(row.get("PSC")),
                "date": row.get("Sub-Award Date") or "",
                "amount": amount if 0 < amount < MAX_CREDIBLE_SUBAWARD else None,
                "description": " ".join((row.get("Sub-Award Description") or "").split()),
            }
        )

    prime_awards = []
    for row in raw_primes:
        if not is_public_health(row):
            continue
        recipient = clean_name(row.get("Recipient Name"))
        if not is_entity(recipient):
            continue
        prime_awards.append(
            {
                "recipient": recipient,
                "award_id": row.get("Award ID") or "",
                "agency": row.get("Awarding Sub Agency") or "",
                "naics": code_of(row.get("NAICS")),
                "naics_title": title_of(row.get("NAICS")),
                "psc": code_of(row.get("PSC")),
                "psc_title": title_of(row.get("PSC")),
                "start": row.get("Start Date") or "",
                "end": row.get("End Date") or "",
                "amount": row.get("Award Amount") or 0,
                "description": " ".join((row.get("Description") or "").split()),
            }
        )

    return Slice(prime_awards=prime_awards, subawards=subawards)


def write_slice(data: Slice, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prime_awards.json").write_text(
        json.dumps(data.prime_awards, indent=1), encoding="utf-8"
    )
    (out / "subawards.json").write_text(
        json.dumps(data.subawards, indent=1), encoding="utf-8"
    )
    stats = data.stats()
    (out / "slice_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_slice(out_dir: str | Path) -> Slice:
    out = Path(out_dir)
    return Slice(
        prime_awards=json.loads((out / "prime_awards.json").read_text(encoding="utf-8")),
        subawards=json.loads((out / "subawards.json").read_text(encoding="utf-8")),
    )
