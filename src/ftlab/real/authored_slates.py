"""Hand-written examples of the shape the blind set actually asks.

A slate question names its candidates and asks you to choose among them. None of
the prose examples in ``authored.py`` teach that, and the first fine-tune showed
what happens without it: on the blind slates it named companies that were not on
the list on nearly every answer, and on one archetype it abandoned the question
entirely to produce a well-formed answer to a different one.

These model the discipline the task needs — read the list, use the evidence in
front of you, and say plainly when the list does not contain a good answer. Two
of the four decline to rank, because a slate with no good option is a real case
and a model that has never seen it will invent an order.

Facts are checked by ``tests/test_authored.py`` alongside the prose set.
"""

from __future__ import annotations

from .authored import CDC, CMS, RTI

SLATES: tuple[tuple[str, str, str, list[tuple]], ...] = (
    (
        "Which of these has RTI actually used on CDC work?\n"
        "1. CARAHSOFT TECHNOLOGY SOLUTIONS\n"
        "2. HR DIRECTIONS\n"
        "3. SNOWFLAKE\n"
        "4. PATIENT PERFORMANCE\n"
        "5. M9 SOLUTIONS\n"
        "6. GERLING & ASSOCIATES",
        "Three of the six. HR Directions is RTI's most-used CDC subcontractor at "
        "ten awards, Patient Performance is on four, Gerling & Associates on "
        "three. Carahsoft, Snowflake and M9 Solutions are all real companies "
        "with real HHS volume, but that volume is CMS and none of it is with RTI.",
        "Three of them:\n\n"
        "- HR Directions (10 CDC awards with RTI)\n"
        "- Patient Performance (4)\n"
        "- Gerling & Associates (3)\n\n"
        "The other three are CMS companies. Carahsoft, Snowflake and M9 Solutions "
        "all have genuine HHS work, which is exactly what makes them plausible on "
        "a list like this, and none has a reported subcontract with RTI at CDC.",
        [
            ("used", RTI, "HR DIRECTIONS", 10, CDC),
            ("used", RTI, "PATIENT PERFORMANCE", 4, CDC),
            ("not_used", RTI, "CARAHSOFT TECHNOLOGY SOLUTIONS", CDC),
            ("not_used", RTI, "SNOWFLAKE", CDC),
            ("not_used", RTI, "M9 SOLUTIONS", CDC),
        ],
    ),
    (
        "Pick the two strongest for a CDC bid under RTI:\n"
        "1. SNOWFLAKE\n"
        "2. GERLING & ASSOCIATES\n"
        "3. TELLIGEN\n"
        "4. HR DIRECTIONS",
        "Only two of the four have any RTI CDC history at all, so the selection "
        "makes itself and the ordering is the only judgement. Ten awards beats "
        "three.",
        "HR Directions first, Gerling & Associates second.\n\n"
        "They are also the only two on this list with any RTI CDC work at all -- "
        "Snowflake is a CMS data-platform supplier, Telligen works AHRQ and CMS. "
        "The choice is not close, and the list is not really offering four "
        "options.",
        [
            ("used", RTI, "HR DIRECTIONS", 10, CDC),
            ("not_used", RTI, "SNOWFLAKE", CDC),
            ("not_used", RTI, "TELLIGEN", CDC),
        ],
    ),
    (
        "Any of these worth approaching for CDC surveillance work?\n"
        "1. SKY TECHNOLOGIES\n"
        "2. M9 SOLUTIONS\n"
        "3. SNOWFLAKE",
        "All three are CMS companies with no reported CDC presence. Any ranking "
        "of them for CDC work would be invented, and saying so is more useful "
        "than producing an order that looks considered.",
        "None of them, on this evidence.\n\n"
        "All three work CMS and none has reported CDC work. I could put them in "
        "an order but it would be fabricated -- nothing in the record "
        "distinguishes them for this customer.\n\n"
        "If the list is fixed, the honest recommendation is to widen it.",
        [
            ("not_used", RTI, "SKY TECHNOLOGIES", CDC),
            ("not_used", RTI, "M9 SOLUTIONS", CDC),
            ("not_used", RTI, "SNOWFLAKE", CDC),
        ],
    ),
    (
        "Which of these work CMS?\n"
        "1. HR DIRECTIONS\n"
        "2. CARAHSOFT TECHNOLOGY SOLUTIONS\n"
        "3. TELLIGEN\n"
        "4. ORACLE AMERICA",
        "All four, as it happens. The list contains no wrong answers, and saying "
        "so beats picking a subset in order to look decisive.",
        "All four.\n\n"
        "Carahsoft is much the largest there by volume; Telligen and Oracle "
        "America both appear at CMS alongside other components; HR Directions "
        "works CMS as well as CDC and SAMHSA.\n\n"
        "This list does not separate anybody. If the intent was to narrow a "
        "field, the filter has to be something other than CMS presence.",
        [
            ("agency_work", "CARAHSOFT TECHNOLOGY SOLUTIONS", CMS),
            ("agency_work", "TELLIGEN", CMS),
            ("agency_work", "ORACLE AMERICA", CMS),
            ("agency_work", "HR DIRECTIONS", CMS),
        ],
    ),
)
