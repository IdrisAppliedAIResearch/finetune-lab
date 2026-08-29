"""Training examples written by hand rather than generated from a template.

The first fine-tune collapsed: eighteen of eighteen blind answers took one of
the seven answer shapes it had been trained on. The diagnosis is uncomfortable
but simple -- scripted generation *is* a template. Measured on the corpus that
produced it, ``prime_candidates`` was 150 numbered lists out of 150,
``sub_candidates`` 276 of 276, ``prior_relationship`` 297 prose answers out of
297. An archetype predicted its format perfectly, so "recognise the type, emit
the shape" was both the easiest rule available and enough to drive the loss
down. The model learned exactly what was on offer.

Adding a shape-randomiser to a generator treats the symptom. These are written
out instead, one at a time, and they differ in the ways real answers differ:
some lead with the conclusion and some with the caveat, some are a sentence and
some a list, some decline to answer, some correct the premise of the question.
What they have in common is reasoning that refers to specific evidence rather
than reciting a structure.

Every factual claim below is checked against the graph by
``tests/test_authored.py``. Authoring prose by hand and asserting facts by
machine is the only combination that gets both -- an unchecked hand-written
corpus would drift from the data within a dozen examples.

Counts come from reported FSRS subcontracts and are therefore lower bounds; the
answers say so where it matters, because a model that never sees hedging never
learns when hedging is the honest answer.
"""

from __future__ import annotations

from typing import Any

# Agency names as the data spells them. Bound to short constants so a fact
# tuple fits on a line and reads as a claim rather than a wall of string.
CDC = "Centers for Disease Control and Prevention"
NIH = "National Institutes of Health"
CMS = "Centers for Medicare and Medicaid Services"
FDA = "Food and Drug Administration"
SAMHSA = "Substance Abuse and Mental Health Services Administration"

RTI = "RESEARCH TRIANGLE INSTITUTE"
GDIT = "GENERAL DYNAMICS INFORMATION TECHNOLOGY"
ACN = "ACCENTURE FEDERAL SERVICES"
DELOITTE = "DELOITTE CONSULTING LLP"

# Each entry: (question, reasoning, answer, facts-to-verify)
# ``facts`` is a list of (kind, *args) checked in the tests:
#   ("team_size", prime, agency, n)      prime used n distinct subs at agency
#   ("used", prime, sub, n)              prime used sub on n reported awards
#   ("agencies", company, n)             company appears at n agencies
#   ("partners", company, n)             company has n distinct partners
#   ("no_work", company, agency)         company has no reported work at agency
#   ("not_used", prime, sub, agency)     prime never used sub at that agency
#   ("agency_work", company, agency)     company does appear at that agency
AUTHORED: tuple[tuple[str, str, str, list[tuple]], ...] = (
    (
        "Who does Research Triangle Institute actually put on its CDC teams?",
        "RTI reports 42 distinct subcontractors on CDC work, which is a lot of "
        "names and not much of a signal on its own. The number that matters is "
        "how often each one comes back: HR Directions appears on ten separate "
        "awards, and nobody else is close.",
        "HR Directions, by a wide margin -- ten reported CDC awards. After that "
        "it thins out fast: Patient Performance on four, Gerling & Associates "
        "and Docusource of North Carolina on three each.\n\n"
        "Read the first name as a standing relationship and the rest as a "
        "working roster. Forty-two subcontractors is breadth, not depth.",
        [("team_size", RTI, CDC, 42),
         ("used", RTI, "HR DIRECTIONS", 10, CDC)],
    ),
    (
        "We're going after CMS work and thinking about approaching GDIT as a prime. "
        "Do they even use subs?",
        "Heavily, and lopsidedly. GDIT reports 63 distinct subcontractors at CMS. "
        "But Carahsoft alone accounts for 28 of those awards, which is a reseller "
        "relationship rather than a teaming one, so the effective bench is smaller "
        "than 63 suggests.",
        "Yes -- 63 distinct subcontractors at CMS, the most of any prime in this "
        "data.\n\n"
        "Temper that before you build a strategy on it. Carahsoft is on 28 of "
        "those awards, and Snowflake, TVAR and CDW Government account for much of "
        "the rest; that pattern reads as product and licence pass-through, not "
        "as capability teaming. The question to ask GDIT is which of their CMS "
        "work actually gets subcontracted for skills.",
        [("team_size", GDIT, CMS, 63),
         ("used", GDIT, "CARAHSOFT TECHNOLOGY SOLUTIONS", 28, CMS)],
    ),
    (
        "Has Accenture Federal Services ever done CDC work?",
        "Checking their agency footprint rather than their size. Accenture Federal "
        "Services appears at CMS, FDA, HRSA and NIH in this data. CDC is not among "
        "them.",
        "No CDC work on record. Where they do appear:\n\n"
        "- CMS (the bulk of it)\n"
        "- FDA\n"
        "- HRSA\n"
        "- NIH\n\n"
        "Worth stating the limit: this is reported subcontract and award data, so "
        "it shows what was filed, not everything that happened. Good evidence "
        "they are not a CDC incumbent; weaker evidence they have never touched "
        "the agency.",
        [("agencies", ACN, 4),
         ("no_work", ACN, CDC)],
    ),
    (
        "What CPARS rating did Accenture Federal Services get on their CDC contract?",
        "Two separate problems with this question. There is no CDC contract on "
        "record for them to be rated on. And CPARS ratings are not public at all "
        "-- they are source-selection information, released only to the government "
        "and the rated contractor.",
        "I can't answer that, for two reasons.\n\n"
        "First, Accenture Federal Services has no reported CDC work, so there is "
        "no contract here to rate. Second, CPARS ratings are not publicly "
        "available for anyone -- they are source-selection information under FAR "
        "42.15, so no amount of digging in award data produces them.\n\n"
        "If you need a performance read, the closest public proxy is whether a "
        "prime hired the same sub again. That is revealed preference rather than "
        "a rating, but it is observable.",
        [("no_work", ACN, CDC)],
    ),
    (
        "Give me one name: who is RTI's most reliable CDC subcontractor?",
        "The question asks for one, so the answer should be one. Frequency is the "
        "only reliability signal available -- CPARS is not public -- and HR "
        "Directions leads on it by a factor of two and a half.",
        "HR Directions. Ten reported CDC awards with RTI, against four for the "
        "next name down.",
        [("used", RTI, "HR DIRECTIONS", 10, CDC)],
    ),
    (
        "Is Deloitte a bigger player than RTI across HHS?",
        "Bigger is ambiguous, and the two plausible readings disagree here. By "
        "agency spread they are close -- Deloitte at six components, RTI at seven. "
        "By teaming volume they are not: RTI reports 131 distinct partners against "
        "Deloitte's 47.",
        "Depends which sense you mean, and the two answers differ.\n\n"
        "By reach they are comparable: RTI appears at seven HHS components, "
        "Deloitte at six. By teaming they are not close -- RTI has 131 distinct "
        "subcontract partners on record, Deloitte 47.\n\n"
        "For a teaming conversation the second number is the relevant one. RTI "
        "subcontracts far more of its work, which means more ways in.",
        [("agencies", RTI, 7),
         ("agencies", DELOITTE, 6),
         ("partners", RTI, 131),
         ("partners", DELOITTE, 47)],
    ),
    (
        "We keep hearing Accenture is everywhere at HHS. True?",
        "The premise overstates it. Accenture Federal Services shows four HHS "
        "components, and their subcontracting is concentrated almost entirely at "
        "one of them -- 40 of their 45 partners are CMS.",
        "Not really. Four HHS components on record -- CMS, FDA, HRSA and NIH -- "
        "and the teaming is lopsided: 40 of their 45 subcontract relationships "
        "are CMS work. FDA shows three, NIH two.\n\n"
        "They are a CMS shop with a few outposts, not a department-wide presence. "
        "If your target is CMS that matters a great deal; anywhere else it "
        "matters much less than the reputation suggests.",
        [("agencies", ACN, 4),
         ("partners", ACN, 45),
         ("team_size", ACN, CMS, 40)],
    ),
    (
        "How many subcontractors does GDIT use at FDA?",
        "A small number, and small enough that the count is the whole answer -- "
        "five distinct subcontractors, with only Sraaya appearing more than once.",
        "Five, of which one -- Sraaya -- appears on more than one award.\n\n"
        "That is thin enough to be worth reading as: GDIT does not routinely "
        "subcontract its FDA work.",
        [("team_size", GDIT, FDA, 5),
         ("used", GDIT, "SRAAYA", 2, FDA)],
    ),
    (
        "Rank RTI's CDC subs for us and say what you're ranking on.",
        "The only ranking signal in public data is repeat use, so that is what "
        "this ranks on, and the answer should say so rather than implying a "
        "quality judgement it cannot support.",
        "Ranked by how many times RTI went back to them, which is the only "
        "performance-adjacent signal that is public:\n\n"
        "1. HR Directions - 10 awards\n"
        "2. Patient Performance - 4\n"
        "3. Gerling & Associates - 3\n"
        "4. Docusource of North Carolina - 3\n\n"
        "What this is not: a quality ranking. CPARS is not public. Repeat hiring "
        "is revealed preference and it is the best available substitute, but a "
        "sub used once may have been excellent and simply not needed again.",
        [("used", RTI, "HR DIRECTIONS", 10, CDC),
         ("used", RTI, "PATIENT PERFORMANCE", 4, CDC)],
    ),
    (
        "Which is the better route into CMS -- Accenture or GDIT?",
        "Both are real CMS primes with substantial benches, so the comparison "
        "turns on how they subcontract rather than whether they do. GDIT reports "
        "63 CMS subcontractors to Accenture's 40, but GDIT's is dominated by "
        "product resale.",
        "GDIT has the larger bench -- 63 CMS subcontractors against Accenture's "
        "40 -- but the shapes differ and Accenture may be the better route.\n\n"
        "GDIT's top relationships are Carahsoft (28 awards), Snowflake, TVAR and "
        "CDW Government: that is licence and hardware pass-through, and it is hard "
        "to enter unless you sell a product. Accenture's are M9 Solutions, Sky "
        "Technologies and ManpowerGroup Public Sector, which reads as staffing and "
        "services.\n\n"
        "If you are selling capability rather than software, Accenture's pattern "
        "is the one you fit.",
        [("team_size", GDIT, CMS, 63),
         ("team_size", ACN, CMS, 40),
         ("used", GDIT, "CARAHSOFT TECHNOLOGY SOLUTIONS", 28, CMS)],
    ),
    (
        "Does RTI work SAMHSA?",
        "Yes, and substantially -- 31 distinct subcontractors there, their second "
        "largest teaming footprint after CDC.",
        "Yes -- 31 distinct subcontractors, their second largest teaming "
        "footprint after CDC. The names that recur:\n\n"
        "- HR Directions (6 awards)\n"
        "- Headway HR Solutions (5)\n"
        "- Perfect Gift (5)\n"
        "- DESA (3)",
        [("team_size", RTI, SAMHSA, 31),
         ("used", RTI, "HR DIRECTIONS", 6, SAMHSA)],
    ),
    (
        "Who should we talk to about NIH work at Deloitte?",
        "Deloitte's NIH teaming is shallow -- 17 subcontractors, none used more "
        "than twice. There is no dominant partner to displace, which cuts both "
        "ways.",
        "There is no incumbent to go through, which is the useful finding.\n\n"
        "1. Oracle America - 2 awards\n"
        "2. Scientific Consulting Group - 2\n"
        "3. Intellipoint Consulting - 2\n"
        "4. Meerutech - 2\n\n"
        "Seventeen subcontractors and nobody used more than twice. No gatekeeper "
        "to get past, and equally no warm introduction available -- you would be "
        "approaching cold.",
        [("team_size", DELOITTE, NIH, 17)],
    ),
    (
        "Summarise GDIT for me in two sentences.",
        "Two sentences was the instruction, so the answer is two sentences. The "
        "load-bearing facts are the breadth (seven components) and the "
        "concentration (CMS dominates the teaming).",
        "GDIT works seven HHS components and reports 90 distinct subcontract "
        "partners, the widest footprint in this data. Its teaming is concentrated "
        "at CMS, where 63 of those relationships sit and a single reseller "
        "accounts for 28 awards.",
        [("agencies", GDIT, 7),
         ("partners", GDIT, 90),
         ("team_size", GDIT, CMS, 63)],
    ),
    (
        "Our CEO says we should chase whoever has the most HHS subcontractors. "
        "Who is that, and is he right?",
        "RTI, at 131 distinct partners. But the strategy behind the question is "
        "weak -- partner count measures how much a prime subcontracts, not how "
        "reachable it is or whether it needs what we sell.",
        "RTI, with 131 distinct subcontract partners across seven components.\n\n"
        "On whether that is the right target: partner count tells you a prime "
        "subcontracts a lot, which is necessary but not sufficient. It says "
        "nothing about whether they need your capability, whether their existing "
        "partners already cover it, or whether you can reach them. GDIT's 90 "
        "partners look comparable and are largely product resellers -- a very "
        "different door.\n\n"
        "Better filter: primes who subcontract *and* have a gap you fill.",
        [("partners", RTI, 131),
         ("partners", GDIT, 90)],
    ),
    (
        "True or false: HR Directions only works with RTI at CDC.",
        "False as stated. HR Directions appears with RTI at CDC ten times, but "
        "also at SAMHSA six times and CMS three -- it is a relationship across "
        "RTI's portfolio, not a CDC-specific one.",
        "False. HR Directions works with RTI across at least three components -- "
        "ten CDC awards, six at SAMHSA, three at CMS.\n\n"
        "That distinction matters for how you read the relationship: it is an RTI "
        "relationship that shows up wherever RTI works, not a CDC speciality.",
        [("used", RTI, "HR DIRECTIONS", 10, CDC),
         ("used", RTI, "HR DIRECTIONS", 6, SAMHSA)],
    ),
)


def all_authored() -> tuple[tuple[str, str, str, list[tuple]], ...]:
    """Prose examples plus the slate ones, which live in their own module.

    Split only to keep each file readable; they are one corpus and are checked
    by one test.
    """
    from .authored_slates import SLATES

    return (*AUTHORED, *SLATES)


def authored_examples(repeat: int = 1) -> list[dict[str, Any]]:
    """The hand-written set, in corpus form.

    ``repeat`` exists because these are few and carry the style the rest of the
    corpus should be judged against; showing them more than once is cheaper than
    writing a thousand of them, and unlike a generated template each repetition
    is of prose that was actually reasoned rather than assembled.
    """
    rows: list[dict[str, Any]] = []
    for _ in range(repeat):
        for question, reasoning, answer, _facts in all_authored():
            rows.append(
                {
                    "question": question,
                    "reasoning": reasoning,
                    "answer": answer,
                    "context": "",
                    "meta": {
                        "archetype": "authored",
                        "gold": [],
                        "tiers": {},
                        "closed_book": True,
                        "authored": True,
                    },
                }
            )
    return rows
