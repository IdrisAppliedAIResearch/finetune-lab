"""Batch two of the open-book examples: more primes, and the rest of HHS.

``authored_context.py`` covers six primes at four components, weighted towards
large IT integrators, because those are where the traps are sharpest. That is a
narrow slice of the graph to generalise from -- 1,793 firms across nine
components -- and a model that has only seen hosting contractors ranked by award
count has been taught a domain rather than a method.

This widens it to ten more primes across every component in the data, including
the four that batch one never touched: AHRQ, ASPR, HRSA and IHS. The primes are
deliberately unlike each other -- a hospital association, a children's research
hospital, a defence integrator, a construction firm, two universities, a
professional association that passes funds to state governments -- because the
right answer looks different in each and the shape of a good roster is part of
what the answer has to explain.

Three lessons here are only available at this width:

* **Tier is a property of the pairing, not of the company.** COX CONSTRUCTION
  is a tier-1 hard negative for Chugach at NIH in batch one. Here it is the
  prime, and CHUGACH WORLD SERVICES, KENT ISLAND MECHANICAL and JOHNSON
  CONTROLS -- Chugach's own deepest partners -- are the hard negatives against
  it. Same companies, same codes, opposite answers. No template can produce
  that pair; it needs two questions written to disagree on purpose.
* **The ranking signal is sometimes absent.** HEALTH RESEARCH AND EDUCATIONAL
  TRUST has 38 subcontractors and uses none of them more than twice; COX
  CONSTRUCTION has 10 and uses none twice. Every generated archetype ranks by
  repeat use, so the corpus has never shown a model the case where that signal
  does not exist and the honest answer is to say so.
* **The same entity can appear twice.** St. Jude's largest research partner is
  filed as both UNIVERSITY OF HONG KONG and UNIVERSITY OF HONG KONG, THE. Read
  separately they are 7 and 5 awards; read together they are 12 and the ranking
  changes.

Conventions are batch one's, unchanged: candidates named in the answer exactly
as the slate spells them, tiers declared and checked against ``tier_for``,
counts checked against the graph.
"""

from __future__ import annotations

from .authored import CDC, CMS, FDA, GDIT, NIH, SAMHSA
from .authored_context import CHUGACH, OpenBook

AHRQ = "Agency for Healthcare Research and Quality"
ASPR = "Office of Assistant Secretary for Preparedness and Response"
HRSA = "Health Resources and Services Administration"
IHS = "Indian Health Service"

HRET = "HEALTH RESEARCH AND EDUCATIONAL TRUST"
STJUDE = "ST. JUDE CHILDREN'S RESEARCH HOSPITAL"
NGIT = "NORTHROP GRUMMAN INFORMATION TECHNOLOGY"
COX = "COX CONSTRUCTION"
UAB = "UNIVERSITY OF ALABAMA AT BIRMINGHAM"
BU = "TRUSTEES OF BOSTON UNIVERSITY"
ACN = "ACCENTURE FEDERAL SERVICES"
DELOITTE = "DELOITTE CONSULTING LLP"
NASMHPD = "NATIONAL ASSOCIATION OF STATE MENTAL HEALTH PROGRAM DIRECTORS"
NGS = "NATIONAL GOVERNMENT SERVICES"

WIDE: tuple[OpenBook, ...] = (
    # ------------------------------------------------------------------
    # AHRQ, and a roster flat enough that ranking is not available. Every
    # generated archetype ranks by repeat use; none can produce this.
    # ------------------------------------------------------------------
    OpenBook(
        ask="HRET runs a lot of AHRQ work. Rank these for a teaming approach.",
        prime=HRET,
        agency=AHRQ,
        slate=(
            "RESEARCH TRIANGLE INSTITUTE",
            "SOUTH CAROLINA HOSPITAL ASSOC",
            "DELOITTE CONSULTING LLP",
            "SOCIETY OF HOSPITAL MEDICINE",
            "NORTHWELL HEALTH",
            "BATTELLE MEMORIAL INSTITUTE",
            "METROHEALTH SYSTEM, THE",
        ),
        tiers={
            "RESEARCH TRIANGLE INSTITUTE": 1,
            "SOUTH CAROLINA HOSPITAL ASSOC": 4,
            "DELOITTE CONSULTING LLP": 1,
            "SOCIETY OF HOSPITAL MEDICINE": 4,
            "NORTHWELL HEALTH": 4,
            "BATTELLE MEMORIAL INSTITUTE": 1,
            "METROHEALTH SYSTEM, THE": 4,
        },
        reasoning="HRET reports 38 distinct AHRQ subcontractors and uses none of "
        "them more than twice. Repeat use is the only ranking signal available in "
        "public data, and here it does not vary, so a ranking would be invented. "
        "Saying that is more useful than producing an order.",
        answer="Thirty-eight subcontractors and not one used more than twice -- so "
        "I can tell you who is on the roster, and I cannot rank them, because the "
        "signal a ranking would use does not vary here.\n\n"
        "On the roster: SOUTH CAROLINA HOSPITAL ASSOC and SOCIETY OF HOSPITAL "
        "MEDICINE at two reported AHRQ awards each, NORTHWELL HEALTH and "
        "METROHEALTH SYSTEM, THE at one. Ordering those four would dress a "
        "one-award difference as a preference.\n\n"
        "Not recommended: RESEARCH TRIANGLE INSTITUTE, DELOITTE CONSULTING LLP "
        "and BATTELLE MEMORIAL INSTITUTE. All three match HRET on its 541611 and "
        "541720 codes and dwarf it in scale; none has a reported subcontract with "
        "it.\n\n"
        "The useful read here is structural. HRET spreads work thin across "
        "hospital associations and professional societies instead of building a "
        "core bench, so there is no incumbent to displace and no warm route in. "
        "Everyone approaches roughly cold, which is better news for you than a "
        "ranking would have been.",
        gold=(
            "SOUTH CAROLINA HOSPITAL ASSOC",
            "SOCIETY OF HOSPITAL MEDICINE",
            "NORTHWELL HEALTH",
            "METROHEALTH SYSTEM, THE",
        ),
        facts=(
            ("team_size", HRET, AHRQ, 38),
            ("used", HRET, "SOUTH CAROLINA HOSPITAL ASSOC", 2, AHRQ),
            ("used", HRET, "SOCIETY OF HOSPITAL MEDICINE", 2, AHRQ),
            ("used", HRET, "NORTHWELL HEALTH", 1, AHRQ),
            ("used", HRET, "METROHEALTH SYSTEM, THE", 1, AHRQ),
            ("not_used", HRET, "RESEARCH TRIANGLE INSTITUTE", AHRQ),
            ("not_used", HRET, "DELOITTE CONSULTING LLP", AHRQ),
            ("not_used", HRET, "BATTELLE MEMORIAL INSTITUTE", AHRQ),
        ),
    ),
    # ------------------------------------------------------------------
    # One institution filed under two names, both on the slate. Reading them
    # separately gets the ranking wrong.
    # ------------------------------------------------------------------
    OpenBook(
        ask="St. Jude has new NIH collaborations coming. Who from this list has "
        "actually worked with them?",
        prime=STJUDE,
        agency=NIH,
        slate=(
            "SEATTLE CHILDREN'S HOSPITAL",
            "UNIVERSITY OF HONG KONG",
            "LEIDOS BIOMEDICAL RESEARCH",
            "REGENTS OF THE UNIVERSITY OF MICHIGAN",
            "UNIVERSITY OF HONG KONG, THE",
            "OHIO STATE UNIVERSITY, THE",
        ),
        tiers={
            "SEATTLE CHILDREN'S HOSPITAL": 1,
            "UNIVERSITY OF HONG KONG": 4,
            "LEIDOS BIOMEDICAL RESEARCH": 1,
            "REGENTS OF THE UNIVERSITY OF MICHIGAN": 4,
            "UNIVERSITY OF HONG KONG, THE": 4,
            "OHIO STATE UNIVERSITY, THE": 4,
        },
        reasoning="Entries 2 and 5 are the same university filed under two names, "
        "at 7 and 5 awards. Counted separately Michigan appears to tie for second; "
        "counted together Hong Kong is the dominant partner at 12 and nothing else "
        "is close. The answer has to say which reading it is using.",
        answer="1. UNIVERSITY OF HONG KONG - 7 reported St. Jude awards at NIH\n"
        "2. REGENTS OF THE UNIVERSITY OF MICHIGAN - 5\n"
        "3. UNIVERSITY OF HONG KONG, THE - 5\n"
        "4. OHIO STATE UNIVERSITY, THE - 2\n\n"
        "Entries 1 and 3 are the same institution under two filings. Read together "
        "that is twelve awards, which makes Hong Kong St. Jude's largest research "
        "partner by a wide margin and puts REGENTS OF THE UNIVERSITY OF MICHIGAN "
        "clearly second. The order above is an artefact of how the awards were "
        "filed; I have left both entries in because both are on your list and you "
        "will meet both in the data.\n\n"
        "Not recommended: SEATTLE CHILDREN'S HOSPITAL and LEIDOS BIOMEDICAL "
        "RESEARCH. A peer children's research hospital and a large NIH-native life "
        "sciences contractor -- both plausible on any capability read, neither "
        "with a reported St. Jude subcontract.",
        gold=(
            "UNIVERSITY OF HONG KONG",
            "REGENTS OF THE UNIVERSITY OF MICHIGAN",
            "UNIVERSITY OF HONG KONG, THE",
            "OHIO STATE UNIVERSITY, THE",
        ),
        facts=(
            ("used", STJUDE, "UNIVERSITY OF HONG KONG", 7, NIH),
            ("used", STJUDE, "UNIVERSITY OF HONG KONG, THE", 5, NIH),
            ("used", STJUDE, "REGENTS OF THE UNIVERSITY OF MICHIGAN", 5, NIH),
            ("used", STJUDE, "OHIO STATE UNIVERSITY, THE", 2, NIH),
            ("not_used", STJUDE, "SEATTLE CHILDREN'S HOSPITAL", NIH),
            ("not_used", STJUDE, "LEIDOS BIOMEDICAL RESEARCH", NIH),
        ),
    ),
    # ------------------------------------------------------------------
    # A defence integrator at CDC: one staffing partner and a thin tail.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Northrop Grumman IT at CDC -- who's on the bench?",
        prime=NGIT,
        agency=CDC,
        slate=(
            "ACCENTURE FEDERAL SERVICES",
            "TEKSYSTEMS",
            "THE ST. JOHN GROUP",
            "BOOZ ALLEN HAMILTON",
            "THE GINN GROUP",
            "IT1 SOURCE",
            "DELOITTE CONSULTING LLP",
        ),
        tiers={
            "ACCENTURE FEDERAL SERVICES": 1,
            "TEKSYSTEMS": 4,
            "THE ST. JOHN GROUP": 4,
            "BOOZ ALLEN HAMILTON": 1,
            "THE GINN GROUP": 4,
            "IT1 SOURCE": 4,
            "DELOITTE CONSULTING LLP": 1,
        },
        reasoning="Fourteen subcontractors and only TEKSYSTEMS used more than "
        "twice. The bench is staff augmentation rather than capability teaming, "
        "which is the finding worth handing over -- it changes what a pitch to "
        "this prime should be about.",
        answer="TEKSYSTEMS, and then a long thin tail:\n\n"
        "- TEKSYSTEMS - 5 reported Northrop Grumman IT awards at CDC\n"
        "- THE ST. JOHN GROUP - 2\n"
        "- THE GINN GROUP - 2\n"
        "- IT1 SOURCE - 1\n\n"
        "Fourteen subcontractors in total and one of them used more than twice. "
        "That is a prime buying staff augmentation rather than assembling a "
        "partner bench, and it tells you what kind of pitch lands: people against "
        "a labour category, not a capability you own.\n\n"
        "Not recommended: ACCENTURE FEDERAL SERVICES, BOOZ ALLEN HAMILTON and "
        "DELOITTE CONSULTING LLP. All three carry the 541512 systems-design "
        "code and work HHS at scale, and none has ever subcontracted to this "
        "prime -- at CDC they are bidding against it, not with it.",
        gold=("TEKSYSTEMS", "THE ST. JOHN GROUP", "THE GINN GROUP", "IT1 SOURCE"),
        facts=(
            ("team_size", NGIT, CDC, 14),
            ("used", NGIT, "TEKSYSTEMS", 5, CDC),
            ("used", NGIT, "THE ST. JOHN GROUP", 2, CDC),
            ("used", NGIT, "THE GINN GROUP", 2, CDC),
            ("used", NGIT, "IT1 SOURCE", 1, CDC),
            ("not_used", NGIT, "ACCENTURE FEDERAL SERVICES", CDC),
            ("not_used", NGIT, "BOOZ ALLEN HAMILTON", CDC),
            ("not_used", NGIT, "DELOITTE CONSULTING LLP", CDC),
        ),
    ),
    # ------------------------------------------------------------------
    # The inversion. Chugach's core NIH bench are this prime's hard negatives,
    # and the batch-one example says the opposite about the same companies.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Cox Construction has Indian Health Service work coming. Which of these "
        "would they bring on?",
        prime=COX,
        agency=IHS,
        slate=(
            "CHUGACH WORLD SERVICES",
            "KITCOR",
            "KENT ISLAND MECHANICAL",
            "A & P DRYWALL",
            "JOHNSON CONTROLS",
            "FORMS & SURFACES",
            "WHITE'S STEEL",
            "POLU KAI SERVICES",
        ),
        tiers={
            "CHUGACH WORLD SERVICES": 1,
            "KITCOR": 4,
            "KENT ISLAND MECHANICAL": 1,
            "A & P DRYWALL": 4,
            "JOHNSON CONTROLS": 1,
            "FORMS & SURFACES": 4,
            "WHITE'S STEEL": 4,
            "POLU KAI SERVICES": 1,
        },
        reasoning="Four have Cox awards and four do not, and the four that do not "
        "are the strongest names on the list -- KENT ISLAND MECHANICAL and JOHNSON "
        "CONTROLS are Chugach's two deepest NIH relationships. Same industry code, "
        "same federal health construction work, no relationship with this prime. "
        "The point worth making is that their tier changed because the question "
        "changed.",
        answer="Four, and all four on equal footing: KITCOR, A & P DRYWALL, FORMS "
        "& SURFACES and WHITE'S STEEL, each with one reported Cox award at Indian "
        "Health Service. Ten subcontractors in total, none used twice, so there is "
        "no repeat-use ordering to give you.\n\n"
        "Not recommended: CHUGACH WORLD SERVICES, KENT ISLAND MECHANICAL, JOHNSON "
        "CONTROLS and POLU KAI SERVICES. Worth spelling out why, because these are "
        "not weak names -- KENT ISLAND MECHANICAL and JOHNSON CONTROLS are the two "
        "deepest subcontract relationships CHUGACH WORLD SERVICES has at NIH, at "
        "21 and 12 awards. They carry the same 236220 building-systems code as "
        "Cox and they do federal health construction. None has ever appeared on a "
        "Cox award.\n\n"
        "Which is the general point: a tier is a fact about the pairing, not about "
        "the company. The same four names that answer \"who does Chugach use at "
        "NIH\" are the wrong answer here, and nothing about them changed.",
        gold=("KITCOR", "A & P DRYWALL", "FORMS & SURFACES", "WHITE'S STEEL"),
        facts=(
            ("team_size", COX, IHS, 10),
            ("used", COX, "KITCOR", 1, IHS),
            ("used", COX, "A & P DRYWALL", 1, IHS),
            ("used", COX, "FORMS & SURFACES", 1, IHS),
            ("used", COX, "WHITE'S STEEL", 1, IHS),
            ("not_used", COX, "CHUGACH WORLD SERVICES", IHS),
            ("not_used", COX, "KENT ISLAND MECHANICAL", IHS),
            ("not_used", COX, "JOHNSON CONTROLS", IHS),
            ("not_used", COX, "POLU KAI SERVICES", IHS),
            ("used", CHUGACH, "KENT ISLAND MECHANICAL", 21, NIH),
            ("used", CHUGACH, "JOHNSON CONTROLS", 12, NIH),
        ),
    ),
    # ------------------------------------------------------------------
    # A university prime, and a four-way tie the answer has to admit to.
    # ------------------------------------------------------------------
    OpenBook(
        ask="UAB is prime on a CDC award. Rank these as potential subs.",
        prime=UAB,
        agency=CDC,
        slate=(
            "RESEARCH TRIANGLE INSTITUTE",
            "THE JOHNS HOPKINS UNIVERSITY",
            "DELOITTE CONSULTING LLP",
            "THE ADMINISTRATORS OF TULANE EDUCATIONAL FUND",
            "THE UNIVERSITY OF CHICAGO",
            "FENWAY COMMUNITY HEALTH CENTER",
            "BATTELLE MEMORIAL INSTITUTE",
        ),
        tiers={
            "RESEARCH TRIANGLE INSTITUTE": 1,
            "THE JOHNS HOPKINS UNIVERSITY": 4,
            "DELOITTE CONSULTING LLP": 1,
            "THE ADMINISTRATORS OF TULANE EDUCATIONAL FUND": 4,
            "THE UNIVERSITY OF CHICAGO": 4,
            "FENWAY COMMUNITY HEALTH CENTER": 4,
            "BATTELLE MEMORIAL INSTITUTE": 1,
        },
        reasoning="Four candidates at three awards each is a tie, not a ranking, "
        "and presenting it as a ranking would invent a preference the data does "
        "not contain. The least prominent of the four has exactly as much history "
        "as the most prominent.",
        answer="1. THE JOHNS HOPKINS UNIVERSITY - 3 reported UAB awards at CDC\n"
        "2. THE ADMINISTRATORS OF TULANE EDUCATIONAL FUND - 3\n"
        "3. THE UNIVERSITY OF CHICAGO - 3\n"
        "4. FENWAY COMMUNITY HEALTH CENTER - 3\n\n"
        "That is a four-way tie and the order above is arbitrary. Nothing in the "
        "record separates them, and FENWAY COMMUNITY HEALTH CENTER -- the least "
        "prominent name of the four -- has exactly the same history with this "
        "prime as Johns Hopkins does. If you need a first call, choose on what the "
        "work needs rather than on this list.\n\n"
        "Not recommended: RESEARCH TRIANGLE INSTITUTE, DELOITTE CONSULTING LLP and "
        "BATTELLE MEMORIAL INSTITUTE. All three are substantial CDC contractors in "
        "their own right, all three overlap UAB's research codes, and none has "
        "subcontracted to it.",
        gold=(
            "THE JOHNS HOPKINS UNIVERSITY",
            "THE ADMINISTRATORS OF TULANE EDUCATIONAL FUND",
            "THE UNIVERSITY OF CHICAGO",
            "FENWAY COMMUNITY HEALTH CENTER",
        ),
        facts=(
            ("used", UAB, "THE JOHNS HOPKINS UNIVERSITY", 3, CDC),
            ("used", UAB, "THE ADMINISTRATORS OF TULANE EDUCATIONAL FUND", 3, CDC),
            ("used", UAB, "THE UNIVERSITY OF CHICAGO", 3, CDC),
            ("used", UAB, "FENWAY COMMUNITY HEALTH CENTER", 3, CDC),
            ("not_used", UAB, "RESEARCH TRIANGLE INSTITUTE", CDC),
            ("not_used", UAB, "DELOITTE CONSULTING LLP", CDC),
            ("not_used", UAB, "BATTELLE MEMORIAL INSTITUTE", CDC),
        ),
    ),
    # ------------------------------------------------------------------
    # ASPR, and a roster made of laboratories rather than contractors.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Boston University, ASPR work. Anyone here they've actually used?",
        prime=BU,
        agency=ASPR,
        slate=(
            "RESEARCH TRIANGLE INSTITUTE",
            "RHODE ISLAND HOSPITAL",
            "UH-OH LABS",
            "LEIDOS BIOMEDICAL RESEARCH",
            "DUKE UNIVERSITY",
            "PRESIDENT AND FELLOWS OF HARVARD COLLEGE",
            "BOOZ ALLEN HAMILTON",
        ),
        tiers={
            "RESEARCH TRIANGLE INSTITUTE": 1,
            "RHODE ISLAND HOSPITAL": 4,
            "UH-OH LABS": 4,
            "LEIDOS BIOMEDICAL RESEARCH": 1,
            "DUKE UNIVERSITY": 4,
            "PRESIDENT AND FELLOWS OF HARVARD COLLEGE": 4,
            "BOOZ ALLEN HAMILTON": 1,
        },
        reasoning="Four of the seven, none more than twice. What matters more than "
        "the ordering is what the roster is made of: hospitals, university labs "
        "and small biotechs on infectious-disease codes, several outside the "
        "United States. The three rejected names are federal services contractors, "
        "which is the wrong kind of organisation as well as the wrong record.",
        answer="Two at two awards each and two more at one: RHODE ISLAND HOSPITAL "
        "and UH-OH LABS, then DUKE UNIVERSITY and PRESIDENT AND FELLOWS OF HARVARD "
        "COLLEGE.\n\n"
        "Worth understanding the roster before you pitch into it. BU's ASPR "
        "subcontracting runs to hospitals, university laboratories and small "
        "biotechs on infectious-disease research codes, several of them outside "
        "the United States, and eighteen names are on it with almost none repeated. "
        "This is a research collaboration network rather than a contracting bench, "
        "and it is joined by having a laboratory rather than a contract vehicle.\n\n"
        "Not recommended: RESEARCH TRIANGLE INSTITUTE, LEIDOS BIOMEDICAL RESEARCH "
        "and BOOZ ALLEN HAMILTON. Large federal health contractors with no "
        "reported work under this prime -- and the wrong shape for this roster "
        "even if the record were silent rather than empty.",
        gold=(
            "RHODE ISLAND HOSPITAL",
            "UH-OH LABS",
            "DUKE UNIVERSITY",
            "PRESIDENT AND FELLOWS OF HARVARD COLLEGE",
        ),
        facts=(
            ("team_size", BU, ASPR, 18),
            ("used", BU, "RHODE ISLAND HOSPITAL", 2, ASPR),
            ("used", BU, "UH-OH LABS", 2, ASPR),
            ("used", BU, "DUKE UNIVERSITY", 1, ASPR),
            ("used", BU, "PRESIDENT AND FELLOWS OF HARVARD COLLEGE", 1, ASPR),
            ("not_used", BU, "RESEARCH TRIANGLE INSTITUTE", ASPR),
            ("not_used", BU, "LEIDOS BIOMEDICAL RESEARCH", ASPR),
            ("not_used", BU, "BOOZ ALLEN HAMILTON", ASPR),
        ),
    ),
    # ------------------------------------------------------------------
    # Accenture at CMS. The top of this bench is the set of names the
    # closed-book slates reject for RTI at CDC -- right company, wrong customer
    # there, right customer here.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Accenture Federal Services, CMS. Rank these for a teaming approach.",
        prime=ACN,
        agency=CMS,
        slate=(
            "M9 SOLUTIONS",
            "LEIDOS",
            "SKY TECHNOLOGIES",
            "CGI FEDERAL",
            "CARAHSOFT TECHNOLOGY",
            "MANPOWERGROUP PUBLIC SECTOR",
            "ICF",
        ),
        tiers={
            "M9 SOLUTIONS": 4,
            "LEIDOS": 1,
            "SKY TECHNOLOGIES": 4,
            "CGI FEDERAL": 1,
            "CARAHSOFT TECHNOLOGY": 4,
            "MANPOWERGROUP PUBLIC SECTOR": 4,
            "ICF": 1,
        },
        reasoning="Four have reported Accenture CMS work and the top of that list "
        "is staffing and services rather than product resale, which is the "
        "practical contrast with the other large CMS primes. The three rejected "
        "names all share Accenture's codes and have no subcontract with it.",
        answer="1. M9 SOLUTIONS - 5 reported Accenture awards at CMS\n"
        "2. SKY TECHNOLOGIES - 4\n"
        "3. CARAHSOFT TECHNOLOGY - 4\n"
        "4. MANPOWERGROUP PUBLIC SECTOR - 3\n\n"
        "The useful thing about that list is its shape. The top of Accenture's CMS "
        "bench is staffing and services rather than licence resale, which is the "
        "opposite of what the other large CMS primes look like -- so if you sell "
        "capability rather than product, this is the bench you can plausibly "
        "join.\n\n"
        "Not recommended: LEIDOS, CGI FEDERAL and ICF. All three carry Accenture's "
        "consulting or systems-design codes and real federal health volume, and "
        "none has a reported subcontract with Accenture at CMS.",
        gold=(
            "M9 SOLUTIONS",
            "SKY TECHNOLOGIES",
            "CARAHSOFT TECHNOLOGY",
            "MANPOWERGROUP PUBLIC SECTOR",
        ),
        facts=(
            ("used", ACN, "M9 SOLUTIONS", 5, CMS),
            ("used", ACN, "SKY TECHNOLOGIES", 4, CMS),
            ("used", ACN, "CARAHSOFT TECHNOLOGY", 4, CMS),
            ("used", ACN, "MANPOWERGROUP PUBLIC SECTOR", 3, CMS),
            ("not_used", ACN, "LEIDOS", CMS),
            ("not_used", ACN, "CGI FEDERAL", CMS),
            ("not_used", ACN, "ICF", CMS),
        ),
    ),
    # ------------------------------------------------------------------
    # The second off-slate example, on a prime batch one never used: a company
    # in the records with a better claim than half the slate, still not an
    # answer, because it was not offered.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Accenture at CMS -- who from this list, and say if the records point "
        "anywhere else.",
        prime=ACN,
        agency=CMS,
        slate=(
            "M9 SOLUTIONS",
            "LEIDOS",
            "SKY TECHNOLOGIES",
            "CGI FEDERAL",
            "CARAHSOFT TECHNOLOGY",
            "MANPOWERGROUP PUBLIC SECTOR",
            "ICF",
        ),
        tiers={
            "M9 SOLUTIONS": 4,
            "LEIDOS": 1,
            "SKY TECHNOLOGIES": 4,
            "CGI FEDERAL": 1,
            "CARAHSOFT TECHNOLOGY": 4,
            "MANPOWERGROUP PUBLIC SECTOR": 4,
            "ICF": 1,
        },
        reasoning="Same four picks as the ranking version. The second half of the "
        "ask invites additions, and the retrieved records do contain Accenture CMS "
        "subcontractors that are not on the slate -- so the honest move is to name "
        "one and be explicit that it is not a pick, because the question was which "
        "of these.",
        answer="From your list: M9 SOLUTIONS (5 reported Accenture awards at CMS), "
        "SKY TECHNOLOGIES (4), CARAHSOFT TECHNOLOGY (4) and MANPOWERGROUP PUBLIC "
        "SECTOR (3).\n\n"
        "You asked whether the records point elsewhere, and they do. The library "
        "in front of me also carries KFORCE, which has three reported Accenture "
        "subcontracts at CMS -- level with MANPOWERGROUP PUBLIC SECTOR and ahead "
        "of three of the names you did offer. It is not one of your options, so I "
        "have not counted it as a pick; a record appearing in the library is not a "
        "nomination, and the question was which of these. Put it on the next "
        "version of the list.\n\n"
        "Not recommended: LEIDOS, CGI FEDERAL and ICF -- Accenture's codes, "
        "Accenture's customer, no Accenture subcontract.",
        gold=(
            "M9 SOLUTIONS",
            "SKY TECHNOLOGIES",
            "CARAHSOFT TECHNOLOGY",
            "MANPOWERGROUP PUBLIC SECTOR",
        ),
        facts=(
            ("used", ACN, "M9 SOLUTIONS", 5, CMS),
            ("used", ACN, "SKY TECHNOLOGIES", 4, CMS),
            ("used", ACN, "CARAHSOFT TECHNOLOGY", 4, CMS),
            ("used", ACN, "MANPOWERGROUP PUBLIC SECTOR", 3, CMS),
            ("used", ACN, "KFORCE", 3, CMS),
            ("not_used", ACN, "LEIDOS", CMS),
            ("not_used", ACN, "CGI FEDERAL", CMS),
            ("not_used", ACN, "ICF", CMS),
        ),
        off_slate_named=("KFORCE",),
    ),
    # ------------------------------------------------------------------
    # A prime that mostly does not subcontract. The answer is about the
    # practice, not about the list.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Deloitte's FDA practice -- who do they sub to?",
        prime=DELOITTE,
        agency=FDA,
        slate=(
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY",
            "PROFICIENT BUSINESS SYSTEMS",
            "ACCENTURE FEDERAL SERVICES",
            "DLT SOLUTIONS",
            "NEEVSYS",
            "BATTELLE MEMORIAL INSTITUTE",
        ),
        tiers={
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY": 1,
            "PROFICIENT BUSINESS SYSTEMS": 4,
            "ACCENTURE FEDERAL SERVICES": 1,
            "DLT SOLUTIONS": 4,
            "NEEVSYS": 4,
            "BATTELLE MEMORIAL INSTITUTE": 1,
        },
        reasoning="Three of the six, and all of them thinly. The number that "
        "actually answers the question is the practice-wide one: thirteen distinct "
        "FDA subcontractors in total, which says Deloitte keeps most of its FDA "
        "delivery in-house and makes teaming a weak route in.",
        answer="Three of the six, and thinly. PROFICIENT BUSINESS SYSTEMS is the "
        "only one with more than a single award, at two; DLT SOLUTIONS and NEEVSYS "
        "have one each.\n\n"
        "The number worth taking away is not from your list, though. Deloitte "
        "reports thirteen distinct subcontractors across all of its FDA work. That "
        "is a prime keeping delivery in-house, so teaming is a weaker route into "
        "its FDA business than it would be into its CMS business, and the "
        "shortlist matters less than the approach does.\n\n"
        "Not recommended: GENERAL DYNAMICS INFORMATION TECHNOLOGY, ACCENTURE "
        "FEDERAL SERVICES and BATTELLE MEMORIAL INSTITUTE. Each is a major FDA "
        "contractor in its own right and each competes with Deloitte rather than "
        "supplying it; none has a reported subcontract with them.",
        gold=("PROFICIENT BUSINESS SYSTEMS", "DLT SOLUTIONS", "NEEVSYS"),
        facts=(
            ("team_size", DELOITTE, FDA, 13),
            ("used", DELOITTE, "PROFICIENT BUSINESS SYSTEMS", 2, FDA),
            ("used", DELOITTE, "DLT SOLUTIONS", 1, FDA),
            ("used", DELOITTE, "NEEVSYS", 1, FDA),
            ("not_used", DELOITTE, "GENERAL DYNAMICS INFORMATION TECHNOLOGY", FDA),
            ("not_used", DELOITTE, "ACCENTURE FEDERAL SERVICES", FDA),
            ("not_used", DELOITTE, "BATTELLE MEMORIAL INSTITUTE", FDA),
        ),
    ),
    # ------------------------------------------------------------------
    # Same prime as batch one's FDA example, a third component, and a bench
    # that looks nothing like its CMS one.
    # ------------------------------------------------------------------
    OpenBook(
        ask="GDIT at HRSA. Same playbook as their CMS work?",
        prime=GDIT,
        agency=HRSA,
        slate=(
            "PERSPECTA ENTERPRISE SOLUTIONS",
            "TECHNOGEN",
            "NATIONAL GOVERNMENT SERVICES",
            "INNOVIEN SOLUTIONS",
            "MYTHICS",
            "CARAHSOFT TECHNOLOGY SOLUTIONS",
            "BOOZ ALLEN HAMILTON",
        ),
        tiers={
            "PERSPECTA ENTERPRISE SOLUTIONS": 1,
            "TECHNOGEN": 4,
            "NATIONAL GOVERNMENT SERVICES": 1,
            "INNOVIEN SOLUTIONS": 4,
            "MYTHICS": 4,
            "CARAHSOFT TECHNOLOGY SOLUTIONS": 4,
            "BOOZ ALLEN HAMILTON": 1,
        },
        reasoning="No. Sixteen subcontractors at HRSA against sixty-three at CMS, "
        "and the composition differs as much as the count: CARAHSOFT TECHNOLOGY "
        "SOLUTIONS carries 28 CMS awards and exactly one at HRSA, behind two "
        "staffing firms. The same prime buys differently per component.",
        answer="TECHNOGEN leads, and no -- the playbook is different:\n\n"
        "- TECHNOGEN - 3 reported GDIT awards at HRSA\n"
        "- INNOVIEN SOLUTIONS - 2\n"
        "- MYTHICS - 1\n"
        "- CARAHSOFT TECHNOLOGY SOLUTIONS - 1\n\n"
        "Sixteen subcontractors at HRSA against sixty-three at CMS, and the shape "
        "differs as much as the size. GDIT's CMS roster is dominated by resellers, "
        "with CARAHSOFT TECHNOLOGY SOLUTIONS alone on 28 awards; at HRSA that same "
        "company appears once, behind two staffing firms. Same prime, same "
        "department, different buying behaviour -- so check the component rather "
        "than assuming one bench across the agency.\n\n"
        "Not recommended: PERSPECTA ENTERPRISE SOLUTIONS, NATIONAL GOVERNMENT "
        "SERVICES and BOOZ ALLEN HAMILTON. None has a reported subcontract with "
        "GDIT at any agency, which for companies of that size is a statement "
        "about position rather than about capability.",
        gold=(
            "TECHNOGEN",
            "INNOVIEN SOLUTIONS",
            "MYTHICS",
            "CARAHSOFT TECHNOLOGY SOLUTIONS",
        ),
        facts=(
            ("team_size", GDIT, HRSA, 16),
            ("team_size", GDIT, CMS, 63),
            ("used", GDIT, "TECHNOGEN", 3, HRSA),
            ("used", GDIT, "INNOVIEN SOLUTIONS", 2, HRSA),
            ("used", GDIT, "MYTHICS", 1, HRSA),
            ("used", GDIT, "CARAHSOFT TECHNOLOGY SOLUTIONS", 1, HRSA),
            ("used", GDIT, "CARAHSOFT TECHNOLOGY SOLUTIONS", 28, CMS),
            ("used", GDIT, "PERSPECTA ENTERPRISE SOLUTIONS", 0),
            ("used", GDIT, "NATIONAL GOVERNMENT SERVICES", 0),
            ("used", GDIT, "BOOZ ALLEN HAMILTON", 0),
        ),
    ),
    # ------------------------------------------------------------------
    # The subcontractors are state governments. The commercial teaming frame
    # the question arrives in does not apply at all.
    # ------------------------------------------------------------------
    OpenBook(
        ask="NASMHPD runs SAMHSA programmes. Which of these could we approach as "
        "a partner?",
        prime=NASMHPD,
        agency=SAMHSA,
        slate=(
            "RESEARCH TRIANGLE INSTITUTE",
            "ALABAMA DEPARTMENT OF HEALTH",
            "DELOITTE CONSULTING LLP",
            "OHIO DEPARTMENT MENTAL HEALTH",
            "IOWA DEPARTMENT OF PUBLIC HEALTH",
            "LEIDOS",
        ),
        tiers={
            "RESEARCH TRIANGLE INSTITUTE": 1,
            "ALABAMA DEPARTMENT OF HEALTH": 4,
            "DELOITTE CONSULTING LLP": 1,
            "OHIO DEPARTMENT MENTAL HEALTH": 4,
            "IOWA DEPARTMENT OF PUBLIC HEALTH": 4,
            "LEIDOS": 1,
        },
        reasoning="Thirty-two subcontractors across thirty-two awards, nearly all "
        "of them state health and mental-health departments. This is programme "
        "funds passing through to states, not a supplier bench, so the question's "
        "framing is what needs correcting rather than its shortlist.",
        answer="These are two different kinds of organisation, and the framing "
        "needs more attention than the shortlist does.\n\n"
        "The three with actual NASMHPD work -- ALABAMA DEPARTMENT OF HEALTH, OHIO "
        "DEPARTMENT MENTAL HEALTH and IOWA DEPARTMENT OF PUBLIC HEALTH -- are "
        "state government agencies, each on a single reported award. That is what "
        "this roster is: thirty-two subcontractors across thirty-two awards, "
        "almost all of them state health departments receiving programme funds. It "
        "is a distribution structure, not a supplier bench, and you do not join it "
        "by being a good vendor.\n\n"
        "Not recommended: RESEARCH TRIANGLE INSTITUTE, DELOITTE CONSULTING LLP and "
        "LEIDOS. None has reported NASMHPD work -- which here tells you less than "
        "usual, since commercial firms are not what this prime subcontracts to in "
        "the first place. Their absence is a fact about the programme rather than "
        "about them.",
        gold=(
            "ALABAMA DEPARTMENT OF HEALTH",
            "OHIO DEPARTMENT MENTAL HEALTH",
            "IOWA DEPARTMENT OF PUBLIC HEALTH",
        ),
        facts=(
            ("team_size", NASMHPD, SAMHSA, 32),
            ("used", NASMHPD, "ALABAMA DEPARTMENT OF HEALTH", 1, SAMHSA),
            ("used", NASMHPD, "OHIO DEPARTMENT MENTAL HEALTH", 1, SAMHSA),
            ("used", NASMHPD, "IOWA DEPARTMENT OF PUBLIC HEALTH", 1, SAMHSA),
            ("not_used", NASMHPD, "RESEARCH TRIANGLE INSTITUTE", SAMHSA),
            ("not_used", NASMHPD, "DELOITTE CONSULTING LLP", SAMHSA),
            ("not_used", NASMHPD, "LEIDOS", SAMHSA),
        ),
    ),
    # ------------------------------------------------------------------
    # A decline where one rejected name is still worth something, so the answer
    # has to separate "no relationship" from "bad fit".
    # ------------------------------------------------------------------
    OpenBook(
        ask="Any of these worth approaching for St. Jude's NIH work?",
        prime=STJUDE,
        agency=NIH,
        slate=(
            "RESEARCH TRIANGLE INSTITUTE",
            "THE MITRE",
            "SOCIAL & SCIENTIFIC SYSTEMS",
            "PERATON",
            "SEATTLE CHILDREN'S HOSPITAL",
        ),
        tiers={
            "RESEARCH TRIANGLE INSTITUTE": 1,
            "THE MITRE": 1,
            "SOCIAL & SCIENTIFIC SYSTEMS": 1,
            "PERATON": 1,
            "SEATTLE CHILDREN'S HOSPITAL": 1,
        },
        reasoning="None of the five has a St. Jude subcontract, so no pick is "
        "supportable. But they fail differently, and collapsing that distinction "
        "would waste the useful half of the answer: four are the wrong kind of "
        "organisation, and one is the right kind with no record.",
        answer="None of the five, though they are not wrong in the same way.\n\n"
        "All five overlap St. Jude's research codes and none has a reported "
        "subcontract with it. St. Jude's NIH subcontracting goes to universities "
        "and research institutes, much of it academic and a good deal of it "
        "overseas -- so most of this list is the wrong kind of organisation before "
        "the question of relationship even arises.\n\n"
        "Not recommended: RESEARCH TRIANGLE INSTITUTE, THE MITRE, SOCIAL & "
        "SCIENTIFIC SYSTEMS and PERATON, all four large federal services "
        "contractors with nothing in this prime's record. SEATTLE CHILDREN'S "
        "HOSPITAL is the interesting one and still not a pick: a peer children's "
        "research hospital with real NIH work, and its record shows 20 awards as "
        "a prime and not one subcontract taken from anybody. It is not a cold "
        "prospect for a sub slot, it is an organisation that does not take them. "
        "If you want a relationship there, the direction that fits the record is "
        "onto its bench rather than onto yours.",
        gold=(),
        facts=(
            ("not_used", STJUDE, "RESEARCH TRIANGLE INSTITUTE", NIH),
            ("not_used", STJUDE, "THE MITRE", NIH),
            ("not_used", STJUDE, "SOCIAL & SCIENTIFIC SYSTEMS", NIH),
            ("not_used", STJUDE, "PERATON", NIH),
            ("not_used", STJUDE, "SEATTLE CHILDREN'S HOSPITAL", NIH),
            ("agency_work", "SEATTLE CHILDREN'S HOSPITAL", NIH),
            ("as_sub", "SEATTLE CHILDREN'S HOSPITAL", 0),
            ("as_prime", "SEATTLE CHILDREN'S HOSPITAL", 20),
        ),
    ),
)

__all__ = ["WIDE"]
