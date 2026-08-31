"""What the firms are, rather than who they worked with.

The slate batches teach choosing between named candidates. They are the rivers,
and every one of them runs over terrain the corpus never describes. A model
shown only "HP ENTERPRISE SERVICES is wrong for Perspecta" has learned a fact
about a pair. A model that can read

    Subcontracts taken: 0; awards where prime: 15

has learned why, and can apply it to a firm it has never seen. These are the
terrain.

The generated corpus has a ``portfolio`` archetype that visits the same records,
and what it does with them is why this file exists -- it prints the fields back
in one sentence frame, 159 times, and never says what any of them mean together.
It was also one of the seven shapes the v1 fine-tune collapsed into. Reading a
record and characterising a firm are different skills and only the first is in
the corpus.

**The constraint that shaped this batch, and what changed.** A first draft of
these profiles named each prime's top subcontractors with award counts, and a
test caught that none of it was in the retrieved context. The record at the time
carried a partner list truncated to six names in *alphabetical* order and no
per-pair counts at all -- so Perspecta's record did not mention HP, its most-used
supplier by a factor of five, and every answer in the corpus citing "27 reported
awards" was asking the model to print a number the prompt did not contain. The
grader reads company names and not numbers, so nothing caught it.

That is why ``company_record`` was widened. It now carries, per component, the
awards and distinct partners; and a partner list ranked by use, eight deep, each
entry with its count and where the work sat:

    Teamed with (33), most used first: HP 27 (CMS); CARAHSOFT TECHNOLOGY 21
    (CMS); SHI INTERNATIONAL 17 (CMS); ...

Every figure in these answers is now on the page the model is handed, and
``test_every_number_cited_is_in_the_context`` holds it there. The widening
reaches all three arms through the same ``context_for``, and it cannot leak the
sealed period: ``Company`` is assembled from training-period rows only.

What remains underivable is the tail. Eight partners of 131 are shown, so the
core of a bench is legible and the long shallow end of it is not -- several
answers below say so, because a model never shown that limit will read a
truncated list as a complete one.

Two kinds of example, and they are not equally useful:

**Bench profiles** ask what kind of firm a prime buys, stated as a type rather
than a roster. These are aimed at the measured task.

**Firm profiles** read one record into a characterisation: which side of the
table a firm sits on, how concentrated it is, which of several similarly-named
entities it actually is. Grounding rather than a lever, and said plainly because
it would be easy to oversell: 94% of the tier-1 traps in the authored slates are
firms that prime more often than they sub, but the blind set filters its
distractors to firms with at least two subcontracts taken, so that rule does not
discriminate there at all. Worth teaching because it is true, because it explains
the traps we write, and because a model that confuses LEIDOS with LEIDOS
BIOMEDICAL RESEARCH is wrong in a way no ranking metric will show.

Several answers state what the record still cannot tell you. That is not filler:
performance, capacity and price appear nowhere, reporting has gaps, and the
partner list stops at eight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .authored import CDC, CMS, NIH, SAMHSA  # noqa: F401  (agency labels)

# Labels for the codes these answers cite. The synth taxonomy has a NAICS map,
# but it is built for the synthetic corpus and covers 54% of the real one,
# missing 541990, 518210 and 236220 -- three of the codes that matter most here.
# Every code below is asserted against the record of the company it describes.
NAICS_LABELS: dict[str, str] = {
    "236220": "Commercial and Institutional Building Construction",
    "518210": "Data Processing, Hosting and Related Services",
    "541512": "Computer Systems Design Services",
    "541611": "Administrative and General Management Consulting",
    "541715": "R&D in Physical, Engineering and Life Sciences",
    "541720": "R&D in the Social Sciences and Humanities",
    "541990": "All Other Professional, Scientific and Technical Services",
}

PERSPECTA = "PERSPECTA ENTERPRISE SOLUTIONS"
GDIT = "GENERAL DYNAMICS INFORMATION TECHNOLOGY"
RTI = "RESEARCH TRIANGLE INSTITUTE"
CHUGACH = "CHUGACH WORLD SERVICES"
BATTELLE = "BATTELLE MEMORIAL INSTITUTE"
ICF = "ICF"
NGIT = "NORTHROP GRUMMAN INFORMATION TECHNOLOGY"
ACN = "ACCENTURE FEDERAL SERVICES"
NASMHPD = "NATIONAL ASSOCIATION OF STATE MENTAL HEALTH PROGRAM DIRECTORS"


@dataclass(frozen=True)
class Profile:
    """One hand-written record-reading example.

    ``kind`` decides which meta key carries the subject, and so which record
    ``context_for`` puts first: a firm profile is about a company, a bench
    profile about a prime at an agency.
    """

    question: str
    subject: str
    kind: str  # "firm" | "bench"
    reasoning: str
    answer: str
    agency: str | None = None
    # Extra companies whose records this question needs. Routed to the "a"/"b"
    # meta keys that ``context_for`` already consults, because BM25 over the
    # question text is not reliable enough to retrieve a named comparison --
    # asking about three Leidos entities returned two of them.
    also: tuple[str, ...] = ()
    facts: tuple[tuple, ...] = ()
    # Every company the answer names. Checked to be present in the supplied
    # context, because a profile of a record that was not shown is recall.
    names_cited: tuple[str, ...] = field(default=())


PROFILES: tuple[Profile, ...] = (
    # ==================================================================
    # bench profiles -- what kind of firm does this prime buy
    # ==================================================================
    Profile(
        question="What kind of subcontractor does Chugach World Services "
        "actually hire on its NIH work?",
        subject=CHUGACH,
        kind="bench",
        agency=NIH,
        reasoning="The record gives one agency, one industry code, 93 awards as "
        "prime, none taken, and 40 partners of which it shows six. Those six are "
        "enough: electrical contractors, mechanical services, engineers, testing. "
        "The type is what transfers to a candidate nobody has seen; the roster "
        "would be out of date by the time it was used.",
        answer="Building trades, and essentially nothing else.\n\n"
        "The record shows Chugach working a single component, NIH, on NAICS "
        "236220, Commercial and Institutional Building Construction, with 93 "
        "awards as the prime and none taken as a subcontractor. Of its 40 partners "
        "the record shows the eight most used: KENT ISLAND MECHANICAL 21, "
        "JOHNSON CONTROLS 12, CARRIER 5, DESA CORP OF VIRGINIA 5, INCON 4, "
        "AMERICAN COMBUSTION INDUSTRIES OF DELAWARE 3, ACM SERVICES 2, BMW "
        "CONSTRUCTION SPECIALISTS 2. Mechanical, controls, HVAC, combustion, "
        "general construction -- every one of them a building trade.\n\n"
        "So for a candidate you have not seen before, the question is whether "
        "they install or maintain building systems on a federal campus. Not "
        "whether they are large, not whether they hold NIH work of another kind, "
        "not whether they are a health company at all. An established health-IT "
        "firm with real NIH past performance is a worse fit here than an unknown "
        "sheet-metal contractor.\n\n"
        "One limit worth carrying: eight of forty are shown and they are the eight "
        "that matter, so you are seeing the core of this bench and none of its "
        "tail. A firm breaking in lands among the thirty-two you cannot see.",
        facts=(
            ("agencies", CHUGACH, 1),
            ("partners", CHUGACH, 40),
            ("as_sub", CHUGACH, 0),
            ("as_prime", CHUGACH, 93),
            ("naics", CHUGACH, "236220"),
        ),
        names_cited=(
            "KENT ISLAND MECHANICAL",
            "JOHNSON CONTROLS",
            "CARRIER",
            "DESA CORP OF VIRGINIA",
            "INCON",
            "AMERICAN COMBUSTION INDUSTRIES OF DELAWARE",
            "ACM SERVICES",
            "BMW CONSTRUCTION SPECIALISTS",
        ),
    ),
    Profile(
        question="If we're chasing ICF's NIH work, what does their typical "
        "subcontractor look like?",
        subject=ICF,
        kind="bench",
        agency=NIH,
        reasoning="Twenty-two partners, 43 awards as prime, none taken, one "
        "component, and the code is management consulting. The six partners shown "
        "include two personal names, which is the whole characterisation: ICF "
        "wins the analytic work and buys production capability from very small "
        "shops.",
        answer="Small, specialised, and sometimes one person.\n\n"
        "- Scale: 43 awards as prime, none taken as a sub, all at NIH\n"
        "- Bench: 22 partners; the eight most used are JML DESIGN 5, CONSOLIDATED "
        "SOLUTIONS & INNOVATIONS 4, DAZA MARTHA S 3, DILTS PARTNERS 3, MOBOMO 3, "
        "WINSLOW LLC, TERESE 3, CARAPELLUCCI, CINDY 2, COMPUTERCRAFT 2\n"
        "- Code: 541611, Administrative and General Management Consulting\n\n"
        "Three of those eight are individuals trading under their own names, and "
        "the rest are small communications and computing shops. ICF holds 541611 and "
        "does the consulting itself; what it buys is delivery around it -- "
        "writing, web, production.\n\n"
        "The trap this sets is the reason to read the bench at all. Sorting a "
        "candidate list by size returns the large consultancies that share ICF's "
        "own code, and those are its competitors. A good candidate here is narrow "
        "enough that ICF would not simply do the work in-house, which is close to "
        "the opposite of impressive.",
        facts=(
            ("agencies", ICF, 1),
            ("partners", ICF, 22),
            ("as_sub", ICF, 0),
            ("as_prime", ICF, 43),
            ("naics", ICF, "541611"),
        ),
        names_cited=(
            "JML DESIGN",
            "CONSOLIDATED SOLUTIONS & INNOVATIONS",
            "DAZA MARTHA S",
            "DILTS PARTNERS",
            "MOBOMO",
            "WINSLOW LLC, TERESE",
            "CARAPELLUCCI, CINDY",
            "COMPUTERCRAFT",
        ),
    ),
    Profile(
        question="Battelle at NIH -- what sort of company ends up on their subcontracts?",
        subject=BATTELLE,
        kind="bench",
        agency=NIH,
        reasoning="Laboratory supply. The six partners shown are instrument, "
        "reagent and gas suppliers, and the code is life-sciences R&D. This is "
        "procurement of materials and capacity rather than teaming for expertise, "
        "which changes what a candidate needs to bring.",
        answer="Laboratory suppliers and testing houses, not research partners.\n\n"
        "Battelle's record shows 59 awards as prime against 4 taken, five HHS "
        "components, and NAICS 541715, R&D in Physical, Engineering and Life "
        "Sciences. Of its 40 partners the eight most used are AMPLIFYBIO 15, "
        "ENVIGO GLOBAL SERVICES 4, ANI LYTICS 2, EMSL ANALYTICAL 2, LIFE "
        "TECHNOLOGIES 2, NORTHWESTERN UNIVERSITY 2, REMPEX PHARMACEUTICALS 2 and "
        "VERSITI BLOOD HEALTH 2 -- animal research, analytical testing, reagents, "
        "blood products. One deep relationship and a flat tail.\n\n"
        "Read that as buying materials and capacity rather than expertise. "
        "Battelle is a research institute; it does not subcontract for analysis "
        "it can do itself, so arriving as a general research contractor is "
        "arriving as a competitor. The route in is a specific assay, instrument "
        "or facility it does not own.\n\n"
        "That generalises past this prime. A bench composed of suppliers tells "
        "you the prime keeps its discipline in-house, and a candidate should "
        "match what is missing from the prime rather than what the prime is "
        "known for.",
        facts=(
            ("agencies", BATTELLE, 5),
            ("partners", BATTELLE, 40),
            ("as_sub", BATTELLE, 4),
            ("as_prime", BATTELLE, 59),
            ("naics", BATTELLE, "541715"),
        ),
        names_cited=(
            "AMPLIFYBIO",
            "ENVIGO GLOBAL SERVICES",
            "ANI LYTICS",
            "EMSL ANALYTICAL",
            "LIFE TECHNOLOGIES",
            "NORTHWESTERN UNIVERSITY",
            "REMPEX PHARMACEUTICALS",
            "VERSITI BLOOD HEALTH",
        ),
    ),
    Profile(
        question="Northrop Grumman IT buys subcontract support at CDC. What are "
        "they buying?",
        subject=NGIT,
        kind="bench",
        agency=CDC,
        reasoning="A small bench -- 14 partners against 20 awards as prime -- and "
        "the visible names mix a reseller, a staffing-adjacent consultancy and "
        "two health-informatics shops. Small and mixed is itself the finding: "
        "there is no standing bench to displace.",
        answer="Not much, and not from many people.\n\n"
        "The record shows 20 awards as prime, none taken, one component, NAICS "
        "541512, Computer Systems Design Services, and just 14 partners -- the "
        "smallest bench of any prime this size in the library. TEKSYSTEMS 5 leads, "
        "then THE GINN GROUP 2 and THE ST. JOHN GROUP 2, then CARAHSOFT "
        "TECHNOLOGY, DB CONSULTING GROUP, FORUM ONE COMMUNICATIONS, "
        "INDUCTIVEHEALTH INFORMATICS and IT1 SOURCE at one apiece.\n\n"
        "One staffing firm and then a mixed tail: a reseller, a consultancy, a "
        "communications shop, a health-informatics specialist, all at a single "
        "award. A prime whose only repeat relationship is with a staffing company "
        "is buying people against requirements rather than assembling a bench, "
        "and the practical consequence is encouraging -- there is no incumbent to "
        "displace and no established type you have to resemble.\n\n"
        "It is also the case where reading the bench tells you least. Fourteen "
        "partners is a thin basis for inferring what they want next, so a "
        "candidate here should be judged on fit to the stated work rather than "
        "on resemblance to a bench type that this record does not establish.",
        facts=(
            ("agencies", NGIT, 1),
            ("partners", NGIT, 14),
            ("as_sub", NGIT, 0),
            ("as_prime", NGIT, 20),
            ("naics", NGIT, "541512"),
        ),
        names_cited=(
            "TEKSYSTEMS",
            "THE GINN GROUP",
            "THE ST. JOHN GROUP",
            "CARAHSOFT TECHNOLOGY",
            "DB CONSULTING GROUP",
            "FORUM ONE COMMUNICATIONS",
            "INDUCTIVEHEALTH INFORMATICS",
            "IT1 SOURCE",
        ),
    ),
    Profile(
        question="Perspecta's CMS bench -- what's actually on it?",
        subject=PERSPECTA,
        kind="bench",
        agency=CMS,
        reasoning="Product. The visible partners are hardware distributors, cable "
        "and connectivity, and a licence reseller, and the prime's code is data "
        "hosting. A services company reading a 33-partner bench as an opportunity "
        "is misreading what those 33 are for.",
        answer="Hardware and licences. This is a supply chain, not a team.\n\n"
        "Perspecta's record: one component, CMS; NAICS 518210, Data Processing, "
        "Hosting and Related Services; 161 awards as prime, the highest count in "
        "this library, and none taken. Of 33 partners the eight most used are HP 27, "
        "CARAHSOFT TECHNOLOGY 21, SHI INTERNATIONAL 17, SMS DATA PRODUCTS GROUP "
        "12, CISCO SYSTEMS 11, ORACLE AMERICA 11, NETAPP 8 and CA 6 -- hardware, "
        "licence resale, network kit, storage. Not one services firm in the top "
        "eight.\n\n"
        "So the honest read for a services firm is discouraging. The bench is "
        "large and it is what you buy to run a hosting estate, which means the "
        "work is not what is being subcontracted. Capability is not the thing "
        "that gets you onto this list; being a supplier is.\n\n"
        "Worth generalising, because the count misleads on its own. A high "
        "partner count can mean a prime teams widely or that it buys a lot of "
        "product through separate paper. Those are identical in the number and "
        "obvious in the names, so read the names.",
        facts=(
            ("agencies", PERSPECTA, 1),
            ("partners", PERSPECTA, 33),
            ("as_sub", PERSPECTA, 0),
            ("as_prime", PERSPECTA, 161),
            ("naics", PERSPECTA, "518210"),
        ),
        names_cited=(
            "HP",
            "CARAHSOFT TECHNOLOGY",
            "SHI INTERNATIONAL",
            "SMS DATA PRODUCTS GROUP",
            "CISCO SYSTEMS",
            "ORACLE AMERICA",
            "NETAPP",
            "CA",
        ),
    ),
    Profile(
        question="Accenture Federal Services at CMS -- who gets onto that bench?",
        subject=ACN,
        kind="bench",
        agency=CMS,
        reasoning="Staffing and IT services firms, which is the opposite "
        "composition to Perspecta at the same customer. Two primes, one agency, "
        "opposite benches is the cleanest evidence available that buying pattern "
        "belongs to the prime and cannot be inferred from the agency.",
        answer="Services and staffing firms, which makes this the most joinable "
        "large bench at CMS.\n\n"
        "The record shows 77 awards as prime against 1 taken, four components, "
        "NAICS 541512 and 541611, and 45 partners. The eight most used are M9 "
        "SOLUTIONS (5), CARAHSOFT TECHNOLOGY (4), SKY TECHNOLOGIES (4), GENERAL "
        "DYNAMICS INFORMATION TECHNOLOGY (3), KFORCE (3), LYNCH CONSULTANTS (3), "
        "MANPOWERGROUP PUBLIC SECTOR (3) and PACE COMPUTER SOLUTIONS (3) -- systems "
        "work, staffing and consulting, with one reseller among them. People and "
        "delivery capability rather than product.\n\n"
        "Set that beside Perspecta at the same agency and the contrast is the "
        "lesson. Same customer, same broad kind of work, and one bench is topped by "
        "hardware and licences while this one is topped by services firms. Note "
        "GENERAL DYNAMICS INFORMATION TECHNOLOGY sitting on it at three awards: a "
        "prime elsewhere, a subcontractor here. The buying pattern is a "
        "property of the prime, so it has to be read per prime; inferring it from "
        "the agency gets it exactly wrong half the time.\n\n"
        "For a services company without CMS past performance this is the bench "
        "worth targeting, and 45 partners with no visible concentration means "
        "there is no incumbent in the way.",
        facts=(
            ("agencies", ACN, 4),
            ("partners", ACN, 45),
            ("as_sub", ACN, 1),
            ("as_prime", ACN, 77),
            ("naics", ACN, "541512"),
            ("naics", ACN, "541611"),
        ),
        names_cited=(
            "M9 SOLUTIONS",
            "CARAHSOFT TECHNOLOGY",
            "SKY TECHNOLOGIES",
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY",
            "KFORCE",
            "LYNCH CONSULTANTS",
            "MANPOWERGROUP PUBLIC SECTOR",
            "PACE COMPUTER SOLUTIONS",
        ),
    ),
    Profile(
        question="NASMHPD shows 33 partners at SAMHSA. Is that a bench?",
        subject=NASMHPD,
        kind="bench",
        agency=SAMHSA,
        reasoning="The partner names are state health departments and a "
        "territorial government. The count looks like a large teaming practice "
        "and is a funding distribution mechanism -- the clearest case in the data "
        "of a number meaning the opposite of what it looks like.",
        answer="No. It is a funding distribution list that counts like a bench.\n\n"
        "The record shows 32 awards as prime, one component, NAICS 541990, and 33 "
        "partners, every one of them at exactly one award -- so the eight the record "
        "shows are simply eight of thirty-three, and they settle what they are: "
        "ALABAMA DEPARTMENT OF HEALTH, AR DEPARTMENT OF HUMAN SERVICES, "
        "DEPARTMENT OF BEHAVIORAL HEALTH AND DEVELOPMENTAL DISABILITIES, "
        "DEPARTMENT OF HEALTH & SOCIAL SERVICES DELAWARE, DEPARTMENT OF PUBLIC "
        "HEALTH CONNECTICUT, GOVERNMENT OF THE COMMONWEALTH OF THE NORTHERN "
        "MARIANA ISLANDS, GOVERNMENT OF THE VIRGIN ISLANDS, GUAM BEHAVIORAL "
        "HEALTH & WELLNESS CENTER.\n\n"
        "Every one is a government body. NASMHPD is an association of state "
        "mental health agencies and these awards are programme money reaching its "
        "members. There is no supplier relationship being recorded, so there is "
        "no incumbent, no selection, and no route in for a commercial firm at "
        "all.\n\n"
        "The transferable point is that a partner count measures paper rather "
        "than partnership. Before reading a roster as a bench, check whether the "
        "names are companies.",
        facts=(
            ("agencies", NASMHPD, 1),
            ("partners", NASMHPD, 33),
            ("as_sub", NASMHPD, 1),
            ("as_prime", NASMHPD, 32),
            ("naics", NASMHPD, "541990"),
        ),
        names_cited=(
            "ALABAMA DEPARTMENT OF HEALTH",
            "AR DEPARTMENT OF HUMAN SERVICES",
            "DEPARTMENT OF BEHAVIORAL HEALTH AND DEVELOPMENTAL DISABILITIES",
            "DEPARTMENT OF HEALTH & SOCIAL SERVICES DELAWARE",
            "DEPARTMENT OF PUBLIC HEALTH CONNECTICUT",
            "GOVERNMENT OF THE COMMONWEALTH OF THE NORTHERN MARIANA ISLANDS",
            "GOVERNMENT OF THE VIRGIN ISLANDS",
            "GUAM BEHAVIORAL HEALTH & WELLNESS CENTER",
        ),
    ),
    Profile(
        question="RTI subcontracts more than anyone here. What do they actually buy?",
        subject=RTI,
        kind="bench",
        agency=CDC,
        reasoning="Not research, which is the surprise. RTI is the most prolific "
        "subcontracting prime in the library and its most-used partners are "
        "staffing, personnel and fulfilment firms. The gap between what a prime "
        "does and what it buys is the transferable lesson, and it is the "
        "commonest way a capability pitch goes to the wrong prime.",
        answer="Operations, not research -- which is the opposite of what the "
        "name suggests.\n\n"
        "Scale first: 215 awards as prime against 2 taken, across seven HHS "
        "components, with 131 distinct partners. That is the widest spread and "
        "the largest bench in this library, and it means RTI is reachable in a "
        "way most primes are not.\n\n"
        "What it buys: HR DIRECTIONS 23, HEADWAY HR SOLUTIONS 8, PERFECT GIFT 6, "
        "GLOVER CORPORATION 5, PATIENT PERFORMANCE 4, TELLIGEN 4, UNIVERSITY OF "
        "NORTH CAROLINA AT CHAPEL HILL 4. Personnel, staffing, incentives and "
        "fulfilment, with a single university among them. RTI holds 541990 and "
        "541720 and does the survey and evaluation science itself; what it "
        "subcontracts is the apparatus that runs a study.\n\n"
        "Generalise that: a prime's own capability tells you what it will not "
        "subcontract. A candidate should match the gap rather than the "
        "reputation, and pitching research to a research institute is the most "
        "common version of getting that backwards.\n\n"
        "One caveat on the sample. Eight of 131 partners are shown, ranked by "
        "use, so the pattern above is the core of the bench and says nothing "
        "about the rest -- a wide, shallow tail that this record cannot "
        "describe.",
        facts=(
            ("agencies", RTI, 7),
            ("partners", RTI, 131),
            ("as_sub", RTI, 2),
            ("as_prime", RTI, 215),
            ("naics", RTI, "541990"),
            ("naics", RTI, "541720"),
        ),
        names_cited=(
            "HR DIRECTIONS",
            "HEADWAY HR SOLUTIONS",
            "PERFECT GIFT",
            "GLOVER CORPORATION",
            "PATIENT PERFORMANCE",
            "TELLIGEN",
            "UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL",
        ),
    ),
    # ==================================================================
    # firm profiles -- reading one record into a characterisation
    # ==================================================================
    Profile(
        question="HP and HP ENTERPRISE SERVICES both show up in CMS records. "
        "What's the difference?",
        subject="HP ENTERPRISE SERVICES",
        kind="firm",
        reasoning="The two records answer this with no outside knowledge. HP "
        "takes 28 subcontracts and primes nothing; HP ENTERPRISE SERVICES takes "
        "none and primes 15 awards with its own partners underneath. Opposite "
        "sides of the table, which is why one can be on a bench and the other "
        "cannot.",
        answer="They sit on opposite sides of the table, and the records say so "
        "outright.\n\n"
        "1. HP: 28 subcontracts taken, 0 awards as prime, 2 partners. It supplies "
        "other people's programmes.\n"
        "2. HP ENTERPRISE SERVICES: 0 subcontracts taken, 15 awards as prime, 9 "
        "partners of its own. It holds contracts and hires other people.\n\n"
        "So a shortlist of candidate subcontractors containing HP ENTERPRISE "
        "SERVICES has made a category error rather than a judgement error. It is "
        "not a weak candidate for a sub slot; it is not a candidate for one, and "
        "no amount of capability match changes that.\n\n"
        "The rule generalises and is cheap to apply: compare \"Subcontracts "
        "taken\" against \"awards where prime\" before reading anything else. Two "
        "firms one word apart in name can be a supplier and a competitor, and the "
        "name will not tell you which.",
        facts=(
            ("as_sub", "HP", 28),
            ("as_prime", "HP", 0),
            ("partners", "HP", 2),
            ("as_sub", "HP ENTERPRISE SERVICES", 0),
            ("as_prime", "HP ENTERPRISE SERVICES", 15),
            ("partners", "HP ENTERPRISE SERVICES", 9),
        ),
        names_cited=("HP", "HP ENTERPRISE SERVICES"),
    ),
    Profile(
        question="LEIDOS, LEIDOS INNOVATIONS and LEIDOS BIOMEDICAL RESEARCH all appear "
        "in here. Are they the same company?",
        subject="LEIDOS BIOMEDICAL RESEARCH",
        kind="firm",
        reasoning="Three records, three businesses. One spans five components and "
        "both roles; one is CMS-only and primes everything; one is NIH-only with "
        "a bench of laboratory suppliers. Treating them as one company gets every "
        "question about any of them wrong.",
        answer="Related, and not interchangeable. For any question you would ask, "
        "they behave as three companies.\n\n"
        "- LEIDOS: 7 subcontracts taken, 25 as prime, five components. The "
        "generalist, and the only one of the three that ever appears as a "
        "subcontractor.\n"
        "- LEIDOS INNOVATIONS: 0 taken, 29 as prime, CMS only, 25 partners. A CMS "
        "prime with a bench of its own.\n"
        "- LEIDOS BIOMEDICAL RESEARCH: 0 taken, 31 as prime, NIH only, 26 "
        "partners, NAICS 541715. Research operations.\n\n"
        "The failure this prevents is concrete. Asked for NIH research partners, "
        "the name that comes to mind is LEIDOS BIOMEDICAL RESEARCH -- and it has "
        "never taken a subcontract from anyone, so it cannot be one. Asked "
        "whether \"Leidos\" has CMS experience, the answer depends entirely on "
        "which of the three was meant, and two of the three would be wrong.\n\n"
        "Shared brand, separate records, different answers.",
        facts=(
            ("as_sub", "LEIDOS", 7),
            ("as_prime", "LEIDOS", 25),
            ("agencies", "LEIDOS", 5),
            ("as_sub", "LEIDOS INNOVATIONS", 0),
            ("as_prime", "LEIDOS INNOVATIONS", 29),
            ("partners", "LEIDOS INNOVATIONS", 25),
            ("as_sub", "LEIDOS BIOMEDICAL RESEARCH", 0),
            ("as_prime", "LEIDOS BIOMEDICAL RESEARCH", 31),
            ("agencies", "LEIDOS BIOMEDICAL RESEARCH", 1),
            ("partners", "LEIDOS BIOMEDICAL RESEARCH", 26),
            ("naics", "LEIDOS BIOMEDICAL RESEARCH", "541715"),
        ),
        also=("LEIDOS", "LEIDOS INNOVATIONS"),
        names_cited=("LEIDOS", "LEIDOS INNOVATIONS", "LEIDOS BIOMEDICAL RESEARCH"),
    ),
    Profile(
        question="Carahsoft keeps appearing on other primes' benches. What are they?",
        subject="CARAHSOFT TECHNOLOGY",
        kind="firm",
        reasoning="52 subcontracts taken, nothing primed, 14 partners across four "
        "components. A firm that only ever appears underneath other people, on "
        "many different people, is a supply channel rather than a teammate -- and "
        "that changes how a prime's award count with it should be read.",
        answer="A reseller, and the shape of the record gives it away before the "
        "name does.\n\n"
        "CARAHSOFT TECHNOLOGY takes 52 subcontracts and has primed nothing, "
        "across four components and 14 different partners. Software and licences "
        "moving onto other companies' contracts. There is a second entry, "
        "CARAHSOFT TECHNOLOGY SOLUTIONS, with 29 subcontracts taken and exactly "
        "one partner.\n\n"
        "Why that matters when reading someone else's bench: a prime with a large "
        "number of Carahsoft awards has a procurement channel, not a deep teaming "
        "relationship. Counting them as evidence of how much that prime "
        "subcontracts overstates its real bench, and the overstatement is not "
        "small.\n\n"
        "The test that catches it: a partner that takes many subcontracts, primes "
        "nothing, and appears under many different primes is selling a product. "
        "Strip those, then judge the bench.",
        facts=(
            ("as_sub", "CARAHSOFT TECHNOLOGY", 52),
            ("as_prime", "CARAHSOFT TECHNOLOGY", 0),
            ("agencies", "CARAHSOFT TECHNOLOGY", 4),
            ("partners", "CARAHSOFT TECHNOLOGY", 14),
            ("as_sub", "CARAHSOFT TECHNOLOGY SOLUTIONS", 29),
            ("as_prime", "CARAHSOFT TECHNOLOGY SOLUTIONS", 0),
            ("partners", "CARAHSOFT TECHNOLOGY SOLUTIONS", 1),
        ),
        names_cited=("CARAHSOFT TECHNOLOGY", "CARAHSOFT TECHNOLOGY SOLUTIONS"),
    ),
    Profile(
        question="HR DIRECTIONS turns up on a lot of RTI's work. How big are they really?",
        subject="HR DIRECTIONS",
        kind="firm",
        reasoning="23 subcontracts taken, nothing primed, five components -- and "
        "exactly one partner. The agency spread reads as breadth and the partner "
        "count says it is one relationship travelling. Two fields that contradict "
        "each other on first reading, and the second one wins.",
        answer="Small, and entirely dependent on a single relationship.\n\n"
        "HR DIRECTIONS has taken 23 subcontracts, primed nothing, and appears at "
        "five HHS components. Five components looks like a firm with reach. Then "
        "the partner count says 1 -- every one of those 23 awards is with the "
        "same prime.\n\n"
        "So it travels wherever that prime works and has no independent federal "
        "presence. Both readings are useful. As a teaming target it is a poor "
        "route into anything except that one relationship. As evidence about the "
        "prime it is strong: a supplier used 23 times across five components is a "
        "settled arrangement rather than a procurement.\n\n"
        "The fields to read together are partner count and agency count. Five "
        "agencies with one partner is a completely different company from five "
        "agencies with twenty, and only one of the two numbers tells you which.",
        facts=(
            ("as_sub", "HR DIRECTIONS", 23),
            ("as_prime", "HR DIRECTIONS", 0),
            ("agencies", "HR DIRECTIONS", 5),
            ("partners", "HR DIRECTIONS", 1),
        ),
        names_cited=("HR DIRECTIONS",),
    ),
    Profile(
        question="Why does GDIT show up as both a prime and a subcontractor?",
        subject=GDIT,
        kind="firm",
        reasoning="178 awards as prime and 11 as sub. Both are real and the ratio "
        "is the characterisation: overwhelmingly a prime, occasionally a sub on "
        "someone else's programme. Role belongs to the award, which a filter "
        "treating 'prime' as a company attribute gets wrong in both directions.",
        answer="Because role belongs to the award, not to the company.\n\n"
        "The record shows 178 awards where GENERAL DYNAMICS INFORMATION "
        "TECHNOLOGY was the prime and 11 where it was the subcontractor, across "
        "seven components, with 90 distinct partners -- the widest footprint in "
        "this library. The characterisation follows: a large integrator that "
        "primes nearly all of its work and will occasionally take a sub position "
        "on someone else's.\n\n"
        "Both numbers are load-bearing. A shortlist that drops it because \"it is "
        "a prime\" loses a firm that demonstrably does take subcontracts. A "
        "shortlist that treats it as an ordinary subcontractor has missed that 11 "
        "against 178 makes that the exception.\n\n"
        "The ratio is the useful quantity, not either number by itself. The same "
        "two fields decide the question outright for firms that only ever do one "
        "or the other; here they decide it by degree.",
        facts=(
            ("as_prime", GDIT, 178),
            ("as_sub", GDIT, 11),
            ("agencies", GDIT, 7),
            ("partners", GDIT, 90),
        ),
        names_cited=(GDIT,),
    ),
    Profile(
        question="SEATTLE CHILDREN'S HOSPITAL looks like an obvious research "
        "partner. Is it?",
        subject="SEATTLE CHILDREN'S HOSPITAL",
        kind="firm",
        reasoning="Zero subcontracts taken against 20 awards as prime, NIH only, "
        "14 partners of its own. Structurally a prime, so approaching it for a "
        "sub slot is the wrong direction. Worth stating explicitly because the "
        "capability read says yes and the record says the relationship would have "
        "to run the other way.",
        answer="An obvious research organisation and a poor candidate for a "
        "subcontract. Those are different questions and only the second one was "
        "asked.\n\n"
        "The record: 0 subcontracts taken, 20 awards as the prime, NIH only, 14 "
        "partners of its own, NAICS 541715. It has never been anyone's "
        "subcontractor in this data.\n\n"
        "That redirects the approach rather than ending it. Asking it onto a team "
        "as a sub asks it to do something it has never done. What the record "
        "supports is the opposite direction -- it holds NIH awards and hires 14 "
        "organisations underneath them, so the route is onto its bench rather "
        "than onto yours.\n\n"
        "Worth saying plainly because the capability read and the structural read "
        "disagree, and the capability read is the one that feels right. A "
        "children's research hospital is a plausible partner for paediatric NIH "
        "work in every sense except the one the record measures.",
        facts=(
            ("as_sub", "SEATTLE CHILDREN'S HOSPITAL", 0),
            ("as_prime", "SEATTLE CHILDREN'S HOSPITAL", 20),
            ("agencies", "SEATTLE CHILDREN'S HOSPITAL", 1),
            ("partners", "SEATTLE CHILDREN'S HOSPITAL", 14),
            ("naics", "SEATTLE CHILDREN'S HOSPITAL", "541715"),
        ),
        names_cited=("SEATTLE CHILDREN'S HOSPITAL",),
    ),
    Profile(
        question="What can't I tell about a company from one of these records?",
        subject=RTI,
        kind="firm",
        reasoning="The limits are as teachable as the contents and nothing in the "
        "corpus states them. Reported subawards have a threshold and gaps, the "
        "codes are self-selected, the partner list is a truncated alphabetical "
        "sample, and none of it carries price, performance or capacity.",
        answer="More than the record's confidence suggests. Five things.\n\n"
        "**The tail of a bench.** The partner list is the eight most-used out of "
        "however many the count says. That is the right eight, so the core of a "
        "relationship is visible -- but for a firm with 131 partners nearly all "
        "of them stay invisible, and a candidate you are asked about is far more "
        "likely to be in the tail than in the head.\n\n"
        "**Performance.** Nothing here is a quality signal. CPARS ratings are "
        "source-selection information and not public, so repeat hiring is the "
        "only performance-adjacent evidence there is, and it is revealed "
        "preference rather than assessment.\n\n"
        "**Completeness.** These are reported subawards, with a dollar threshold "
        "and real filing gaps. No record of work at an agency may mean work below "
        "the threshold or under another corporate name. Absence supports \"not an "
        "established partner\", not \"never worked together\".\n\n"
        "**Codes.** NAICS is self-selected per award and coarse. 541990 covers "
        "RESEARCH TRIANGLE INSTITUTE's survey science and a great deal that "
        "resembles it not at all, so a code match on its own is close to no "
        "evidence.\n\n"
        "**Capacity and price.** Neither appears anywhere. A firm on 23 awards "
        "may be three people or three hundred.\n\n"
        "What the record does measure directly is roles and relationships, and "
        "those are what the ranking questions turn on.",
        facts=(
            ("naics", RTI, "541990"),
            ("as_sub", RTI, 2),
            ("partners", RTI, 131),
        ),
        names_cited=(RTI,),
    ),
)


def profile_examples(graph: Any, index: Any, repeat: int = 1) -> list[dict[str, Any]]:
    """The profile set, in corpus form, with real retrieved context.

    Open-book for the same reason the slate examples are: the closed-book result
    was decisive -- 29% fact recall, below the random floor -- so teaching a
    model to state these numbers from memory would repeat a failure the project
    has already measured. What is being taught is reading a record, and the
    record has to be there.
    """
    from ..shared.records import context_for
    from ..shared.questions import Question

    rows: list[dict[str, Any]] = []
    for _ in range(repeat):
        for item in PROFILES:
            meta: dict[str, Any] = {"authored": True, "profile_kind": item.kind}
            if item.kind == "bench":
                meta["prime"] = item.subject
                meta["agency"] = item.agency
            else:
                meta["company"] = item.subject
            for key, extra in zip(("a", "b"), item.also):
                meta[key] = extra
            question = Question(
                question=item.question,
                answer=item.answer,
                reasoning=item.reasoning,
                archetype="authored_profile",
                gold=[],
                tiers={},
                meta=meta,
            )
            record = question.to_record()
            record["context"] = context_for(graph, question, index)
            record["meta"]["closed_book"] = False
            rows.append(record)
    return rows
