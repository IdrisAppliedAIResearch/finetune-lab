"""Hand-written examples that reason over the retrieved records, not from memory.

Every authored example written so far is closed-book: ``context`` is empty and
the answer comes out of the weights. That was the right target when the question
was whether the graph could be memorised. It is the wrong target now, because
the arm worth improving is the one that is served *with* records --

    arm A  tuned + retrieval   precision 0.445, off-slate 0.298
    arm C  base + retrieval    precision 0.477, off-slate 0.550

-- and every open-book training example it learned from was machine-generated.
The 152 hand-written rows carry the reasoning quality and teach it in the mode
that scored below the random floor. The good prose was aimed at the losing arm.

These are aimed at the winning one. Each carries a real slate, a real context
block assembled by the same ``context_for`` used at serving time, and an answer
that reasons over the records in front of it.

Three failures are targeted specifically, all measured on the v2 blind run:

* **Off-slate naming** (0.298 of arm A's answers). Fourteen of fourteen
  off-slate picks were companies whose records appeared in the supplied
  context -- BM25 fill, not hallucination. The model was reading the library
  and answering from it, having never been shown that a record's presence is
  not a nomination. Two examples below say so outright, and one names a
  company that would be a *good* answer if it had been offered.
* **Hard negatives recommended** (1.647 against arm C's 1.039). Tier-1
  candidates match on NAICS and agency scale and have no relationship, so a
  name-shaped heuristic takes them every time. The slates here are chosen for
  the traps that punish that heuristic hardest: ``HP`` is a Perspecta sub 27
  times over, ``HP ENTERPRISE SERVICES`` never; ``PERATON ENTERPRISE
  SOLUTIONS`` shares the prime's own naming convention and its industry code.
* **Trap rejection** (0.169). Only 18% of open-book training answers name what
  they turned down. An answer that rejects out loud is both better analysis and
  the only form the grader can see.

One convention that looks cosmetic and is not: candidates are named in the
answer exactly as the slate spells them, in capitals. ``find_companies`` matches
by literal substring, so "HR Directions" is invisible to it where "HR
DIRECTIONS" is not -- the whole generated corpus writes candidates in capitals,
every model answer measured so far does too, and a hand-written set that quietly
switched to title case would teach a model to produce correct answers that score
as naming nobody. Primes are referred to in ordinary prose, since the grader
excludes the question's own subject from the picks either way.

The discipline from ``authored.py`` carries over unchanged: prose by hand, facts
by machine. Declared tiers are checked against ``tier_for`` and every count is
checked against the graph by ``tests/test_authored_context.py``. A number that
drifts fails the build rather than teaching the model something false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .authored import CDC, CMS, FDA, GDIT, NIH, RTI, SAMHSA

PERSPECTA = "PERSPECTA ENTERPRISE SOLUTIONS"
NGS = "NATIONAL GOVERNMENT SERVICES"
BATTELLE = "BATTELLE MEMORIAL INSTITUTE"
CHUGACH = "CHUGACH WORLD SERVICES"
ICF = "ICF"
LEWIN = "LEWIN GROUP, INC., THE"


@dataclass(frozen=True)
class OpenBook:
    """One hand-written open-book example.

    ``tiers`` is declared rather than computed so that the prose and the label
    can disagree and be caught. Writing "this one is a hard negative" and having
    a test confirm the graph agrees is the point; computing the tier and then
    describing it would make the claim unfalsifiable.
    """

    ask: str
    prime: str
    agency: str
    slate: tuple[str, ...]
    tiers: dict[str, int]
    reasoning: str
    answer: str
    gold: tuple[str, ...] = ()
    facts: tuple[tuple, ...] = ()
    # Companies the answer names as present-in-context-but-not-on-the-slate.
    # Checked against the generated context, since the whole lesson depends on
    # the record actually being there, and BM25 chose it rather than I did.
    off_slate_named: tuple[str, ...] = field(default=())

    @property
    def question(self) -> str:
        slate = "\n".join(f"{i}. {n}" for i, n in enumerate(self.slate, start=1))
        return f"{self.ask}\n{slate}"


EXAMPLES: tuple[OpenBook, ...] = (
    # ------------------------------------------------------------------
    # Two names one word apart with opposite answers. The clearest case in the
    # data for reading the record instead of the name.
    # ------------------------------------------------------------------
    OpenBook(
        ask="We're pitching Perspecta Enterprise Solutions on a CMS re-compete "
        "and need a teaming shortlist. Which of these are actually on their bench?",
        prime=PERSPECTA,
        agency=CMS,
        slate=(
            "HP ENTERPRISE SERVICES",
            "CARAHSOFT TECHNOLOGY",
            "LEIDOS",
            "HP",
            "BROAD INSTITUTE, INC., THE",
            "SHI INTERNATIONAL",
            "PERATON ENTERPRISE SOLUTIONS",
            "SMS DATA PRODUCTS GROUP",
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY",
            "ORACLE AMERICA",
        ),
        tiers={
            "HP ENTERPRISE SERVICES": 1,
            "CARAHSOFT TECHNOLOGY": 4,
            "LEIDOS": 1,
            "HP": 4,
            "BROAD INSTITUTE, INC., THE": 1,
            "SHI INTERNATIONAL": 4,
            "PERATON ENTERPRISE SOLUTIONS": 1,
            "SMS DATA PRODUCTS GROUP": 4,
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY": 2,
            "ORACLE AMERICA": 4,
        },
        reasoning="Two pairs on this list differ by a word and by nothing a "
        "name-matcher can see. HP has 27 reported Perspecta awards at CMS; HP "
        "ENTERPRISE SERVICES has none. PERATON ENTERPRISE SOLUTIONS copies the "
        "prime's own naming convention and its 518210 code and has never "
        "subcontracted to it. The records settle both, so the only judgement "
        "left is the order of the four that do have a relationship.",
        answer="1. HP - 27 reported Perspecta awards at CMS\n"
        "2. CARAHSOFT TECHNOLOGY - 21\n"
        "3. SHI INTERNATIONAL - 17\n"
        "4. SMS DATA PRODUCTS GROUP - 12\n\n"
        "ORACLE AMERICA is a fifth genuine option at 11 if you want the depth.\n\n"
        "Not recommended: HP ENTERPRISE SERVICES and PERATON ENTERPRISE "
        "SOLUTIONS. Each is one word away from a name that belongs on this list, "
        "each carries the same 518210 hosting code, and neither has a single "
        "reported subcontract with Perspecta. GENERAL DYNAMICS INFORMATION "
        "TECHNOLOGY is a much larger CMS company than any of the four above and "
        "has also never worked under this prime -- at CMS it competes with "
        "Perspecta rather than teaming with it.\n\n"
        "One record not to act on: the library also returned CISCO SYSTEMS, and "
        "Perspecta's own record shows it on 11 awards -- better evidenced than "
        "half your list. It is still not one of your options, so I have not "
        "counted it. A record turning up in the library is not a nomination. "
        "Worth adding to the next version of the list.",
        gold=("HP", "CARAHSOFT TECHNOLOGY", "SHI INTERNATIONAL", "SMS DATA PRODUCTS GROUP"),
        facts=(
            ("used", PERSPECTA, "HP", 27, CMS),
            ("used", PERSPECTA, "CARAHSOFT TECHNOLOGY", 21, CMS),
            ("used", PERSPECTA, "SHI INTERNATIONAL", 17, CMS),
            ("used", PERSPECTA, "SMS DATA PRODUCTS GROUP", 12, CMS),
            ("used", PERSPECTA, "ORACLE AMERICA", 11, CMS),
            ("not_used", PERSPECTA, "HP ENTERPRISE SERVICES", CMS),
            ("not_used", PERSPECTA, "PERATON ENTERPRISE SOLUTIONS", CMS),
            ("not_used", PERSPECTA, "GENERAL DYNAMICS INFORMATION TECHNOLOGY", CMS),
            ("used", PERSPECTA, "CISCO SYSTEMS", 11, CMS),
        ),
        off_slate_named=("CISCO SYSTEMS",),
    ),
    # ------------------------------------------------------------------
    # The biggest name on the slate is the hard negative. Scale at the agency
    # and a relationship with the prime point in opposite directions.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Which of these would you put on a National Government Services CMS bid?",
        prime=NGS,
        agency=CMS,
        slate=(
            "PERSPECTA ENTERPRISE SOLUTIONS",
            "USA INFOTECH",
            "CGI FEDERAL",
            "MOSER CONSULTING",
            "LEIDOS",
            "RELI GROUP",
            "BEACON HILL SOLUTIONS GROUP",
            "MYTHICS",
        ),
        tiers={
            "PERSPECTA ENTERPRISE SOLUTIONS": 1,
            "USA INFOTECH": 4,
            "CGI FEDERAL": 1,
            "MOSER CONSULTING": 4,
            "LEIDOS": 1,
            "RELI GROUP": 4,
            "BEACON HILL SOLUTIONS GROUP": 4,
            "MYTHICS": 4,
        },
        reasoning="Five of the eight have reported NGS work at CMS and three do "
        "not, and the split does not follow size. Perspecta is the largest CMS "
        "prime in this data by award volume and shares NGS's hosting and "
        "integration codes, which is what makes it the wrong answer rather than "
        "the safe one.",
        answer="Five of the eight have actually worked for NGS at CMS, and they "
        "are not the five you would pick off a capability statement. USA INFOTECH "
        "leads at nine reported awards, then MOSER CONSULTING at six, RELI GROUP "
        "and BEACON HILL SOLUTIONS GROUP at five each, and MYTHICS at four.\n\n"
        "PERSPECTA ENTERPRISE SOLUTIONS is the trap. It is the largest CMS prime "
        "in these records, it matches NGS on industry code, and it has never "
        "appeared on an NGS subcontract. CGI FEDERAL and LEIDOS are wrong in the "
        "same shape: real CMS presence, no relationship with this prime.\n\n"
        "Volume at the agency tells you a company can do the work. It says "
        "nothing about whether this prime has ever chosen them, and only the "
        "second question is the one you asked.",
        gold=(
            "USA INFOTECH",
            "MOSER CONSULTING",
            "RELI GROUP",
            "BEACON HILL SOLUTIONS GROUP",
            "MYTHICS",
        ),
        facts=(
            ("used", NGS, "USA INFOTECH", 9, CMS),
            ("used", NGS, "MOSER CONSULTING", 6, CMS),
            ("used", NGS, "RELI GROUP", 5, CMS),
            ("used", NGS, "BEACON HILL SOLUTIONS GROUP", 5, CMS),
            ("used", NGS, "MYTHICS", 4, CMS),
            ("not_used", NGS, "PERSPECTA ENTERPRISE SOLUTIONS", CMS),
            ("not_used", NGS, "CGI FEDERAL", CMS),
            ("not_used", NGS, "LEIDOS", CMS),
        ),
    ),
    # ------------------------------------------------------------------
    # The obscure name is right and the famous adjacent one is wrong.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Battelle is standing up new NIH lab work. Anyone here worth a call?",
        prime=BATTELLE,
        agency=NIH,
        slate=(
            "LEIDOS BIOMEDICAL RESEARCH",
            "AMPLIFYBIO",
            "THE MITRE",
            "ANI LYTICS",
            "RESEARCH TRIANGLE INSTITUTE",
            "EMSL ANALYTICAL",
            "ENVIGO GLOBAL SERVICES",
        ),
        tiers={
            "LEIDOS BIOMEDICAL RESEARCH": 1,
            "AMPLIFYBIO": 4,
            "THE MITRE": 1,
            "ANI LYTICS": 4,
            "RESEARCH TRIANGLE INSTITUTE": 1,
            "EMSL ANALYTICAL": 4,
            "ENVIGO GLOBAL SERVICES": 4,
        },
        reasoning="AMPLIFYBIO at fifteen awards against two for everyone else is "
        "less a ranking than one relationship and a tail. The interesting half of "
        "this slate is the rejected half: LEIDOS BIOMEDICAL RESEARCH is an "
        "NIH-native life sciences contractor at real scale, which makes it the "
        "name most likely to be picked and one with no Battelle work at all.",
        answer="AMPLIFYBIO, and it is not close -- fifteen reported Battelle NIH "
        "awards against two for anyone else on the roster. ANI LYTICS, EMSL "
        "ANALYTICAL and ENVIGO GLOBAL SERVICES fill out the bench at two apiece.\n\n"
        "LEIDOS BIOMEDICAL RESEARCH is the name I would expect you to reach for "
        "and the one to leave alone. Large, NIH-native, matching research codes, "
        "and no reported subcontract with Battelle at any agency. THE MITRE and "
        "RESEARCH TRIANGLE INSTITUTE are the same story: real, adjacent, "
        "unrelated.\n\n"
        "Read the tail carefully, though. The gap between fifteen and two is a "
        "signal; the gap between two and zero is thin enough that one unfiled "
        "subcontract would close it.",
        gold=("AMPLIFYBIO", "ANI LYTICS", "EMSL ANALYTICAL", "ENVIGO GLOBAL SERVICES"),
        facts=(
            ("used", BATTELLE, "AMPLIFYBIO", 15, NIH),
            ("used", BATTELLE, "ANI LYTICS", 2, NIH),
            ("used", BATTELLE, "EMSL ANALYTICAL", 2, NIH),
            ("used", BATTELLE, "ENVIGO GLOBAL SERVICES", 2, NIH),
            ("not_used", BATTELLE, "LEIDOS BIOMEDICAL RESEARCH", NIH),
            ("not_used", BATTELLE, "THE MITRE", NIH),
            ("not_used", BATTELLE, "RESEARCH TRIANGLE INSTITUTE", NIH),
        ),
    ),
    # ------------------------------------------------------------------
    # Not an IT prime. Construction trades, and traps sharing the prime's
    # industry code and corporate profile while working a different customer.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Chugach World Services has NIH facilities work coming up. Which of these fit?",
        prime=CHUGACH,
        agency=NIH,
        slate=(
            "POLU KAI SERVICES",
            "KENT ISLAND MECHANICAL",
            "DOYON MANAGEMENT SERVICES",
            "JOHNSON CONTROLS",
            "COX CONSTRUCTION",
            "CARRIER",
            "DESA CORP OF VIRGINIA",
            "INCON",
        ),
        tiers={
            "POLU KAI SERVICES": 1,
            "KENT ISLAND MECHANICAL": 4,
            "DOYON MANAGEMENT SERVICES": 1,
            "JOHNSON CONTROLS": 4,
            "COX CONSTRUCTION": 1,
            "CARRIER": 4,
            "DESA CORP OF VIRGINIA": 4,
            "INCON": 4,
        },
        reasoning="Five of the eight are on Chugach's NIH roster, and the three "
        "that are not share more with the prime than the five do -- the same "
        "236220 building-systems code, a similar corporate profile, comparable "
        "size. What they do not share is an award.",
        answer="Five fit, and the roster runs deeper than the list suggests:\n\n"
        "- KENT ISLAND MECHANICAL - 21 reported Chugach awards at NIH, the anchor\n"
        "- JOHNSON CONTROLS - 12\n"
        "- CARRIER - 5\n"
        "- DESA CORP OF VIRGINIA - 5\n"
        "- INCON - 4\n\n"
        "Not recommended, and not near-misses either. DOYON MANAGEMENT SERVICES "
        "and COX CONSTRUCTION carry the same 236220 building-systems code and do "
        "their federal work at Indian Health Service. POLU KAI SERVICES does "
        "appear at NIH in its own right, which makes it the most convincing of "
        "the three and still leaves it without a Chugach award. Shared industry, "
        "shared customer and shared business type are each one step short of the "
        "thing you are buying, which is a prime that has hired them before.",
        gold=(
            "KENT ISLAND MECHANICAL",
            "JOHNSON CONTROLS",
            "CARRIER",
            "DESA CORP OF VIRGINIA",
            "INCON",
        ),
        facts=(
            ("used", CHUGACH, "KENT ISLAND MECHANICAL", 21, NIH),
            ("used", CHUGACH, "JOHNSON CONTROLS", 12, NIH),
            ("used", CHUGACH, "CARRIER", 5, NIH),
            ("used", CHUGACH, "DESA CORP OF VIRGINIA", 5, NIH),
            ("used", CHUGACH, "INCON", 4, NIH),
            ("not_used", CHUGACH, "POLU KAI SERVICES", NIH),
            ("not_used", CHUGACH, "DOYON MANAGEMENT SERVICES", NIH),
            ("not_used", CHUGACH, "COX CONSTRUCTION", NIH),
            ("agency_work", "POLU KAI SERVICES", NIH),
        ),
    ),
    # ------------------------------------------------------------------
    # Nothing on the slate is supported. A model never shown this case invents
    # an order instead, which is what the v1 run did.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Any of these a sensible sub for the Lewin Group at CMS?",
        prime=LEWIN,
        agency=CMS,
        slate=(
            "BATTELLE MEMORIAL INSTITUTE",
            "DUKE UNIVERSITY",
            "SOCIAL & SCIENTIFIC SYSTEMS",
            "UNIVERSITY OF ALABAMA AT BIRMINGHAM",
        ),
        tiers={
            "BATTELLE MEMORIAL INSTITUTE": 1,
            "DUKE UNIVERSITY": 1,
            "SOCIAL & SCIENTIFIC SYSTEMS": 1,
            "UNIVERSITY OF ALABAMA AT BIRMINGHAM": 1,
        },
        reasoning="All four match Lewin's 541720 research code, which is why the "
        "list looks reasonable, and none has a reported subcontract with Lewin at "
        "any agency. A ranking here would be a ranking of nothing, and it would "
        "look exactly like a ranking of evidence.",
        answer="None of the four, on this evidence.\n\n"
        "They are all real health-policy research organisations and all four "
        "match Lewin's 541720 code, which is what makes the list plausible. Not "
        "one has a reported subcontract with the Lewin Group, at CMS or anywhere "
        "else.\n\n"
        "I could order them by size, or by CMS volume, and the result would rank "
        "how prominent they are rather than how likely Lewin is to call them. You "
        "would have no way to tell those two apart by looking at the output, "
        "which is the reason not to produce it.\n\n"
        "Not recommended, and in this order of credibility: SOCIAL & SCIENTIFIC "
        "SYSTEMS and BATTELLE MEMORIAL INSTITUTE both work CMS for other primes, "
        "so they are the ones I would keep looking at; DUKE UNIVERSITY and "
        "UNIVERSITY OF ALABAMA AT BIRMINGHAM do not. Credible is not evidenced. "
        "If the shortlist has to come from this set, treat all four as cold "
        "approaches and choose on capability instead.",
        gold=(),
        facts=(
            ("not_used", LEWIN, "BATTELLE MEMORIAL INSTITUTE", CMS),
            ("not_used", LEWIN, "DUKE UNIVERSITY", CMS),
            ("not_used", LEWIN, "SOCIAL & SCIENTIFIC SYSTEMS", CMS),
            ("not_used", LEWIN, "UNIVERSITY OF ALABAMA AT BIRMINGHAM", CMS),
        ),
    ),
    # ------------------------------------------------------------------
    # Role is a property of the award, not of the company: a large prime turns
    # up on another prime's sub roster.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Lewin Group, CMS work -- rank these by how likely they are to end up on the team.",
        prime=LEWIN,
        agency=CMS,
        slate=(
            "BATTELLE MEMORIAL INSTITUTE",
            "KNG HEALTH CONSULTING",
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY",
            "INSTITUTE FOR HEALTHCARE IMPROVEMENT",
            "ABT GLOBAL",
            "DUKE UNIVERSITY",
        ),
        tiers={
            "BATTELLE MEMORIAL INSTITUTE": 1,
            "KNG HEALTH CONSULTING": 4,
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY": 4,
            "INSTITUTE FOR HEALTHCARE IMPROVEMENT": 4,
            "ABT GLOBAL": 4,
            "DUKE UNIVERSITY": 1,
        },
        reasoning="The third name is the one worth stopping on. GDIT is the prime "
        "on 178 awards of its own in this data and appears as a subcontractor to "
        "Lewin at CMS five times. A shortlist filtered by who counts as a prime "
        "would drop a real partner on a category error.",
        answer="1. KNG HEALTH CONSULTING - 6 reported Lewin awards at CMS\n"
        "2. INSTITUTE FOR HEALTHCARE IMPROVEMENT - 6\n"
        "3. GENERAL DYNAMICS INFORMATION TECHNOLOGY - 5\n"
        "4. ABT GLOBAL - 4\n\n"
        "The third entry looks like a mistake and is not. GDIT is the prime on "
        "178 awards of its own in these records, and on Lewin's CMS work it shows "
        "up five times as the subcontractor. Role belongs to the award, not to "
        "the company, and a filter treating \"prime\" as a property of a firm "
        "would have thrown this one out.\n\n"
        "Not recommended: BATTELLE MEMORIAL INSTITUTE and DUKE UNIVERSITY. Both "
        "match Lewin on research codes, neither has a reported award with them.",
        gold=(
            "KNG HEALTH CONSULTING",
            "INSTITUTE FOR HEALTHCARE IMPROVEMENT",
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY",
            "ABT GLOBAL",
        ),
        facts=(
            ("used", LEWIN, "KNG HEALTH CONSULTING", 6, CMS),
            ("used", LEWIN, "INSTITUTE FOR HEALTHCARE IMPROVEMENT", 6, CMS),
            ("used", LEWIN, "GENERAL DYNAMICS INFORMATION TECHNOLOGY", 5, CMS),
            ("used", LEWIN, "ABT GLOBAL", 4, CMS),
            ("as_prime", GDIT, 178),
            ("not_used", LEWIN, "BATTELLE MEMORIAL INSTITUTE", CMS),
            ("not_used", LEWIN, "DUKE UNIVERSITY", CMS),
        ),
    ),
    # ------------------------------------------------------------------
    # The premise is wrong, and tier 3 -- a real relationship with this prime
    # at a different customer -- is worth naming as its own category.
    # ------------------------------------------------------------------
    OpenBook(
        ask="We want in on GDIT's FDA work through one of their regular subs. "
        "Which of these is the way in?",
        prime=GDIT,
        agency=FDA,
        slate=(
            "CARAHSOFT TECHNOLOGY SOLUTIONS",
            "SRAAYA",
            "LEIDOS",
            "SNOWFLAKE",
            "DELOITTE CONSULTING LLP",
        ),
        tiers={
            "CARAHSOFT TECHNOLOGY SOLUTIONS": 3,
            "SRAAYA": 4,
            "LEIDOS": 1,
            "SNOWFLAKE": 3,
            "DELOITTE CONSULTING LLP": 1,
        },
        reasoning="The premise does not survive the records: GDIT reports five "
        "distinct subcontractors across all of its FDA work and only one of them "
        "more than once. There is no regular FDA bench to route through. "
        "CARAHSOFT TECHNOLOGY SOLUTIONS and SNOWFLAKE are the useful finding "
        "instead -- substantial GDIT relationships, none of it at this agency.",
        answer="The question assumes a bench that isn't there. GDIT reports five "
        "distinct subcontractors across its FDA work, and only SRAAYA appears "
        "more than once, at two awards. There is no regular FDA sub to go "
        "through, so it is the plan that needs changing rather than the name.\n\n"
        "SRAAYA is the only company on this list with reported GDIT work at FDA. "
        "CARAHSOFT TECHNOLOGY SOLUTIONS and SNOWFLAKE are worth understanding "
        "rather than dismissing: both hold genuine GDIT relationships -- "
        "CARAHSOFT TECHNOLOGY SOLUTIONS on 29 awards, 28 of them at CMS -- and "
        "none of that work is FDA, and most of it is licence resale rather than "
        "services teaming. An existing relationship with the prime at another "
        "customer is a warmer start than nothing. It is not FDA past performance "
        "and should not be sold as one.\n\n"
        "Not recommended: LEIDOS and DELOITTE CONSULTING LLP, which have neither "
        "the agency nor the relationship.",
        gold=("SRAAYA",),
        facts=(
            ("team_size", GDIT, FDA, 5),
            ("used", GDIT, "SRAAYA", 2, FDA),
            ("used", GDIT, "CARAHSOFT TECHNOLOGY SOLUTIONS", 29),
            ("used", GDIT, "CARAHSOFT TECHNOLOGY SOLUTIONS", 28, CMS),
            ("not_used", GDIT, "CARAHSOFT TECHNOLOGY SOLUTIONS", FDA),
            ("not_used", GDIT, "SNOWFLAKE", FDA),
            ("not_used", GDIT, "LEIDOS", FDA),
            ("not_used", GDIT, "DELOITTE CONSULTING LLP", FDA),
        ),
    ),
    # ------------------------------------------------------------------
    # Tier 3 again, as the answer to a question the asker did not quite ask.
    # ------------------------------------------------------------------
    OpenBook(
        ask="RTI, SAMHSA work. Who are the safe picks, and is there anyone I'm underrating?",
        prime=RTI,
        agency=SAMHSA,
        slate=(
            "TELLIGEN",
            "HR DIRECTIONS",
            "SNOWFLAKE",
            "PERFECT GIFT",
            "DELOITTE CONSULTING LLP",
            "HEADWAY HR SOLUTIONS",
            "DESA",
        ),
        tiers={
            "TELLIGEN": 3,
            "HR DIRECTIONS": 4,
            "SNOWFLAKE": 0,
            "PERFECT GIFT": 4,
            "DELOITTE CONSULTING LLP": 1,
            "HEADWAY HR SOLUTIONS": 4,
            "DESA": 4,
        },
        reasoning="Four have SAMHSA work with RTI and the ordering is by award "
        "count. TELLIGEN is what the second half of the question is about: no "
        "SAMHSA work with RTI, four reported RTI subcontracts elsewhere, which "
        "makes it a different kind of candidate rather than a worse one.",
        answer="Safe picks, by how often RTI has gone back to them at SAMHSA:\n\n"
        "- HR DIRECTIONS - 6 reported awards\n"
        "- HEADWAY HR SOLUTIONS - 5\n"
        "- PERFECT GIFT - 5\n"
        "- DESA - 3\n\n"
        "The one you are underrating is TELLIGEN. It has no SAMHSA work with RTI, "
        "so it does not belong in that list -- but it holds four reported RTI "
        "subcontracts at other components. That is an established relationship "
        "with the prime looking for a new customer, which is usually an easier "
        "conversation than a cold introduction and a different pitch entirely: "
        "you are asking them to extend something rather than to start it.\n\n"
        "Not recommended: SNOWFLAKE and DELOITTE CONSULTING LLP. Neither has "
        "worked with RTI anywhere. They are on this list because they are large "
        "HHS names, which is the only thing they have in common with the rest.",
        gold=("HR DIRECTIONS", "HEADWAY HR SOLUTIONS", "PERFECT GIFT", "DESA"),
        facts=(
            ("used", RTI, "HR DIRECTIONS", 6, SAMHSA),
            ("used", RTI, "HEADWAY HR SOLUTIONS", 5, SAMHSA),
            ("used", RTI, "PERFECT GIFT", 5, SAMHSA),
            ("used", RTI, "DESA", 3, SAMHSA),
            ("used", RTI, "TELLIGEN", 4),
            ("not_used", RTI, "TELLIGEN", SAMHSA),
            ("used", RTI, "SNOWFLAKE", 0),
            ("used", RTI, "DELOITTE CONSULTING LLP", 0),
        ),
    ),
    # ------------------------------------------------------------------
    # Two names were asked for. The corpus needs cases where brevity is the
    # correct behaviour and not an abbreviation of the right answer.
    # ------------------------------------------------------------------
    OpenBook(
        ask="Just give me the top two for a Perspecta CMS bid, and tell me why GDIT isn't one of them.",
        prime=PERSPECTA,
        agency=CMS,
        slate=(
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY",
            "HP",
            "LEIDOS",
            "ORACLE AMERICA",
            "CARAHSOFT TECHNOLOGY",
            "CISCO SYSTEMS",
        ),
        tiers={
            "GENERAL DYNAMICS INFORMATION TECHNOLOGY": 2,
            "HP": 4,
            "LEIDOS": 1,
            "ORACLE AMERICA": 4,
            "CARAHSOFT TECHNOLOGY": 4,
            "CISCO SYSTEMS": 4,
        },
        reasoning="Two names were asked for, so two names is the answer. GDIT's "
        "exclusion is the substantive half: it works CMS at greater scale than "
        "anything on this list and has never subcontracted to Perspecta.",
        answer="HP and CARAHSOFT TECHNOLOGY -- 27 and 21 reported Perspecta "
        "awards at CMS.\n\n"
        "GENERAL DYNAMICS INFORMATION TECHNOLOGY is not among them because it has "
        "never subcontracted to Perspecta, at CMS or anywhere. It is one of the "
        "largest CMS contractors in this data and it does the same hosting and "
        "integration work, which makes it a competitor for this prime's business "
        "rather than a candidate for its team. Working the same agency and "
        "working for the same prime are different facts, and only the second one "
        "predicts who Perspecta brings on.",
        gold=("HP", "CARAHSOFT TECHNOLOGY"),
        facts=(
            ("used", PERSPECTA, "HP", 27, CMS),
            ("used", PERSPECTA, "CARAHSOFT TECHNOLOGY", 21, CMS),
            ("not_used", PERSPECTA, "GENERAL DYNAMICS INFORMATION TECHNOLOGY", CMS),
            ("not_used", PERSPECTA, "LEIDOS", CMS),
        ),
    ),
    # ------------------------------------------------------------------
    # Sorting by size returns exactly the wrong answer, and the asker has said
    # out loud that they expect it to, so the answer has to disagree.
    # ------------------------------------------------------------------
    OpenBook(
        ask="ICF at NIH. These all look small to me -- am I missing the real players?",
        prime=ICF,
        agency=NIH,
        slate=(
            "RESEARCH TRIANGLE INSTITUTE",
            "JML DESIGN",
            "ACCENTURE FEDERAL SERVICES",
            "CONSOLIDATED SOLUTIONS & INNOVATIONS",
            "MOBOMO",
            "PERATON",
            "DILTS PARTNERS",
        ),
        tiers={
            "RESEARCH TRIANGLE INSTITUTE": 1,
            "JML DESIGN": 4,
            "ACCENTURE FEDERAL SERVICES": 1,
            "CONSOLIDATED SOLUTIONS & INNOVATIONS": 4,
            "MOBOMO": 4,
            "PERATON": 1,
            "DILTS PARTNERS": 4,
        },
        reasoning="The small names are the real ones. ICF's NIH subcontracting "
        "goes to design and communications shops, several of them sole "
        "proprietors, across 22 distinct subcontractors. The three large firms on "
        "the slate share ICF's 541611 consulting code, dwarf it in scale, and "
        "have no reported subcontract with it.",
        answer="You are not missing them. The small names are the real players.\n\n"
        "1. JML DESIGN - 5 reported ICF awards at NIH\n"
        "2. CONSOLIDATED SOLUTIONS & INNOVATIONS - 4\n"
        "3. MOBOMO - 3\n"
        "4. DILTS PARTNERS - 3\n\n"
        "ICF subcontracts its NIH work to design, web and communications shops, "
        "several of them individuals trading under their own names. Twenty-two "
        "distinct subcontractors and not a household name among them.\n\n"
        "Not recommended: RESEARCH TRIANGLE INSTITUTE, ACCENTURE FEDERAL SERVICES "
        "and PERATON. All three carry ICF's 541611 management-consulting code, "
        "all three are far larger than the firms above, and none has a reported "
        "subcontract with ICF. A shortlist sorted by size returns exactly these "
        "three, which is a good demonstration that size is the wrong sort key "
        "for this question.",
        gold=("JML DESIGN", "CONSOLIDATED SOLUTIONS & INNOVATIONS", "MOBOMO", "DILTS PARTNERS"),
        facts=(
            ("used", ICF, "JML DESIGN", 5, NIH),
            ("used", ICF, "CONSOLIDATED SOLUTIONS & INNOVATIONS", 4, NIH),
            ("used", ICF, "MOBOMO", 3, NIH),
            ("used", ICF, "DILTS PARTNERS", 3, NIH),
            ("team_size", ICF, NIH, 22),
            ("not_used", ICF, "RESEARCH TRIANGLE INSTITUTE", NIH),
            ("not_used", ICF, "ACCENTURE FEDERAL SERVICES", NIH),
            ("not_used", ICF, "PERATON", NIH),
        ),
    ),
    # ------------------------------------------------------------------
    # The direct fix for off-slate naming: a company in the records that would
    # be a good answer, and still is not an answer, because it wasn't offered.
    # ------------------------------------------------------------------
    OpenBook(
        ask="ICF, NIH task orders -- who from this list, and flag anything I should "
        "have put on it.",
        prime=ICF,
        agency=NIH,
        slate=(
            "RESEARCH TRIANGLE INSTITUTE",
            "JML DESIGN",
            "ACCENTURE FEDERAL SERVICES",
            "CONSOLIDATED SOLUTIONS & INNOVATIONS",
            "MOBOMO",
            "PERATON",
            "DILTS PARTNERS",
        ),
        tiers={
            "RESEARCH TRIANGLE INSTITUTE": 1,
            "JML DESIGN": 4,
            "ACCENTURE FEDERAL SERVICES": 1,
            "CONSOLIDATED SOLUTIONS & INNOVATIONS": 4,
            "MOBOMO": 4,
            "PERATON": 1,
            "DILTS PARTNERS": 4,
        },
        reasoning="Same four picks as the ranking question, and the second half "
        "of the ask is the interesting one. The retrieved records include ICF "
        "subcontractors that are not on the slate; naming them as picks would be "
        "answering a question that was not asked, and not naming them at all "
        "would waste the fact that the asker invited additions.",
        answer="From your list: JML DESIGN (5 reported ICF awards at NIH), "
        "CONSOLIDATED SOLUTIONS & INNOVATIONS (4), MOBOMO (3) and DILTS PARTNERS "
        "(3). RESEARCH TRIANGLE INSTITUTE, ACCENTURE FEDERAL SERVICES and PERATON "
        "have no ICF subcontracts at all and are on the list because they are "
        "large.\n\n"
        "You did ask what you missed, so: the records in front of me also cover "
        "ICF subcontractors that are not among your options -- DAZA MARTHA S is "
        "one, with three reported ICF awards at NIH, which would put it level "
        "with MOBOMO and DILTS PARTNERS. I have not counted it as a pick, because "
        "a record appearing in the library is not the same as a name appearing on "
        "your slate, and the question was which of these. It belongs on the next "
        "version of the list.",
        gold=("JML DESIGN", "CONSOLIDATED SOLUTIONS & INNOVATIONS", "MOBOMO", "DILTS PARTNERS"),
        facts=(
            ("used", ICF, "JML DESIGN", 5, NIH),
            ("used", ICF, "CONSOLIDATED SOLUTIONS & INNOVATIONS", 4, NIH),
            ("used", ICF, "MOBOMO", 3, NIH),
            ("used", ICF, "DILTS PARTNERS", 3, NIH),
            ("used", ICF, "DAZA MARTHA S", 3, NIH),
            ("not_used", ICF, "RESEARCH TRIANGLE INSTITUTE", NIH),
            ("not_used", ICF, "ACCENTURE FEDERAL SERVICES", NIH),
            ("not_used", ICF, "PERATON", NIH),
        ),
        off_slate_named=("DAZA MARTHA S",),
    ),
    # ------------------------------------------------------------------
    # One name asked for, one name given, in three sentences.
    # ------------------------------------------------------------------
    OpenBook(
        ask="One name for an RTI CDC bid. Who?",
        prime=RTI,
        agency=CDC,
        slate=(
            "CARAHSOFT TECHNOLOGY SOLUTIONS",
            "HR DIRECTIONS",
            "SNOWFLAKE",
            "PATIENT PERFORMANCE",
            "GERLING & ASSOCIATES",
            "M9 SOLUTIONS",
        ),
        tiers={
            "CARAHSOFT TECHNOLOGY SOLUTIONS": 1,
            "HR DIRECTIONS": 4,
            "SNOWFLAKE": 0,
            "PATIENT PERFORMANCE": 4,
            "GERLING & ASSOCIATES": 4,
            "M9 SOLUTIONS": 0,
        },
        reasoning="One was asked for and the data supports one: HR DIRECTIONS at "
        "ten CDC awards with RTI, two and a half times the next name.",
        answer="HR DIRECTIONS -- ten reported RTI awards at CDC, against four for "
        "PATIENT PERFORMANCE and three for GERLING & ASSOCIATES.\n\n"
        "Not recommended: CARAHSOFT TECHNOLOGY SOLUTIONS, SNOWFLAKE and M9 "
        "SOLUTIONS, all CMS companies with no RTI work at CDC. They are on this "
        "list because they are large, not because they fit.",
        gold=("HR DIRECTIONS",),
        facts=(
            ("used", RTI, "HR DIRECTIONS", 10, CDC),
            ("used", RTI, "PATIENT PERFORMANCE", 4, CDC),
            ("used", RTI, "GERLING & ASSOCIATES", 3, CDC),
            ("not_used", RTI, "CARAHSOFT TECHNOLOGY SOLUTIONS", CDC),
            ("not_used", RTI, "SNOWFLAKE", CDC),
            ("not_used", RTI, "M9 SOLUTIONS", CDC),
        ),
    ),
    # ------------------------------------------------------------------
    # What an absent record does and does not license you to say. The corpus
    # has almost nothing on bounding a negative claim.
    # ------------------------------------------------------------------
    OpenBook(
        ask="None of these show up in RTI's CDC records. Does that mean they've never worked together?",
        prime=RTI,
        agency=CDC,
        slate=(
            "SNOWFLAKE",
            "M9 SOLUTIONS",
            "CARAHSOFT TECHNOLOGY SOLUTIONS",
        ),
        tiers={
            "SNOWFLAKE": 0,
            "M9 SOLUTIONS": 0,
            "CARAHSOFT TECHNOLOGY SOLUTIONS": 1,
        },
        reasoning="The correct answer is about the strength of the evidence "
        "rather than about the companies. Subaward reporting has thresholds and "
        "gaps, so absence supports 'not an established partner' and not 'never "
        "worked together'. Stating the weaker claim accurately is the whole job.",
        answer="No -- it means there is no reported subcontract, which is a "
        "weaker claim than it sounds.\n\n"
        "Subaward reporting has thresholds and gaps. Work below the reporting "
        "floor, work filed under a different corporate name, and work simply "
        "never filed all look identical in this data to work that never happened. "
        "The absence supports one conclusion firmly -- none of these three is an "
        "established RTI partner at CDC -- and cannot support the stronger one "
        "you asked about.\n\n"
        "Not recommended either way, for a reason that has nothing to do with "
        "the gaps: SNOWFLAKE, M9 SOLUTIONS and CARAHSOFT TECHNOLOGY SOLUTIONS are "
        "CMS companies. It is the customer mismatch rather than the missing "
        "record that makes them poor fits for CDC work, so the reporting question "
        "does not change the recommendation here even though it is a fair "
        "question to have asked.",
        gold=(),
        facts=(
            ("not_used", RTI, "SNOWFLAKE", CDC),
            ("not_used", RTI, "M9 SOLUTIONS", CDC),
            ("not_used", RTI, "CARAHSOFT TECHNOLOGY SOLUTIONS", CDC),
        ),
    ),
)


def all_open_book() -> tuple[OpenBook, ...]:
    """Both batches. Split across two files only to keep each readable; they are
    one corpus and are checked by one test."""
    from .authored_context_wide import WIDE

    return (*EXAMPLES, *WIDE)


def context_examples(graph: Any, index: Any, repeat: int = 1) -> list[dict[str, Any]]:
    """The hand-written open-book set, in corpus form.

    Context is built by the same ``context_for`` the generated questions and the
    blind set use, so these rows are indistinguishable in shape from what the
    model is served at inference. Writing a context block by hand instead would
    teach a prompt format that never occurs.
    """
    from ..shared.questions import Question
    from ..shared.records import context_for

    rows: list[dict[str, Any]] = []
    for _ in range(repeat):
        for example in all_open_book():
            question = Question(
                question=example.question,
                answer=example.answer,
                reasoning=example.reasoning,
                archetype="authored_slate",
                gold=list(example.gold),
                tiers=dict(example.tiers),
                meta={
                    "prime": example.prime,
                    "agency": example.agency,
                    "authored": True,
                },
            )
            record = question.to_record()
            record["context"] = context_for(graph, question, index)
            record["meta"]["closed_book"] = False
            rows.append(record)
    return rows
