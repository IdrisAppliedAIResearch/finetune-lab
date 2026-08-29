"""Answer-shape variation, and general instruction data to hold the rest open.

The first fine-tune collapsed to templates: eighteen of eighteen blind answers
were forced into one of the seven answer shapes it had been trained on, and the
one blind question type absent from training scored zero where random scored
0.36. Asked which companies were new to NIH, it produced a well-formed list of
primes to approach.

That is not a mysterious failure. Every training answer for a given archetype
had an identical structure, so "recognise the archetype, emit its template" is
both the easiest rule to learn and sufficient to drive the loss down. The model
learned the rule that was available.

Two counterweights here.

``render_list`` and ``phrase`` vary the *shape* of an answer without touching
its content, so the same fact arrives as a numbered list, as bullets, as a
sentence, with the conclusion first or last. An archetype stops predicting a
format, which leaves the content as the only thing worth learning.

``general_examples`` supplies instruction-following that has nothing to do with
contracting -- following an explicit format, answering briefly, declining to
invent, working a short problem step by step. A model fine-tuned only on seven
domain templates has no reason to keep any of that, and arm C's advantage on the
unseen question type was precisely that it still had it.
"""

from __future__ import annotations

import random
from typing import Any

LIST_STYLES = ("numbered", "bulleted", "inline", "ranked")


def render_list(names: list[str], style: str, rng: random.Random | None = None) -> str:
    """One list of names, rendered several ways.

    Content identical, shape different. This is the whole point: if every
    answer of a kind looks the same, the shape becomes the answer.
    """
    if not names:
        return "nothing on record"
    if style == "numbered":
        return "\n".join(f"{i}. {n}" for i, n in enumerate(names, start=1))
    if style == "bulleted":
        return "\n".join(f"- {n}" for n in names)
    if style == "ranked":
        head, *rest = names
        lines = [f"1. {head} (strongest fit)"]
        lines += [f"{i}. {n}" for i, n in enumerate(rest, start=2)]
        return "\n".join(lines)
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def phrase(key: str, rng: random.Random, **fields: Any) -> str:
    """One of several equivalent sentences, chosen per example.

    Deliberately not a single house style. A fixed opener is another template
    for the model to key on, and it will take it.
    """
    options = PHRASINGS[key]
    return rng.choice(options).format(**fields)


PHRASINGS: dict[str, tuple[str, ...]] = {
    "team_lead": (
        "{prime}'s {short} team, most-used first:",
        "On {short} work, {prime} has used these subcontractors:",
        "Who {prime} brings to {short}:",
        "{prime} reports these subs on {short} awards:",
    ),
    "sub_lead": (
        "Sub candidates for {prime} on {short} work:",
        "For a {prime} bid at {short}, put these forward:",
        "The credible subs here:",
        "Recommended for the {prime} {short} team:",
    ),
    "reject_lead": (
        "Not recommended, despite looking like obvious picks:",
        "Worth rejecting out loud:",
        "Leave these off, and why:",
        "These look right and are not:",
    ),
    "no_record": (
        "Nothing on record supports that.",
        "The records do not show it.",
        "No reported evidence either way.",
        "Not something the library settles.",
    ),
}


# ---------------------------------------------------------------------------
# general instruction data
# ---------------------------------------------------------------------------

# Deliberately not about contracting. The point is to keep the general
# instruction-following the base model already has, which a diet of seven domain
# templates otherwise overwrites -- that capability is exactly what let the
# untuned model beat both fine-tuned arms on an unfamiliar question.
GENERAL: tuple[tuple[str, str, str], ...] = (
    (
        "Answer in one sentence: why do ranked lists need a stated criterion?",
        "Because without one the order is unfalsifiable -- a reader cannot tell a "
        "considered ranking from an arbitrary one.",
        "The instruction is the constraint: one sentence, so the answer is one "
        "sentence rather than a paragraph that happens to start well.",
    ),
    (
        "List three things, as bullets, that make a claim checkable.",
        "- a stated source\n- a number that could have come out differently\n"
        "- a method someone else could repeat",
        "Format asked for bullets and a count of three; both are part of the answer.",
    ),
    (
        "What is the population of the city I am thinking of?",
        "I have no way to know which city you mean. Tell me which one and I can "
        "answer, or I can tell you where to look it up.",
        "The question cannot be answered as asked. Guessing a number would be "
        "worse than saying so.",
    ),
    (
        "A team of 4 reviews 3 documents each per day. How many in a 5-day week?",
        "60. Four reviewers times three documents is twelve a day, times five days.",
        "Small arithmetic, shown rather than asserted, so the reader can check the "
        "step where it would go wrong.",
    ),
    (
        "Summarise in under 20 words: a metric with a high random baseline can look "
        "like a strong result when it is not.",
        "A metric scores well by chance unless you report what chance scores.",
        "A length limit is a real constraint, not a suggestion.",
    ),
    (
        "Rewrite this as a question: The contract ended in 2024.",
        "When did the contract end?",
        "A transformation task: the output form is specified, and the content is "
        "carried over unchanged.",
    ),
    (
        "I think 7 is a prime number. Am I right?",
        "Yes. Seven has no divisors other than 1 and itself.",
        "Confirming a correct claim plainly, without hedging it into ambiguity.",
    ),
    (
        "I think 9 is a prime number. Am I right?",
        "No. Nine is 3 times 3, so it has a divisor other than 1 and itself.",
        "Correcting an incorrect claim directly, and saying what makes it wrong.",
    ),
    (
        "Give me the answer as JSON with keys 'verdict' and 'why': is a sample of "
        "18 enough to separate two close results?",
        '{"verdict": "no", "why": "differences of a few points are within noise at '
        'that sample size"}',
        "An explicit output format, followed exactly.",
    ),
    (
        "Explain to someone non-technical what a held-out test set is, in two "
        "sentences.",
        "It is a portion of the data put aside and never used while building the "
        "model. You check against it at the end, so a good score means the model "
        "generalised rather than memorised.",
        "Audience and length are both specified; jargon has to go.",
    ),
    (
        "What did I ask you in our first conversation?",
        "I have no memory of previous conversations, so I cannot say.",
        "The honest answer is a limitation, not an invented recollection.",
    ),
    (
        "Reply with exactly the word DONE and nothing else.",
        "DONE",
        "An instruction about the whole output, including what must not be added.",
    ),
)


def general_examples(count: int = 120, seed: int = 42) -> list[dict[str, Any]]:
    """Repeat the general set to a target size, tagged so it can be filtered.

    Kept small relative to the domain data -- enough to hold instruction
    following open, not enough to crowd out what the model is being taught.
    """
    rng = random.Random(seed)
    pool = list(GENERAL)
    rows: list[dict[str, Any]] = []
    while len(rows) < count:
        rng.shuffle(pool)
        for question, answer, reasoning in pool:
            if len(rows) >= count:
                break
            rows.append(
                {
                    "question": question,
                    "reasoning": reasoning,
                    "answer": answer,
                    "context": "",
                    "meta": {
                        "archetype": "general_instruction",
                        "gold": [],
                        "tiers": {},
                        "closed_book": True,
                        "general": True,
                    },
                }
            )
    return rows
