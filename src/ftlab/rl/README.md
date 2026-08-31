# Reinforcement learning

**Can training improve a model's judgement, if it cannot put facts into it?**

The [first experiment](../sft/README.md) found that fine-tuning did not teach the
model facts — the untrained model with a keyword search was just as accurate.
What training *did* buy was discipline.

So this one stops trying to teach facts. The facts stay outside, in documents the
model is handed. What gets trained is the **reasoning** it does over them.

---

## The test

Take a subcontract that really happened after the cutoff date. Hide who got it.
Show the model the prime contractor, twelve plausible candidate firms, and the
public record for each one. Ask which company actually won the work.

```
CGI FEDERAL took on a new subcontractor for CMS work.
Which of these was it? Rank your top five, most likely first.
1. CATAPULT STAFFING     5. CISCO SYSTEMS       9. SAMTEK
2. ORACLE AMERICA        6. IMPETUS TECH       10. ORAN        ← the answer
3. SHI INTERNATIONAL     7. RESOLVESOFT        11. THE ACI GROUP
4. SPERIDIAN             8. CARAHSOFT          12. HP
```

Right or wrong is history, not opinion. That is the whole reason this works:
the model can be scored **automatically, thousands of times**, without anyone
writing an answer key. Training against a score nobody authored is the difference
between reinforcement learning and guesswork.

**872 questions, 162 prime contractors.** They split in two, and only one half is
interesting:

| | questions | what it measures |
|---|---|---|
| the prime had used this firm before | 587 | can the model read a partner list |
| **the prime had never used this firm** | **285** | can the model actually reason |

On the first half you can score well by counting who someone worked with last
time. On the second half that strategy scores **zero by construction** — the
relationship does not exist yet.

## The starting point

We measured the untrained model first, because a result without a baseline is not
a result.

| | hit rate | random guessing |
|---|---|---|
| all questions | 43.3% | 8.3% |
| had worked together before | 57.7% | 8.3% |
| **never worked together** | **8.3%** | **8.3%** |

**Exactly chance.** 13 correct out of 156 — one in twelve, to four decimal
places.

The headline 43% is entirely look-up. Given a prime contractor it had never seen
paired with any of these firms, the model with full access to every company's
record has **no signal at all**. That is the honest starting line, and it means
there is a lot of room to improve.

## How training works

For each question the model writes several different answers. Each is scored, and
the ones that beat the group's average get reinforced. Nothing is compared
against a "correct" essay — only against whether the right company was named, and
how near the top.

The score is deliberately plain:

- **1.0** for naming the right firm first, 0.5 for second, down to 0.2 for fifth
- **−0.1** for each company named that was not on the list
- **−0.2** for not answering at all

It scores the ranking and nothing else. Rewarding the *explanation* would mean
rewarding whatever we think an explanation should look like, which is the
hand-written answer key problem wearing a disguise.

**Training and test share no companies.** The 338 training questions and the 534
test questions come from separate prime contractors. Splitting them any other way
would let the model memorise one firm's suppliers during training and then score
on that same firm at test time.

## Where it stands

A first training run is under way. **There is no result yet**, and it may well be
that there is no improvement to find — the test was built to be able to say so.

One thing already learned from watching it: **just over half of the training
steps produce no signal.** When all of a question's answers come out equally
right or equally wrong, there is nothing to learn from that question. Most of
those are the easy look-up questions the model already gets right.

---

## What is in here

| | |
|---|---|
| `reward.py` | turns an answer into a score |
| `rollout.py` | runs the model over a set of questions and scores the results |
| `train.py` | the training loop |

## Running it

```bash
uv run ftlab masked-build                       # build the questions
uv run ftlab masked-run -c real-3arm.yaml       # measure a model
uv run ftlab masked-train -c real-3arm.yaml     # train one
```

**Before starting a long run, read the efficiency notes in
[AGENTS.md](../../../AGENTS.md).** Two runs were lost to the same failure: the
graphics card quietly runs out of memory, reports no error, and everything just
gets four times slower. There is a checklist for it.
