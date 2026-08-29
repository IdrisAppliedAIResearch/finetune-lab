# Three-arm result — v2, 51 blind questions

Answers from subcontracts dated after the training cutoff; 71% of the teaming
pairs involved appear nowhere in training. Floor is a random pick from each
question's own slate.

| metric | floor | **C** base+RAG | **B** tuned only | **A** tuned+RAG |
|---|---|---|---|---|
| **precision@4** | 0.369 | **0.477** | 0.306 | 0.445 |
| recall of key | 0.333 | 0.272 | 0.276 | **0.331** |
| mean tier of picks | 1.477 | **1.908** | 1.224 | 1.780 |
| **hard negatives recommended** *(lower better)* | 2.510 | **1.039** | 2.627 | 1.647 |
| hard negatives rejected | 0.000 | 0.194 | 0.172 | 0.169 |
| answers naming anything | 1.000 | 0.784 | 0.961 | 0.922 |
| named off the slate | 0.000 | 0.550 | **0.000** | 0.298 |
| answers in the WRONG template | 0.000 | **0.000** | 0.039 | 0.098 |
| **truncated** | — | **51/51** | 0/51 | 0/51 |
| output length (chars) | — | 8,353 | 1,233 | **1,015** |

## Reading

**Fine-tuning did not buy ranking accuracy.** Arm C — the untouched base model
with the same retrieved records — scores 0.477 against arm A's 0.445, and
recommends fewer hard negatives (1.04 against 1.65). Both clear the floor. On
the measure the demo was built around, retrieval is doing the work.

Arm C's picks match the slate's listed order in only 3 of 51 answers, so this is
real discrimination and not an artefact of reading its enumeration in order —
which is what the equivalent v1 number turned out to be.

**Fine-tuning bought answer discipline, and a great deal of it.** Arm A
terminates on every question where arm C terminates on none, even at a 2,500
token budget; it stays on the offered slate far better (0.298 against 0.550
off-slate names); it produces an answer more often; and it does so in **1,015
characters against 8,353** — eight times cheaper to serve, per query, forever.

**Closed-book does not work.** Arm B, the same adapter without retrieval, sits
*below* the random floor at 0.306. The graph is not in the weights, which
matches the closed-book finding from the earlier run: 29% fact recall, accuracy
tracking the entropy of the answer space rather than anything learned.

## Caveat that cuts one way

Arm C is truncated on every question, so its numbers come from unfinished
analyses. That is a real limitation of the comparison — but it biases *against*
arm C, since a cut-off answer can only lose picks. Its 0.477 is therefore closer
to a lower bound than an inflated figure, which makes the negative result on
accuracy stronger rather than weaker.

That arm C never terminates at any budget tried is itself a finding, not just a
measurement problem: the base model does not produce a bounded answer to these
questions.

## What this says about the original question

Asked whether a fine-tuned local model out-reasons a base model plus a
deterministic retrieval architecture: **on accuracy, no.** On cost and
deployability — bounded, on-slate, terminating answers at an eighth of the
tokens — **yes, clearly.**
