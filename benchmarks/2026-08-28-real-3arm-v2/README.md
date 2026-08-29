# v2 corpus, trained — arms not yet measured

**2026-08-28** · `google/gemma-4-12B-it` + QLoRA r=32 · authored + widened corpus

Training complete and recorded. **No three-arm result yet**: the first run's
arms measurement was found invalid and the re-measurement has not been run.

## What changed from v1

| | v1 | v2 |
|---|---|---|
| archetypes | 7 | **13** |
| train examples | 1,335 | **1,829** |
| authored examples | 0 | **152** (19 written by hand, repeated) |
| general instruction data | 0 | **150** |
| answer shapes | one per archetype | 899 numbered / 851 prose / 79 bulleted |
| blind questions | 18 | **51** |

Motivated by the v1 finding, which stands: 18 of 18 blind answers were forced
into one of seven trained templates. Asked which companies were new to NIH, the
model produced a well-formed list of primes to approach. That is not a
truncation artefact — a cut-off answer is an unfinished answer to the *right*
question.

## Training

| | |
|---|---|
| wall time | 1h 24m 54s (230 steps, 2 epochs) |
| cost | $0.13 · peak allocated 22.02 GB |
| train loss | 3.7604 → 0.0824 |
| eval loss | 0.7399 → 0.2983 (best **0.2923 @ step 175**) |

Better eval loss than v1's 0.3546 on a materially harder corpus. Both
checkpoints kept: `checkpoint-115` (epoch 1) and `checkpoint-230` (epoch 2).

### Gate: STOP, on two checks

```
[PASS] still learning      +4.93%
[FAIL] not memorising      eval 0.2983 vs train 0.0824 = 72.4% (limit 10%)
[FAIL] still at the floor  last 0.2983 vs best 0.2923 @ step 175
```

The best model was mid-second-epoch and the run got slightly worse after it.
`checkpoint-115` is therefore a genuine candidate rather than a fallback.

## Why the v1 arms numbers were struck

Two defects in the measurement harness, both found by inspecting generations
before re-running:

1. **Truncation.** The 900-token budget was 420. Answers cut off mid-sentence:
   90% (arm C), 70% (B), 90% (A). Every arm was scored on answers it had not
   finished writing.
2. **Graded on notes, not conclusions.** The base model enumerated all twelve
   slate candidates in order before choosing; the grader read the first four
   names it found. That is close to a random draw, and close to the score it
   received — so arm C sitting at the floor was likely an artefact of reading,
   not a fact about the model.

Both fixed and applied identically to every arm. `conclusion_of` reads the text
after the last concluding phrase, falling back to the whole answer when a model
never concludes. Truncation and template collapse are now reported metrics
rather than things noticed by eye.

A third defect was found in the fix itself: the word-boundary escapes in the
conclusion regex had been written as literal 0x08 backspace bytes, so it matched
nothing and silently returned the whole answer — the exact no-op it existed to
prevent. Caught by a unit test after the code had already been committed.

## Open

- Arms across both checkpoints, 51 blind questions (~1 hour GPU)
- Unverified: whether 900 tokens lets the base model reach a conclusion. If not,
  arm C is under-measured again — but the truncation metric now says so in the
  table.
