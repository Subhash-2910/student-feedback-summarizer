# Results

Everything here is produced by `make all` (data → train → evaluate → merge). Files:

| file | made by | what it holds |
|---|---|---|
| [metrics.md](metrics.md) / [metrics.json](metrics.json) | `python -m app.evaluate` | scores of every system on the held-out test set |
| [samples.md](samples.md) | `python -m app.evaluate` | example outputs, including hand-written comments |
| [training.json](training.json) | `python -m app.train` | loss per epoch and training time |
| [merge.json](merge.json) | `python -m app.merge` | merged vs. adapter: equality check, speed, file sizes |

## How the test is set up

60 test comments are generated with **phrasings that never appear in training** (the last phrasing of every
aspect/sentiment pair is held out), so the model must recognise e.g. *"the queue at office hours was so long I
never got help"* as an office-hours concern without having seen that sentence.

## Main result (GPT-2 + LoRA, r=16, 10 epochs, 2.4M trainable parameters = 1.9%)

| system | ROUGE-L | aspect F1 | overall sentiment acc. | follows format | words |
|---|---|---|---|---|---|
| lead-1 baseline (first sentence) | 0.052 | 0.000 | 0.000 | 0% | 9.8 |
| base GPT-2, same prompt, no fine-tuning | 0.010 | 0.000 | 0.000 | 0% | 23.1 |
| **GPT-2 + LoRA** | **0.850** | **0.689** | **0.883** | **100%** | **6.2** |

The base model just continues the text (it rambles, often repeating the feedback). After LoRA it produces a
short structured summary every time, at about a quarter of the length of the feedback (compression 0.25).

## What changed the result (training runs we tried)

| base model | rank | epochs | final train loss | ROUGE-L | aspect F1 | overall acc. | train time (M3) |
|---|---|---|---|---|---|---|---|
| distilgpt2 (82M) | 8 | 3 | 0.50 | 0.629 | 0.351 | 0.467 | ~1 min |
| distilgpt2 (82M) | 16 | 10 | 0.15 | 0.720 | 0.456 | 0.733 | 2.4 min |
| **gpt2 (124M)** | 16 | 10 | 0.08 | **0.850** | **0.689** | **0.883** | 3.0 min |

Training longer helped, and a bigger base model helped more: the larger model generalises better to the
unseen phrasings.

## Hand-written comments (checked by hand)

These five were written by hand, not by the dataset script, to see how the model copes with real-sounding text
(slang, aspects named with other words such as "prof", "midterm", "project").

| comment (short) | model summary | verdict |
|---|---|---|
| Prof was great ... but the midterm was brutal | mixed · praises instructor · concerns exams | correct |
| Too much reading ... slides from 2015. Labs were ok | mixed · praises labs · concerns course materials | mostly right; misses the workload concern, "ok" read as praise |
| Loved this class. TA office hours saved me, assignments taught me | positive · praises assignments | misses office hours |
| Couldn't keep up, three chapters a week, grades took a month | negative · concerns grading | misses pacing |
| Lectures were boring but the project was the best part | positive · praises assignments | misses the lectures concern, so the overall label is wrong |

**Score: 1 of 5 fully correct, 4 of 5 with no wrong aspect named; it misses aspects more often than it invents them.**

### Rubric for judging summaries by hand

For instructors who want to spot-check, rate each summary 0–2 on:

1. **Relevance:** every aspect named is really in the comment, with the right sentiment.
2. **Coverage:** no aspect the student raised is missing.
3. **Conciseness:** one line; no copied sentences or filler.

## Limits

- **Synthetic data.** 400 template-built comments about 10 fixed aspects. Real feedback raises things outside
  that list (room, timetable, software), which the model cannot name.
- **It misses aspects more often than it invents them.** When a comment has 3–4 points, the last one is often
  dropped. Aspect F1 0.69 means roughly 1 in 3 aspect mentions is wrong or missing.
- **One comment at a time.** Each comment is summarised alone; the UI's "across all comments" chart just counts
  the per-comment results.
- **ROUGE is generous here.** The structured format means every summary shares words like "Overall" and
  "Praises", so aspect F1 is the metric to trust.
- **Small test.** 60 test comments: differences of a few points between runs are within noise.
