# Student Feedback Summarizer

[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/mbgxgavm8n/badge.svg)](https://ask.thili.ai/b/mbgxgavm8n)

Course instructors get dozens of free-text comments at the end of a term. This project fine-tunes **GPT-2 with
LoRA** so it turns each comment into a one-line summary an instructor can scan, and a Streamlit app shows what
the whole class praised and complained about.

```
Feedback: Prof was great, super clear explanations, but the midterm was brutal and nothing like the practice ones.
Summary:  Overall: mixed. Praises: instructor. Concerns: exams.
```

The LoRA code (adapter injection, training and merging) is written from scratch in [app/lora.py](app/lora.py)
in about 130 lines, with no PEFT library.

## Results

On 60 held-out test comments written with **phrasings the model never saw in training**
(full table: [results/metrics.md](results/metrics.md), write-up: [results/README.md](results/README.md)):

| system | ROUGE-L | aspect F1 | overall sentiment acc. | follows format | words |
|---|---|---|---|---|---|
| baseline: first sentence of the comment | 0.05 | 0.00 | 0.00 | 0% | 9.8 |
| baseline: GPT-2, same prompt, no fine-tuning | 0.01 | 0.00 | 0.00 | 0% | 23.1 |
| **GPT-2 + LoRA (this project)** | **0.85** | **0.69** | **0.88** | **100%** | **6.2** |

- **Training:** 2.4M of 127M parameters trained (1.9%), 10 epochs, about 3 minutes on an Apple M3. The training loss
  went from 1.44 to 0.08 ([results/training.json](results/training.json)).
- **Merging:** the merged model gives identical summaries on 30 of 30 test comments, runs 6% faster and loads as a
  plain `transformers` model ([results/merge.json](results/merge.json)).
- **Sample outputs** for test and hand-written comments: [results/samples.md](results/samples.md).

**Limits.** The data is synthetic (400 template-built comments about 10 fixed aspects), so the model can only
name those 10 aspects. On comments with 3–4 points it often drops one: about 1 in 3 aspect mentions is wrong or
missing. Of 5 hand-written comments, 1 was summarised perfectly and 4 named no wrong aspect; it misses things
more often than it invents them.
See [results/README.md](results/README.md#limits).

## Run it

You need Python 3.11 or 3.12 and about 2 GB of disk space (libraries, GPT-2 and the merged model).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make app            # or: streamlit run app/main.py
```

The trained adapter is in the repo ([adapters/default](adapters/default), 9.5 MB), so the app works right away.
GPT-2 is downloaded once (about 500 MB) into `.cache/` inside the project.

**Rebuild everything from scratch** (dataset → training → evaluation → merge, about 5 minutes on an M-series Mac,
longer on CPU):

```bash
make all
make test           # 25 tests; they use a tiny random GPT-2, so they run offline in seconds
```

| step | command | output |
|---|---|---|
| 1. dataset | `python -m app.dataset` | `dataset/train.jsonl` (400), `dataset/test.jsonl` (60) |
| 2. LoRA training | `python -m app.train` | `adapters/default/` |
| 3. UI | `streamlit run app/main.py` | the app at http://localhost:8501 |
| 4. evaluation | `python -m app.evaluate` | `results/metrics.*`, `results/samples.md` |
| 5. merging | `python -m app.merge` | `merged/default/` (fp16), `results/merge.json` |

**Docker** (builds the merged model into the image; needs about 3 GB of free disk. This build was not run while the
project was developed because the dev machine was short on disk space, so the local commands above are the tested path):

```bash
docker build -t feedback-summarizer . && docker run -p 8080:8080 feedback-summarizer   # http://localhost:8080
```

## Using the app

1. Paste comments (one per line or paragraph) or upload a `.txt`, or a `.csv` with a `feedback` column.
2. Optionally choose an adapter, the maximum summary length and the temperature in the sidebar.
3. Click **Summarize**. You get a bar chart of praises and concerns per aspect across all comments, a table of
   per-comment summaries and a CSV download. Past runs appear under **History** (saved in `data/app.json`).

**One adapter per course:** `python -m app.train --name chemistry --course "Organic Chemistry"` saves
`adapters/chemistry/`. It then shows up in the app's adapter menu, and `python -m app.merge chemistry` merges it.

## How it works

| milestone | code | tests |
|---|---|---|
| 1. Synthetic dataset | [app/dataset.py](app/dataset.py): 10 aspects × 2 sentiments × 5 phrasings, openers, filler, lowercase noise; the 5th phrasing of each is held out for testing | [tests/test_dataset.py](tests/test_dataset.py) |
| 2. LoRA injection and training | [app/lora.py](app/lora.py) `LoRALinear`, `inject_lora`; [app/train.py](app/train.py): loss only on summary tokens | [tests/test_lora.py](tests/test_lora.py) |
| 3. Streamlit UI | [app/main.py](app/main.py), [app/summarize.py](app/summarize.py) | [tests/test_ui.py](tests/test_ui.py) |
| 4. Evaluation | [app/evaluate.py](app/evaluate.py): ROUGE-1/2/L, aspect F1, sentiment accuracy, compression | [tests/test_evaluate.py](tests/test_evaluate.py) |
| 5. Adapter merging | [app/lora.py](app/lora.py) `merge_lora`, [app/merge.py](app/merge.py) | [tests/test_merge.py](tests/test_merge.py) |

**LoRA in one line:** each target layer computes `x W + (alpha/r) · x Aᵀ Bᵀ`. W is frozen; only the small A
(r × in) and B (out × r) are trained. B starts at zero, so training starts from the unchanged base model.
Merging adds `(alpha/r) · B A` into W once, so inference costs exactly as much as plain GPT-2. Adapters are
injected into all attention and MLP projections (`c_attn`, `c_proj`, `c_fc`) with r=16 and alpha=32.

**Why aspect F1:** ROUGE rewards word overlap, and every structured summary shares words like "Overall" and
"Praises". Aspect F1 parses the summary and checks that the right (aspect, praise/concern) pairs are named,
which is what an instructor actually needs.

## Settings

Nothing is required and no API key is used. `.env.example` lists the two optional settings: where the history
is stored (`DATA_PATH`) and where models are cached (`HF_HOME`).
