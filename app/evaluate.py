"""Milestone 4: measure summary quality on the held-out test set and compare against baselines.

    python -m app.evaluate                 # writes results/metrics.json, results/metrics.md, results/samples.md

Systems compared
- lead-1:          the first sentence of the feedback (a classic extractive summarisation baseline)
- base GPT-2:      the same base model with the same prompt, no fine-tuning
- LoRA (adapter):  base model + our adapter

Metrics (all in [0, 1] except length)
- ROUGE-1 / ROUGE-2 / ROUGE-L F1 against the reference summary (word overlap; implemented here)
- aspect F1:    are the right (aspect, praise/concern) pairs named? This is *relevance*: a summary that says
                "Concerns: pacing" when the student complained about grading scores 0 even if it reads well.
- overall acc:  is the overall sentiment (positive / negative / mixed) right?
- format ok:    share of outputs that follow the structured format an instructor can scan
- compression:  summary words / feedback words (lower = more concise)
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from app.dataset import load_jsonl
from app.summarize import parse_summary

WORD = re.compile(r"[a-z0-9]+")

# Not from the dataset generator: written by hand to see how the model copes with real-sounding comments.
HANDWRITTEN = [
    "Prof was great, super clear explanations, but the midterm was brutal and nothing like the practice ones.",
    "Too much reading every week and the slides are from like 2015. Labs were ok I guess.",
    "Loved this class. The TA office hours saved me, and the assignments actually taught me stuff.",
    "I couldn't keep up. We covered three chapters a week and grades took a month to come back.",
    "Lectures were boring but the project was the best part of my semester.",
]


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if overlap == 0 or n_pred == 0 or n_ref == 0:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_n(pred: str, ref: str, n: int = 1) -> float:
    def grams(ws):
        out = {}
        for g in zip(*(ws[i:] for i in range(n))):
            out[g] = out.get(g, 0) + 1
        return out
    p, r = grams(words(pred)), grams(words(ref))
    return _f1(sum(min(c, r.get(g, 0)) for g, c in p.items()), sum(p.values()), sum(r.values()))


def rouge_l(pred: str, ref: str) -> float:
    a, b = words(pred), words(ref)
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return _f1(prev[-1], len(a), len(b))


def aspect_f1(pred: str, record: dict) -> float:
    parsed = parse_summary(pred)
    got = {(a, "pos") for a in parsed["praises"]} | {(a, "neg") for a in parsed["concerns"]}
    gold = {(x["aspect"], x["sentiment"]) for x in record["aspects"]}
    return _f1(len(got & gold), len(got), len(gold))


def score(pred: str, record: dict) -> dict:
    ref = record["summary"]
    parsed, gold = parse_summary(pred), parse_summary(ref)
    return {
        "rouge1": rouge_n(pred, ref, 1), "rouge2": rouge_n(pred, ref, 2), "rougeL": rouge_l(pred, ref),
        "aspect_f1": aspect_f1(pred, record),
        "overall_acc": float(parsed["overall"] == gold["overall"]),
        "format_ok": float(parsed["overall"] is not None),
        "compression": len(words(pred)) / max(1, len(words(record["feedback"]))),
        "words": len(words(pred)),
    }


def aggregate(scores: list[dict]) -> dict:
    return {k: round(statistics.mean(s[k] for s in scores), 3) for k in scores[0]}


def lead_1(feedback: str) -> str:
    return re.split(r"(?<=[.!?])\s+", feedback.strip())[0]


def run_system(name: str, predict, test: list[dict]) -> tuple[dict, list[str]]:
    start = time.time()
    preds = [predict(r["feedback"]) for r in test]
    result = aggregate([score(p, r) for p, r in zip(preds, test)])
    result["ms_per_summary"] = round(1000 * (time.time() - start) / len(test), 1)
    print(f"{name:<16} {result}")
    return result, preds


def to_markdown(metrics: dict, n: int) -> str:
    cols = ["rouge1", "rouge2", "rougeL", "aspect_f1", "overall_acc", "format_ok", "compression", "words"]
    lines = [f"Held-out test set: {n} comments written with phrasings never seen in training.\n",
             "| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in metrics.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" if c != "words" else f"{m[c]:.1f}" for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main(adapter: str = "default", test_path: str = "dataset/test.jsonl", out_dir: str = "results"):
    from app.summarize import load_model, summarize

    test = load_jsonl(test_path)
    metrics, outputs = {}, {}
    metrics["lead-1 baseline"], outputs["lead-1"] = run_system("lead-1", lead_1, test)
    for label, name in [("base GPT-2", None), (f"LoRA ({adapter})", adapter)]:
        model, tok = load_model(name, merged=False)
        metrics[label], outputs[label] = run_system(label, lambda f: summarize(model, tok, f), test)
        outputs[label + " handwritten"] = [summarize(model, tok, f) for f in HANDWRITTEN]

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"n_test": len(test), "systems": metrics}, indent=2))
    (out / "metrics.md").write_text(to_markdown(metrics, len(test)))
    lora = f"LoRA ({adapter})"
    sample = ["# Sample outputs\n", "## Held-out test set (first 8)\n"]
    for i, r in enumerate(test[:8]):
        sample += [f"**Feedback:** {r['feedback']}  ", f"**Reference:** {r['summary']}  ",
                   f"**LoRA:** {outputs[lora][i]}  ", f"**Base GPT-2:** {outputs['base GPT-2'][i][:150]}\n"]
    sample.append("## Hand-written comments (not generated by the dataset script)\n")
    for i, f in enumerate(HANDWRITTEN):
        sample += [f"**Feedback:** {f}  ", f"**LoRA:** {outputs[lora + ' handwritten'][i]}  ",
                   f"**Base GPT-2:** {outputs['base GPT-2 handwritten'][i][:150]}\n"]
    (out / "samples.md").write_text("\n".join(sample))
    print(f"wrote {out}/metrics.json, metrics.md, samples.md")
    return metrics


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
