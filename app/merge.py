"""Milestone 5: merge a LoRA adapter into the base model for deployment.

    python -m app.merge                 # adapters/default -> merged/default
    python -m app.merge chemistry       # adapters/chemistry -> merged/chemistry

It checks that the merged model gives the same next-token predictions as base + adapter, times both,
and writes results/merge.json. The merged model is saved in float16 (half the disk space).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from app.dataset import PROMPT, load_jsonl
from app.lora import load_adapter, merge_lora
from app.summarize import ADAPTERS_DIR, MERGED_DIR, summarize


def time_generation(model, tok, texts: list[str]) -> tuple[float, list[str]]:
    start = time.time()
    outs = [summarize(model, tok, t) for t in texts]
    return 1000 * (time.time() - start) / len(texts), outs


def main(adapter: str = "default", n_check: int = 30):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((ADAPTERS_DIR / adapter / "adapter_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])
    load_adapter(model, ADAPTERS_DIR / adapter)
    model.eval()

    texts = [r["feedback"] for r in load_jsonl("dataset/test.jsonl")[:n_check]]
    batch = tok([PROMPT.format(feedback=t) for t in texts[:8]], return_tensors="pt", padding=True)
    with torch.no_grad():
        before_logits = model(**batch).logits
    ms_adapter, outs_adapter = time_generation(model, tok, texts)

    merge_lora(model)
    model.eval()
    with torch.no_grad():
        after_logits = model(**batch).logits
    ms_merged, outs_merged = time_generation(model, tok, texts)

    out = MERGED_DIR / adapter
    model.half().save_pretrained(out)
    tok.save_pretrained(out)
    size_mb = sum(f.stat().st_size for f in out.iterdir()) / 1e6

    report = {
        "adapter": adapter, "base_model": cfg["base_model"],
        "max_logit_difference": float((before_logits - after_logits).abs().max()),
        "same_summaries": sum(a == b for a, b in zip(outs_adapter, outs_merged)), "checked": len(texts),
        "ms_per_summary_adapter": round(ms_adapter, 1), "ms_per_summary_merged": round(ms_merged, 1),
        "adapter_file_mb": round((ADAPTERS_DIR / adapter / "adapter.pt").stat().st_size / 1e6, 2),
        "merged_model_mb_fp16": round(size_mb, 1),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/merge.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved merged model to {out}")
    return report


if __name__ == "__main__":
    main(*sys.argv[1:2])
