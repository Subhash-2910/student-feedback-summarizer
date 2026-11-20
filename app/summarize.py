"""Inference helpers shared by the Streamlit UI, the evaluation and the merge step."""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from pathlib import Path

import torch

from app.dataset import ASPECTS, PROMPT

ADAPTERS_DIR = Path("adapters")
MERGED_DIR = Path("merged")


@torch.no_grad()
def summarize(model, tok, feedback: str, max_new_tokens: int = 40, temperature: float = 0.0,
              max_prompt_tokens: int = 400) -> str:
    """Generate one summary. temperature=0 means greedy decoding (deterministic)."""
    device = next(model.parameters()).device
    ids = tok(PROMPT.format(feedback=feedback.strip()), return_tensors="pt")["input_ids"][:, -max_prompt_tokens:]
    kwargs = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 else {"do_sample": False}
    out = model.generate(ids.to(device), attention_mask=torch.ones_like(ids).to(device),
                         max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                         eos_token_id=tok.eos_token_id, **kwargs)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text.split("\n")[0].strip()


def parse_summary(summary: str) -> dict:
    """Read the structured summary back into {'overall', 'praises': [...], 'concerns': [...]}.
    Unknown aspect names are dropped, so free-form text from an untrained model parses to empty lists."""
    def aspects_after(label: str) -> list[str]:
        m = re.search(label + r":\s*([^.]*)", summary, flags=re.I)
        items = [s.strip().lower() for s in m.group(1).split(",")] if m else []
        return [a for a in items if a in ASPECTS]
    m = re.search(r"overall:\s*(positive|negative|mixed)", summary, flags=re.I)
    return {"overall": m.group(1).lower() if m else None,
            "praises": aspects_after("praises"), "concerns": aspects_after("concerns")}


def split_feedback(raw: str | bytes, filename: str = "") -> list[str]:
    """Turn an upload or pasted text into a list of comments.
    CSV: the column named feedback/comment/text (or the first column). Otherwise: one comment per
    non-empty line, or per paragraph when comments span several lines."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if filename.lower().endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            return []
        cols = {c.lower().strip(): c for c in rows[0]}
        col = next((cols[c] for c in ("feedback", "comment", "comments", "text") if c in cols), list(rows[0])[0])
        return [r[col].strip() for r in rows if r.get(col) and r[col].strip()]
    if "\n\n" in text.strip():
        parts = re.split(r"\n\s*\n", text)
        return [" ".join(p.split()) for p in parts if p.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def tally(summaries: list[str]) -> list[dict]:
    """Count how often each aspect is praised or raised as a concern across many summaries."""
    praises, concerns = Counter(), Counter()
    for s in summaries:
        p = parse_summary(s)
        praises.update(p["praises"])
        concerns.update(p["concerns"])
    rows = [{"aspect": a, "praised": praises[a], "concerns": concerns[a]} for a in ASPECTS
            if praises[a] or concerns[a]]
    return sorted(rows, key=lambda r: (-r["concerns"], -r["praised"]))


def list_adapters() -> list[str]:
    return sorted(p.parent.name for p in ADAPTERS_DIR.glob("*/adapter.pt"))


def load_model(adapter: str | None = "default", merged: bool = True, device: str = "cpu"):
    """Load a model ready for inference.

    - adapter=None: the plain base model (the baseline)
    - merged=True and merged/<adapter> exists: the pre-merged model (fastest, no LoRA code needed)
    - otherwise: base model + adapter, merged in memory (same outputs, no LoRA overhead)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from app.lora import load_adapter, merge_lora
    from app.train import BASE_MODEL

    if adapter and merged and (MERGED_DIR / adapter / "config.json").exists():
        path = MERGED_DIR / adapter
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(path)
    else:
        import json
        base = BASE_MODEL
        if adapter:
            base = json.loads((ADAPTERS_DIR / adapter / "adapter_config.json").read_text()).get("base_model", base)
        model = AutoModelForCausalLM.from_pretrained(base)
        tok = AutoTokenizer.from_pretrained(base)
        if adapter:
            load_adapter(model, ADAPTERS_DIR / adapter)
            merge_lora(model)
    tok.pad_token = tok.eos_token
    return model.to(device).eval(), tok
