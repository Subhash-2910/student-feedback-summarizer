"""Milestone 2: train a LoRA adapter on GPT-2 to summarise student feedback.

    python -m app.train                       # GPT-2, adapter saved to adapters/default
    python -m app.train --name chemistry --course "Organic Chemistry"
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from app.dataset import PROMPT, load_jsonl
from app.lora import count_parameters, inject_lora, save_adapter

BASE_MODEL = "gpt2"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base(name: str = BASE_MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tok


def encode(record: dict, tok, max_len: int = 256) -> tuple[list[int], list[int]]:
    """Token ids for prompt + summary + EOS. Labels are -100 on the prompt so loss is only on the summary."""
    prompt = tok(PROMPT.format(feedback=record["feedback"]))["input_ids"]
    target = tok(" " + record["summary"])["input_ids"] + [tok.eos_token_id]
    prompt = prompt[-(max_len - len(target)):]
    return prompt + target, [-100] * len(prompt) + target


def batches(examples: list, batch_size: int, pad_id: int, rng: random.Random):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [examples[j] for j in order[i:i + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id)
        labels = torch.full((len(chunk), width), -100)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for k, (x, y) in enumerate(chunk):
            ids[k, :len(x)], labels[k, :len(y)], mask[k, :len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids, labels, mask


def train_lora(model, tok, records: list[dict], epochs: int = 3, lr: float = 2e-4, batch_size: int = 8,
               r: int = 8, alpha: int = 16, device: str = "cpu", seed: int = 0, log=print) -> list[float]:
    """Inject LoRA into `model` and train it in place. Returns the mean loss of each epoch."""
    torch.manual_seed(seed)
    n = inject_lora(model, r=r, alpha=alpha)
    trainable, total = count_parameters(model)
    log(f"LoRA on {n} layers: {trainable:,} trainable of {total:,} parameters ({100 * trainable / total:.2f}%)")
    model.to(device).train()
    examples = [encode(r_, tok) for r_ in records]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng, history = random.Random(seed), []
    for epoch in range(epochs):
        losses = []
        for ids, labels, mask in batches(examples, batch_size, tok.pad_token_id, rng):
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(out.loss.item())
        history.append(sum(losses) / len(losses))
        log(f"epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}")
    model.eval()
    return history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/train.jsonl")
    ap.add_argument("--name", default="default", help="adapter name: saved to adapters/<name>")
    ap.add_argument("--course", default=None, help="train only on feedback for this course")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args(argv)

    records = load_jsonl(args.data)
    if args.course:
        records = [r for r in records if r["course"] == args.course]
    device = pick_device()
    model, tok = load_base(args.base)
    print(f"training on {len(records)} examples, device={device}")
    start = time.time()
    history = train_lora(model, tok, records, epochs=args.epochs, lr=args.lr, r=args.rank,
                         alpha=args.alpha, device=device)
    out = save_adapter(model, Path("adapters") / args.name, extra={
        "base_model": args.base, "course": args.course, "examples": len(records), "epochs": args.epochs,
        "lr": args.lr, "loss_history": history, "train_seconds": round(time.time() - start, 1)})
    print(f"saved adapter to {out}")
    Path("results").mkdir(exist_ok=True)
    Path("results/training.json").write_text(json.dumps(
        {"adapter": args.name, "loss_history": history, "seconds": round(time.time() - start, 1)}, indent=2))


if __name__ == "__main__":
    main()
