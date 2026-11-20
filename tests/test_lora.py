"""Milestone 2: LoRA injection and training. Uses a tiny randomly initialised GPT-2 (no download)."""
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from app.lora import LoRALinear, count_parameters, inject_lora, load_adapter, save_adapter
from app.train import encode, train_lora


def tiny_gpt2(seed=0):
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=300, n_positions=128)).eval()


class FakeTokenizer:
    """Byte-level stand-in for the GPT-2 tokenizer so tests run offline."""
    eos_token_id = pad_token_id = 0

    def __call__(self, text):
        return {"input_ids": [b + 1 for b in text.encode()[:100]]}


def test_injection_wraps_target_layers_and_freezes_the_rest():
    model = tiny_gpt2()
    total_before = count_parameters(model)[1]
    n = inject_lora(model, r=4)
    assert n == 2 * 4            # per block: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
    assert sum(isinstance(m, LoRALinear) for m in model.modules()) == n
    trainable_names = [k for k, p in model.named_parameters() if p.requires_grad]
    assert trainable_names and all("lora_" in k for k in trainable_names)
    trainable, total = count_parameters(model)
    assert trainable < 0.2 * total_before and total > total_before


def test_fresh_adapter_does_not_change_outputs():
    model, ref = tiny_gpt2(), tiny_gpt2()
    inject_lora(model, r=4, dropout=0.0)
    x = torch.randint(0, 300, (2, 16))
    assert torch.allclose(model(x).logits, ref(x).logits, atol=1e-6)    # B starts at zero


def test_encode_masks_the_prompt():
    ids, labels = encode({"feedback": "Labs were fun.", "summary": "Overall: positive."}, FakeTokenizer())
    assert len(ids) == len(labels)
    n_target = sum(label != -100 for label in labels)
    assert n_target == len(" Overall: positive.") + 1                     # summary + EOS
    assert labels[-n_target:] == ids[-n_target:]


def test_training_lowers_loss_and_only_moves_lora_weights():
    model = tiny_gpt2()
    frozen_before = model.transformer.h[0].attn.c_attn.weight.clone()
    records = [{"feedback": f"Comment {i} about labs.", "summary": "Overall: positive. Praises: labs."}
               for i in range(16)]
    history = train_lora(model, FakeTokenizer(), records, epochs=4, lr=5e-3, r=4, log=lambda *_: None)
    assert history[-1] < history[0]
    assert torch.equal(model.transformer.h[0].attn.c_attn.base.weight, frozen_before)
    assert model.transformer.h[0].attn.c_attn.lora_B.abs().sum() > 0


def test_adapter_save_and_load_round_trip(tmp_path):
    model = tiny_gpt2()
    inject_lora(model, r=4, dropout=0.0)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_()
    save_adapter(model, tmp_path / "a", extra={"course": "Labs"})
    assert (tmp_path / "a" / "adapter.pt").stat().st_size < 200_000
    fresh = tiny_gpt2()
    cfg = load_adapter(fresh, tmp_path / "a")
    assert cfg["course"] == "Labs" and cfg["r"] == 4
    x = torch.randint(0, 300, (1, 10))
    assert torch.allclose(model(x).logits, fresh.eval()(x).logits, atol=1e-5)
