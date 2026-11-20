"""Milestone 5: merging the adapter into the base weights."""
import torch
from torch import nn

from app.lora import LoRALinear, inject_lora, merge_lora
from tests.test_lora import tiny_gpt2


def trained_looking_lora(seed=1):
    model = tiny_gpt2()
    inject_lora(model, r=4, dropout=0.0)
    torch.manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.1)     # pretend training moved B away from zero
    return model.eval()


def test_merged_model_matches_base_plus_adapter():
    model = trained_looking_lora()
    x = torch.randint(0, 300, (2, 12))
    with torch.no_grad():
        before = model(x).logits
        merge_lora(model)
        after = model(x).logits
    assert torch.allclose(before, after, atol=1e-4)
    assert not torch.allclose(after, tiny_gpt2()(x).logits, atol=1e-3)     # and it differs from the base


def test_merge_removes_all_lora_layers_and_parameters():
    model = trained_looking_lora()
    n_base = sum(p.numel() for p in tiny_gpt2().parameters())
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert not any("lora_" in k for k in model.state_dict())
    assert sum(p.numel() for p in model.parameters()) == n_base


def test_merge_works_for_nn_linear_too():
    lin = nn.Linear(6, 4)
    wrapped = LoRALinear(lin, r=2, dropout=0.0)
    with torch.no_grad():
        wrapped.lora_B.normal_()
    model = nn.Sequential(wrapped)
    x = torch.randn(3, 6)
    before = model(x)
    merge_lora(model)
    assert isinstance(model[0], nn.Linear) and torch.allclose(model(x), before, atol=1e-5)


def test_merged_model_saves_and_reloads(tmp_path):
    from transformers import GPT2LMHeadModel
    model = trained_looking_lora()
    merge_lora(model)
    model.save_pretrained(tmp_path)
    reloaded = GPT2LMHeadModel.from_pretrained(tmp_path).eval()
    x = torch.randint(0, 300, (1, 8))
    assert torch.allclose(model(x).logits, reloaded(x).logits, atol=1e-5)
