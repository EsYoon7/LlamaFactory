"""Sanity-check a base VLM in bf16: load model, run a few forwards, look for inf/nan
and report logit magnitudes at each layer where it might explode.

Run inside es_nemo2502:
  cd /home/qid/esyoon/workspace/web-agent-training/libs/LLaMA-Factory
  source .venv/bin/activate
  python scripts/web_agent_cu/sanity_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

import os
MODEL_PATH = "/home/qid/esyoon/workspace/models/Qwen3.5-9B"
DEVICE = os.environ.get("SANITY_DEVICE", "cuda:0")


def stat(name: str, t: torch.Tensor) -> dict:
    flat = t.detach().float()
    return {
        "name": name,
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "abs_max": flat.abs().max().item(),
        "any_inf": bool(torch.isinf(flat).any().item()),
        "any_nan": bool(torch.isnan(flat).any().item()),
    }


def show(name: str, t: torch.Tensor) -> None:
    s = stat(name, t)
    flag = ""
    if s["any_inf"]:
        flag += " [INF!]"
    if s["any_nan"]:
        flag += " [NAN!]"
    print(f"  {name:40s} shape={s['shape']!s:30s} dtype={s['dtype']:14s} "
          f"abs_max={s['abs_max']:.3e} min={s['min']:.3e} max={s['max']:.3e}{flag}")


def scan_weights(model: torch.nn.Module) -> None:
    print("\n=== Weight scan ===")
    bad = []
    big = []
    for n, p in model.named_parameters():
        s = stat(n, p)
        if s["any_inf"] or s["any_nan"]:
            bad.append(s)
        if s["abs_max"] > 100:
            big.append((n, s["abs_max"]))
    print(f"  total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  inf/nan weights: {len(bad)}")
    for s in bad[:5]:
        print(f"    {s['name']}: {s}")
    print(f"  weights with abs_max > 100: {len(big)}")
    for n, m in sorted(big, key=lambda x: -x[1])[:10]:
        print(f"    {n}: abs_max={m:.3e}")


def forward_text_only(model, tokenizer):
    print("\n=== Text-only forward ===")
    text = "Hello, please respond with one short sentence."
    ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"  input_ids shape: {ids.shape}")
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    show("logits", out.logits)
    if out.hidden_states is not None:
        for i, h in enumerate(out.hidden_states):
            if i in (0, 1, len(out.hidden_states) // 2, len(out.hidden_states) - 1):
                show(f"hidden[{i}]", h)


def forward_with_image(model, processor):
    print("\n=== Image+text forward ===")
    img_dir = Path("/home/ubuntu/.cache/web_agent_cu/images")
    pngs = list(img_dir.rglob("*.png"))[:1]
    if not pngs:
        print("  no cached image found, skipping")
        return
    from PIL import Image
    img = Image.open(pngs[0]).convert("RGB")
    print(f"  using image: {pngs[0]} size={img.size}")

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pngs[0].as_posix()},
            {"type": "text", "text": "Describe what you see briefly."},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    print(f"  input_ids shape: {inputs.input_ids.shape}")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    show("logits", out.logits)
    if out.hidden_states is not None:
        for i, h in enumerate(out.hidden_states):
            if i in (0, 1, len(out.hidden_states) // 2, len(out.hidden_states) - 1):
                show(f"hidden[{i}]", h)


def main():
    print(f"Loading {MODEL_PATH} in bf16 on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"  AutoProcessor failed: {e}; will skip image forward")
        processor = None
    attn_impl = os.environ.get("SANITY_ATTN", "flash_attention_2")
    print(f"  attn_implementation: {attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(DEVICE).eval()
    print(f"  model class: {type(model).__name__}")
    print(f"  config.model_type: {model.config.model_type}")
    print(f"  vocab_size (tokenizer): {len(tokenizer)}")
    print(f"  vocab_size (config): {getattr(model.config, 'vocab_size', None)}")

    scan_weights(model)
    forward_text_only(model, tokenizer)
    if processor is not None:
        forward_with_image(model, processor)
    print("\nDone.")


if __name__ == "__main__":
    main()
