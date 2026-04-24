"""Compute per-token CE loss on one real training sample to check for explosion.

Run after stopping training:
  cd /home/qid/esyoon/workspace/web-agent-training/libs/LLaMA-Factory
  source .venv/bin/activate
  SANITY_DEVICE=cuda:0 python scripts/web_agent_cu/loss_check.py
"""
from __future__ import annotations

import json, os, sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

MODEL_PATH = "/home/qid/esyoon/workspace/models/Qwen3.5-9B"
DEVICE = os.environ.get("SANITY_DEVICE", "cuda:0")
ATTN = os.environ.get("SANITY_ATTN", "sdpa")
JSONL = "/home/qid/esyoon/workspace/web-agent-training/libs/LLaMA-Factory/data/web_agent_cu/val.jsonl"

# minimal local replica of LF qwen3_5 template tool prompt + slots so we don't drag
# in the full LF stack. We use processor.apply_chat_template + simple loss-mask.

def main():
    sys.path.insert(0, "/home/qid/esyoon/workspace/web-agent-training/libs/LLaMA-Factory/src")
    from llamafactory.data.tool_utils import Qwen35ToolUtils
    from llamafactory.data.formatter import FunctionFormatter
    fn_fmt = FunctionFormatter(slots=["{{content}}"], tool_format="qwen3_5")
    tool_utils = Qwen35ToolUtils()
    thought_words = ("<think>\n", "\n</think>\n\n")
    tool_call_words = ("<tool_call>", "</tool_call>")

    print(f"Loading model on {DEVICE} attn={ATTN}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=ATTN,
    ).to(DEVICE).eval()

    sample = next(open(JSONL))
    sample = json.loads(sample)
    print(f"Sample: {len(sample['messages'])} messages, {len(sample['images'])} images")

    # Render assistant turns through LF formatters, build raw text
    tools_str = tool_utils.tool_formatter(json.loads(sample['tools']))
    rendered = []
    for m in sample['messages']:
        role = m['role']
        content = m['content']
        if role == 'function_call':
            content = fn_fmt.apply(content=content, thought_words=thought_words, tool_call_words=tool_call_words)[0]
            role = 'assistant'
        rendered.append({"role": role, "content": content})
    # Prepend system with tools
    sys_msg = {"role": "system", "content": "You are a helpful assistant." + tools_str}
    rendered = [sys_msg] + rendered

    # Apply qwen-style chat template manually (simple <|im_start|>/<|im_end|>)
    parts = []
    label_mask: list[bool] = []  # True = compute loss
    for m in rendered:
        role = m['role']
        text = m['content']
        if role in ('user', 'system'):
            chunk = f"<|im_start|>{role}\n{text}<|im_end|>\n"
            ids = tok(chunk, add_special_tokens=False).input_ids
            parts.extend(ids)
            label_mask.extend([False] * len(ids))
        elif role == 'assistant':
            prefix = f"<|im_start|>assistant\n"
            suffix = f"<|im_end|>\n"
            pre_ids = tok(prefix, add_special_tokens=False).input_ids
            mid_ids = tok(text, add_special_tokens=False).input_ids
            suf_ids = tok(suffix, add_special_tokens=False).input_ids
            parts.extend(pre_ids); label_mask.extend([False] * len(pre_ids))
            parts.extend(mid_ids); label_mask.extend([True] * len(mid_ids))
            parts.extend(suf_ids); label_mask.extend([True] * len(suf_ids))
    print(f"text-only token count: {len(parts)}, labeled tokens: {sum(label_mask)}")

    # Now we need the multimodal version: use processor with images so <image> -> image_pad expansion
    images = [Image.open(p).convert("RGB") for p in sample['images']]
    # Build messages list for processor with image chunks
    proc_messages = []
    img_idx = 0
    for m in rendered:
        if m['role'] == 'user' and '<image>' in m['content']:
            chunks: list = []
            text_parts = m['content'].split('<image>')
            for i, t in enumerate(text_parts):
                if t:
                    chunks.append({"type": "text", "text": t})
                if i < len(text_parts) - 1:
                    chunks.append({"type": "image", "image": sample['images'][img_idx]})
                    img_idx += 1
            proc_messages.append({"role": m['role'], "content": chunks})
        else:
            proc_messages.append({"role": m['role'], "content": m['content']})

    text = proc.apply_chat_template(proc_messages, tokenize=False, add_generation_prompt=False)
    inputs = proc(text=[text], images=images, return_tensors="pt").to(DEVICE)
    print(f"final input_ids shape: {inputs.input_ids.shape}")

    # Build labels: mask everything that's outside <|im_start|>assistant ... <|im_end|>
    # Quick approximation — find spans between assistant role markers in input_ids
    ids_list = inputs.input_ids[0].tolist()
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    asst_token_id = tok("assistant", add_special_tokens=False).input_ids[0]
    labels = torch.full_like(inputs.input_ids, -100)
    in_asst = False
    i = 0
    n = len(ids_list)
    while i < n:
        if not in_asst and ids_list[i] == im_start and i + 1 < n and ids_list[i+1] == asst_token_id:
            in_asst = True
            i += 2
            # skip newline token
            if i < n and ids_list[i] in tok.encode("\n", add_special_tokens=False):
                i += 1
            continue
        if in_asst and ids_list[i] == im_end:
            labels[0, i] = ids_list[i]
            in_asst = False
            i += 1
            continue
        if in_asst:
            labels[0, i] = ids_list[i]
        i += 1

    n_labeled = (labels != -100).sum().item()
    print(f"labeled tokens (assistant turns): {n_labeled}")

    with torch.no_grad():
        out = model(**inputs, labels=labels)
    loss = out.loss
    logits = out.logits
    print(f"\nloss = {loss.item():.4f}")
    print(f"logits abs_max = {logits.float().abs().max().item():.3e}")
    print(f"logits min/max = {logits.float().min().item():.3e} / {logits.float().max().item():.3e}")
    print(f"logits has inf: {bool(torch.isinf(logits).any())}, nan: {bool(torch.isnan(logits).any())}")


if __name__ == "__main__":
    main()
