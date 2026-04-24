"""Run LLaMA-Factory's data pipeline on the configured dataset and print
per-sample token / label statistics. Use this to verify that loss is being
computed on a sane number of assistant tokens and that the rendered text
looks correct.

Usage:
  cd /home/qid/esyoon/workspace/web-agent-training/libs/LLaMA-Factory
  source .venv/bin/activate
  python scripts/web_agent_cu/inspect_tokens.py \
      examples/train_lora/qwen3_5_web_agent_cu_lora_sft.yaml --num 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llamafactory.data import get_dataset  # noqa: E402
from llamafactory.data.template import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402
from llamafactory.hparams import get_train_args, read_args  # noqa: E402
from llamafactory.model import load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    # Mirror the launcher: feed the YAML path as sys.argv[1] so read_args picks it up.
    sys.argv = [sys.argv[0], args.yaml]
    parsed = read_args(None)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(parsed)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, "sft", **tokenizer_module)
    train_ds = dataset_module["train_dataset"]
    print(f"\nTotal train samples (after preprocessing/drops): {len(train_ds)}")

    for i in range(min(args.num, len(train_ds))):
        s = train_ds[i]
        ids = np.array(s["input_ids"])
        labels = np.array(s["labels"])
        n_total = len(labels)
        n_trained = int((labels != IGNORE_INDEX).sum())
        n_images = len(s.get("images", []) or [])
        # Decode the labeled tokens (assistant turns) so we can eyeball them
        labeled_ids = labels[labels != IGNORE_INDEX].tolist()
        labeled_text = tokenizer.decode(labeled_ids, skip_special_tokens=False)
        # Decode the full input (truncated for display)
        full_text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        print("=" * 80)
        print(f"[sample {i}] total_tokens={n_total} trained_tokens={n_trained} images={n_images}")
        print(f"-- LABELED (assistant) tokens, first 800 chars --")
        print(labeled_text[:800])
        print(f"-- INPUT (first 600 chars) --")
        print(full_text[:600])
        print(f"-- INPUT (last 600 chars) --")
        print(full_text[-600:])


if __name__ == "__main__":
    main()
