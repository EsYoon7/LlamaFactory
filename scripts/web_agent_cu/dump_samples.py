"""Dump a handful of post-template training samples to disk for human review.

Caps the dataset size via data_args.max_samples to keep the run short.

Usage:
  cd /home/qid/esyoon/workspace/web-agent-training/libs/LLaMA-Factory
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29501 \
    scripts/web_agent_cu/dump_samples.py \
    examples/train_lora/qwen3_5_web_agent_cu_lora_sft.yaml \
    --num 5 \
    --out /home/qid/esyoon/workspace/web-agent-training/training_samples
"""
from __future__ import annotations

import argparse
import json
import shutil
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
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.argv = [sys.argv[0], args.yaml]
    parsed = read_args(None)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(parsed)
    # Limit dataset preprocessing to the requested few samples
    data_args.max_samples = args.num
    training_args.dispatch_batches = False

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, "sft", **tokenizer_module)
    train_ds = dataset_module["train_dataset"]
    print(f"\nGot {len(train_ds)} samples; dumping {min(args.num, len(train_ds))} to {args.out}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num, len(train_ds))):
        s = train_ds[i]
        ids = np.array(s["input_ids"])
        labels = np.array(s["labels"])
        n_total = len(labels)
        n_trained = int((labels != IGNORE_INDEX).sum())
        images = s.get("images", []) or []

        labeled_ids = labels[labels != IGNORE_INDEX].tolist()
        labeled_text = tokenizer.decode(labeled_ids, skip_special_tokens=False)
        full_text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)

        meta = {
            "index": i,
            "n_total_tokens": n_total,
            "n_trained_tokens": n_trained,
            "n_images": len(images),
            "image_paths": [str(p) for p in images] if isinstance(images, list) else None,
        }

        sample_dir = out / f"sample_{i:03d}"
        sample_dir.mkdir(exist_ok=True)
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        (sample_dir / "full_input_decoded.txt").write_text(full_text)
        (sample_dir / "labeled_assistant_decoded.txt").write_text(labeled_text)
        # Also save raw token id arrays for any downstream debugging
        np.save(sample_dir / "input_ids.npy", ids)
        np.save(sample_dir / "labels.npy", labels)
        # Copy each image so the sample is self-contained for review
        img_dir = sample_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for j, p in enumerate(images):
            try:
                src = Path(p)
                if src.exists():
                    shutil.copy2(src, img_dir / f"{j:02d}_{src.name}")
            except Exception as exc:
                print(f"  warn: could not copy image {p}: {exc}")
        print(f"[{i}] tokens={n_total} trained={n_trained} images={len(images)} -> {sample_dir}")


if __name__ == "__main__":
    main()
