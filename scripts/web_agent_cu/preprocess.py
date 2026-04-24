"""Preprocess the web-agent computer-use dataset into sharegpt jsonl.

Downloads parquet + images from Azure blob (caches locally), converts each
trajectory via ``llamafactory.data.web_agent_cu``, and writes a jsonl at
``<output_dir>/<split>.jsonl``. The same output directory is registered in
``data/dataset_info.json`` via the helper at the bottom of this file.

Run inside es_nemo2502:
  cd /home/qid/esyoon/workspace/web-agent-training/libs/LLaMA-Factory
  source .venv/bin/activate
  python scripts/web_agent_cu/preprocess.py --split val --image_history_n 3

Re-run with different --image_history_n / --action_split_mode /
--action_mapping_mode to regenerate the jsonl.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import multiprocessing as mp
import os
import struct
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

# Make "llamafactory" importable when running as a plain script
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from llamafactory.data.web_agent_cu import (  # noqa: E402
    StepInfo,
    TrajectorySpec,
    convert_specs,
)

DEFAULT_ACCOUNT_URL = "https://t2vgusw2.blob.core.windows.net"
DEFAULT_APP_ID = "8cafa2b1-a2a7-4ad9-814a-ffe4aed7e800"
DEFAULT_DATA_CONTAINER = "esyoon"
DEFAULT_IMAGE_CONTAINER = "v-zhihongtan"
DEFAULT_PARQUET_PREFIX = "dataset/gpt5-4_v260409/processed_raw_good_with_thinking_train_val_split"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--account_url", default=DEFAULT_ACCOUNT_URL)
    p.add_argument("--app_id", default=DEFAULT_APP_ID)
    p.add_argument("--data_container", default=DEFAULT_DATA_CONTAINER)
    p.add_argument("--image_container", default=DEFAULT_IMAGE_CONTAINER)
    p.add_argument("--parquet_prefix", default=DEFAULT_PARQUET_PREFIX)
    p.add_argument("--cache_dir", default=str(Path.home() / ".cache" / "web_agent_cu"))
    p.add_argument("--output_dir", default=str(_REPO_ROOT / "data" / "web_agent_cu"))
    # Converter knobs
    p.add_argument("--image_history_n", type=int, default=3)
    p.add_argument("--action_split_mode", choices=["all_at_once", "split_per_action"], default="all_at_once")
    p.add_argument("--action_mapping_mode", choices=["extend_schema", "decompose"], default="extend_schema")
    # Performance knobs
    p.add_argument("--num_workers", type=int, default=64, help="Processes for converter stage.")
    p.add_argument("--download_workers", type=int, default=64, help="Threads for image download stage.")
    # Debug
    p.add_argument("--limit_tasks", type=int, default=0, help="If >0, process only N tasks (for dry-run).")
    p.add_argument("--force_redownload", action="store_true")
    p.add_argument("--max_cache_gb", type=float, default=50.0, help="Warn if image cache exceeds this many GB.")
    p.add_argument(
        "--no_download",
        action="store_true",
        help="Skip image download; only use images already present in the local cache. "
        "Trajectories with any missing before_image are dropped.",
    )
    return p.parse_args()


def _get_clients(args: argparse.Namespace):
    from azure.storage.blob import ContainerClient
    from azure.identity import ManagedIdentityCredential
    cred = ManagedIdentityCredential(client_id=args.app_id)
    data = ContainerClient(account_url=args.account_url, container_name=args.data_container, credential=cred)
    img = ContainerClient(account_url=args.account_url, container_name=args.image_container, credential=cred)
    return data, img


def _cached_download_parquet(data_client, blob_path: str, cache_dir: Path, force: bool) -> Path:
    local = cache_dir / "parquet" / blob_path.replace("/", "__")
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and not force:
        print(f"[parquet] cache hit: {local}")
        return local
    print(f"[parquet] downloading {blob_path} -> {local}")
    data = data_client.get_blob_client(blob_path).download_blob().readall()
    local.write_bytes(data)
    return local


def _download_one_image(img_client, blob_path: str, local_path: Path, force: bool) -> None:
    if local_path.exists() and not force:
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    data = img_client.get_blob_client(blob_path).download_blob().readall()
    local_path.write_bytes(data)


def _download_images(img_client, rel_paths: list[str], cache_dir: Path, n_workers: int, force: bool) -> dict[str, Path]:
    images_root = cache_dir / "images"
    mapping: dict[str, Path] = {}
    todo: list[tuple[str, Path]] = []
    for rel in rel_paths:
        local = images_root / rel
        mapping[rel] = local
        if force or not local.exists():
            todo.append((rel, local))
    if not todo:
        print(f"[images] all {len(rel_paths)} cached")
        return mapping
    print(f"[images] downloading {len(todo)}/{len(rel_paths)} with {n_workers} threads")
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_download_one_image, img_client, rel, loc, force): rel for rel, loc in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            rel = futs[fut]
            try:
                fut.result()
            except Exception as exc:
                errors += 1
                if errors < 10:
                    print(f"[images] failed {rel}: {exc}")
            if (i + 1) % 500 == 0:
                print(f"[images] {i+1}/{len(todo)}")
    if errors:
        print(f"[images] total errors: {errors}")
    return mapping


_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_IEND = b"IEND\xaeB`\x82"


def _verify_and_size(path_str: str) -> tuple[str, tuple[int, int] | None]:
    """Validate a PNG and return its width/height without full pixel decode.

    Strategy (all cheap disk I/O, no PIL decode on the hot path):
      1. Non-empty file check.
      2. 8-byte PNG signature.
      3. IHDR chunk at fixed offset → pull width/height directly.
      4. Last 12 bytes contain the IEND terminator — guarantees the file
         wasn't cut off mid-IDAT by an interrupted download.

    This is ~2 orders of magnitude faster than PIL full-decode and still
    rejects the exact failure mode we care about (truncated downloads).
    Returns ``(path_str, None)`` on any failure.
    """
    try:
        st = os.stat(path_str)
        if st.st_size < 57:  # minimum valid PNG ≈ 57 bytes
            return (path_str, None)
        with open(path_str, "rb") as f:
            header = f.read(24)
            if header[:8] != _PNG_SIG:
                return (path_str, None)
            # IHDR chunk: bytes 8..12 length, 12..16 type "IHDR", 16..20 width, 20..24 height
            if header[12:16] != b"IHDR":
                return (path_str, None)
            width = struct.unpack(">I", header[16:20])[0]
            height = struct.unpack(">I", header[20:24])[0]
            # Confirm IEND trailer
            f.seek(-12, os.SEEK_END)
            tail = f.read(12)
            if _PNG_IEND not in tail:
                return (path_str, None)
        if width < 32 or height < 32:
            # Reject degenerate / placeholder images (e.g. 1x1 transparent PNGs)
            # that also fall below the smart_resize minimum factor.
            return (path_str, None)
        return (path_str, (width, height))
    except Exception:
        return (path_str, None)


def _probe_image_sizes(
    image_paths: list[Path], n_workers: int, delete_corrupt: bool = False
) -> dict[Path, tuple[int, int] | None]:
    """Return size dict; corrupt/missing files map to ``None``.

    Uses a process pool — PNG header parsing is CPU-bound per file but IO
    dominates; a process pool still scales well because each worker does its
    own syscalls without GIL contention.
    """
    sizes: dict[Path, tuple[int, int] | None] = {}
    str_paths = [str(p) for p in image_paths]
    bad: list[Path] = []
    total = len(str_paths)
    chunksize = max(1, total // (n_workers * 8)) if n_workers > 0 else 1
    with mp.Pool(processes=max(1, n_workers)) as pool:
        for i, (ps, result) in enumerate(pool.imap_unordered(_verify_and_size, str_paths, chunksize=chunksize)):
            p = Path(ps)
            sizes[p] = result
            if result is None:
                bad.append(p)
            if (i + 1) % 20000 == 0:
                print(f"[verify] {i+1}/{total} (bad so far: {len(bad)})")
    print(f"[verify] total bad images: {len(bad)}/{total}")
    if delete_corrupt and bad:
        removed = 0
        for p in bad:
            try:
                if p.exists():
                    p.unlink()
                    removed += 1
            except Exception:
                pass
        print(f"[verify] removed {removed} corrupt files")
    return sizes


def _cache_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / 1e9


def _df_to_specs(
    df: pd.DataFrame,
    image_local: dict[str, Path],
    sizes: dict[Path, tuple[int, int] | None],
) -> tuple[list[TrajectorySpec], int, int]:
    """Return (specs, n_dropped_tasks, n_bad_step_images).

    Drops a whole trajectory if ANY step's ``before_image`` is missing or
    failed verification (corrupt/partial download).
    """
    specs: list[TrajectorySpec] = []
    dropped = 0
    bad_steps = 0
    for task_id, task_df in df.groupby("task_id", sort=False):
        task_df = task_df.sort_values(["step_index", "action_index"])
        # dedupe per step (thought & all_actions are shared across action_index rows)
        step_rows = task_df.groupby("step_index", sort=True).first().reset_index()
        instruction = str(task_df.iloc[0]["instruction"])
        answer = task_df.iloc[0].get("answer")
        answer_str = str(answer) if isinstance(answer, str) and answer else None

        steps: list[StepInfo] = []
        last_idx = len(step_rows) - 1
        task_ok = True
        for i, row in step_rows.iterrows():
            try:
                all_actions = json.loads(row["all_actions"])
            except Exception:
                all_actions = []
            img_rel = row["before_image"]
            local = image_local.get(img_rel)
            size = sizes.get(local) if local is not None else None
            if local is None or not local.exists() or size is None:
                # Missing or corrupt/partial image → drop the whole trajectory.
                bad_steps += 1
                task_ok = False
                break
            w, h = size
            steps.append(StepInfo(
                step_index=int(row["step_index"]),
                thought=str(row["thought"] or ""),
                all_actions=all_actions,
                image_path=str(local),
                image_w=w,
                image_h=h,
                answer=answer_str if i == last_idx else None,
            ))
        if task_ok and steps:
            specs.append(TrajectorySpec(task_id=str(task_id), instruction=instruction, steps=steps))
        else:
            dropped += 1
    return specs, dropped, bad_steps


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_client, img_client = _get_clients(args)

    blob_path = f"{args.parquet_prefix}/{args.split}.parquet"
    parquet_local = _cached_download_parquet(data_client, blob_path, cache_dir, args.force_redownload)
    df = pd.read_parquet(parquet_local)
    print(f"[parquet] {args.split}: {len(df)} rows, {df['task_id'].nunique()} tasks")

    if args.limit_tasks > 0:
        keep = df["task_id"].drop_duplicates().head(args.limit_tasks).tolist()
        df = df[df["task_id"].isin(keep)].copy()
        print(f"[parquet] limited to {args.limit_tasks} tasks ({len(df)} rows)")

    rel_paths = sorted(df["before_image"].dropna().unique().tolist())
    if args.no_download:
        print(f"[images] --no_download: skipping download, using local cache only")
        images_root = cache_dir / "images"
        image_local = {rel: images_root / rel for rel in rel_paths}
    else:
        image_local = _download_images(img_client, rel_paths, cache_dir, args.download_workers, args.force_redownload)
    cache_gb = _cache_size_gb(cache_dir / "images")
    print(f"[images] cache dir size: {cache_gb:.2f} GB")
    if cache_gb > args.max_cache_gb:
        print(f"[images] WARNING: exceeds --max_cache_gb={args.max_cache_gb}")

    local_list = [image_local[rel] for rel in rel_paths if image_local[rel].exists()]
    print(f"[images] {len(local_list)}/{len(rel_paths)} files present; verifying integrity...")
    # Delete corrupt/partial PNGs so a future --force_redownload (or plain rerun)
    # re-fetches them instead of silently reusing truncated bytes.
    sizes = _probe_image_sizes(local_list, n_workers=args.download_workers, delete_corrupt=True)

    specs, n_dropped, n_bad = _df_to_specs(df, image_local, sizes)
    print(f"[convert] {len(specs)} trajectories kept; {n_dropped} dropped (due to {n_bad} missing/corrupt step images)")
    print(f"[convert] running {args.num_workers}-way pool")

    samples = convert_specs(
        specs,
        image_history_n=args.image_history_n,
        action_split_mode=args.action_split_mode,
        action_mapping_mode=args.action_mapping_mode,
        num_workers=args.num_workers,
    )

    out_path = output_dir / f"{args.split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[output] {out_path}: {len(samples)} samples")

    # Record settings alongside for reproducibility
    (output_dir / f"{args.split}.meta.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
