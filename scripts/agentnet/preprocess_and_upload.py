"""One-time prep: stream AgentNet → sharegpt parquet → upload to Azure blob.

Nothing touches local disk:
  * Raw JSONL is streamed from the ``esyoon`` container chunk-by-chunk via
    ``ContainerClient.download_blob(...).chunks()``.
  * Converted sharegpt rows are collected into an in-memory ``pyarrow.Table``.
  * Parquet bytes are written into a ``BytesIO`` sink.
  * Those bytes are uploaded back to the same blob container at
    ``dataset/AgentNet/processed_sharegpt/{train,val}.parquet``.

After this script succeeds, training reads the parquet via the blobfuse
mount (``/home/qid/esyoon/blob_data/dataset/AgentNet/processed_sharegpt/...``).
No per-training-run bootstrap. Multi-node safe (each node just mounts
blobfuse and reads the same parquet).

Usage:
  cd /home/qid/esyoon/workspace/web-agent-training/libs/LLaMA-Factory
  .venv/bin/python scripts/agentnet/preprocess_and_upload.py \
      --val_pct 1 --image_history_n 3
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Side-load the converter module (skip llamafactory package __init__)
# ---------------------------------------------------------------------------

def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_DATA_DIR = _REPO_ROOT / "src" / "llamafactory" / "data"
_pkg_stub = importlib.util.module_from_spec(
    importlib.util.spec_from_loader("llamafactory_agentnet_prep", loader=None)
)
_pkg_stub.__path__ = [str(_DATA_DIR)]
sys.modules["llamafactory_agentnet_prep"] = _pkg_stub
_load_module("llamafactory_agentnet_prep.web_agent_cu", _DATA_DIR / "web_agent_cu.py")
_an = _load_module("llamafactory_agentnet_prep.agentnet", _DATA_DIR / "agentnet.py")
AgentNetSpec = _an.AgentNetSpec
AgentNetStep = _an.AgentNetStep
build_agentnet_sample = _an.build_agentnet_sample
_build_osworld_system_prompt = _an._build_osworld_system_prompt


DEFAULT_ACCOUNT_URL = "https://t2vgusw2.blob.core.windows.net"
DEFAULT_APP_ID = "8cafa2b1-a2a7-4ad9-814a-ffe4aed7e800"
DEFAULT_CONTAINER = "esyoon"

# (raw_jsonl_blob, image_subdir_in_blob)
SOURCES = [
    ("dataset/AgentNet/agentnet_ubuntu_5k.jsonl",
     "dataset/AgentNet/images_extracted/ubuntu_images"),
    ("dataset/AgentNet/agentnet_win_mac_18k.jsonl",
     "dataset/AgentNet/images_extracted/win_mac_images"),
]

# Output blob paths (under the same container) + corresponding blobfuse paths
# that training will read from.
DEFAULT_OUTPUT_PREFIX = "dataset/AgentNet/processed_sharegpt"
DEFAULT_BLOBFUSE_ROOT = "/home/qid/esyoon/blob_data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--account_url", default=DEFAULT_ACCOUNT_URL)
    p.add_argument("--app_id", default=DEFAULT_APP_ID)
    p.add_argument("--container", default=DEFAULT_CONTAINER)
    p.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX,
                   help="Blob key prefix for the output parquet shards.")
    p.add_argument("--blobfuse_root", default=DEFAULT_BLOBFUSE_ROOT,
                   help="Mount point used to resolve image paths embedded in "
                        "the parquet. Training containers must mount this path.")
    p.add_argument("--val_pct", type=int, default=1,
                   help="0..99, percent of trajectories for val (hash-based).")
    p.add_argument("--image_history_n", type=int, default=3)
    p.add_argument("--require_completed", action="store_true")
    p.add_argument("--min_alignment_score", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, stop after N trajectories (dry-run).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-upload even if the destination parquet already exists.")
    return p.parse_args()


def _get_client(args):
    from azure.storage.blob import ContainerClient
    from azure.identity import ManagedIdentityCredential
    cred = ManagedIdentityCredential(client_id=args.app_id)
    return ContainerClient(account_url=args.account_url,
                           container_name=args.container, credential=cred)


def _stream_jsonl(client, blob_name: str):
    downloader = client.download_blob(blob_name)
    buf = b""
    for chunk in downloader.chunks():
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line, buf = buf[:nl], buf[nl + 1:]
            line = line.strip()
            if line:
                yield json.loads(line.decode("utf-8"))
    tail = buf.strip()
    if tail:
        yield json.loads(tail.decode("utf-8"))


def _bucket(task_id: str) -> int:
    return int(hashlib.md5(task_id.encode("utf-8")).hexdigest()[:8], 16) % 100


def _spec_from_record(d: dict, image_root: Path, args) -> "AgentNetSpec | None":
    traj = d.get("traj") or []
    if not traj:
        return None
    if args.require_completed and not d.get("task_completed"):
        return None
    if d.get("alignment_score") is not None and float(d["alignment_score"]) < args.min_alignment_score:
        return None
    steps: list[AgentNetStep] = []
    for step in traj:
        val = step.get("value") or {}
        img_rel = step.get("image")
        if not img_rel:
            return None
        steps.append(AgentNetStep(
            thought=val.get("thought", ""),
            action_nl=val.get("action", ""),
            code=val.get("code", ""),
            image_path=str(image_root / img_rel),
        ))
    if not steps:
        return None
    return AgentNetSpec(
        task_id=str(d.get("task_id", "unknown")),
        instruction=str(d.get("instruction") or d.get("natural_language_task") or ""),
        steps=steps,
    )


def _convert(args) -> tuple[list[dict], list[dict]]:
    """Stream + convert everything once; split by task_id hash."""
    client = _get_client(args)
    blobfuse_root = Path(args.blobfuse_root)
    # Freeze one system prompt for the whole run so the baked date string is
    # identical across every sample in the shard.
    system_prompt = _build_osworld_system_prompt()
    train, val = [], []
    seen = 0
    dropped = 0
    for blob_name, img_subdir in SOURCES:
        print(f"[blob] streaming {args.container}/{blob_name}", flush=True)
        image_root = blobfuse_root / img_subdir
        for d in _stream_jsonl(client, blob_name):
            seen += 1
            spec = _spec_from_record(d, image_root, args)
            if spec is None:
                dropped += 1
                continue
            try:
                sample = build_agentnet_sample(
                    spec,
                    image_history_n=args.image_history_n,
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                print(f"[skip] task={spec.task_id}: {exc}", flush=True)
                dropped += 1
                continue
            if sample is None:
                dropped += 1
                continue
            in_val = _bucket(spec.task_id) < args.val_pct
            (val if in_val else train).append(sample)
            if seen % 2000 == 0:
                print(f"[stream] seen={seen} kept={len(train)+len(val)} (train={len(train)}, val={len(val)})",
                      flush=True)
            if args.limit and seen >= args.limit:
                break
        if args.limit and seen >= args.limit:
            break
    print(f"[stream] done. seen={seen}  train={len(train)}  val={len(val)}  dropped={dropped}",
          flush=True)
    return train, val


def _to_parquet_bytes(samples: list[dict]) -> bytes:
    """Serialize sharegpt samples as parquet in memory.

    Parquet schema matches the sharegpt fields LLaMA-Factory expects:
      * messages: list<struct<role: string, content: string>>
      * images:   list<string>
      * tools:    string
      * task_id:  string
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    msg_struct = pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
    schema = pa.schema([
        pa.field("messages", pa.list_(msg_struct)),
        pa.field("images", pa.list_(pa.string())),
        pa.field("tools", pa.string()),
        pa.field("task_id", pa.string()),
    ])

    cols = {"messages": [], "images": [], "tools": [], "task_id": []}
    for s in samples:
        cols["messages"].append(s["messages"])
        cols["images"].append(s["images"])
        cols["tools"].append(s["tools"])
        cols["task_id"].append(s["task_id"])
    table = pa.Table.from_pydict(cols, schema=schema)

    sink = io.BytesIO()
    # snappy is widely supported and doesn't require an extra dep.
    pq.write_table(table, sink, compression="snappy")
    return sink.getvalue()


def _upload(client, blob_name: str, data: bytes, overwrite: bool):
    if not overwrite:
        try:
            client.get_blob_client(blob_name).get_blob_properties()
            print(f"[upload] exists, skipping: {blob_name} (pass --overwrite to replace)")
            return
        except Exception:
            pass
    print(f"[upload] {blob_name}: {len(data)/1e6:.1f} MB", flush=True)
    client.upload_blob(name=blob_name, data=data, overwrite=True)


def main() -> None:
    args = parse_args()
    train, val = _convert(args)
    if not train:
        raise SystemExit("No train samples produced; check filters / blob access.")

    print("[pack] encoding train parquet...", flush=True)
    train_bytes = _to_parquet_bytes(train)
    print("[pack] encoding val parquet...", flush=True)
    val_bytes = _to_parquet_bytes(val) if val else b""

    client = _get_client(args)
    train_blob = f"{args.output_prefix}/train.parquet"
    val_blob = f"{args.output_prefix}/val.parquet"

    _upload(client, train_blob, train_bytes, args.overwrite)
    if val_bytes:
        _upload(client, val_blob, val_bytes, args.overwrite)

    # Sidecar meta so we know when/how the shards were built. Upload only;
    # no local write.
    meta = {
        **vars(args),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_bytes": len(train_bytes),
        "val_bytes": len(val_bytes),
    }
    client.upload_blob(
        name=f"{args.output_prefix}/_meta.json",
        data=json.dumps(meta, indent=2).encode("utf-8"),
        overwrite=True,
    )
    print("[done] uploaded.")
    print(f"  train: az://{args.container}/{train_blob}")
    print(f"    -> blobfuse: {args.blobfuse_root}/{train_blob}")
    if val_bytes:
        print(f"  val:   az://{args.container}/{val_blob}")
        print(f"    -> blobfuse: {args.blobfuse_root}/{val_blob}")


if __name__ == "__main__":
    main()
