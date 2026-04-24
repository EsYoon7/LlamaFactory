"""Probe schema of computer-use parquet on Azure blob.

Run inside es_nemo2502:
  cd /home/qid/esyoon/workspace/web-agent-training/libs/LLaMA-Factory
  source .venv/bin/activate
  python scripts/web_agent_cu/probe_schema.py
"""
import io
import json
import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import ManagedIdentityCredential

ACCOUNT_URL = "https://t2vgusw2.blob.core.windows.net"
APP_ID = "8cafa2b1-a2a7-4ad9-814a-ffe4aed7e800"
CONTAINER = "esyoon"
BLOB_NAME = "dataset/gpt5-4_v260409/processed_raw_train_val_split/val.parquet"

cred = ManagedIdentityCredential(client_id=APP_ID)
client = ContainerClient(account_url=ACCOUNT_URL, container_name=CONTAINER, credential=cred)

print(f"Downloading {BLOB_NAME} ...")
blob_data = client.get_blob_client(BLOB_NAME).download_blob().readall()
print(f"  size: {len(blob_data)/1e6:.2f} MB")

df = pd.read_parquet(io.BytesIO(blob_data))
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")

print("\n=== Row 0 high-level ===")
row = df.iloc[0]
for col in df.columns:
    val = row[col]
    if isinstance(val, (bytes, bytearray)):
        print(f"  {col}: <bytes len={len(val)}>")
    elif isinstance(val, str):
        s = val[:300].replace("\n", "\\n")
        print(f"  {col} (str, len={len(val)}): {s}{'...' if len(val) > 300 else ''}")
    elif isinstance(val, list):
        print(f"  {col}: list(len={len(val)})")
        if len(val) > 0:
            first = val[0]
            print(f"    first item type: {type(first).__name__}")
            if isinstance(first, dict):
                print(f"    first item keys: {list(first.keys())}")
                for k, v in first.items():
                    if isinstance(v, (bytes, bytearray)):
                        print(f"      {k}: <bytes len={len(v)}>")
                    elif isinstance(v, str):
                        ss = v[:200].replace("\n", "\\n")
                        print(f"      {k} (str, len={len(v)}): {ss}{'...' if len(v) > 200 else ''}")
                    elif isinstance(v, list):
                        print(f"      {k}: list(len={len(v)})")
                        if len(v) > 0:
                            print(f"        first: {repr(v[0])[:200]}")
                    elif isinstance(v, dict):
                        print(f"      {k}: dict keys={list(v.keys())}")
                    else:
                        print(f"      {k}: {repr(v)[:200]}")
    elif isinstance(val, dict):
        print(f"  {col}: dict keys={list(val.keys())}")
    else:
        print(f"  {col}: {repr(val)[:200]}")

print("\n=== Row 0 full dump (truncated bytes) ===")
def safe(o):
    if isinstance(o, (bytes, bytearray)):
        return f"<bytes len={len(o)}>"
    if isinstance(o, dict):
        return {k: safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [safe(x) for x in o[:3]] + ([f"... and {len(o)-3} more"] if len(o) > 3 else [])
    if hasattr(o, "tolist"):
        try:
            return safe(o.tolist())
        except Exception:
            pass
    if isinstance(o, str) and len(o) > 500:
        return o[:500] + f"... <truncated, total len={len(o)}>"
    return o

dump = {col: safe(row[col]) for col in df.columns}
print(json.dumps(dump, indent=2, default=str, ensure_ascii=False))
