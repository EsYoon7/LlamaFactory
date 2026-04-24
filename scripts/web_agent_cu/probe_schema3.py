"""Probe the with_thinking dataset + image container."""
import io, json
import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import ManagedIdentityCredential

ACCOUNT_URL = "https://t2vgusw2.blob.core.windows.net"
APP_ID = "8cafa2b1-a2a7-4ad9-814a-ffe4aed7e800"
DATA_CONTAINER = "esyoon"
IMG_CONTAINER = "v-zhihongtan"
BLOB = "dataset/gpt5-4_v260409/processed_raw_with_thinking_train_val_split/val.parquet"

cred = ManagedIdentityCredential(client_id=APP_ID)
dc = ContainerClient(account_url=ACCOUNT_URL, container_name=DATA_CONTAINER, credential=cred)
ic = ContainerClient(account_url=ACCOUNT_URL, container_name=IMG_CONTAINER, credential=cred)

df = pd.read_parquet(io.BytesIO(dc.get_blob_client(BLOB).download_blob().readall()))
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

row = df.iloc[0]
print(f"\n=== Row 0 ===")
for col in df.columns:
    v = row[col]
    if isinstance(v, str):
        s = v[:500].replace("\n", "\\n")
        print(f"  {col} (len={len(v)}): {s}{'...' if len(v) > 500 else ''}")
    else:
        print(f"  {col}: {repr(v)[:200]}")

# Search for reasoning/thought-like columns
thinking_cols = [c for c in df.columns if "think" in c.lower() or "reason" in c.lower() or "thought" in c.lower()]
print(f"\nThinking-like columns: {thinking_cols}")
if thinking_cols:
    for c in thinking_cols:
        for i in range(min(3, len(df))):
            v = df.iloc[i][c]
            if isinstance(v, str) and v:
                print(f"\n  {c}[{i}] (len={len(v)}): {v[:400]}")
                break

# Verify image container
img_path = row['before_image']
print(f"\n=== Image probe ===\nPath: {img_path}")
try:
    bc = ic.get_blob_client(img_path)
    props = bc.get_blob_properties()
    print(f"  FOUND in container '{IMG_CONTAINER}': {props.size} bytes")
except Exception as e:
    print(f"  miss: {type(e).__name__}: {e}")
    # try listing prefix
    try:
        it = ic.list_blob_names(name_starts_with="results/")
        for i, name in enumerate(it):
            if i < 5:
                print(f"    sample: {name}")
            else:
                break
    except Exception as e2:
        print(f"  list failed: {e2}")

print(f"\nTotal tasks: {df['task_id'].nunique()}, rows: {len(df)}")
print(f"num_actions_in_step: {df['num_actions_in_step'].value_counts().sort_index().to_dict()}")
