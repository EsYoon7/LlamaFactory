"""Deeper probe: multi-action steps, reasoning, image storage location."""
import io, json
import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import ManagedIdentityCredential

ACCOUNT_URL = "https://t2vgusw2.blob.core.windows.net"
APP_ID = "8cafa2b1-a2a7-4ad9-814a-ffe4aed7e800"
CONTAINER = "esyoon"
BLOB = "dataset/gpt5-4_v260409/processed_raw_train_val_split/val.parquet"

cred = ManagedIdentityCredential(client_id=APP_ID)
client = ContainerClient(account_url=ACCOUNT_URL, container_name=CONTAINER, credential=cred)
df = pd.read_parquet(io.BytesIO(client.get_blob_client(BLOB).download_blob().readall()))

print(f"Total rows: {len(df)}, unique tasks: {df['task_id'].nunique()}")
print(f"\nnum_actions_in_step distribution:\n{df['num_actions_in_step'].value_counts().sort_index()}")
print(f"\naction_type counts:\n{df['action_type'].value_counts()}")

# Check rows where num_actions_in_step > 1 (multi-action steps)
multi = df[df['num_actions_in_step'] > 1]
print(f"\n=== Multi-action step rows: {len(multi)} ===")
if len(multi) > 0:
    sample_task = multi.iloc[0]['task_id']
    sample_step = multi.iloc[0]['step_index']
    print(f"Sample task={sample_task}, step={sample_step}")
    sub = df[(df['task_id']==sample_task) & (df['step_index']==sample_step)].sort_values('action_index')
    print(f"Rows for this step: {len(sub)}")
    for _, r in sub.iterrows():
        print(f"  action_index={r['action_index']}, type={r['action_type']}")
        print(f"    action_detail: {r['action_detail']}")
        print(f"    before_image: {r['before_image']}")
        print(f"    after_image: {r['after_image']}")
    print(f"  all_actions (shared): {sub.iloc[0]['all_actions'][:500]}")
    print(f"  all_images (shared): {sub.iloc[0]['all_images'][:500]}")

# Look at one whole task end-to-end
print(f"\n=== Whole task trajectory sample ===")
tid = df['task_id'].iloc[0]
traj = df[df['task_id']==tid].sort_values(['step_index','action_index'])
print(f"Task {tid}: {len(traj)} rows, total_steps={traj.iloc[0]['total_steps']}")
print(f"step_indices: {sorted(traj['step_index'].unique().tolist())}")
print(f"\nFirst 6 actions:")
for _, r in traj.head(6).iterrows():
    print(f"  step={r['step_index']} act={r['action_index']}/{r['num_actions_in_step']} type={r['action_type']}")
    print(f"    detail: {r['action_detail'][:200]}")
    print(f"    before: {r['before_image']}")

# Try to find image file on blob
print(f"\n=== Probing image blob location ===")
img_path = traj.iloc[0]['before_image']
print(f"Image rel path: {img_path}")
candidates = [
    img_path,
    f"dataset/gpt5-4_v260409/{img_path}",
    f"dataset/gpt5-4_v260409/final/{img_path}",
    f"dataset/gpt5-4_v260409/raw/{img_path}",
]
for c in candidates:
    try:
        bc = client.get_blob_client(c)
        props = bc.get_blob_properties()
        print(f"  FOUND: {c} ({props.size} bytes)")
        break
    except Exception as e:
        print(f"  miss: {c}  ({type(e).__name__})")

# Check if action_detail/all_actions has any 'reasoning' / 'thought' field
print(f"\n=== Looking for reasoning fields ===")
sample_actions = []
for _, r in df.head(20).iterrows():
    try:
        sample_actions.extend(json.loads(r['all_actions']))
    except Exception:
        pass
keys = set()
for a in sample_actions:
    if isinstance(a, dict):
        keys.update(a.keys())
print(f"Union of action keys (first 20 rows): {sorted(keys)}")
print(f"\nDistinct action types seen: {sorted({a.get('type') for a in sample_actions if isinstance(a, dict)})}")

# Show 5 distinct example actions
seen = set()
print(f"\n=== Example actions per type ===")
for a in sample_actions:
    if not isinstance(a, dict): continue
    t = a.get('type')
    if t in seen: continue
    seen.add(t)
    print(f"  {t}: {json.dumps(a)[:300]}")
