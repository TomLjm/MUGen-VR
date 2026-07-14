from huggingface_hub import snapshot_download

model_id = "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers"
local_dir = "models/anyflow-local"
print(f"downloading {model_id} -> {local_dir}", flush=True)
path = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    max_workers=1,
)
print(f"download complete: {path}", flush=True)
