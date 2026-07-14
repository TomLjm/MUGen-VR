from huggingface_hub import snapshot_download

model_id = "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers"
print(f"starting snapshot_download: {model_id}", flush=True)
path = snapshot_download(repo_id=model_id, resume_download=True, max_workers=1)
print(f"download complete: {path}", flush=True)
