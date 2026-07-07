#!/usr/bin/env python3
"""
HierarchicalConditionFusion + ImageBind 集成验证
"""
import sys, os, torch, numpy as np

PROJ = "/data14/jiaming.lin.2601/MUGen-VR"
IB_DIR = os.path.join(PROJ, "third_party/ImageBind")
sys.path.insert(0, IB_DIR)

from imagebind import data as ib_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import torch.nn.functional as F

sys.path.insert(0, PROJ)
from src.fusion.fusion_module import HierarchicalConditionFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATHS = [os.path.join(PROJ, "third_party/VBench/VBench-2.0/vbench2/third_party/YOLO-World/mmyolo/demo/demo.jpg")]
TEXTS = ["a dog running in the park", "a busy city street at night", "a peaceful mountain landscape", "a cat sitting on a windowsill"]
AUDIO_PATHS = []
IB_DIM = 1024
FUSION_HIDDEN = 768

print(f"Device: {DEVICE}  ImageBind dim: {IB_DIM}  Fusion hidden: {FUSION_HIDDEN}\n")

# Step 1: Load ImageBind
print("=" * 60)
print("Step 1: Loading ImageBind model...")
print("=" * 60)
os.chdir(IB_DIR)
model_ib = imagebind_model.imagebind_huge(pretrained=True)
model_ib.eval()
model_ib.to(DEVICE)
print(f"  ImageBind loaded! Params: {sum(p.numel() for p in model_ib.parameters()):,}\n")

# Step 2: Extract real embeddings
print("=" * 60)
print("Step 2: Extracting real ImageBind embeddings...")
print("=" * 60)
text_tokens = ib_data.load_and_transform_text(TEXTS, DEVICE)
image_tensors = ib_data.load_and_transform_vision_data(IMAGE_PATHS, DEVICE)
print(f"  Text tokens: {text_tokens.shape}, Image tensors: {image_tensors.shape}")

with torch.no_grad():
    embeddings = model_ib({ModalityType.TEXT: text_tokens, ModalityType.VISION: image_tensors})

text_emb = embeddings[ModalityType.TEXT]
image_emb = embeddings[ModalityType.VISION].expand(text_emb.size(0), -1)
print(f"  Text embedding: {text_emb.shape}, Image embedding: {image_emb.shape}")
print(f"  Text norm (sample): {text_emb[0].norm():.4f}, Image norm: {image_emb[0].norm():.4f}\n")

# Step 3: Build Fusion
print("=" * 60)
print("Step 3: Building HierarchicalConditionFusion...")
print("=" * 60)
fusion = HierarchicalConditionFusion(
    dims={"text": IB_DIM, "image": IB_DIM, "audio": IB_DIM},
    hidden_dim=FUSION_HIDDEN, num_heads=8, modality_dropout=0.0,
)
fusion.eval()
fusion.to(DEVICE)
print(f"  Fusion module built! Params: {sum(p.numel() for p in fusion.parameters()):,}\n")

# Step 4: Run fusion
print("=" * 60)
print("Step 4: Running fusion with real embeddings...")
print("=" * 60)
with torch.no_grad():
    fused_text = fusion({"text": (text_emb, None)})
    fused_image = fusion({"image": (image_emb, None)})
    fused_combined, layer_weights = fusion({"text": (text_emb, None), "image": (image_emb, None)}, return_weights=True)

print(f"  Text-only fused:    {fused_text.shape}  norm={fused_text[0].norm():.4f}")
print(f"  Image-only fused:   {fused_image.shape}  norm={fused_image[0].norm():.4f}")
print(f"  Text+Image fused:   {fused_combined.shape}  norm={fused_combined[0].norm():.4f}")
print()
for i in range(layer_weights.size(0)):
    w = layer_weights[i, 0]
    print(f"  Layer L{i}: Text={w[0]:.4f}, Image={w[1]:.4f}")
print()

# Step 5: Baseline comparison (project all to 768-dim)
print("=" * 60)
print("Step 5: Comparison with baselines...")
print("=" * 60)

avg_fused = (text_emb + image_emb) / 2
concat_fused = torch.cat([text_emb, image_emb], dim=-1)
linear_proj = torch.nn.Linear(IB_DIM * 2, FUSION_HIDDEN).to(DEVICE)
proj_to_768 = torch.nn.Linear(IB_DIM, FUSION_HIDDEN).to(DEVICE)

with torch.no_grad():
    concat_fused = linear_proj(concat_fused)
    text_proj = proj_to_768(text_emb)
    img_proj = proj_to_768(image_emb)
    avg_proj = proj_to_768(avg_fused)

def mc(a, b):
    return float(F.cosine_similarity(a, b).mean())

print(f"  {'Method':<30} {'Sim-to-Text':>12} {'Sim-to-Image':>13} {'Fused-Norm':>11}")
print(f"  {'-'*30} {'-'*12} {'-'*13} {'-'*11}")
print(f"  {'Average Baseline':<30} {mc(avg_proj, text_proj):>12.4f} {mc(avg_proj, img_proj):>13.4f} {avg_proj[0].norm():>11.4f}")
print(f"  {'Concat+Linear':<30} {mc(concat_fused, text_proj):>12.4f} {mc(concat_fused, img_proj):>13.4f} {concat_fused[0].norm():>11.4f}")
print(f"  {'HierarchicalConditionFusion':<30} {mc(fused_combined, text_proj):>12.4f} {mc(fused_combined, img_proj):>13.4f} {fused_combined[0].norm():>11.4f}")

print("\n  Per-sample fused norms:")
for i, txt in enumerate(TEXTS):
    print(f"    [{i}] '{txt[:30]:30s}' norm={fused_combined[i].norm():.4f}")

print("\n" + "=" * 60)
print("PASSED: HierarchicalConditionFusion + ImageBind integration!")
print("=" * 60)
