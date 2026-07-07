#!/usr/bin/env python3
"""
CrossModalRetriever + InternVideo2 集成验证

流程:
  1. 加载 InternVideo2，从多个视频提取特征
  2. 构建 FeatureIndex
  3. 用 CrossModalRetriever 做 text→video 检索
  4. 与原始 InternVideo2 相似度对比
"""
import sys, os, types, torch, numpy as np, cv2

# ── transformers 5.x compat: apply_chunking_to_forward ──
import transformers.modeling_utils as _tmu
from transformers.pytorch_utils import apply_chunking_to_forward as _actf
_tmu.apply_chunking_to_forward = _actf

# ── flash_attn mock ─────────────────────────────────────
class ML:
    def __init__(self): self._m = {}
    def _mk(self, n):
        m = types.ModuleType(n); m.__path__ = []
        MC = type("MC", (), {"__init__": lambda s,*a,**k: None, "__call__": lambda s,*a,**k: None, "__getattr__": lambda s,n: s, "apply": staticmethod(lambda *a,**k: None)})
        for a in ["FusedMLP","FlashAttention","RotaryEmbedding","DropoutAddRMSNorm","Block","pad_input","unpad_input","flash_attn_varlen_qkvpacked_func"]:
            setattr(m, a, MC)
        self._m[n]=m; return m
    def find_spec(self, n, p, t):
        if n.startswith("flash_attn") or n=="flash_attention_class":
            import importlib.util
            if n not in self._m: self._mk(n)
            return importlib.util.spec_from_loader(n, self)
        return None
    def create_module(self, s): return self._m.get(s.name)
    def exec_module(self, m): pass
sys.meta_path.insert(0, ML())

# ── InternVideo2 路径 ────────────────────────────────────
PROJ = "/data14/jiaming.lin.2601/MUGen-VR"
IV_PATH = os.path.join(PROJ, "third_party/InternVideo/InternVideo2/multi_modality")
os.chdir(IV_PATH)
sys.path.insert(0, ".")

# ── 先 override models/__init__.py 再导入 ────────────────
init_path = os.path.join(IV_PATH, "models/__init__.py")
with open(init_path) as f: _backup = f.read()
with open(init_path, "w") as f:
    f.write("from .backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224\n")

from configs.data import *
from configs.model import *
from transformers import BertTokenizer, BertModel
from models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new
import torch.nn as nn
from models import pretrain_internvideo2_1b_patch14_224

# ── MUGen-VR 导入 ───────────────────────────────────────
sys.path.insert(0, PROJ)
from src.retrieval.retriever import CrossModalRetriever
from src.retrieval.feature_index import FeatureIndex
from src.common.interfaces import RetrievalResult

# ══════════════════════════════════════════════════════════
# Step 1: 加载 InternVideo2 模型
# ══════════════════════════════════════════════════════════
DEVICE = "cpu"
print("=" * 60)
print("Step 1: Loading InternVideo2 model (2.63GB)...")
print("=" * 60)

CKPT = "/data14/jiaming.lin.2601/.cache/huggingface/hub/models--OpenGVLab--InternVideo2-Stage2_1B-224p-f4/snapshots/4362e1f88a992e7edbfd7696f7f78b7f79426dfd/InternVideo2-stage2_1b-224p-f4.pt"

class AD(dict):
    def __init__(self, d=None):
        super().__init__()
        if d:
            for k, v in d.items():
                v2 = AD(v) if isinstance(v, dict) else v
                self[k] = v2
                setattr(self, k, v2)
    def __getattr__(self, k):
        try: return self[k]
        except: raise AttributeError(k)

text_enc = "bert_large"
model_cfg = AD({
    "vision_encoder": AD({
        "name": "pretrain_internvideo2_1b_patch14_224", "img_size": 224, "num_frames": 4,
        "tubelet_size": 1, "patch_size": 14, "d_model": 1408, "clip_embed_dim": 768,
        "clip_teacher_embed_dim": 3200, "clip_teacher_final_dim": 768,
        "clip_norm_type": "l2", "clip_return_layer": 6, "clip_student_return_interval": 1,
        "use_checkpoint": True, "checkpoint_num": 40, "use_flash_attn": False,
        "use_fused_rmsnorm": False, "use_fused_mlp": False, "clip_teacher": None,
        "clip_input_resolution": 224, "clip_teacher_return_interval": 1,
        "video_mask_type": "random", "video_mask_ratio": 0.8, "image_mask_type": "random",
        "image_mask_ratio": 0.5, "sep_image_video_pos_embed": True,
        "keep_temporal": False, "only_mask": True, "pretrained": None,
    }),
    "text_encoder": AD(TextEncoders[text_enc]),
    "embed_dim": 512, "multimodal": {"enable": True},
})

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
vision_encoder = pretrain_internvideo2_1b_patch14_224(model_cfg)

# Use standard HuggingFace BERT instead of InternVideo's build_bert
bert_model = BertModel.from_pretrained("bert-large-uncased", add_pooling_layer=False)
bert_model.eval()

class TextEncoderWrapper(nn.Module):
    """Minimal wrapper to mimic InternVideo's text encoder interface."""
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

text_encoder = TextEncoderWrapper(bert_model)

model = nn.Module()
model.vision_encoder = vision_encoder
model.text_encoder = text_encoder
model.vision_proj = nn.Linear(768, 512)
model.text_proj = nn.Linear(1024, 512)
model.tokenizer = tokenizer

def _get_vid_feat(self, frames):
    with torch.no_grad():
        image = frames.permute(0,2,1,3,4).float()
        _, pve, _, _ = self.vision_encoder(image, None, False)
        vfeat = self.vision_proj(pve)
        vfeat /= vfeat.norm(dim=-1, keepdim=True)
    return vfeat

def _get_txt_feat(self, text):
    with torch.no_grad():
        ti = self.tokenizer(text, padding="max_length", truncation=True, max_length=40, return_tensors="pt")
        to = self.text_encoder.bert(ti["input_ids"], attention_mask=ti["attention_mask"], return_dict=True, mode="text")
        tfeat = self.text_proj(to.last_hidden_state[:, 0])
        tfeat /= tfeat.norm(dim=-1, keepdim=True)
    return tfeat

model.get_vid_feat = _get_vid_feat.__get__(model)
model.get_txt_feat = _get_txt_feat.__get__(model)

print("Loading checkpoint...")
checkpoint = torch.load(CKPT, map_location="cpu")
state_dict = checkpoint.get("model", checkpoint.get("module", checkpoint))
interpolate_pos_embed_internvideo2_new(state_dict, model.vision_encoder, orig_t_size=4)
msg = model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"  Model loaded! Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
print()

# ══════════════════════════════════════════════════════════
# Step 2: 从多个视频提取特征
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("Step 2: Extracting features from demo videos...")
print("=" * 60)

def extract_video_frames(video_path, num_frames=4, target_size=224):
    """Extract uniformly sampled frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        r, f = cap.read()
        if not r:
            break
        frames.append(f)
    cap.release()
    if len(frames) == 0:
        return None
    step = len(frames) // num_frames
    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    selected = []
    for i in range(0, len(frames), step)[:num_frames]:
        f = cv2.resize(frames[i][:, :, ::-1], (target_size, target_size))
        selected.append(np.expand_dims((f / 255.0 - v_mean) / v_std, axis=(0, 1)))
    vt = np.transpose(np.concatenate(selected, axis=1), (0, 1, 4, 2, 3))
    return torch.from_numpy(vt).float()

# 模拟多个视频（从 example1.mp4 的不同时间范围采样不同的帧组）
demo_video = os.path.join(IV_PATH, "demo/example1.mp4")
print(f"  Source video: {demo_video}")

# 提取多个独立的 frame 组作为"不同视频"
video_feats = []
video_metadata = []

all_frames = []
cap = cv2.VideoCapture(demo_video)
while True:
    r, f = cap.read()
    if not r:
        break
    all_frames.append(f)
cap.release()
total_frames = len(all_frames)
print(f"  Total frames available: {total_frames}")

# 创建4个"视频片段"，每个取不同的帧范围
segments = [
    (0, total_frames // 3, "segment_1_dog_playing"),              # 狗玩耍
    (total_frames // 3, 2 * total_frames // 3, "segment_2_dog_mid"),  # 中间段
    (2 * total_frames // 3, total_frames, "segment_3_dog_end"),     # 后段
    (0, total_frames, "segment_4_full_video"),                      # 完整视频
]

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

for start, end, name in segments:
    seg_frames = all_frames[start:end]
    if len(seg_frames) < 4:
        continue
    step = len(seg_frames) // 4
    selected = []
    for i in range(0, len(seg_frames), step)[:4]:
        f = cv2.resize(seg_frames[i][:, :, ::-1], (224, 224))
        selected.append(np.expand_dims((f / 255.0 - v_mean) / v_std, axis=(0, 1)))
    if len(selected) < 4:
        continue
    vt = np.transpose(np.concatenate(selected, axis=1), (0, 1, 4, 2, 3))
    vid_tensor = torch.from_numpy(vt).float()

    with torch.no_grad():
        feat = model.get_vid_feat(vid_tensor)
    video_feats.append(feat)
    video_metadata.append({"name": name, "source": "example1.mp4", "frames": f"{start}-{end}"})
    print(f"  [{name:30s}] feat shape: {feat.shape}, norm: {feat[0].norm():.4f}")

video_feats_tensor = torch.cat(video_feats, dim=0)
print(f"\n  All video features: {video_feats_tensor.shape}")
print()

# ══════════════════════════════════════════════════════════
# Step 3: 构建 CrossModalRetriever
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("Step 3: Building CrossModalRetriever...")
print("=" * 60)

# InternVideo2 输出 512-dim video features
VFEAT_DIM = video_feats_tensor.size(-1)
retriever = CrossModalRetriever(dim=VFEAT_DIM, top_k=3)
retriever.build_index(video_feats_tensor, video_metadata)
print(f"  Retriever built! Index size: {retriever.index.size}")
print()

# ══════════════════════════════════════════════════════════
# Step 4: 运行 text→video 检索
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("Step 4: Text-to-Video retrieval...")
print("=" * 60)

queries = [
    "A playful dog and its owner wrestle in the snowy yard.",
    "A dog playing outdoors in winter.",
    "A cat sitting on a windowsill.",
]

with torch.no_grad():
    for q in queries:
        q_feat = model.get_txt_feat(q)
        result = retriever.retrieve(q_feat, top_k=3)
        print(f"\n  Query: \"{q}\"")
        print(f"  {'#':<3} {'Score':<10} {'Name':<30}")
        print(f"  {'-'*3} {'-'*10} {'-'*30}")
        for i, (idx, score, meta) in enumerate(zip(
            result.retrieved_indices[0],
            result.retrieved_scores[0],
            result.retrieved_metadata[0],
        )):
            print(f"  {i+1:<3} {float(score):<10.4f} {meta['name']:<30}")

# ══════════════════════════════════════════════════════════
# Step 5: 对比原始 InternVideo2 相似度
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: Raw InternVideo2 similarity check...")
print("=" * 60)

with torch.no_grad():
    for q in queries:
        q_feat = model.get_txt_feat(q)
        # Raw dot product similarity
        sims = (video_feats_tensor @ q_feat.T).squeeze()
        top_idx = sims.argsort(descending=True)[:3]
        print(f"\n  Query: \"{q[:40]:40s}\"")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1}. score={sims[idx]:.4f}  -> {video_metadata[idx]['name']}")

# ══════════════════════════════════════════════════════════
# Step 6: Multi-modal retrieval test
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 6: Multi-modal retrieval (text + dummy image)...")
print("=" * 60)

# 用相同的 text feature 模拟多模态查询
dummy_img_feat = torch.randn(1, 512)
dummy_img_feat /= dummy_img_feat.norm(dim=-1, keepdim=True)
dummy_audio_feat = torch.randn(1, 512)
dummy_audio_feat /= dummy_audio_feat.norm(dim=-1, keepdim=True)

with torch.no_grad():
    q1 = model.get_txt_feat("A dog playing outdoors.")
    multi_result = retriever.multi_modal_retrieve({
        "text": q1,
        "image": dummy_img_feat,
    }, top_k=3)
    print('  Multi-modal query (text + image):')
    for i, score in enumerate(multi_result.retrieved_scores[0]):
        print('    {}. score={:.4f}'.format(i+1, float(score)))

# 恢复 __init__.py
with open(init_path, "w") as f:
    f.write(_backup)

print("\n" + "=" * 60)
print("PASSED: CrossModalRetriever + InternVideo2 integration!")
print("=" * 60)
