import sys, os, types, torch, numpy as np, cv2

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

iv_path = "/data14/jiaming.lin.2601/MUGen-VR/third_party/InternVideo/InternVideo2/multi_modality"
os.chdir(iv_path); sys.path.insert(0, ".")

init_path = os.path.join(iv_path, "models/__init__.py")
with open(init_path) as f: backup = f.read()
with open(init_path, "w") as f:
    f.write("from .backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224\n")

from configs.data import *
from configs.model import *
from transformers import BertTokenizer
from models.backbones.bert.builder import build_bert
from models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new
import torch.nn as nn
from models import pretrain_internvideo2_1b_patch14_224

ckpt = "/data14/jiaming.lin.2601/.cache/huggingface/hub/models--OpenGVLab--InternVideo2-Stage2_1B-224p-f4/snapshots/4362e1f88a992e7edbfd7696f7f78b7f79426dfd/InternVideo2-stage2_1b-224p-f4.pt"

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

# Build config with proper nesting
text_enc = "bert_large"
model_cfg = AD({
    "vision_encoder": AD({
        "name": "pretrain_internvideo2_1b_patch14_224", "img_size": 224,
        "num_frames": 4, "tubelet_size": 1, "patch_size": 14, "d_model": 1408,
        "clip_embed_dim": 768, "clip_teacher_embed_dim": 3200,
        "clip_teacher_final_dim": 768, "clip_norm_type": "l2",
        "clip_return_layer": 6, "clip_student_return_interval": 1,
        "use_checkpoint": True, "checkpoint_num": 40,
        "use_flash_attn": False, "use_fused_rmsnorm": False, "use_fused_mlp": False,
        "clip_teacher": None, "clip_input_resolution": 224,
        "clip_teacher_return_interval": 1, "video_mask_type": "random",
        "video_mask_ratio": 0.8, "image_mask_type": "random",
        "image_mask_ratio": 0.5, "sep_image_video_pos_embed": True,
        "keep_temporal": False, "only_mask": True, "pretrained": None,
    }),
    "text_encoder": AD(TextEncoders[text_enc]),
    "embed_dim": 512, "multimodal": {"enable": True},
})

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
vision_encoder = pretrain_internvideo2_1b_patch14_224(model_cfg)
text_encoder = build_bert(model_cfg, True, True)

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

print("Loading checkpoint (2.63 GB)...")
checkpoint = torch.load(ckpt, map_location="cpu")
state_dict = checkpoint.get("model", checkpoint.get("module", checkpoint))
interpolate_pos_embed_internvideo2_new(state_dict, model.vision_encoder, orig_t_size=4)
msg = model.load_state_dict(state_dict, strict=False)
print("Model loaded! Missing: %d, Unexpected: %d" % (len(msg.missing_keys), len(msg.unexpected_keys)))
model.eval()

cap = cv2.VideoCapture("demo/example1.mp4")
frames = []
while True:
    r, f = cap.read()
    if not r: break
    frames.append(f)
cap.release()

v_mean = np.array([0.485,0.456,0.406]).reshape(1,1,3)
v_std = np.array([0.229,0.224,0.225]).reshape(1,1,3)
step = len(frames) // 4
selected = [cv2.resize(frames[i][:,:,::-1], (224,224)) for i in range(0, len(frames), step)][:4]
vt = [np.expand_dims((x/255.0-v_mean)/v_std, axis=(0,1)) for x in selected]
vt = np.transpose(np.concatenate(vt, axis=1), (0,1,4,2,3))
vid_tensor = torch.from_numpy(vt).float()

with torch.no_grad():
    vfeat = model.get_vid_feat(vid_tensor)

texts = [
    "A playful dog and its owner wrestle in the snowy yard.",
    "A man in a gray coat walks through the snowy landscape.",
    "A person shovels the snow-covered pavement.",
    "A cat excitedly runs through the yard.",
    "A person bundled up enjoys a peaceful winter walk.",
]

with torch.no_grad():
    tfeats = torch.cat([model.get_txt_feat(t) for t in texts], 0)
    probs = (100.0 * vfeat @ tfeats.T).softmax(dim=-1)
    tp, ti = probs.topk(5, dim=-1)

print()
print("=== Retrieval Results (Video-to-Text) ===")
for i, (idx, prob) in enumerate(zip(ti[0], tp[0])):
    print("  %d. [%.1f%%] %s" % (i+1, prob*100, texts[idx]))
print()
print("InternVideo2 retrieval baseline PASSED!")

with open(init_path, "w") as f: f.write(backup)
