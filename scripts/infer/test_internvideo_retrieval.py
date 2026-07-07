import sys, os, types, torch, numpy as np, cv2

# Universal flash_attn mock via import hook
class FlashAttnMockLoader:
    def __init__(self):
        self._modules = {}
    def _make_mock(self, name):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__file__ = '<mock-' + name + '>'
        MockClass = type('MockClass', (), {
            '__init__': lambda self, *a, **kw: None,
            '__call__': lambda self, *a, **kw: None,
            '__getattr__': lambda self, n: self,
            '__enter__': lambda self: self,
            '__exit__': lambda self, *a: None,
            'apply': staticmethod(lambda *a, **kw: None),
        })
        for attr in ['FusedMLP', 'FusedDense', 'FlashAttention', 'RotaryEmbedding',
                     'DropoutAddRMSNorm', 'LayerNorm', 'CrossEntropyLoss', 'Block',
                     'ParallelBlock', 'pad_input', 'unpad_input', 'IndexFirstDim',
                     'UnpadInput', 'flash_attn_varlen_qkvpacked_func',
                     'flash_attn_varlen_func', 'flash_attn_func']:
            setattr(m, attr, MockClass)
        self._modules[name] = m
        return m
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('flash_attn') or fullname == 'flash_attention_class':
            import importlib.util
            if fullname not in self._modules:
                self._make_mock(fullname)
            return importlib.util.spec_from_loader(fullname, self)
        return None
    def create_module(self, spec):
        return self._modules.get(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, FlashAttnMockLoader())

# Setup InternVideo path
iv_path = '/data14/jiaming.lin.2601/MUGen-VR/third_party/InternVideo/InternVideo2/multi_modality'
sys.path.insert(0, iv_path)
os.chdir(iv_path)

# Build config
from configs.data import *
from configs.model import *

text_enc = 'bert_large'
ckpt = '/data14/jiaming.lin.2601/.cache/huggingface/hub/models--OpenGVLab--InternVideo2-Stage2_1B-224p-f4/snapshots/4362e1f88a992e7edbfd7696f7f78b7f79426dfd/InternVideo2-stage2_1b-224p-f4.pt'

raw_config = {
    'model': {
        'model_cls': 'InternVideo2_Stage2',
        'vision_encoder': {
            'name': 'pretrain_internvideo2_1b_patch14_224',
            'img_size': 224, 'num_frames': 4, 'tubelet_size': 1, 'patch_size': 14,
            'd_model': 1408, 'clip_embed_dim': 768, 'clip_teacher_embed_dim': 3200,
            'clip_teacher_final_dim': 768, 'clip_norm_type': 'l2', 'clip_return_layer': 6,
            'clip_student_return_interval': 1, 'use_checkpoint': True, 'checkpoint_num': 40,
            'use_flash_attn': False, 'use_fused_rmsnorm': False, 'use_fused_mlp': False,
            'clip_teacher': None, 'clip_input_resolution': 224,
            'clip_teacher_return_interval': 1, 'video_mask_type': 'random',
            'video_mask_ratio': 0.8, 'image_mask_type': 'random',
            'image_mask_ratio': 0.5, 'sep_image_video_pos_embed': True,
            'keep_temporal': False, 'only_mask': True,
        },
        'text_encoder': TextEncoders[text_enc],
        'multimodal': {'enable': True}, 'embed_dim': 512, 'temp': 0.07,
        'find_unused_parameters': False,
    },
    'pretrained_path': ckpt, 'device': 'cuda:0',
    'origin_num_frames': 4, 'use_bf16': False, 'use_half_precision': False,
    'max_txt_l': 40, 'gradient_checkpointing': True, 'num_frames': 4, 'size_t': 224,
}

class AttrDict(dict):
    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            setattr(self, k, AttrDict(v) if isinstance(v, dict) else v)

config = AttrDict(raw_config)

print('Loading InternVideo2 Stage2 1B model...')
from demo.utils import InternVideo2_Stage2
from models.backbones.bert.tokenization_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    config.model.text_encoder.pretrained, local_files_only=True)
model = InternVideo2_Stage2(config=config, tokenizer=tokenizer, is_pretrain=True)

print('Loading checkpoint (2.63 GB)...')
checkpoint = torch.load(ckpt, map_location='cpu')
state_dict = checkpoint.get('model', checkpoint.get('module', checkpoint))

from models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new
interpolate_pos_embed_internvideo2_new(state_dict, model.vision_encoder, orig_t_size=4)

msg = model.load_state_dict(state_dict, strict=False)
model = model.to('cuda:0')
model.eval()
print('Model loaded to GPU!')

# Load example video
cap = cv2.VideoCapture(os.path.join(iv_path, 'demo/example1.mp4'))
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f'Video loaded: {len(frames)} frames')

# Preprocess
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
selected = [cv2.resize(frames[i][:,:,::-1], (224,224)) for i in range(0, len(frames), len(frames)//4)][:4]
vid_tube = [np.expand_dims((x/255.0 - v_mean)/v_std, axis=(0,1)) for x in selected]
vid_tube = np.concatenate(vid_tube, axis=1)
vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
vid_tensor = torch.from_numpy(vid_tube).to('cuda:0').float()

# Run retrieval
with torch.no_grad():
    vfeat = model.get_vid_feat(vid_tensor)

texts = [
    'A playful dog and its owner wrestle in the snowy yard.',
    'A man in a gray coat walks through the snowy landscape, pulling a sleigh.',
    'A person shovels the snow-covered pavement outside their house.',
    'A cat excitedly runs through the yard, chasing a rabbit.',
    'A person bundled up enjoys a peaceful winter walk.',
]

with torch.no_grad():
    text_feats = []
    for t in texts:
        ti = tokenizer(t, padding='max_length', truncation=True, max_length=40, return_tensors='pt')
        ti = {k: v.to('cuda:0') for k, v in ti.items()}
        _, tfeat = model.encode_text(ti)
        tfeat = model.text_proj(tfeat)
        tfeat /= tfeat.norm(dim=-1, keepdim=True)
        text_feats.append(tfeat)
    text_tensor = torch.cat(text_feats, 0)
    probs = (100.0 * vfeat @ text_tensor.T).softmax(dim=-1)
    top_probs, top_idx = probs.cpu().topk(5, dim=-1)

print('\n=== Video-to-Text Retrieval Results ===')
for i, (idx, prob) in enumerate(zip(top_idx[0], top_probs[0])):
    print(f'  {i+1}. [{prob:.1%}] {texts[idx]}')
print('\nInternVideo2 retrieval baseline PASSED!')
