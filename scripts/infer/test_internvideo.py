"""
Minimal InternVideo2 retrieval test using HuggingFace integration.
Phase 1: verify the model can load and run retrieval.
"""
import os
import sys
import cv2
import torch
import numpy as np

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')

# Test 1: Check if transformers can load InternVideo2
print('\n=== Test 1: Loading InternVideo2 via HuggingFace ===')
try:
    from transformers import AutoModel, AutoImageProcessor
    print('Loading InternVideo2-Stage2_1B-224p-f4 model...')
    model = AutoModel.from_pretrained(
        'OpenGVLab/InternVideo2-Stage2_1B-224p-f4',
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f'Model loaded: {type(model).__name__}')
    print(f'Model device: {next(model.parameters()).device}')
    
except Exception as e:
    print(f'HuggingFace direct loading failed: {e}')
    print('Trying to build from local repo source...')
    
    # Try local import approach
    sys.path.insert(0, '/data14/jiaming.lin.2601/MUGen-VR/third_party/InternVideo/InternVideo2/multi_modality')
    try:
        from demo.utils import setup_internvideo2, InternVideo2_Stage2, frames2tensor, retrieve_text, _frame_from_video
        print('Local import successful, but needs config file.')
    except Exception as e2:
        print(f'Local import also failed: {e2}')

print('\n=== Test done ===')
