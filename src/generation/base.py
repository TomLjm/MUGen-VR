import torch
import torch.nn as nn
from ..common.interfaces import GenerationResult

class BaseVideoGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def generate(self, conditions, **kwargs):
        raise NotImplementedError
    def generate_from_text(self, text, **kwargs):
        raise NotImplementedError
    def generate_from_image(self, image, **kwargs):
        raise NotImplementedError
