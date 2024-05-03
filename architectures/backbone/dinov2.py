import timm
import torch.nn as nn

from huggingface_hub.hf_api import HfFolder
access_token = "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"
HfFolder.save_token(access_token)

class DINOv2(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True)
        self.outdim = self.model.embed_dim
        # self.outdim = 224
        # breakpoint()
        # self.embed_dim = self.model.parameters()['embed_dim']

    def forward(self, x):
        return self.model(x)

def create_model(name="vit_base_patch14_dinov2.lvd142m"):
    return DINOv2(name)