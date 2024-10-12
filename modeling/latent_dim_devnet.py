import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM
from taming.modules.losses.vqperceptual import *
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import os

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


class MLP(nn.Module):
    def __init__(self, args, dims):
        super(MLP, self).__init__()
        self.args = args
        layers = [nn.Linear(dims[i - 1], dims[i], bias=False) for i in range(1, len(dims))]
        self.hidden = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.hidden:
            x = F.leaky_relu(layer(x))
        return x

class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        
        resume_path = "./models/epoch=5-step=119.ckpt"
        resume_path = "./models/epoch=157-step=3159.ckpt"
        resume_path = "./models/PACS_epoch=999-step=19999.ckpt"

        # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        self.AE = create_model('./models/autoencoder_kl_32x32x4.yaml').cpu()
        self.AE.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
        self.AE = self.AE.cuda()
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, 4, stride=2, padding=1),  # Output: 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # Output: 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, stride=2, padding=1), # Output: 4x8x8
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1, padding=1),  # Output: 4x8x8
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, stride=1, padding=1), # Output: 1x8x8
            nn.ReLU()
        )


    def forward(self, image):

        if self.args.n_scales == 0:
            raise ValueError

        with torch.no_grad():
            posterior = self.AE.encode(image)
            z = posterior.sample().detach() #4*32*32

        scores = self.conv(z)
        if self.args.topk > 0:
            scores = scores.view(int(scores.size(0)), -1)
            topk = max(int(scores.size(1) * self.args.topk), 1)
            scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
            scores = torch.mean(scores, dim=1).view(-1, 1)
        else:
            scores = scores.view(int(scores.size(0)), -1)
            scores = torch.mean(scores, dim=1).view(-1, 1)
        score = torch.mean(scores, dim=1)
        return score.view(-1, 1)