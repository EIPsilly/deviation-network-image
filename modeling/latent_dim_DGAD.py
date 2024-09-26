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

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 4, stride=2, padding=1),  # Output: 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # Output: 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), # Output: 32x8x8
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # Output: 32x8x8
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), # Output: 32x4x4
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), # Output: 32x4x4
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32 * 4 * 4, 128)
        self.fc_logvar = nn.Linear(32 * 4 * 4, 128)
        self.fc_decode = nn.Linear(128, 32 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), # Output: 32x4x4
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), # Output: 32x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), # Output: 32x8x8
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # Output: 16*16*16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1), # Output: 16x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, 4, stride=2, padding=1),   # Output: 4x32x32
            nn.Sigmoid()  # For output normalization
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_decode(z).view(-1, 32, 4, 4)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        
        resume_path = "./models/epoch=5-step=119.ckpt"
        resume_path = "./models/epoch=157-step=3159.ckpt"

        # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        self.AE = create_model('./models/autoencoder_kl_32x32x4.yaml').cpu()
        self.AE.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
        self.AE = self.AE.cuda()
        self.VAE = VAE(args)
        self.domain_classifier = nn.utils.weight_norm(nn.Linear(64, self.args.domain_cnt, bias=False))
        self.domain_classifier.weight_g.data.fill_(1)
        self.domain_classifier.weight_g.requires_grad = False
        self.anomaly_score_net = MLP(self.args, [128, 32, 1])
        self.domain_classification_net = MLP(self.args, [128, 32, 3])


    def calc_anomaly(self, x):
        return self.anomaly_score_net(x)

    def domain_classification(self, x):
        return self.domain_classification_net(x)

    def forward(self, image, domain_label, target, weights=None):
        with torch.no_grad():
            posterior = self.AE.encode(image)
            z = posterior.sample().detach() #4*32*32
        recon, mu, logvar = self.VAE(z)
        
        return z, recon, mu, logvar