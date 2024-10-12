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
            nn.Conv2d(4, 32, 4, stride=2, padding=1),  # Output: 32x16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), # Output: 32x8x8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # Output: 64x4x4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 256)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 256)
        )
        self.fc_decode = nn.Linear(256, 64 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # Output: 32x8x8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), # Output: 32x16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1),   # Output: 4x32x32
            nn.BatchNorm2d(4),
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
        z = self.fc_decode(z).view(-1, 64, 4, 4)
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
        resume_path = "./models/PACS_epoch=999-step=19999.ckpt"

        # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        self.AE = create_model('./models/autoencoder_kl_32x32x4.yaml').cpu()
        self.AE.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
        self.AE = self.AE.cuda()
        self.VAE = VAE(args)
        self.perceptual_loss = LPIPS().eval()
        logvar_init = 0.0
        self.perceptual_weight = 1.0
        self.kl_weight = 0.000001
        # self.kl_weight = 1.0
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        

    def forward(self, image, domain_label, target, weights=None):
        with torch.no_grad():
            posterior = self.AE.encode(image)
            z = posterior.sample().detach() #4*32*32
        recon, mu, logvar = self.VAE(z)
        
        with torch.no_grad():
            reconstructions = self.AE.decode(recon)
        
        rec_loss = torch.abs(image.contiguous() - reconstructions.contiguous())
        p_loss = self.perceptual_loss(image.contiguous(), reconstructions.contiguous())
        rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] * self.kl_weight
        
        return z, reconstructions, mu, logvar, weighted_nll_loss, kl_loss