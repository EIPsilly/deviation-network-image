import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM
from taming.modules.losses.vqperceptual import *

# class MLP(nn.Module):
#     def __init__(self, args, dims):
#         super(MLP, self).__init__()
#         self.args = args
#         layers = [nn.Linear(dims[i - 1], dims[i], bias=False) for i in range(1, len(dims))]
#         self.hidden = nn.ModuleList(layers)

#     def forward(self, x):
#         for layer in self.hidden:
#             x = F.leaky_relu(layer(x))
#         return x

class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        self.VAE = build_feature_extractor(self.args)
        logvar_init = 0.0
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.kl_weight = 0.000001
        self.perceptual_weight = 1
        self.perceptual_loss = LPIPS().eval()

    def forward(self, image, domain_label, weights=None):
        reconstructions, posterior = self.VAE(image)
        rec_loss = torch.abs(image.contiguous() - reconstructions.contiguous())
        p_loss = self.perceptual_loss(image.contiguous(), reconstructions.contiguous())
        rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = weighted_nll_loss + self.kl_weight * kl_loss

            
        return reconstructions, loss, weighted_nll_loss, self.kl_weight * kl_loss