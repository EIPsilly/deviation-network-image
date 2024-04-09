import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM

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
        self.CVAE = build_feature_extractor(self.args)
        self.score_net = MLP(self.args, [1000, 100, 10, 1])

    def forward(self, image, domain_label):
        rec_img, mu, logvar, mu_prior, logvar_prior, z_feature, reconstruction_loss = self.CVAE(image, domain_label)
        score = self.score_net(z_feature)
            
        return rec_img, mu, logvar, mu_prior, logvar_prior, z_feature, score, reconstruction_loss