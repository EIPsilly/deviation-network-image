import sys
sys.path.append("/home/hzw/DGAD/deviation-network-image/modeling/networks")
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from .res_encoder import ResNet as resnet_encoder
from .res_decoder import ResNet as resnet_decoder
from .res_encoder import Bottleneck as Bottleneck_encoder
from .res_decoder import Bottleneck as Bottleneck_decoder

class PriorNetwork(nn.Module):
    def __init__(self, dff, d_latent):
        super(PriorNetwork, self).__init__()
        self.hidden_layer = nn.Linear(dff, dff // 2)
        self.hidden_layer_mu = nn.Linear(dff // 2, dff // 4)
        self.hidden_layer_logvar = nn.Linear(dff // 2, dff // 4)
        self.output_layer_mu = nn.Linear(dff // 4, d_latent * 2)
        self.output_layer_logvar = nn.Linear(dff // 4, d_latent * 2)
    
    def forward(self, input):
        h = F.leaky_relu(self.hidden_layer(input))

        h_mu = F.leaky_relu(self.hidden_layer_mu(h))
        mu = torch.tanh(self.output_layer_mu(h_mu))

        h_logvar = F.leaky_relu(self.hidden_layer_logvar(h))
        logvar = torch.tanh(self.output_layer_logvar(h_logvar))

        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        return z, mu, logvar

class RecognitionNetwork(nn.Module):
    def __init__(self, dff, d_latent):
        super(RecognitionNetwork, self).__init__()

        self.hidden_layer = nn.Linear(dff, dff // 2)
        self.hidden_layer_mu = nn.Linear(dff // 2, dff // 4)
        self.hidden_layer_logvar = nn.Linear(dff // 2, dff // 4)
        self.output_layer_mu = nn.Linear(dff // 4, d_latent * 2)
        self.output_layer_logvar = nn.Linear(dff // 4, d_latent * 2)

    def forward(self, cond, inp):
        x = torch.cat([cond, inp], dim=-1)
        h = F.leaky_relu(self.hidden_layer(x))
        h_mu = F.leaky_relu(self.hidden_layer_mu(h))
        mu = torch.tanh(self.output_layer_mu(h_mu))
        h_logvar = F.leaky_relu(self.hidden_layer_logvar(h))
        logvar = torch.tanh(self.output_layer_logvar(h_logvar))
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        return z, mu, logvar

class BeforeReconstructionNetwork(nn.Module):
    def __init__(self, dff, d_latent):
        super(BeforeReconstructionNetwork, self).__init__()

        self.fc1 = nn.Linear(d_latent, d_latent * 2)
        self.fc2 = nn.Linear(d_latent * 2, d_latent * 4)
        self.fc3 = nn.Linear(d_latent * 4, dff)
        self.hidden = nn.ModuleList([self.fc1, self.fc2,self.fc3])
        
    def forward(self, cond, inp):
        x = torch.cat([cond, inp], dim=-1)
        for layer in self.hidden:
            x = F.leaky_relu(layer(x))
        return x

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class CVAE(nn.Module):
    def __init__(self, args):
        super(CVAE, self).__init__()
        self.args = args
        self.d_latent = 64

        self.encoder = resnet_encoder(Bottleneck_encoder, [3, 4, 6, 3], return_indices=True, width_per_group = 64 * 2)
        state_dict = load_state_dict_from_url(model_urls["wide_resnet50_2"])
        # for k,v in list(state_dict.items()):
        #    if 'layer4' in k or 'fc' in k:
        #        state_dict.pop(k)
        self.encoder.load_state_dict(state_dict)


        self.decoder = resnet_decoder(Bottleneck_decoder, [3, 6, 4, 3], width_per_group = 64 * 2)
        
        self.condition_embedding = nn.Embedding(self.args.domain_cnt, 300)
        
        self.prior_net = PriorNetwork(300, self.d_latent)
        self.recog_net = RecognitionNetwork(1000 + 300, self.d_latent)
        self.before_reconstruction = BeforeReconstructionNetwork(1000, self.d_latent + 300)

    def forward(self, input, condition):
        encoder_list, feature, indices = self.encoder(input)
        cond_embed = self.condition_embedding(condition)

        _, mu_prior, logvar_prior = self.prior_net(cond_embed)
        z, mu, logvar = self.recog_net(cond_embed, feature)
        
        z_feature = self.before_reconstruction(cond_embed, z[:,:self.d_latent])
        decoder_list, rec_img = self.decoder(z_feature, indices)
        
        reconstruction_loss = loss_fucntion(encoder_list, decoder_list)

        return rec_img, mu, logvar, mu_prior, logvar_prior, z_feature, reconstruction_loss

    # def loss_function(self, recon, x, mu, log_std) -> torch.Tensor:
    #     kl_div = 0.5 * torch.sum(logvar_p - logvar_r - 1
    #                      + torch.exp(logvar_r - logvar_p)
    #                      + (mu_p - mu_r) ** 2 / torch.exp(logvar_p), dim=-1)

    #     kl_div = torch.max(5.0, kl_div)