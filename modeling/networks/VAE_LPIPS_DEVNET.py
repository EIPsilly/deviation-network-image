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
import numpy as np

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

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, partition = 4):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)
        mean_list = torch.chunk(self.mean, partition, dim=1)
        std_list = torch.chunk(self.mean, partition, dim=1)
        self.class_mean = torch.cat(mean_list[:-1], dim=1)
        self.class_std = torch.cat(std_list[:-1], dim=1)
        self.domain_mean = mean_list[-1]
        self.domain_std = std_list[-1]

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x
    
    def sample_class_feature(self):
        x = self.class_mean + self.class_std * torch.randn(self.class_mean .shape).to(device=self.parameters.device)
        return x

    def sample_domain_feature(self):
        x = self.domain_mean + self.domain_std * torch.randn(self.domain_mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def class_mode(self):
        return self.class_mean, self.class_std

    def domain_mode(self):
        return self.domain_mean, self.domain_std

    def mode(self):
        return self.mean, self.std

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.d_latent = 64
        self.partition = 4

        self.encoder = resnet_encoder(Bottleneck_encoder, [3, 4, 6, 3], return_indices=True, width_per_group = 64 * 2)
        self.encoder.load_state_dict(load_state_dict_from_url(model_urls["wide_resnet50_2"]))
        self.decoder = resnet_decoder(Bottleneck_decoder, [3, 6, 4, 3], width_per_group = 64 * 2)
        self.quant_conv = nn.Conv2d(2048, 2048, 1)
        self.post_quant_conv = nn.Conv2d(768, 2048, 1)

    def encode(self, x):
        h, indices = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments, partition=self.partition)
        return posterior, indices

    def decode(self, z, indices):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, indices)
        return dec
    
    def forward(self, input, sample_posterior=True):
        posterior, indices = self.encode(input)
        if sample_posterior:
            z = posterior.sample_class_feature()
        else:
            z, _ = posterior.mode()
        dec = self.decode(z, indices)
        return dec, posterior