import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM

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

    def forward(self, image, domain_label):
        reconstructions, posterior = self.VAE(image)
        rec_loss = torch.abs(image.contiguous() - reconstructions.contiguous())
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            
        return reconstructions, rec_loss, kl_loss