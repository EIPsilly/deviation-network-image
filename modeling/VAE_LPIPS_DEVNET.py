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

# class PriorNetwork(nn.Module):
#     def __init__(self, dff=1024):
#         super(PriorNetwork, self).__init__()
        
#         # 定义隐藏层
#         self.hidden_layer = nn.Sequential(
#             nn.Linear(dff, dff * 8),
#             nn.BatchNorm1d(dff * 8),
#             nn.LeakyReLU(),
#             nn.Linear(dff * 8, dff * 8 * 8),
#             nn.BatchNorm1d(dff * 8 * 8),
#             nn.LeakyReLU(),
#         )
        
#           # ReLU 在 forward 里加
#         self.hidden_layer_mu = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
#         self.hidden_layer_logvar = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        
#         # 输出层
#         self.output_layer_mu = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
#         self.output_layer_logvar = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        
#         # 激活函数
#         self.relu = nn.LeakyReLU()
#         self.tanh = nn.Tanh()

#     def forward(self, inp):
#         # 前向传播
#         h = self.relu(self.hidden_layer(inp))
#         h = h.view(-1, 1024, 8, 8)

#         h_mu = self.relu(self.hidden_layer_mu(h))
#         mu = self.tanh(self.output_layer_mu(h_mu))
        
#         h_logvar = self.relu(self.hidden_layer_logvar(h))
#         logvar = self.tanh(self.output_layer_logvar(h_logvar))
        
#         # 重参数化技巧
#         std = torch.exp(0.5 * logvar)  # 计算标准差
#         eps = torch.randn_like(std)    # 随机噪声
#         z = mu + eps * std             # 生成 z
        
#         return z, mu, logvar

class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        self.VAE = build_feature_extractor(self.args)
        logvar_init = 0.0
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.perceptual_weight = 1
        self.kl_weight = 0.000001
        self.perceptual_loss = LPIPS().eval()
        self.class_conv = nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, padding=0)
        self.domain_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0)
        
        self.domain_classifier = nn.utils.weight_norm(nn.Linear(64, self.args.domain_cnt, bias=False))
        self.domain_classifier.weight_g.data.fill_(1)
        self.domain_classifier.weight_g.requires_grad = False

        # self.prior_embeddings = nn.Embedding(2, 1024)
        # self.prior_net = PriorNetwork(1024)


    def calc_anomaly_score(self, scores):
        if self.args.topk > 0:
            scores = scores.view(int(scores.size(0)), -1)
            topk = max(int(scores.size(1) * self.args.topk), 1)
            scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
            scores = torch.mean(scores, dim=1).view(-1, 1)
        else:
            scores = scores.view(int(scores.size(0)), -1)
            scores = torch.mean(scores, dim=1).view(-1, 1)
        
        score = torch.mean(scores, dim=1)

        return score

    def domain_classification(self, feature):
        feature = torch.flatten(feature, 1)
        return self.domain_classifier(F.normalize(feature))


    def forward(self, image, domain_label, target, weights=None):
        idx = torch.where(target == 0)[0]

        reconstructions, posterior = self.VAE(image)
        rec_loss = torch.abs(image.contiguous() - reconstructions.contiguous())
        p_loss = self.perceptual_loss(image.contiguous(), reconstructions.contiguous())
        rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = weighted_nll_loss[idx]
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        kl_loss = posterior.kl()
        kl_loss = kl_loss[idx]
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] * self.kl_weight
        # logvar = posterior.logvar
        # mu = posterior.mean

        # prior_embeddings = self.prior_embeddings(target)
        # _, mu_p, logvar_p = self.prior_net(prior_embeddings)

        # kl_loss = 0.5 * (logvar_p - logvar - 1
        #                  + torch.exp(logvar - logvar_p)
        #                  + (mu_p - mu) ** 2 / torch.exp(logvar_p))
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        
        class_feature, _ = posterior.class_mode()
        # domain_feature, _ = posterior.domain_mode()
        class_feature = self.class_conv(class_feature)
        # domain_feature = self.domain_conv(domain_feature)
        class_score = self.calc_anomaly_score(class_feature)
        # domain_score = self.calc_anomaly_score(domain_feature)
        
        # class_classification = self.domain_classification(class_feature)
        # domain_classification = self.domain_classification(domain_feature)

        domain_score = None
        class_classification = None
        domain_classification = None

        return reconstructions, weighted_nll_loss, kl_loss, class_score, domain_score, class_classification, domain_classification