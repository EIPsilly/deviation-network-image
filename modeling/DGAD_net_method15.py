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

class DGAD_net(nn.Module):
    def __init__(self, args):
        super(DGAD_net, self).__init__()
        self.args = args
        self.encoder, self.shallow_conv = build_feature_extractor(self.args)
        # self.conv = nn.Conv2d(in_channels=NET_OUT_DIM[self.args.backbone], out_channels=32, kernel_size=1, padding=0)
        self.score_net = MLP(args, [NET_OUT_DIM[self.args.backbone], 512, 64, 1])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.origin_fc = nn.Linear(NET_OUT_DIM[self.args.backbone], 1000)
        self.texture_fc = nn.Linear(NET_OUT_DIM[self.args.backbone], 1000)
        self.origin_reg_fc = nn.Linear(NET_OUT_DIM[self.args.backbone], 1000)

        self.domain_prototype = nn.utils.weight_norm(nn.Linear(1000, self.args.domain_cnt, bias=False))
        self.domain_prototype.weight_g.data.fill_(1)
        self.domain_prototype.weight_g.requires_grad = False
    

    def forward(self, image):
        if self.args.n_scales == 0:
            raise ValueError
        
        Intermediate_feature, origin_feature = self.encoder(image)
        texture_feature = self.shallow_conv(Intermediate_feature)
        
        scores = self.avgpool(origin_feature)
        scores = torch.flatten(scores, 1)
        scores = self.score_net(scores)
        
        texture_scores = self.avgpool(texture_feature)
        texture_scores = torch.flatten(texture_scores, 1)
        texture_scores = self.score_net(texture_scores)
        
        origin_feature = self.avgpool(origin_feature)
        origin_feature = torch.flatten(origin_feature, 1)
        origin_reg_feature = self.origin_reg_fc(origin_feature)
        origin_feature = self.origin_fc(origin_feature)

        texture_feature = self.avgpool(texture_feature)
        texture_feature = torch.flatten(texture_feature, 1)
        texture_feature = self.texture_fc(texture_feature)
        
        return scores, texture_scores, origin_feature, texture_feature, origin_reg_feature