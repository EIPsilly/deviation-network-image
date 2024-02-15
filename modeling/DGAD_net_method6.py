import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM

class DGAD_net(nn.Module):
    def __init__(self, args):
        super(DGAD_net, self).__init__()
        self.args = args
        self.encoder, self.bn = build_feature_extractor(self.args.backbone)
        self.conv = nn.Conv2d(in_channels=NET_OUT_DIM[self.args.backbone], out_channels=1, kernel_size=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.origin_fc = nn.Linear(NET_OUT_DIM[self.args.backbone], 1000)
        self.texture_fc = nn.Linear(NET_OUT_DIM[self.args.backbone], 1000)

        self.domain_prototype = nn.utils.weight_norm(nn.Linear(1000, self.args.domain_cnt, bias=False))
        self.domain_prototype.weight_g.data.fill_(1)
        self.domain_prototype.weight_g.requires_grad = False
    

    def forward(self, image):

        if self.args.n_scales == 0:
            raise ValueError

        image_pyramid = list()
        texture_scores_list = []
        for s in range(self.args.n_scales):
            image_scaled = F.interpolate(image, size=self.args.img_size // (2 ** s)) if s > 0 else image
            
            Intermediate_feature, origin_feature = self.encoder(image_scaled)
            texture_feature = self.bn(Intermediate_feature)

            scores = self.conv(origin_feature)
            if self.args.topk > 0:
                scores = scores.view(int(scores.size(0)), -1)
                topk = max(int(scores.size(1) * self.args.topk), 1)
                scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
                scores = torch.mean(scores, dim=1).view(-1, 1)
            else:
                scores = scores.view(int(scores.size(0)), -1)
                scores = torch.mean(scores, dim=1).view(-1, 1)

            texture_scores = self.conv(texture_feature)
            if self.args.topk > 0:
                texture_scores = texture_scores.view(int(texture_scores.size(0)), -1)
                topk = max(int(texture_scores.size(1) * self.args.topk), 1)
                texture_scores = torch.topk(torch.abs(texture_scores), topk, dim=1)[0]
                texture_scores = torch.mean(texture_scores, dim=1).view(-1, 1)
            else:
                texture_scores = texture_scores.view(int(texture_scores.size(0)), -1)
                texture_scores = torch.mean(texture_scores, dim=1).view(-1, 1)

            image_pyramid.append(scores)
            texture_scores_list.append(texture_scores)
        scores = torch.cat(image_pyramid, dim=1)
        score = torch.mean(scores, dim=1)
        
        texture_scores = torch.cat(texture_scores_list, dim=1)
        texture_score = torch.mean(texture_scores, dim=1)
        return score.view(-1, 1), texture_score.view(-1, 1)
        
        # return score.view(-1, 1), None

    def CL(self, image):
        Intermediate_feature, origin_feature = self.encoder(image)
        texture_feature = self.bn(Intermediate_feature)

        origin_feature = self.avgpool(origin_feature)
        origin_feature = torch.flatten(origin_feature, 1)
        origin_feature = self.origin_fc(origin_feature)

        texture_feature = self.avgpool(texture_feature)
        texture_feature = torch.flatten(texture_feature, 1)
        texture_feature = self.texture_fc(texture_feature)
        return origin_feature, texture_feature