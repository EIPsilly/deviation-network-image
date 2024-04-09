import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM

class DGAD_net(nn.Module):
    def __init__(self, args):
        super(DGAD_net, self).__init__()
        self.args = args
        self.encoder, self.bn = build_feature_extractor(self.args)
        self.conv = nn.Conv2d(in_channels=NET_OUT_DIM[self.args.backbone], out_channels=1, kernel_size=1, padding=0)
    

    def forward(self, image):

        if self.args.n_scales == 0:
            raise ValueError

        image_pyramid = list()
        invariant_scores_list = []
        for s in range(self.args.n_scales):
            image_scaled = F.interpolate(image, size=self.args.img_size // (2 ** s)) if s > 0 else image
            
            features_intermediate = self.encoder(image_scaled)
            texture_feature, invariant_feature, origin_feature = self.bn(features_intermediate)

            scores = self.conv(origin_feature)
            if self.args.topk > 0:
                scores = scores.view(int(scores.size(0)), -1)
                topk = max(int(scores.size(1) * self.args.topk), 1)
                scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
                scores = torch.mean(scores, dim=1).view(-1, 1)
            else:
                scores = scores.view(int(scores.size(0)), -1)
                scores = torch.mean(scores, dim=1).view(-1, 1)

            # invariant_scores = self.conv(invariant_feature)
            # if self.args.topk > 0:
            #     invariant_scores = invariant_scores.view(int(invariant_scores.size(0)), -1)
            #     topk = max(int(invariant_scores.size(1) * self.args.topk), 1)
            #     invariant_scores = torch.topk(torch.abs(invariant_scores), topk, dim=1)[0]
            #     invariant_scores = torch.mean(invariant_scores, dim=1).view(-1, 1)
            # else:
            #     invariant_scores = invariant_scores.view(int(invariant_scores.size(0)), -1)
            #     invariant_scores = torch.mean(invariant_scores, dim=1).view(-1, 1)

            image_pyramid.append(scores)
            # invariant_scores_list.append(invariant_scores)
        scores = torch.cat(image_pyramid, dim=1)
        score = torch.mean(scores, dim=1)
        
        # invariant_scores = torch.cat(invariant_scores_list, dim=1)
        # invariant_score = torch.mean(invariant_scores, dim=1)
        # return score.view(-1, 1), invariant_score.view(-1, 1)
        
        return score.view(-1, 1), None