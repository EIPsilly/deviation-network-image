import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM
from taming.modules.losses.vqperceptual import *

from omegaconf import OmegaConf
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        config_path = 'autoencoder.yaml'
        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config.model)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        self.class_conv = nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, padding=0)
        self.domain_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0)
        
        self.domain_classifier = nn.utils.weight_norm(nn.Linear(64, self.args.domain_cnt, bias=False))
        self.domain_classifier.weight_g.data.fill_(1)
        self.domain_classifier.weight_g.requires_grad = False


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


    def forward(self, image, domain_label, target):
        # with torch.no_grad():
        #     encoder_posterior = self.first_stage_model.encode(image)
        #     z = encoder_posterior.sample()

        dec, posterior = self.first_stage_model(image)

        # idx = torch.where(target == 0)[0]

        # reconstructions, posterior = self.VAE(image)
        # rec_loss = torch.abs(image.contiguous() - reconstructions.contiguous())
        # p_loss = self.perceptual_loss(image.contiguous(), reconstructions.contiguous())
        # rec_loss = rec_loss + self.perceptual_weight * p_loss

        # nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # weighted_nll_loss = nll_loss
        # if weights is not None:
        #     weighted_nll_loss = weights*nll_loss
        # weighted_nll_loss = weighted_nll_loss[idx]
        # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        # kl_loss = posterior.kl()
        # kl_loss = kl_loss[idx]
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] * self.kl_weight
        
        # class_feature, _ = posterior.class_mode()
        # domain_feature, _ = posterior.domain_mode()
        # class_feature = self.class_conv(class_feature)
        # domain_feature = self.domain_conv(domain_feature)
        # class_score = self.calc_anomaly_score(class_feature)
        # domain_score = self.calc_anomaly_score(domain_feature)
        
        # class_classification = self.domain_classification(class_feature)
        # domain_classification = self.domain_classification(domain_feature)

        return dec