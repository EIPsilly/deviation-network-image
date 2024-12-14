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
        self.encoder, self.bn, self.decoder, self.inference_encoder = build_feature_extractor(self.args)
        self.conv = nn.Conv2d(in_channels=2048, out_channels=32, kernel_size=1, padding=0)
        # self.score_net = MLP(args, [NET_OUT_DIM[self.args.backbone], 512, 64, 1])

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.origin_fc = nn.Linear(NET_OUT_DIM[self.args.backbone], 1000)
        # self.texture_fc = nn.Linear(NET_OUT_DIM[self.args.backbone], 1000)
        # self.origin_reg_fc = nn.Linear(NET_OUT_DIM[self.args.backbone], 1000)

        # self.domain_prototype = nn.utils.weight_norm(nn.Linear(1000, self.args.domain_cnt, bias=False))
        # self.domain_prototype.weight_g.data.fill_(1)
        # self.domain_prototype.weight_g.requires_grad = False
    
    def inference(self, image, normal_image, type_of_test='EFDM_test', lamda = 0.5):
        inputs = self.inference_encoder(image, normal_image, type_of_test=type_of_test, lamda=lamda)
        outputs = self.decoder(self.bn(inputs))

        return inputs, outputs

        
    def forward(self, normal, augmix_img):
        inputs_normal = self.encoder(normal) # [(256,64,64), (512,32,32), (1024,16,16)]
        bn_normal = self.bn(inputs_normal) # (2048,8,8)
        outputs_normal = self.decoder(bn_normal)  # [(256,64,64), (512,32,32), (1024,16,16)]


        inputs_augmix = self.encoder(augmix_img) # [(256,64,64), (512,32,32), (1024,16,16)]
        bn_augmix = self.bn(inputs_augmix) # (2048,8,8)
        outputs_augmix = self.decoder(bn_augmix) # [(256,64,64), (512,32,32), (1024,16,16)]
        
        return inputs_normal, bn_normal, outputs_normal, bn_augmix, outputs_augmix