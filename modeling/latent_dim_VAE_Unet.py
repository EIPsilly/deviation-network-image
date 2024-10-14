import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM
from taming.modules.losses.vqperceptual import *
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import os

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


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

class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,
                              kernel_size = kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class deconv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(deconv2d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
    
class Unet(nn.Module):
    def __init__(self, args):
        super(Unet,self).__init__()
        self.layer1_conv = double_conv2d_bn(4,8)
        self.layer2_conv = double_conv2d_bn(8,16)
        self.layer3_conv = double_conv2d_bn(16,32)
        self.layer4_conv = double_conv2d_bn(32,64)
        self.layer5_conv = double_conv2d_bn(64,128)
        self.layer6_conv = double_conv2d_bn(128,64)
        self.layer7_conv = double_conv2d_bn(64,32)
        self.layer8_conv = double_conv2d_bn(32,16)
        self.layer9_conv = double_conv2d_bn(16,8)
        self.layer10_conv = nn.Conv2d(8, 4,kernel_size=3,stride=1,padding=1,bias=True)
        
        self.deconv1 = deconv2d_bn(128,64)
        self.deconv2 = deconv2d_bn(64,32)
        self.deconv3 = deconv2d_bn(32,16)
        self.deconv4 = deconv2d_bn(16,8)
        
        self.fc_mu = nn.Sequential(
            nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
            # nn.Linear(128, 128)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
            # nn.Linear(128, 128)
        )
        self.fc_decode = nn.Linear(128, 512)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1,2)
        
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2,2)
        
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3,2)
        
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4,2)
        
        conv5 = self.layer5_conv(pool4)

        conv5 = conv5.view(conv5.size(0), -1)

        mu = self.fc_mu(conv5)
        logvar = self.fc_logvar(conv5)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        conv5 = self.fc_decode(z).view(-1, 128, 2, 2)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1,conv4],dim=1)
        conv6 = self.layer6_conv(concat1)
        
        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2,conv3],dim=1)
        conv7 = self.layer7_conv(concat2)
        
        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3,conv2],dim=1)
        conv8 = self.layer8_conv(concat3)
        
        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4,conv1],dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)
        return outp, mu, logvar


class PriorNetwork(nn.Module):
    def __init__(self, dff=256, d_latent=256):
        super(PriorNetwork, self).__init__()
        
        # 定义隐藏层
        self.hidden_layer = nn.Linear(dff, dff)  # ReLU 在 forward 里加
        self.hidden_layer_mu = nn.Linear(dff, dff)
        self.hidden_layer_logvar = nn.Linear(dff, dff)
        
        # 输出层
        self.output_layer_mu = nn.Linear(dff, d_latent)
        self.output_layer_logvar = nn.Linear(dff, d_latent)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, inp):
        # 前向传播
        h = self.relu(self.hidden_layer(inp))
        
        h_mu = self.relu(self.hidden_layer_mu(h))
        mu = self.tanh(self.output_layer_mu(h_mu))
        
        h_logvar = self.relu(self.hidden_layer_logvar(h))
        logvar = self.tanh(self.output_layer_logvar(h_logvar))
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)    # 随机噪声
        z = mu + eps * std             # 生成 z
        
        return z, mu, logvar


class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        
        resume_path = "./models/epoch=5-step=119.ckpt"
        resume_path = "./models/epoch=157-step=3159.ckpt"
        resume_path = "./models/PACS_epoch=999-step=19999.ckpt"

        # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        self.AE = create_model('./models/autoencoder_kl_32x32x4.yaml').cpu()
        self.AE.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
        self.Unet = Unet(args)
        self.perceptual_loss = LPIPS().eval()
        logvar_init = 0.0
        self.perceptual_weight = 1.0
        # self.kl_weight = 0.000001
        # self.kl_weight = 1.0
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        # self.domain_classifier = nn.utils.weight_norm(nn.Linear(64, self.args.domain_cnt, bias=False))
        # self.domain_classifier.weight_g.data.fill_(1)
        # self.domain_classifier.weight_g.requires_grad = False
        self.anomaly_score_net = nn.Sequential(
            nn.Conv2d(4, 16, 4, stride=2, padding=1),  # Output: 16x16x16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), # Output: 16x8x8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1, padding=0)
        )
        self.domain_classification_net = MLP(self.args, [128, 32, 3])
        self.prior_embeddings = nn.Embedding((args.domain_cnt + 1) * 2, 256)
        self.prior_net = PriorNetwork(256, 128)


    def calc_anomaly(self, x):
        scores = self.anomaly_score_net(x)
        if self.args.topk > 0:
            scores = scores.view(int(scores.size(0)), -1)
            topk = max(int(scores.size(1) * self.args.topk), 1)
            scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
            scores = torch.mean(scores, dim=1).view(-1, 1)
        else:
            scores = scores.view(int(scores.size(0)), -1)
            scores = torch.mean(scores, dim=1).view(-1, 1)
        return scores

    def domain_classification(self, x):
        return self.domain_classification_net(x)

    def forward(self, image, domain_label, target, weights=None):
        with torch.no_grad():
            posterior = self.AE.encode(image)
            z = posterior.sample().detach() #4*32*32
        recon, mu, logvar = self.Unet(z)
        # 随机选择几个下标，这里假设随机选择5个下标
        num_indices = 5  
        indices = torch.randperm(image.shape[0])[:num_indices]  # 随机选取5个索引
        domain_label[indices] = 3
        prior_embeddings = self.prior_embeddings(target * self.args.domain_cnt + domain_label)
        _, mu_p, logvar_p = self.prior_net(prior_embeddings)

        with torch.no_grad():
            reconstructions = self.AE.decode(recon)
        
        rec_loss = torch.abs(image.contiguous() - reconstructions.contiguous())
        p_loss = self.perceptual_loss(image.contiguous(), reconstructions.contiguous())
        rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        # kl_loss = posterior.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] * self.kl_weight
        
        return z, recon, mu, logvar, mu_p, logvar_p, weighted_nll_loss