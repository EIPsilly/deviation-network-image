import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM

class MLP(nn.Module):
    def __init__(self, args, dims):
        super(MLP, self).__init__()
        self.args = args
        # if self.args.backbone == "mlp4":
        #     dims = [NET_OUT_DIM[self.args.backbone], 1000, 256, 64]
        # elif self.args.backbone == "mlp2":
        #     dims = [NET_OUT_DIM[self.args.backbone], 64]
        # elif self.args.backbone == "mlp1":
        #     dims = [NET_OUT_DIM[self.args.backbone]]
        
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
        self.encoder, self.shallow_conv = build_feature_extractor(self.args.backbone)
        self.conv = nn.Conv2d(in_channels=NET_OUT_DIM[self.args.backbone], out_channels=1, kernel_size=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.origin_fc = MLP(args, [NET_OUT_DIM[self.args.backbone], 1024, 512, 64])
        self.shallow_fc = MLP(args, [NET_OUT_DIM[self.args.backbone], 1024, 512, 64])
        self.texture_fc = MLP(args, [128, 64, 64])
        self.class_fc = MLP(args, [128, 64, 64])
        self.origin_class_fc = MLP(args, [64, 64, 64])

    def forward(self, image):
        Intermediate_feature, origin_feature = self.encoder(image)
        shallow_feature = self.shallow_conv(Intermediate_feature)

        origin_feature = self.avgpool(origin_feature)
        origin_feature = torch.flatten(origin_feature, 1)
        origin_feature = self.origin_fc(origin_feature)

        shallow_feature = self.avgpool(shallow_feature)
        shallow_feature = torch.flatten(shallow_feature, 1)
        shallow_feature = self.shallow_fc(shallow_feature)
        
        texture_feature = self.texture_fc(torch.cat([shallow_feature, shallow_feature - self.center], dim=1))
        similarity_matrix = torch.cat([torch.sum((texture_feature - self.domain_prototype[i])**2, dim=1).reshape(-1, 1) for i in range(self.args.domain_cnt)], dim = 1)
        
        category = torch.argmax(F.softmax(similarity_matrix, dim = 1), dim=1)
        domain_prototype_loss = nn.CrossEntropyLoss()(similarity_matrix, category)
        
        class_feature = torch.cat([torch.cat([origin_feature[i], origin_feature[i] - self.domain_prototype[category[i]]]).reshape(1, -1) for i in range(category.shape[0])],dim=0)
        class_feature = self.class_fc(class_feature)
        class_svdd_loss = torch.sum((class_feature - self.center) ** 2, dim=1)

        origin_svdd_loss = torch.sum((self.origin_class_fc(origin_feature) - self.center) ** 2, dim=1)
        align_loss = torch.abs(origin_svdd_loss - class_svdd_loss)
        
        return torch.cat([domain_prototype_loss.reshape(-1, 1), torch.mean(origin_svdd_loss).reshape(-1, 1), torch.mean(class_svdd_loss).reshape(-1, 1), torch.mean(align_loss).reshape(-1, 1)])

    def test(self, image):
        _, origin_feature = self.encoder(image)
        origin_feature = self.avgpool(origin_feature)
        origin_feature = torch.flatten(origin_feature, 1)
        origin_feature = self.origin_fc(origin_feature)

        origin_svdd_loss = torch.sum((self.origin_class_fc(origin_feature) - self.center) ** 2, dim=1)
        return origin_svdd_loss