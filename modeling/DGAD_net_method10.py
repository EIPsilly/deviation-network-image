import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        # 控制温度
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        # student_output:[batchsize *2, dim], [原始表征的相似度+增强后的表征的相似度,dim]
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

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
        self.DistillLoss = DistillLoss(int(self.args.epochs * 0.25), self.args.epochs, warmup_teacher_temp=0.7, teacher_temp=0.4, student_temp = 1)
        self.domain_prototype = None
    
    def _forward_impl(self, image):
        # 原始样本计算
        Intermediate_feature, origin_feature = self.encoder(image)
        shallow_feature = self.shallow_conv(Intermediate_feature)

        origin_feature = self.avgpool(origin_feature)
        origin_feature = torch.flatten(origin_feature, 1)
        origin_feature = self.origin_fc(origin_feature)

        shallow_feature = self.avgpool(shallow_feature)
        shallow_feature = torch.flatten(shallow_feature, 1)
        shallow_feature = self.shallow_fc(shallow_feature)
        
        texture_feature = self.texture_fc(torch.cat([shallow_feature, shallow_feature - self.center], dim=1))
        # 计算样本到原型的距离，作为相似度
        similarity_matrix = torch.cat([-torch.sum((texture_feature - self.domain_prototype[i])**2, dim=1).reshape(-1, 1) for i in range(self.args.domain_cnt)], dim = 1)
        # category表示样本所属类型
        category = torch.argmax(F.softmax(similarity_matrix.detach(), dim = 1), dim=1)

        # 原始特征-原型特征
        class_feature = torch.cat([torch.cat([origin_feature[i], origin_feature[i] - self.domain_prototype[category[i]].detach()]).reshape(1, -1) for i in range(category.shape[0])],dim=0)
        class_feature = self.class_fc(class_feature)
        class_svdd_loss = torch.sum((class_feature - self.center) ** 2, dim=1)

        origin_svdd_loss = torch.sum((self.origin_class_fc(origin_feature) - self.center) ** 2, dim=1)
        align_loss = torch.abs(origin_svdd_loss - class_svdd_loss)
        
        return similarity_matrix, torch.cat([torch.mean(origin_svdd_loss).reshape(-1, 1), torch.mean(class_svdd_loss).reshape(-1, 1), torch.mean(align_loss).reshape(-1, 1)])

    def forward(self, image, augimg, epoch):
        similarity_matrix, loss = self._forward_impl(image)
        aug_similarity_matrix, aug_loss = self._forward_impl(augimg)
        student = torch.cat([similarity_matrix, aug_similarity_matrix])
        teacher = student.detach()
        domain_prototype_loss = self.DistillLoss(student, teacher, epoch)

        return torch.cat([domain_prototype_loss.reshape(-1,1), loss + aug_loss])
        

    def test(self, image):
        _, origin_feature = self.encoder(image)
        origin_feature = self.avgpool(origin_feature)
        origin_feature = torch.flatten(origin_feature, 1)
        origin_feature = self.origin_fc(origin_feature)

        origin_svdd_loss = torch.sum((self.origin_class_fc(origin_feature) - self.center) ** 2, dim=1)
        return origin_svdd_loss