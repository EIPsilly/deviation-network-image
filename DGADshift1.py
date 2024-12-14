import os
import random
import time
from line_profiler import LineProfiler
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from PIL import Image
import argparse
from dataloaders.dataloader import build_dataloader
from modeling.DGADshift_1 import DGAD_net
from tqdm import tqdm
from utils import aucPerformance
from modeling.layers import build_criterion
import yaml
import torchvision.transforms as T

with open("../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss

def loss_fucntion_last(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    item = 0
    loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                    b[item].view(b[item].shape[0], -1)))
    return loss

def gaussian_kernel(x, y, kernel_bandwidth=1.0):
    """Compute the Gaussian kernel between two sets of data points."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)

    kernel_input = ((tiled_x - tiled_y) ** 2).sum(2) / (2.0 * kernel_bandwidth ** 2.0)
    return torch.exp(-kernel_input)

def mmd_loss(x, y, kernel_bandwidth=1.0):
    """Compute the Maximum Mean Discrepancy (MMD) loss between two distributions."""
    xx_kernel = gaussian_kernel(x, x, kernel_bandwidth)
    yy_kernel = gaussian_kernel(y, y, kernel_bandwidth)
    xy_kernel = gaussian_kernel(x, y, kernel_bandwidth)
    # print(xx_kernel)

    mmd = xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()
    # mmd = 2 * xy_kernel.mean()
    return mmd

class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.val_loader, self.test_loader, self.unlabeled_loader = build_dataloader(args, **kwargs)
        
        self.model = DGAD_net(args)
        
        # self.criterion = build_criterion(args.criterion, args)
        # self.uniform_criterion = build_criterion("uniform", args)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.pre_lr, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min = args.lr * 1e-6)

        if args.cuda:
            self.model = self.model.cuda()
            # self.criterion = self.criterion.cuda()
            # self.uniform_criterion = self.uniform_criterion.cuda()
    
    def train(self, epochs):
        self.epochs = epochs
        print(f"epochs:{epochs}")
        train_start = time.time()
        train_loss = 0.0
        self.model.train()
        train_loss_list = []
        sub_train_loss_list = []
        class_feature_list = []
        texture_feature_list = []
        target_list = []
        domain_label_list = []
        scores_list = []
        texture_scores_list = []
        for i, sample in enumerate(self.unlabeled_loader):
            idx, image, augimg, target, domain_label, semi_domain_label = sample

            if self.args.cuda:
                image, target, augimg, domain_label = image.cuda(), target.cuda(), augimg.cuda(), domain_label.cuda()

            inputs_normal, bn_normal, outputs_normal, bn_augmix, outputs_augmix = self.model(image, augimg)
            
            # 对应论文的 L_abs
            L_abs = loss_fucntion([bn_normal], [bn_augmix])

            # 对应论文的 L_lowf
            L_lowf = loss_fucntion_last(outputs_normal, outputs_augmix)
            # 对应论文的 L_ori
            loss_normal = loss_fucntion(inputs_normal, outputs_normal)
            
            bn_normal = bn_normal.view(self.args.batch_size, -1)
            bn_normal = F.normalize(bn_normal, p=2, dim=1)
            L_mmd = 0
            for domain_idx in range(self.args.domain_cnt):
                other_indices = [i for i in range(self.args.domain_cnt) if i!= domain_idx]
                selected_idx = torch.where(domain_label == random.choice(other_indices))
                selected_features = bn_normal[selected_idx]
                current_features = bn_normal[torch.where(domain_label == domain_idx)]
                
                mmd = mmd_loss(current_features, selected_features, kernel_bandwidth=1.0)
                L_mmd += mmd * 50
            
            loss = loss_normal*0.9 + L_abs*0.05 + L_lowf*0.05 + L_mmd

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            train_loss_list.append(loss.item())
            sub_train_loss_list.append([L_abs.item(),L_lowf.item(),loss_normal.item()])
            
        self.domain_key = "val"
        val_loss_list, val_auroc, val_auprc, total_pred, total_target, file_name_list = self.eval(self.val_loader)
        
        if self.args.save_embedding == 1:
            np.savez(f"./results/intermediate_results/{self.args.results_save_path}/{self.args.file_name},epoch={epoch}.npz",
                     domain_prototype = self.model.domain_prototype.weight.cpu().detach().numpy(),
                     center = self.model.center.cpu().numpy(),
                     class_feature = np.concatenate(class_feature_list),
                     texture_feature = np.concatenate(texture_feature_list),
                     target_list=np.concatenate(target_list),
                     domain_label_list=np.concatenate(domain_label_list),
                     total_pred = total_pred,
                     total_target = total_target,
                     file_name_list = file_name_list)
        
        if (epochs == 0)  or ((epochs + 1) %  self.args.test_epoch == 0):
            test_start = time.time()
            test_metric = self.test()
            end = time.time()
            print(f'train time: {end - train_start}\t test time: {end - test_start}')
        else:
            test_metric=None
            end = time.time()
            print(f'train time: {end - train_start}')

        return train_loss_list, val_loss_list, val_auroc, val_auprc, test_metric, sub_train_loss_list
    
    def test(self):
        test_metric = {}
        for key in trainer.test_loader:
            self.domain_key = key
            print(key)
            test_loss_list, test_auroc, test_auprc, total_pred, total_target, file_name_list = self.eval(trainer.test_loader[key])
            test_metric[key] = {
                "test_loss_list": test_loss_list,
                "AUROC": test_auroc,
                "AUPRC": test_auprc,
                "pred":total_pred,
                "target":total_target,
                "file_name_list": file_name_list
            }
        return test_metric

    def eval(self, dataloader):
        self.model.eval()
        # tbar = tqdm(dataloader, desc='\r')
        test_loss = 0.0
        total_pred = np.array([])
        total_target = np.array([])
        loss_list = []
        class_feature_list = []
        texture_feature_list = []
        target_list = []
        domain_label_list = []
        file_name_list = []
        if self.args.data_name.__contains__('PACS'):
            img_path = self.train_loader.dataset.image_paths[np.random.choice(np.where(self.train_loader.dataset.labels == 0)[0], 1)]
            link_to_normal_sample = config["PACS_root"] + img_path.item() #update the link here
            normal_image = Image.open(link_to_normal_sample).convert("RGB")
            mean_train = [0.485, 0.456, 0.406]
            std_train = [0.229, 0.224, 0.225]
            trans = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=mean_train,
                                    std=std_train)
            ])
            normal_image = trans(normal_image)
            normal_image = torch.unsqueeze(normal_image, 0)
            normal_image = normal_image.cuda()

        for i, sample in enumerate(dataloader):
            idx, image, _, target, domain_label, semi_domain_label = sample
            if self.args.cuda:
                image, target, domain_label = image.cuda(), target.cuda(), domain_label.cuda()
            with torch.no_grad():
                inputs, outputs = self.model.inference(image, normal_image)
                anomaly_map, _ = cal_anomaly_map(inputs, outputs, image.shape[-1], amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                output = np.max(anomaly_map)

                # class_feature_list.append(class_feature.cpu().detach().numpy())
                # texture_feature_list.append(texture_feature.cpu().detach().numpy())

                target_list.append(target.cpu().numpy())
                domain_label_list.append(domain_label.cpu().numpy())
                file_name_list.append(dataloader.dataset.image_paths[idx])

            # loss = self.criterion(output, target.unsqueeze(1).float())
            # test_loss += loss.item()
            # loss_list.append(loss.item())
            # tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_pred = np.append(total_pred, output)
            total_target = np.append(total_target, target.cpu().numpy())
        file_name_list=np.array(file_name_list)
        roc, pr = aucPerformance(total_pred, total_target)
        if self.args.save_embedding == 1:
            np.savez(f"./results/intermediate_results/{self.args.results_save_path}/{self.args.file_name},epoch={self.epoch},{self.domain_key}.npz",
                    #  class_feature_list=np.concatenate(class_feature_list),
                    #  texture_feature_list=np.concatenate(texture_feature_list),
                     target_list=np.concatenate(target_list),
                     domain_label_list=np.concatenate(domain_label_list),
                     total_pred=total_pred,
                     total_target=total_target,
                     AUROC=np.array(roc),
                     AUPRC=np.array(pr),
                     file_name_list=file_name_list
                     )
        return loss_list, roc, pr, total_pred, total_target, file_name_list
    
    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(os.path.join(args.experiment_dir, filename)))

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="PACS_with_domain_label")
    parser.add_argument("--contamination_rate", type=float ,default=0.12)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--checkitew", type=str, default="bottle")
    parser.add_argument("--pre_lr",type=float,default=0.005)
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--batch_size", type=int, default=30, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--pre_epochs", type=int, default=2, help="the number of pretrain epochs")
    parser.add_argument("--epochs", type=int, default=2, help="the number of epochs")
    parser.add_argument("--cnt", type=int, default=0)
    parser.add_argument("--tau1",type=float,default=0.07)
    parser.add_argument("--tau2",type=float,default=0.07)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--NCE_lambda", type=float, default=1.0)
    parser.add_argument("--PL_lambda", type=float, default=1.0)
    parser.add_argument("--class_lambda", type=float, default=1.0)
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--confidence_margin", type=int, default=5)

    parser.add_argument("--random_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='/experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=448, help="the image size of input")
    parser.add_argument("--save_embedding", type=int, default=0, help="No intermediate results are saved")
    
    parser.add_argument("--normal_class", nargs="+", type=int, default=[0])
    parser.add_argument("--anomaly_class", nargs="+", type=int, default=[1,2,3,4,5,6])
    parser.add_argument("--n_anomaly", type=int, default=13, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='DGADshift1', help="the backbone network")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--gpu",type=str, default="2")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--domain_cnt", type=int, default=3)
    parser.add_argument("--method", type=int, default=1)

    args = parser.parse_args()
    
    if args.pretrained == 1:
        args.pretrained = True
    else:
        args.pretrained = False
    args.experiment_dir = f"experiment{args.results_save_path}"
    if args.data_name.__contains__("PACS"):
        file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},epochs={args.epochs},lr={args.lr},contamination={args.contamination_rate},reg_lambda={args.reg_lambda},NCE_lambda={args.NCE_lambda},PL_lambda={args.PL_lambda},class_lambda={args.class_lambda},cnt={args.cnt}'
    if args.data_name.__contains__("MVTEC"):
        file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},checkitew={args.checkitew},tau1={args.tau1},tau2={args.tau2},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},epochs={args.epochs},lr={args.lr},reg_lambda={args.reg_lambda},NCE_lambda={args.NCE_lambda},PL_lambda={args.PL_lambda},cnt={args.cnt}'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    args.file_name = file_name
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)
    torch.manual_seed(args.random_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if not os.path.exists(f"results{args.results_save_path}"):
        os.makedirs(f"results{args.results_save_path}")
    
    if args.save_embedding == 1:
        if not os.path.exists(f"results/intermediate_results/{args.results_save_path}"):
            os.makedirs(f"results/intermediate_results/{args.results_save_path}")

    argsDict = args.__dict__
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    val_max_metric = {"AUROC": -1,
                      "AUPRC": -1}
    train_results_loss = []
    sub_train_results_loss = []
    val_results_loss = []
    val_AUROC_list = []
    val_AUPRC_list = []
    
    test_results_list = []
    for epoch in range(0, trainer.args.epochs):
        train_loss_list, val_loss_list, val_auroc, val_auprc, test_metric, sub_train_loss_list = trainer.train(epoch)
        # train_loss_list, sub_train_loss_list = trainer.train(epoch)
        if val_max_metric["AUPRC"] <= val_auprc:
            val_max_metric["AUROC"] = val_auroc
            val_max_metric["AUPRC"] = val_auprc
            val_max_metric["epoch"] = epoch
            trainer.save_weights(f'{file_name}.pt')
        train_results_loss.append(train_loss_list)
        sub_train_results_loss.append(sub_train_loss_list)
        
        val_results_loss.append(val_loss_list)
        val_AUROC_list.append(val_auroc)
        val_AUPRC_list.append(val_auprc)

        test_results_list.append(test_metric)
    
    trainer.load_weights(f'{file_name}.pt')
    val_max_metric["metric"] = trainer.test()
    # test_metric = trainer.test()
    
    print(f'results{args.results_save_path}/{file_name}.npz')
    np.savez(f'results{args.results_save_path}/{file_name}.npz',
             val_max_metric = np.array(val_max_metric),
             train_results_loss = np.array(train_results_loss),
             sub_train_results_loss = np.array(sub_train_results_loss),
             val_results_loss = np.array(val_results_loss),
             val_AUROC_list = np.array(val_AUROC_list),
             val_AUPRC_list = np.array(val_AUPRC_list),
             test_results_list = np.array(test_results_list),
             test_metric = np.array(test_metric),
             args = np.array(args.__dict__),)