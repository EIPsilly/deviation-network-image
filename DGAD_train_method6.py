import time
from line_profiler import LineProfiler
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os

from dataloaders.dataloader import build_dataloader
from modeling.DGAD_net_method6 import DGAD_net
from tqdm import tqdm
from utils import aucPerformance
from modeling.layers import build_criterion

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = torch.mean(1 - cos_loss(a.view(a.shape[0], -1),
                                    b.view(b.shape[0], -1)))
    return loss

class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.val_loader, self.test_loader, self.unlabeled_loader = build_dataloader(args, **kwargs)
        
        self.model = DGAD_net(args)
        
        self.criterion = build_criterion(args.criterion, args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def init_center(self):
        self.model.eval()
        # tbar = tqdm(self.train_loader)
        feature_list = []
        for i, sample in enumerate(self.unlabeled_loader):
            idx, image, _, target, domain_label, semi_domain_label = sample

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                class_feature, _ = self.model.CL(image)
            
            feature_list.append(class_feature)
        
        self.model.center = F.normalize(torch.mean(torch.concat(feature_list), dim=0), dim=0)
    
    def train(self, epoch):
        self.epoch = epoch
        print(f"epoch:{epoch}")
        train_start = time.time()
        train_loss = 0.0
        self.model.train()
        # tbar = tqdm(self.train_loader)
        train_loss_list = []
        sub_train_loss_list = []
        class_feature_list = []
        texture_feature_list = []
        target_list = []
        domain_label_list = []
        for i, sample in enumerate(self.train_loader):
            idx, image, augimg, target, domain_label, semi_domain_label = sample

            if self.args.cuda:
                image, target, augimg, domain_label = image.cuda(), target.cuda(), augimg.cuda(), domain_label.cuda()

            output, texture_score = self.model(image)
            devnet_loss = self.criterion(output, target.unsqueeze(1).float())
            # reg_loss = self.criterion(texture_score, target.unsqueeze(1).float()) * self.args.reg_lambda
            # reg_loss = torch.mean(torch.abs(texture_score)) * self.args.reg_lambda
            reg_loss = torch.mean(torch.abs(texture_score)[torch.where(target == 0)[0]]) * self.args.reg_lambda
            
            class_feature, texture_feature = self.model.CL(image)
            aug_class_feature, _ = self.model.CL(augimg)

            class_feature_list.append(class_feature.cpu().detach().numpy())
            texture_feature_list.append(texture_feature.cpu().detach().numpy())
            target_list.append(target.cpu().detach().numpy())
            domain_label_list.append(domain_label.cpu().detach().numpy())

            class_feature = F.normalize(class_feature) - self.model.center
            aug_class_feature = F.normalize(aug_class_feature) - self.model.center
            similarity_matrix = torch.matmul(class_feature, aug_class_feature.T) / self.args.tau1
            NCE_loss = nn.CrossEntropyLoss()(similarity_matrix, torch.arange(class_feature.shape[0]).cuda()) * self.args.NCE_lambda
            
            domain_similarity_matrix = self.model.domain_prototype(F.normalize(texture_feature)) / self.args.tau2
            PL_loss = nn.CrossEntropyLoss()(domain_similarity_matrix, domain_label) * self.args.PL_lambda

            loss = devnet_loss + reg_loss + NCE_loss + PL_loss
            
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            # tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
            train_loss_list.append(loss.item())
            sub_train_loss_list.append([devnet_loss.item(),reg_loss.item(),NCE_loss.item(),PL_loss.item(),])
            
        self.scheduler.step()
        self.domain_key = "val"
        val_loss_list, val_auroc, val_auprc, total_pred, total_target = self.eval(self.val_loader)
        if epoch %  self.args.test_epoch == 0:
            test_start = time.time()
            test_metric = self.test()
            end = time.time()
            print(f'train time: {end - train_start}\t test time: {end - test_start}')
        else:
            test_metric=None
            end = time.time()
            print(f'train time: {end - train_start}')

        if self.args.save_embedding == 1:
            np.savez(f"./results/intermediate_results/epoch={epoch}.npz",
                     domain_prototype = self.model.domain_prototype.weight.cpu().detach().numpy(),
                     center = self.model.center.cpu().numpy(),
                     class_feature = np.concatenate(class_feature_list),
                     texture_feature = np.concatenate(texture_feature_list),
                     target_list=np.concatenate(target_list),
                     domain_label_list=np.concatenate(domain_label_list),
                     total_pred = total_pred,
                     total_target = total_target)
        
        
        return train_loss_list, val_loss_list, val_auroc, val_auprc, test_metric, sub_train_loss_list
    
    def test(self):
        test_metric = {}
        for key in trainer.test_loader:
            self.domain_key = key
            print(key)
            test_loss_list, test_auroc, test_auprc, total_pred, total_target = self.eval(trainer.test_loader[key])
            test_metric[key] = {
                "test_loss_list": test_loss_list,
                "AUROC": test_auroc,
                "AUPRC": test_auprc,
                "pred":total_pred,
                "target":total_target
            }
        return test_metric

    def eval(self, dataset):
        self.model.eval()
        # tbar = tqdm(dataset, desc='\r')
        test_loss = 0.0
        total_pred = np.array([])
        total_target = np.array([])
        loss_list = []
        class_feature_list = []
        texture_feature_list = []
        target_list = []
        domain_label_list = []
        for i, sample in enumerate(dataset):
            idx, image, _, target, domain_label, semi_domain_label = sample
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output, invariant_score = self.model(image)
                class_feature, texture_feature = self.model.CL(image)
                class_feature_list.append(class_feature.cpu().detach().numpy())
                texture_feature_list.append(texture_feature.cpu().detach().numpy())
                target_list.append(target.cpu().numpy())
                domain_label_list.append(domain_label.cpu().numpy())

            loss = self.criterion(output, target.unsqueeze(1).float())
            # loss2 = torch.mean(torch.abs(output - invariant_score))
            # loss += loss2
            test_loss += loss.item()
            loss_list.append(loss.item())
            # tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, target.cpu().numpy())
        roc, pr = aucPerformance(total_pred, total_target)
        if self.args.save_embedding == 1:
            np.savez(f"./results/intermediate_results/epoch={self.epoch},{self.domain_key}.npz",
                     class_feature_list=np.concatenate(class_feature_list),
                     texture_feature_list=np.concatenate(texture_feature_list),
                     target_list=np.concatenate(target_list),
                     domain_label_list=np.concatenate(domain_label_list),
                     total_pred=total_pred,
                     AUROC=np.array(roc),
                     AUPRC=np.array(pr),
                     )
        return loss_list, roc, pr, total_pred, total_target
    
    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(os.path.join(args.experiment_dir, filename)))

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="PACS_with_domain_label")
    parser.add_argument("--lr",type=float,default=0.0002)
    parser.add_argument("--batch_size", type=int, default=30, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=2, help="the number of epochs")
    parser.add_argument("--cnt", type=int, default=0)
    parser.add_argument("--tau1",type=float,default=0.07)
    parser.add_argument("--tau2",type=float,default=0.07)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--NCE_lambda", type=float, default=1.0)
    parser.add_argument("--PL_lambda", type=float, default=1.0)
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--test_epoch", type=int, default=5)

    parser.add_argument("--random_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=32, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=448, help="the image size of input")
    parser.add_argument("--save_embedding", type=int, default=0, help="No intermediate results are saved")
    
    parser.add_argument("--normal_class", nargs="+", type=int, default=[0])
    parser.add_argument("--anomaly_class", nargs="+", type=int, default=[1,2,3,4,5,6])
    parser.add_argument("--n_anomaly", type=int, default=13, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='DGAD6', help="the backbone network")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--gpu",type=str, default="2")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--domain_cnt", type=int, default=3)
    parser.add_argument("--method", type=int, default=6)

    # args = parser.parse_args(["--epochs", "2", "--lr", "0.00001"])
    args = parser.parse_args()
    # args = parser.parse_args(["--epochs", "40", "--lr", "0.0001", "--normal", "1", "--anomaly_class", "0", "2", "3", "4", "5", "6"])
    # args = parser.parse_args(["--epochs", "30", "--lr", "5e-5", "--tau1", "0.07", "--tau2", "0.07", "--reg_lambda", "2.0", "--NCE_lambda", "1.0", "--PL_lambda", "1.0", "--gpu", "1", "--cnt", "0", "--save_embedding", "1", "--results_save_path", "/intermediate_results"])
    
    if args.pretrained == 1:
        args.pretrained = True
    else:
        args.pretrained = False

    model_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},reg_lambda={args.reg_lambda},NCE_lambda={args.NCE_lambda},PL_lambda={args.PL_lambda}'
    file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},epochs={args.epochs},lr={args.lr},tau1={args.tau1},tau2={args.tau2},reg_lambda={args.reg_lambda},NCE_lambda={args.NCE_lambda},PL_lambda={args.PL_lambda},cnt={args.cnt}'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)
    torch.manual_seed(args.random_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if not os.path.exists(f"results{args.results_save_path}"):
        os.makedirs(f"results{args.results_save_path}")
    
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
    
    trainer.init_center()
    test_results_list = []
    for epoch in range(0, trainer.args.epochs):
        # lp = LineProfiler()
        # lp_wrapper = lp(trainer.train)
        # train_loss_list, val_loss_list, val_auroc, val_auprc, test_metric, sub_train_loss_list = lp_wrapper(epoch)
        # lp.print_stats()
        train_loss_list, val_loss_list, val_auroc, val_auprc, test_metric, sub_train_loss_list = trainer.train(epoch)
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