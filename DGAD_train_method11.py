from torch.nn.parameter import Parameter
import time
from line_profiler import LineProfiler
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

import argparse
import os

from dataloaders.dataloader import build_dataloader
from modeling.DGAD_net_method11 import DGAD_net
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
        
        self.criterion = build_criterion(args.criterion)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.beta_list = np.sin(np.linspace(0, np.pi / 2, self.args.epochs)) * (self.args.beta_end - self.args.beta_begin) + self.args.beta_begin

        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def init_center(self):
        self.model.eval()
        # tbar = tqdm(self.train_loader)
        feature_list = []
        target_list = []
        domain_label_list = []
        semi_domain_label_list = []
        for i, sample in enumerate(self.unlabeled_loader):
            idx, image, _, target, domain_label, semi_domain_label = sample

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                _, class_feature = self.model.encoder(image)
                
                class_feature = self.model.avgpool(class_feature)
                class_feature = torch.flatten(class_feature, 1)
                class_feature = self.model.origin_fc(class_feature)
            
            feature_list.append(class_feature)
            target_list.append(target)
            domain_label_list.append(domain_label)
            semi_domain_label_list.append(semi_domain_label)
        
        semi_domain_label_list = torch.concat(semi_domain_label_list)
        feature_list = torch.concat(feature_list)
        target_list = torch.concat(target_list)
        domain_label_list = torch.concat(domain_label_list)
        
        # np.savez("init_feature.npz",
        #          feature_list = feature_list.cpu().numpy(),
        #          target_list = target_list.cpu().numpy(),
        #          domain_label_list = domain_label_list.cpu().numpy())
        
        # self.model.center = F.normalize(torch.mean(feature_list[torch.where(target_list == 0)[0]], dim=0), dim=0)
        self.model.center = torch.mean(feature_list[torch.where(target_list == 0)[0]], dim=0)
        
        estimator = KMeans(n_clusters=3)  # 构造聚类器
        estimator.fit(feature_list.cpu().numpy())
        predict = estimator.predict(feature_list.cpu().numpy())
        # predict = domain_label_list

        self.model.domain_prototype = []
        for i in range(self.args.domain_cnt):
            self.model.domain_prototype.append(torch.mean(feature_list[np.where(predict == i)[0]], dim=0).reshape(1, -1))
        
        self.model.domain_prototype = torch.cat(self.model.domain_prototype)
        # self.model.domain_prototype = torch.rand(self.args.domain_cnt, self.model.center.shape[0]).cuda()
        # scale = torch.norm(self.model.domain_prototype, dim=1) / torch.norm(self.model.center)
        # self.model.domain_prototype = self.model.domain_prototype / scale.reshape(-1, 1)
    
    def update(self, epoch):
        self.model.eval()
        # tbar = tqdm(self.train_loader)
        category_list = []
        feature_list = []
        target_list = []
        semi_domain_label_list = []
        domain_label_list = []
        for i, sample in enumerate(self.unlabeled_loader):
            idx, image, _, target, domain_label, semi_domain_label = sample

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                Intermediate_feature, _ = self.model.encoder(image)
                shallow_feature = self.model.shallow_conv(Intermediate_feature)

                shallow_feature = self.model.avgpool(shallow_feature)
                shallow_feature = torch.flatten(shallow_feature, 1)
                shallow_feature = self.model.shallow_fc(shallow_feature)
                
                texture_feature = self.model.texture_fc(torch.cat([shallow_feature, shallow_feature - self.model.center], dim=1))
                similarity_matrix = torch.cat([-torch.sum((texture_feature - self.model.domain_prototype[i])**2, dim=1).reshape(-1, 1) for i in range(self.args.domain_cnt)], dim = 1)
                category = torch.argmax(F.softmax(similarity_matrix, dim = 1), dim=1)
            
            feature_list.append(texture_feature)
            category_list.append(category)
            target_list.append(target)
            domain_label_list.append(domain_label)
            semi_domain_label_list.append(semi_domain_label)
        
        semi_domain_label_list = torch.concat(semi_domain_label_list)
        feature_list = torch.concat(feature_list)
        target_list = torch.concat(target_list)
        category_list = torch.concat(category_list)
        
        for i in range(self.args.domain_cnt):
            if torch.sum(category_list == i).item() > 0:
                # self.model.domain_prototype[i] = self.model.domain_prototype[i] * self.beta_list[epoch] + (1 - self.beta_list[epoch]) * torch.mean(feature_list[torch.where(category_list == i)[0]], dim=0)
                self.model.domain_prototype[i] = self.model.domain_prototype[i] * self.args.beta + (1 - self.args.beta) * torch.mean(feature_list[torch.where(category_list == i)[0]], dim=0)

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
        loss_lambda = torch.Tensor([1, self.args.origin_svdd_lambda, self.args.class_svdd_lambda, self.args.align_lambda]).cuda().reshape(-1, 1)
        for i, sample in enumerate(self.unlabeled_loader):
            idx, image, augimg, target, domain_label, semi_domain_label = sample

            if self.args.cuda:
                image, target, augimg, domain_label = image.cuda(), target.cuda(), augimg.cuda(), domain_label.cuda()
            
            loss_list = self.model(image)
            loss2_list = self.model(augimg)
            loss_list += loss2_list
            loss_list = torch.mul(loss_list, loss_lambda)
            loss = torch.sum(loss_list)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            # tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
            train_loss_list.append(loss.item())
            sub_train_loss_list.append([loss_list[0].item(), loss_list[1].item(), loss_list[2].item(), loss_list[3].item(),])
            
        self.scheduler.step()
        self.domain_key = "val"
        val_loss_list, val_auroc, val_auprc, total_pred, total_target = self.eval(self.val_loader)
        test_start = time.time()
        test_metric = self.test()
        end = time.time()
        print(f'train time: {end - train_start}\t test time: {end - test_start}')

        if self.args.save_embedding == 1:
            np.savez(f"./results/intermediate_results/epoch={epoch}.npz",
                     domain_prototype = self.model.domain_prototype.cpu().numpy(),
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
                output = self.model.test(image)
                # class_feature, texture_feature = self.model(image)
                # class_feature_list.append(class_feature.cpu().detach().numpy())
                # texture_feature_list.append(texture_feature.cpu().detach().numpy())
                target_list.append(target.cpu().numpy())
                domain_label_list.append(domain_label.cpu().numpy())

            # loss = self.criterion(output, target.unsqueeze(1).float())
            # loss2 = torch.mean(torch.abs(output - invariant_score))
            # loss += loss2
            loss = torch.mean(output)
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

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename + '.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="PACS_with_domain_label")
    parser.add_argument("--lr",type=float,default=0.0002)
    parser.add_argument("--batch_size", type=int, default=30, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=2, help="the number of epochs")
    parser.add_argument("--cnt", type=int, default=0)
    # parser.add_argument("--tau1",type=float,default=0.07)
    # parser.add_argument("--tau2",type=float,default=0.07)
    parser.add_argument("--origin_svdd_lambda", type=float, default=1.0)
    parser.add_argument("--class_svdd_lambda", type=float, default=1.0)
    parser.add_argument("--align_lambda", type=float, default=1.0)
    parser.add_argument("--beta_begin",type=float,default=0.5)
    parser.add_argument("--beta_end",type=float,default=0.92)
    parser.add_argument("--beta",type=float,default=0.92)

    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
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
    parser.add_argument("--domain_label_ratio", type=float, default=0.02)
    parser.add_argument("--unsupervised", type=int, default=1)
    parser.add_argument("--n_anomaly", type=int, default=13, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='DGAD9', help="the backbone network")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--gpu",type=str, default="3")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--domain_cnt", type=int, default=3)
    parser.add_argument("--method", type=int, default=11)

    # args = parser.parse_args(["--epochs", "2", "--lr", "0.00001"])
    args = parser.parse_args()
    # args = parser.parse_args(["--epochs", "30", "--lr", "5e-5", "--origin_svdd_lambda", "2.0", "--class_svdd_lambda", "1.0", "--align_lambda", "1.0", "--gpu", "1", "--cnt", "0"])
    
    # model_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},origin_svdd_lambda={args.origin_svdd_lambda},class_svdd_lambda={args.class_svdd_lambda},align_lambda={args.align_lambda},beta_begin={args.beta_begin},beta_end={args.beta_end}'
    # file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},epochs={args.epochs},lr={args.lr},origin_svdd_lambda={args.origin_svdd_lambda},class_svdd_lambda={args.class_svdd_lambda},align_lambda={args.align_lambda},beta_begin={args.beta_begin},beta_end={args.beta_end},cnt={args.cnt}'
    model_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},origin_svdd_lambda={args.origin_svdd_lambda},class_svdd_lambda={args.class_svdd_lambda},align_lambda={args.align_lambda},beta={args.beta}'
    file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},epochs={args.epochs},lr={args.lr},origin_svdd_lambda={args.origin_svdd_lambda},class_svdd_lambda={args.class_svdd_lambda},align_lambda={args.align_lambda},beta={args.beta},cnt={args.cnt}'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)
    torch.manual_seed(args.ramdn_seed)

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
        trainer.update(epoch)
        if val_max_metric["AUROC"] <= val_auroc:
            val_max_metric["AUROC"] = val_auroc
            val_max_metric["AUPRC"] = val_auprc
            val_max_metric["epoch"] = epoch
            # trainer.save_weights(f'{file_name}.pkl')
        train_results_loss.append(train_loss_list)
        sub_train_results_loss.append(sub_train_loss_list)
        
        val_results_loss.append(val_loss_list)
        val_AUROC_list.append(val_auroc)
        val_AUPRC_list.append(val_auprc)

        test_results_list.append(test_metric)
        

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