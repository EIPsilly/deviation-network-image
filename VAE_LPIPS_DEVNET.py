import os

import time
from line_profiler import LineProfiler
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from dataloaders.dataloader import build_dataloader
from modeling.VAE_LPIPS_DEVNET import SemiADNet
from tqdm import tqdm
from utils import aucPerformance
from modeling.layers import build_criterion
import torchvision.transforms as transforms

with open("../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    import yaml
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

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
        
        self.model = SemiADNet(args)
        
        self.criterion = build_criterion(args.criterion, args)
        self.uniform_criterion = build_criterion("uniform", args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        
        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            self.uniform_criterion = self.uniform_criterion.cuda()

    def log_image(self, x, y, file_name, epoch, normal_anomaly):
        un_norm = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                           std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                      transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                           std = [ 1., 1., 1. ]),
                               ])

        x = np.uint8(un_norm(x).cpu().numpy() * 255).transpose((2,1,0))
        y = np.uint8(un_norm(y).cpu().numpy() * 255).transpose((2,1,0))
        from PIL import Image
        input_img = Image.fromarray(x)
        rec_img = Image.fromarray(y)
        
        input_img.save(f"images_log{args.results_save_path}/{file_name}/{epoch}_{normal_anomaly}input_img.jpg")
        rec_img.save(f"images_log{args.results_save_path}/{file_name}/{epoch}_{normal_anomaly}rec_img.jpg")

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
                image, target, domain_label = image.cuda(), target.cuda(), domain_label.cuda()
            reconstructions, rec_loss, kl_loss, class_score, domain_score, class_classification, domain_classification = self.model(image, domain_label, target)
            
            devnet_loss = self.criterion(class_score, target.unsqueeze(1).float())
            # reg_loss = self.uniform_criterion(domain_score)
            
            # PL_loss = nn.CrossEntropyLoss()(domain_classification / self.args.tau2, domain_label) * self.args.PL_lambda
            # class_classification = (class_classification / self.args.tau2).softmax(dim=1)
            # class_reg_loss = -torch.mean(torch.sum(-class_classification * torch.log(class_classification), dim=1))
            reg_loss = torch.zeros_like(devnet_loss)
            PL_loss = torch.zeros_like(devnet_loss)
            class_reg_loss = torch.zeros_like(devnet_loss)

            loss = (self.args.rec_lambda * (rec_loss + kl_loss)) + devnet_loss + reg_loss + PL_loss + class_reg_loss
            
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            # tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
            train_loss_list.append(loss.item())
            sub_train_loss_list.append([rec_loss.item(), kl_loss.item(), devnet_loss.item(), reg_loss.item(), PL_loss.item(), class_reg_loss.item()])
        
        print(f"train_loss:{np.mean(train_loss_list)}\tsubloss:{np.array(sub_train_loss_list).mean(axis = 0)}")
        x = image[-1].detach()
        y = reconstructions[-1].detach()
        self.log_image(x, y, file_name, epoch, "A")
        x = image[0].detach()
        y = reconstructions[0].detach()
        self.log_image(x, y, file_name, epoch, "N")
        
        self.scheduler.step()
        self.domain_key = "val"
        val_loss_list, val_auroc, val_auprc, total_pred, total_target = self.eval("val", self.val_loader)
        if (epoch == 0)  or ((epoch - 1) %  self.args.test_epoch == 0):
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
            test_loss_list, test_auroc, test_auprc, total_pred, total_target = self.eval(key, trainer.test_loader[key])
            test_metric[key] = {
                "test_loss_list": test_loss_list,
                "AUROC": test_auroc,
                "AUPRC": test_auprc,
                "pred":total_pred,
                "target":total_target
            }
        return test_metric

    def eval(self, domain, dataset):
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
                image, target, domain_label = image.cuda(), target.cuda(), domain_label.cuda()
            with torch.no_grad():
                reconstructions, rec_loss, kl_loss, class_score, domain_score, class_classification, domain_classification = self.model(image, domain_label, target)
            
                devnet_loss = self.criterion(class_score, target.unsqueeze(1).float())
                # reg_loss = self.uniform_criterion(domain_score)
                
                # PL_loss = nn.CrossEntropyLoss()(domain_classification / self.args.tau2, domain_label) * self.args.PL_lambda
                # class_classification = (class_classification / self.args.tau2).softmax(dim=1)
                # class_reg_loss = -torch.mean(torch.sum(-class_classification * torch.log(class_classification), dim=1))
                reg_loss = torch.zeros_like(devnet_loss)
                PL_loss = torch.zeros_like(devnet_loss)
                class_reg_loss = torch.zeros_like(devnet_loss)

                loss = (self.args.rec_lambda * (rec_loss + kl_loss)) + devnet_loss + reg_loss + PL_loss + class_reg_loss
                
                target_list.append(target.cpu().numpy())
                domain_label_list.append(domain_label.cpu().numpy())
            
            # loss = torch.mean(loss)
            # test_loss += loss.item()
            loss_list.append(loss.item())
            # tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_pred = np.append(total_pred, class_score.cpu().numpy())
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
    
    def load_pretrain_weights(self, filename):
        self.model.load_state_dict(torch.load(filename))

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
    parser.add_argument("--rec_lambda", type=float, default=1e-6)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--NCE_lambda", type=float, default=1.0)
    parser.add_argument("--PL_lambda", type=float, default=1.0)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--confidence_margin", type=int, default=5)

    parser.add_argument("--random_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=32, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=448, help="the image size of input")
    parser.add_argument('--input_img_size', type=int, default=253, help="the image size of input")
    parser.add_argument("--save_embedding", type=int, default=0, help="No intermediate results are saved")
    parser.add_argument("--BalancedBatchSampler", type=int, default=1)
    
    
    parser.add_argument("--normal_class", nargs="+", type=int, default=[0])
    parser.add_argument("--anomaly_class", nargs="+", type=int, default=[1,2,3,4,5,6])
    parser.add_argument("--n_anomaly", type=int, default=13, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='VAE_LPIPS_DEVNET', help="the backbone network")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--gpu",type=str, default="3")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--domain_cnt", type=int, default=3)
    parser.add_argument("--method", type=str, default="VAE_LPIPS_DEVNET")

    # args = parser.parse_args(["--epochs", "2", "--lr", "0.00001"])
    args = parser.parse_args()
    # args = parser.parse_args(["--epochs", "30", "--lr", "5e-5", "--tau1", "0.07", "--tau2", "0.07", "--reg_lambda", "2.0", "--NCE_lambda", "1.0", "--PL_lambda", "1.0", "--gpu", "1", "--cnt", "0", "--save_embedding", "1", "--results_save_path", "/intermediate_results"])
    
    args.experiment_dir = f"experiment{args.results_save_path}"
    model_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},rec_lambda={args.rec_lambda},reg_lambda={args.reg_lambda},NCE_lambda={args.NCE_lambda},PL_lambda={args.PL_lambda},BalancedBatchSampler={args.BalancedBatchSampler}'
    file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},epochs={args.epochs},lr={args.lr},tau1={args.tau1},tau2={args.tau2},rec_lambda={args.rec_lambda},reg_lambda={args.reg_lambda},NCE_lambda={args.NCE_lambda},PL_lambda={args.PL_lambda},BalancedBatchSampler={args.BalancedBatchSampler},cnt={args.cnt}'
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)
    # pre_train_model_name = 'experiment/DGAD/VAE_LPIPS/method=VAE_LPIPS,backbone=VAE,domain_cnt=3,normal_class=[0],anomaly_class=[1, 2, 3, 4, 5, 6],batch_size=30,steps_per_epoch=20,epochs=250,lr=0.0001,tau1=0.07,tau2=0.07,reg_lambda=1.0,NCE_lambda=1.0,PL_lambda=1.0,BalancedBatchSampler=1,cnt=2'
    # trainer.load_pretrain_weights(pre_train_model_name)

    torch.manual_seed(args.random_seed)
    

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if not os.path.exists(f"results{args.results_save_path}"):
        os.makedirs(f"results{args.results_save_path}")
    
    if not os.path.exists(f"images_log{args.results_save_path}/{file_name}"):
        os.makedirs(f"images_log{args.results_save_path}/{file_name}")
    
    argsDict = args.__dict__
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    val_max_metric = {"AUROC": -1,
                      "AUPRC": -1,
                      "loss": 2147483647}
    train_results_loss = []
    sub_train_results_loss = []
    val_results_loss = []
    val_AUROC_list = []
    val_AUPRC_list = []
    
    test_results_list = []
    for epoch in range(0, trainer.args.epochs):
        train_loss_list, val_loss_list, val_auroc, val_auprc, test_metric, sub_train_loss_list = trainer.train(epoch)
        loss = np.mean(val_loss_list)
        print(f"val_loss:{loss}")
        if val_max_metric["AUPRC"] <= val_auprc:
            val_max_metric["AUROC"] = val_auroc
            val_max_metric["AUPRC"] = val_auprc
            val_max_metric["epoch"] = epoch
            val_max_metric["loss"] = loss
            trainer.save_weights(f'{file_name}.pt')
        train_results_loss.append(train_loss_list)
        sub_train_results_loss.append(sub_train_loss_list)
        
        val_results_loss.append(val_loss_list)
        val_AUROC_list.append(val_auroc)
        val_AUPRC_list.append(val_auprc)

        test_results_list.append(test_metric)
        
    trainer.save_weights(f'{file_name},epoch={args.epochs}.pt')
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