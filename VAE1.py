import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
from line_profiler import LineProfiler
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from dataloaders.dataloader import build_dataloader
from modeling.VAE import SemiADNet
from tqdm import tqdm
from utils import aucPerformance
from modeling.layers import build_criterion
import torchvision.transforms as transforms
from PIL import Image

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    
    def eval(self, key, data_loader):
        x_list = []
        y_list = []
        self.model.eval()
        for i, sample in enumerate(data_loader):
            idx, image, augimg, target, domain_label, semi_domain_label = sample

            if self.args.cuda:
                image, target, domain_label = image.cuda(), target.cuda(), domain_label.cuda()
            
            with torch.no_grad():
                reconstructions, rec_loss, kl_loss = self.model(image, domain_label)
            
            x = image.detach()
            y = reconstructions.detach()
            un_norm = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ]),
                                ])

            x = np.uint8(un_norm(x).cpu().numpy() * 255).transpose((0,2,3,1))
            y = np.uint8(un_norm(y).cpu().numpy() * 255).transpose((0,2,3,1))
            
            x_list.append(x)
            y_list.append(y)

        x_list = np.concatenate(x_list)
        y_list = np.concatenate(y_list)

        if not os.path.exists(f'images/{file_name}/{key}'):
            os.makedirs(f'images/{file_name}/{key}')

        for idx, x in enumerate(x_list[:5]):
            input_img = Image.fromarray(x)
            input_img.save(f"images/{file_name}/{key}/{idx}input_img.jpg")
        
        for idx, y in enumerate(y_list[:5]):
            input_img = Image.fromarray(y)
            input_img.save(f"images/{file_name}/{key}/{idx}rec_img.jpg")
        
        return
    
    def test(self):
        self.eval("train", self.train_loader)
        self.eval("val", self.val_loader)
        for key in self.test_loader:
            self.domain_key = key
            print(key)
            self.eval(key, self.test_loader[key])
        return

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
    parser.add_argument("--BalancedBatchSampler", type=int, default=1)
    
    
    parser.add_argument("--normal_class", nargs="+", type=int, default=[0])
    parser.add_argument("--anomaly_class", nargs="+", type=int, default=[1,2,3,4,5,6])
    parser.add_argument("--n_anomaly", type=int, default=13, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='VAE', help="the backbone network")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--gpu",type=str, default="1")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--domain_cnt", type=int, default=3)
    parser.add_argument("--method", type=int, default=0)

    # args = parser.parse_args(["--epochs", "2", "--lr", "0.00001"])
    args = parser.parse_args()
    # args = parser.parse_args(["--epochs", "30", "--lr", "5e-5", "--tau1", "0.07", "--tau2", "0.07", "--reg_lambda", "2.0", "--NCE_lambda", "1.0", "--PL_lambda", "1.0", "--gpu", "1", "--cnt", "0", "--save_embedding", "1", "--results_save_path", "/intermediate_results"])
    
    args.experiment_dir = f"experiment{args.results_save_path}"
    model_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},reg_lambda={args.reg_lambda},NCE_lambda={args.NCE_lambda},PL_lambda={args.PL_lambda},BalancedBatchSampler={args.BalancedBatchSampler}'
    file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},epochs={args.epochs},lr={args.lr},tau1={args.tau1},tau2={args.tau2},reg_lambda={args.reg_lambda},NCE_lambda={args.NCE_lambda},PL_lambda={args.PL_lambda},BalancedBatchSampler={args.BalancedBatchSampler},cnt={args.cnt}'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)
    torch.manual_seed(args.random_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if not os.path.exists(f"results{args.results_save_path}"):
        os.makedirs(f"results{args.results_save_path}")
    
    if not os.path.exists(f"images/{file_name}"):
        os.makedirs(f"images/{file_name}")
    
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
    # for epoch in range(0, trainer.args.epochs):
    #     train_loss_list, val_loss_list, val_auroc, val_auprc, test_metric, sub_train_loss_list = trainer.train(epoch)
    #     loss = np.mean(val_loss_list)
    #     print(f"val_loss:{loss}")
    #     if val_max_metric["loss"] >= loss:
    #         val_max_metric["AUROC"] = val_auroc
    #         val_max_metric["AUPRC"] = val_auprc
    #         val_max_metric["epoch"] = epoch
    #         val_max_metric["loss"] = loss
    #         trainer.save_weights(f'{file_name}.pt')
    #     train_results_loss.append(train_loss_list)
    #     sub_train_results_loss.append(sub_train_loss_list)
        
    #     val_results_loss.append(val_loss_list)
    #     val_AUROC_list.append(val_auroc)
    #     val_AUPRC_list.append(val_auprc)

    #     test_results_list.append(test_metric)
        

    trainer.load_weights(f'{file_name}.pt')
    # val_max_metric["metric"] = trainer.test()
    trainer.test()
    
    # print(f'results{args.results_save_path}/{file_name}.npz')
    # np.savez(f'results{args.results_save_path}/{file_name}.npz',
    #          val_max_metric = np.array(val_max_metric),
    #          train_results_loss = np.array(train_results_loss),
    #          sub_train_results_loss = np.array(sub_train_results_loss),
    #          val_results_loss = np.array(val_results_loss),
    #          val_AUROC_list = np.array(val_AUROC_list),
    #          val_AUPRC_list = np.array(val_AUPRC_list),
    #          test_results_list = np.array(test_results_list),
    #          test_metric = np.array(test_metric),
    #          args = np.array(args.__dict__),)