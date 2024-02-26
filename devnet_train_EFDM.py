from collections import Counter
import numpy as np
import torch
import torch.nn as nn

import argparse
import os

from dataloaders.dataloader import build_dataloader
from modeling.net import SemiADNet
from modeling.EFDM_net import EFDM_net
from tqdm import tqdm
from utils import aucPerformance
from modeling.layers import build_criterion

class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.val_loader, self.test_loader, _ = build_dataloader(args, **kwargs)

        self.model = SemiADNet(args)
        self.EFDM_mode = EFDM_net(args, "EFDM_DGAD")

        self.criterion = build_criterion(args.criterion)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if args.cuda:
           self.model = self.model.cuda()
           self.EFDM_mode = self.EFDM_mode.cuda()
           self.criterion = self.criterion.cuda()

    def train(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        train_loss_list = []
        for i, sample in enumerate(tbar):
            # image, target = sample['image'], sample['label']
            idx, image, augimg, target = sample

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            output = self.model(image)
            loss = self.criterion(output, target.unsqueeze(1).float())
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
            train_loss_list.append(loss.item())

        self.scheduler.step()
        val_loss_list, val_auroc, val_auprc = self.eval(self.val_loader)
        test_metric = self.test()
        
        return train_loss_list, val_loss_list, val_auroc, val_auprc, test_metric
    
    def test(self):
        self.EFDM_mode.feature_extractor.load_state_dict(self.model.feature_extractor.net.state_dict())
        self.EFDM_mode.conv.load_state_dict(self.model.conv.state_dict())
        for sample in self.train_loader:
            # image, target = sample['image'], sample['label']
            idx, image, augimg, target = sample

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            normal_image = image[0]
            break
        
        test_metric = {}
        for key in trainer.test_loader:
            self.EFDM_mode.eval()
            
            tbar = tqdm(trainer.test_loader[key], desc='\r')
            test_loss = 0.0
            total_pred = np.array([])
            total_target = np.array([])
            test_loss_list = []
            for i, sample in enumerate(tbar):
                # image, target = sample['image'], sample['label']
                idx, image, augimg, target = sample
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    normal_images = normal_image.unsqueeze(0).repeat(image.size(0), 1, 1, 1)
                    output = self.EFDM_mode(image.float(), normal_images.float())
                loss = self.criterion(output, target.unsqueeze(1).float())
                test_loss += loss.item()
                test_loss_list.append(loss.item())
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                total_pred = np.append(total_pred, output.data.cpu().numpy())
                total_target = np.append(total_target, target.cpu().numpy())
            test_auroc, test_auprc = aucPerformance(total_pred, total_target)
            
            test_metric[key] = {
                "test_loss_list": test_loss_list,
                "AUROC": test_auroc,
                "AUPRC": test_auprc,
            }
        return test_metric

    def eval(self, dataset):
        self.model.eval()
        tbar = tqdm(dataset, desc='\r')
        test_loss = 0.0
        total_pred = np.array([])
        total_target = np.array([])
        loss_list = []
        for i, sample in enumerate(tbar):
            # image, target = sample['image'], sample['label']
            idx, image, augimg, target = sample
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image.float())
            loss = self.criterion(output, target.unsqueeze(1).float())
            test_loss += loss.item()
            loss_list.append(loss.item())
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, target.cpu().numpy())
        roc, pr = aucPerformance(total_pred, total_target)
        return loss_list, roc, pr

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="PACS")
    parser.add_argument("--lr",type=float,default=0.0002)
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=5, help="the number of epochs")
    parser.add_argument("--cnt", type=int, default=0)

    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=448, help="the image size of input")
    
    parser.add_argument("--normal_class", nargs="+", type=int, default=[0])
    parser.add_argument("--anomaly_class", nargs="+", type=int, default=[1,2,3,4,5,6])
    parser.add_argument("--n_anomaly", type=int, default=13, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='wide_resnet50_2', help="the backbone network")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--gpu",type=str, default="1")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--domain_cnt", type=int, default=1)

    # args = parser.parse_args(["--backbone", "DGAD", "--epochs", "15", "--lr", "0.00001"])
    args = parser.parse_args()
    
    file_name = f'EFDM,backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},steps_per_epoch={args.steps_per_epoch},epochs={args.epochs},lr={args.lr},cnt={args.cnt}'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)
    torch.manual_seed(args.ramdn_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    argsDict = args.__dict__
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    val_max_metric = {"AUROC": -1,
                      "AUPRC": -1}
    train_results_loss = []
    val_results_loss = []
    val_AUROC_list = []
    val_AUPRC_list = []
    
    test_results_list = []
    for epoch in range(0, trainer.args.epochs):
        train_loss_list, val_loss_list, val_auroc, val_auprc, test_metric = trainer.train(epoch)
        if val_max_metric["AUROC"] < val_auroc:
            val_max_metric["AUROC"] = val_auroc
            val_max_metric["AUPRC"] = val_auprc
            val_max_metric["epoch"] = epoch
            # trainer.save_weights(f'{file_name}.pkl')
        train_results_loss.append(train_loss_list)
        
        val_results_loss.append(val_loss_list)
        val_AUROC_list.append(val_auroc)
        val_AUPRC_list.append(val_auprc)

        test_results_list.append(test_metric)
        

    # test_metric = trainer.test()
    
    print(f'results{args.results_save_path}/{file_name}.npz')
    np.savez(f'results{args.results_save_path}/{file_name}.npz',
             val_max_metric = np.array(val_max_metric),
             train_results_loss = np.array(train_results_loss),
             val_results_loss = np.array(val_results_loss),
             val_AUROC_list = np.array(val_AUROC_list),
             val_AUPRC_list = np.array(val_AUPRC_list),
             test_results_list = np.array(test_results_list),
             test_metric = np.array(test_metric),
             args = np.array(args.__dict__),)