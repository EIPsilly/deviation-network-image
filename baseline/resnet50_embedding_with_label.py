import os

import time
from line_profiler import LineProfiler
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import sys
sys.path.append("/home/hzw/DGAD/deviation-network-image")
from dataloaders.dataloader import build_dataloader
from torchvision import models

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = torch.mean(1 - cos_loss(a.view(a.shape[0], -1),
                                    b.view(b.shape[0], -1)))
    return loss

class Trainer(object):

    def __init__(self, args):
        self.args = args

        kwargs = {'num_workers': args.workers}
        self.train_loader, self.val_loader, self.test_loader, self.unlabeled_loader = build_dataloader(args, **kwargs)
        
        self.model = models.wide_resnet50_2(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        if args.cuda:
            self.model = self.model.cuda()
    
    def train(self, data_loader):
        embeddings = []
        labels = []
        domain_labels = []
        for i, sample in enumerate(data_loader):
            idx, image, augimg, target, domain_label, semi_domain_label = sample

            if self.args.cuda:
                image = image.cuda()

            with torch.no_grad():
                embedding = self.model(image)
                embeddings.append(embedding.view(-1, 2048).cpu().numpy())
                
            labels.append(target.view(-1, 1))
            domain_labels.append(domain_label.view(-1, 1))

        return np.concatenate(embeddings), np.concatenate(labels), np.concatenate(domain_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="PACS_with_domain_label")
    parser.add_argument("--contamination_rate", type=float ,default=0)
    parser.add_argument("--checkitew", type=str, default="bottle")
    parser.add_argument("--severity", type=int, default=3)
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
    parser.add_argument("--class_lambda", type=float, default=1.0)
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--confidence_margin", type=int, default=5)
    parser.add_argument("--BalancedBatchSampler", type=int, default=0)

    parser.add_argument("--random_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=16, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='/experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=448, help="the image size of input")
    parser.add_argument("--save_embedding", type=int, default=0, help="No intermediate results are saved")
    
    parser.add_argument("--normal_class", nargs="+", type=int, default=[5])
    parser.add_argument("--anomaly_class", nargs="+", type=int, default=[0,1,2,3,4,6])
    parser.add_argument("--n_anomaly", type=int, default=13, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='wide_resnet50_2', help="the backbone network")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--gpu",type=str, default="3")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--domain_cnt", type=int, default=3)
    parser.add_argument("--method", type=int, default=0)

    args = parser.parse_args()
    
    if args.pretrained == 1:
        args.pretrained = True
    else:
        args.pretrained = False
    args.experiment_dir = f"experiment{args.results_save_path}"
    if args.data_name.__contains__("PACS"):
        if args.contamination_rate != 0:
            file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},contamination={args.contamination_rate}'
        else:
            file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class}'
        domain_list = ['photo', 'art_painting', 'cartoon', 'sketch']
    if args.data_name.__contains__("MVTEC"):
        file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},checkitew={args.checkitew}'
        domain_list = ['origin', 'brightness', 'contrast', 'defocus_blur', 'gaussian_noise']
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)
    torch.manual_seed(args.random_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if not os.path.exists(f"results{args.results_save_path}"):
        os.makedirs(f"results{args.results_save_path}")

    train_embeddings, train_labels, train_domain_label = trainer.train(trainer.train_loader)
    val_embeddings, val_labels, val_domain_label = trainer.train(trainer.val_loader)
    test_dict = dict()
    
    for key in domain_list:
        embeddings, labels, domain_label = trainer.train(trainer.test_loader[key])
        test_dict[key] = {
            "embeddings" : embeddings,
            "labels" : labels,
            "domain_labels": domain_label
        }
    
    print(f'results{args.results_save_path}/{file_name}.npz')
    if args.data_name.__contains__("PACS"):
        np.savez(f'results{args.results_save_path}/{file_name}.npz',
                train_embeddings = train_embeddings,
                val_embeddings = val_embeddings,
                train_labels = train_labels,
                val_labels = val_labels,
                train_domain_label = train_domain_label,
                val_domain_label = val_domain_label,
                test_photo = test_dict["photo"]["embeddings"],
                test_photo_labels = test_dict["photo"]["labels"],
                test_photo_domain_labels = test_dict["photo"]["domain_labels"],
                test_art_painting = test_dict["art_painting"]["embeddings"],
                test_art_painting_labels = test_dict["art_painting"]["labels"],
                test_art_painting_domain_labels = test_dict["art_painting"]["domain_labels"],
                test_cartoon = test_dict["cartoon"]["embeddings"],
                test_cartoon_labels = test_dict["cartoon"]["labels"],
                test_cartoon_domain_labels = test_dict["cartoon"]["domain_labels"],
                test_sketch = test_dict["sketch"]["embeddings"],
                test_sketch_labels = test_dict["sketch"]["labels"],
                test_sketch_domain_labels = test_dict["sketch"]["domain_labels"],
                )
    if args.data_name.__contains__("MVTEC"):
        np.savez(f'results{args.results_save_path}/{file_name}.npz',
                train_embeddings = train_embeddings,
                val_embeddings = val_embeddings,
                train_labels = train_labels,
                val_labels = val_labels,
                test_origin = test_dict["origin"]["embeddings"],
                test_origin_labels = test_dict["origin"]["labels"],
                test_brightness = test_dict["brightness"]["embeddings"],
                test_brightness_labels = test_dict["brightness"]["labels"],
                test_contrast = test_dict["contrast"]["embeddings"],
                test_contrast_labels = test_dict["contrast"]["labels"],
                test_defocus_blur = test_dict["defocus_blur"]["embeddings"],
                test_defocus_blur_labels = test_dict["defocus_blur"]["labels"],
                test_gaussian_noise = test_dict["gaussian_noise"]["embeddings"],
                test_gaussian_noise_labels = test_dict["gaussian_noise"]["labels"]
                )