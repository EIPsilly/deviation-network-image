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
from deepod.models.tabular import *
from deepod.metrics import tabular_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="PACS_with_domain_label")
parser.add_argument("--severity", type=int, default=3)
parser.add_argument("--checkitew", type=str, default="bottle")
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
parser.add_argument('--backbone', type=str, default='DeepSAD', help="the backbone network")
parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
parser.add_argument("--gpu",type=str, default="3")
parser.add_argument("--results_save_path", type=str, default="/DEBUG")
parser.add_argument("--domain_cnt", type=int, default=3)
parser.add_argument("--method", type=int, default=0)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.data_name.__contains__("PACS"):
    data_name = f'method=0,backbone=wide_resnet50_2,domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class}'
    file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},normal_class={args.normal_class},anomaly_class={args.anomaly_class},batch_size={args.batch_size},epochs={args.epochs},lr={args.lr},cnt={args.cnt}'
    data = np.load(f'results/PACS_embedding/{data_name}.npz', allow_pickle=True)
    domain_list = ['photo', 'art_painting', 'cartoon', 'sketch']
if args.data_name.__contains__("MVTEC"):
    data_name = f'method=0,backbone=wide_resnet50_2,domain_cnt={args.domain_cnt},checkitew={args.checkitew}'
    file_name = f'method={args.method},backbone={args.backbone},domain_cnt={args.domain_cnt},checkitew={args.checkitew},batch_size={args.batch_size},epochs={args.epochs},lr={args.lr},cnt={args.cnt}'
    data = np.load(f'results/PACS_embedding_embedding/{data_name}.npz', allow_pickle=True)
    domain_list = ['origin', 'brightness', 'contrast', 'defocus_blur', 'gaussian_noise']

if not os.path.exists(f"results{args.results_save_path}"):
    os.makedirs(f"results{args.results_save_path}")

if args.backbone == "DeepSAD":
    clf = DeepSAD(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, verbose = 2)
    clf.fit(data["train_embeddings"], data["train_labels"].squeeze())
if args.backbone == "PReNet":
    clf = PReNet(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, verbose = 2)
    clf.fit(data["train_embeddings"], data["train_labels"])
if args.backbone == "RoSAS":
    clf = RoSAS(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, verbose = 2)
    clf.fit(data["train_embeddings"], data["train_labels"])

result = dict()

scores = clf.decision_function(data["val_embeddings"])
auc, ap, f1 = tabular_metrics(data["val_labels"], scores)

result["val"]={
    'AUC': auc, 'AP': ap, 'F1': f1
}

for key in domain_list:
    scores = clf.decision_function(data[f"test_{key}"])
    auc, ap, f1 = tabular_metrics(data[f"test_{key}_labels"], scores)
    result[key]={
        'AUC': auc, 'AP': ap, 'F1': f1
    }

np.savez(f'results{args.results_save_path}/{file_name}.npz',
         result = np.array(result),
         args = np.array(args.__dict__),)