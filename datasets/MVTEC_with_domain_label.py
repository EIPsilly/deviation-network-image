import argparse
import random
from PIL import Image, ImageOps, ImageEnhance
import glob
import torch
import logging
import os
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from collections import Counter
from datasets.augmix import augmvtec

with open("../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    import yaml
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
domain_to_idx = config["MVTEC_domain_to_idx"]

class MVTEC_Dataset(Dataset):
    def __init__(self, args, x, y, transform=None, target_transform=None, augment_transform = None):
        self.image_paths = x
        self.labels = y
        self.transform = transform
        self.target_transform = target_transform
        self.augment_transform = augment_transform
        
        self.img_list = []
        resize_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            ])
        self.domain_labels = np.empty_like(self.labels)

        if ("severity" in args == False) or (args.severity == 3):
            root = config["mvtec_ood_root"]
        elif args.severity == 4:
            root = config["mvtec_ood_root_severity4"]
        elif args.severity == 5:
            root = config["mvtec_ood_root_severity5"]

        for idx, img_path in enumerate(self.image_paths):
            self.domain_labels[idx] = domain_to_idx[img_path.split("/")[1].replace("mvtec_", "")]
            img = Image.open(root + img_path).convert('RGB')
            img = resize_transform(img)
            self.img_list.append(img)
        
        self.semi_domain_labels = self.domain_labels.copy()
        if "domain_label_ratio" in args:
            from sklearn.model_selection import train_test_split
            mask_set, unmask_set, mask_idx, unmask_idx = train_test_split(self.semi_domain_labels, np.arange(self.semi_domain_labels.shape[0]), test_size=max(3 / self.semi_domain_labels.shape[0], args.domain_label_ratio), random_state=42, stratify=self.semi_domain_labels)
            self.semi_domain_labels[mask_idx] = -1
        
        self.normal_idx = np.where(self.labels==0)[0]
        self.outlier_idx = np.where(self.labels==1)[0]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        label = self.labels[idx]
        
        augimg = img

        if self.augment_transform is not None:
            augimg = augmvtec(augimg, self.augment_transform)
            # augimg = self.augment_transform(augimg)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return idx, img, augimg, label, self.domain_labels[idx], self.semi_domain_labels[idx]
    
class MVTEC_with_domain_label():

    def __init__(self, args):
            
        if args.domain_cnt == 1:
            train_path = f'../domain-generalization-for-anomaly-detection/data/mvtec/semi-supervised/1domain/20240412-MVTEC-{args.checkitew}.npz'
        elif args.domain_cnt == 4:
            train_path = f'../domain-generalization-for-anomaly-detection/data/mvtec/semi-supervised/4domain/20240412-MVTEC-{args.checkitew}.npz'
        
        data = np.load(train_path, allow_pickle=True)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_size = 256

        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
        ])
        
        augment_transform = train_transform
        
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(image_size),
                transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
        ])
        
        self.train_data = MVTEC_Dataset(args, data["train_set_path"], data["train_labels"], transform=train_transform, target_transform=None, augment_transform = augment_transform)
        unlabeled_idx = np.where(data["train_labels"] == 0)[0]
        self.unlabeled_data = MVTEC_Dataset(args, data["train_set_path"][unlabeled_idx], data["train_labels"][unlabeled_idx], transform=train_transform, target_transform=None, augment_transform = augment_transform)
        self.val_data = MVTEC_Dataset(args, data["val_set_path"], data["val_labels"], transform=train_transform, target_transform=None, augment_transform = augment_transform)

        logging.info("y_train\t" + str(dict(sorted(Counter(data["train_labels"]).items()))))
        logging.info("y_val\t" + str(dict(sorted(Counter(data["val_labels"]).items()))))
        print("y_train\t" + str(dict(sorted(Counter(data["train_labels"]).items()))))
        print("y_val\t" + str(dict(sorted(Counter(data["val_labels"]).items()))))
        self.test_dict = {}
        for domain in ["origin", "brightness", "contrast", "defocus_blur", "gaussian_noise"]:
            self.test_dict[domain] = MVTEC_Dataset(args, data[f"test_{domain}"], data[f"test_{domain}_labels"], transform=test_transform, target_transform=None, augment_transform = augment_transform)
            logging.info(domain + "\ty_test\t" + str(dict(sorted(Counter(data[f"test_{domain}_labels"]).items()))))
            print(domain + "\ty_test\t" + str(dict(sorted(Counter(data[f"test_{domain}_labels"]).items()))))
            
        
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 8, drop_last_train = True):

        self.train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=shuffle_train,
                                                            num_workers=num_workers, drop_last=drop_last_train)

        self.val_loader = DataLoader(dataset=self.val_data, batch_size=batch_size, shuffle=shuffle_train,
                                                            num_workers=num_workers, drop_last=drop_last_train)

        self.test_loader_dict = {}
        for domain_type, test_set in self.test_dict.items():
            self.test_loader_dict[domain_type] = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers, drop_last=False)
        
        self.unlabeled_loader = DataLoader(dataset=self.unlabeled_data, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, drop_last=False)

        return self.train_loader, self.val_loader, self.test_loader_dict
    
    

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:    %(message)s', datefmt='%Y-%m-%d %H:%M:%S ')
    logging.getLogger().setLevel(logging.INFO)

    args = argparse.ArgumentParser()
    args.add_argument("--data_name", type=str, default="MVTEC")
    args.add_argument("--checkitew", type=str, default="bottle")
    args.add_argument("--domain_cnt", type=int, default=4)

    args.add_argument("--contamination_rate", type=float ,default=0)
    args.add_argument("--labeled_rate", type=float, default=0.02)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--train_binary", type=bool, default=True)
    args.add_argument("--in_domain_type", nargs="+", type=str, default=["origin", "brightness", "contrast", "defocus_blur"], choices=["origin", "brightness", "contrast", "defocus_blur", "gaussian_noise"])
    args = args.parse_args()
    

    data = MVTEC_with_domain_label(args)

    train_loader, val_loader, test_loader_dict = data.loaders(batch_size = 16)

    label_list = []
    domain_labels_list = []
    for batch in train_loader:
        idx, img, aug, label, domain_labels, semi_domain_labels = batch
        label_list.append(label)
        domain_labels_list.append(domain_labels)
    
    logging.info("train_loader\t" + str(dict(sorted(Counter(np.concatenate(label_list)).items()))))
    logging.info("train_loader domain label\t" + str(dict(sorted(Counter(np.concatenate(domain_labels_list)).items()))))
    

    label_list = []
    domain_labels_list = []
    for batch in val_loader:
        idx, img, aug, label, domain_labels, semi_domain_labels = batch
        label_list.append(label)
        domain_labels_list.append(domain_labels)
    
    logging.info("val_loader\t" + str(dict(sorted(Counter(np.concatenate(label_list)).items()))))
    logging.info("val_loader domain label\t" + str(dict(sorted(Counter(np.concatenate(domain_labels_list)).items()))))
    

    for domain_type, test_loader in test_loader_dict.items():
        label_list = []
        domain_labels_list = []
        for batch in test_loader:
            idx, img, aug, label, domain_labels, semi_domain_labels = batch
            label_list.append(label)
            domain_labels_list.append(domain_labels)
        
        logging.info(domain_type + "\ttest_loader\t" + str(dict(sorted(Counter(np.concatenate(label_list)).items()))))
        logging.info(f"{domain_type}\tval_loader domain label\t" + str(dict(sorted(Counter(np.concatenate(domain_labels_list)).items()))))