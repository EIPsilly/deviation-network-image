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
from datasets.augmix import augpacs

with open("../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    import yaml
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
class_to_idx = config["PACS_class_to_idx"]

class PACS_Dataset(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None, augment_transform = None):
        self.image_paths = x
        self.labels = y
        self.transform = transform
        self.target_transform = target_transform
        self.augment_transform = augment_transform
        
        self.img_list = []
        resize_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            ])
        for img_path in self.image_paths:
            img = Image.open(config["PACS_root"] + img_path).convert('RGB')
            img = resize_transform(img)
            self.img_list.append(img)
        
        self.normal_idx = np.where(self.labels==0)[0]
        self.outlier_idx = np.where(self.labels==1)[0]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        label = self.labels[idx]
        
        augimg = img

        if self.augment_transform is not None:
            augimg = augpacs(augimg, self.augment_transform)
            # augimg = self.augment_transform(augimg)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return idx, img, augimg, label
    
class PACS_Data():

    def __init__(self, args):
            
        normal_class = "".join(list(map(str,args.normal_class)))
        anomaly_class = "".join(list(map(str,args.anomaly_class)))
        if args.domain_cnt == 1:
            train_path = f'../domain-generalization-for-anomaly-detection/data/pacs/semi-supervised/1domain/20240412-PACS-{normal_class}-{anomaly_class}.npz'
        elif args.domain_cnt == 3:
            train_path = f'../domain-generalization-for-anomaly-detection/data/pacs/semi-supervised/3domain/20240412-PACS-{normal_class}-{anomaly_class}.npz'
        
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
        
        self.train_data = PACS_Dataset(data["train_set_path"], data["train_labels"], transform=train_transform, target_transform=None, augment_transform = augment_transform)
        unlabeled_idx = np.where(data["train_labels"] == 0)[0]
        self.unlabeled_data = PACS_Dataset(data["train_set_path"][unlabeled_idx], data["train_labels"][unlabeled_idx], transform=train_transform, target_transform=None, augment_transform = augment_transform)
        self.val_data = PACS_Dataset(data["val_set_path"], data["val_labels"], transform=train_transform, target_transform=None, augment_transform = augment_transform)

        logging.info("y_train\t" + str(dict(sorted(Counter(data["train_labels"]).items()))))
        logging.info("y_val\t" + str(dict(sorted(Counter(data["val_labels"]).items()))))
        print("y_train\t" + str(dict(sorted(Counter(data["train_labels"]).items()))))
        print("y_val\t" + str(dict(sorted(Counter(data["val_labels"]).items()))))
        self.test_dict = {}
        for domain in ["photo", "art_painting", "cartoon", "sketch"]:
            self.test_dict[domain] = PACS_Dataset(data[f"test_{domain}"], data[f"test_{domain}_labels"], transform=test_transform, target_transform=None, augment_transform = augment_transform)
            logging.info(domain + "\ty_test\t" + str(dict(sorted(Counter(data[f"test_{domain}_labels"]).items()))))
            print(domain + "\ty_test\t" + str(dict(sorted(Counter(data[f"test_{domain}_labels"]).items()))))
            
        # for domain in os.listdir(f'{config["PACS_root"]}/test'):
        #     # if domain in self.in_domain_type:
        #     #     continue
        #     test_img_path_list = []
        #     test_label = []
        #     for _class_ in os.listdir(f'{config["PACS_root"]}/test/{domain}'):
        #         img_paths = glob.glob(f'{config["PACS_root"]}/test/{domain}/{_class_}/*.[jp][pn]g')
        #         img_paths = [path.split("PACS")[1] for path in img_paths]
        #         test_img_path_list += img_paths
        #         test_label += [int(class_to_idx[_class_] in args.anomaly_class)] * len(img_paths)
        #     logging.info(domain + "\ty_test\t" + str(dict(sorted(Counter(test_label).items()))))
        #     self.test_dict[domain] = PACS_Dataset(test_img_path_list, test_label, transform=test_transform, target_transform=None, augment_transform = augment_transform)


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
    args.add_argument("--data_name", type=str, default="PACS")
    args.add_argument("--normal_class", nargs="+", type=int, default=[0,1,2,3])
    args.add_argument("--anomaly_class", nargs="+", type=int, default= [4,5,6])
    args.add_argument("--contamination_rate", type=float ,default=0)
    args.add_argument("--labeled_rate", type=float, default=0.02)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--train_binary", type=bool, default=True)
    args.add_argument("--in_domain_type", nargs="+", type=str, default=["photo", "art_painting", "cartoon"], choices=["art_painting", "cartoon", "photo", "sketch"])
    

    args = args.parse_args(["--data_name", "PACS",
                            "--normal_class", "0", "1", "2", "3",
                            "--anomaly_class", "4", "5", "6",
                            "--train_binary", "True",
                            "--in_domain_type", "photo", "art_painting", "cartoon",
                            ])

    data = PACS_Data(args)