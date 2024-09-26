import os
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets.augmix import augpacs

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
def data_transforms(size):
    datatrans =  transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.CenterCrop(size),
    #transforms.CenterCrop(args.input_size),
    transforms.Normalize(mean=mean_train,
                         std=std_train)])
    return datatrans
def gt_transforms(size):
    gttrans =  transforms.Compose([
    transforms.Resize((size, size)),
    transforms.CenterCrop(size),
    transforms.ToTensor()])
    return gttrans

with open("../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    import yaml
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
domain_to_idx = config["PACS_domain_to_idx"]
class_to_idx = config["PACS_class_to_idx"]
idx_to_class = config["PACS_idx_to_class"]

# 计算不同domain 和 class 对应的labal
def calc_label_idx(domain, idx):
    return domain_to_idx[domain] * 7 + class_to_idx[idx]

class PACSDataset(Dataset):
    def __init__(self, args, type):
        root = config["PACS_root"]
        self.args = args

        normal_class = "".join(list(map(str,args.normal_class)))
        anomaly_class = "".join(list(map(str,args.anomaly_class)))
        if args.domain_cnt == 1:
            train_path = f'../domain-generalization-for-anomaly-detection/data/pacs/semi-supervised/1domain/20240412-PACS-{normal_class}-{anomaly_class}.npz'
        elif args.domain_cnt == 3:
            train_path = f'../domain-generalization-for-anomaly-detection/data/pacs/semi-supervised/3domain/20240412-PACS-{normal_class}-{anomaly_class}.npz'
        
        data = np.load(train_path, allow_pickle=True)

        self.data = []
        if type == 'train':
            unlabeled_idx = np.where(data["train_labels"] == 0)[0]
            for filename in data["train_set_path"][unlabeled_idx]:
                filename = str(filename)
                splits = filename.split("/")
                self.data.append({
                    "filename": filename,
                    "label": 0,
                    "clsname": ",".join([splits[2], splits[3]]),
                    "label_name": "good"
                    })
        else:
            for name in ["test_photo", "test_art_painting", "test_cartoon", "test_sketch"]:
                unlabeled_idx = np.where(data[f"{name}_labels"] == 0)[0]
                for filename in data[name][unlabeled_idx]:
                    filename = str(filename)
                    splits = filename.split("/")
                    self.data.append({
                        "filename": filename,
                        "label": 0,
                        "clsname": ",".join([splits[2], splits[3]]),
                        "label_name": "good"
                        })
        
        self.image_size = (256, 256)
        self.root = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['filename']
        target_filename = item['filename']
        label = item["label"]
        if item.get("maskname", None):
            mask = cv2.imread( self.root + item['maskname'], cv2.IMREAD_GRAYSCALE)
        else:
            if label == 0:  # good
                mask = np.zeros(self.image_size).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones(self.image_size)).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        prompt = ""
        source = cv2.imread(self.root + source_filename)
        target = cv2.imread(self.root + target_filename)
        source = cv2.cvtColor(source, 4)
        target = cv2.cvtColor(target, 4)
        source = Image.fromarray(source, "RGB")
        target = Image.fromarray(target, "RGB")
        mask = Image.fromarray(mask, "L")
        # transform_fn = transforms.Resize(256, Image.BILINEAR)
        transform_fn = transforms.Resize(self.image_size)
        source = transform_fn(source)
        target = transform_fn(target)
        mask = transform_fn(mask)
        source = transforms.ToTensor()(source)
        target = transforms.ToTensor()(target)
        mask = transforms.ToTensor()(mask)
        normalize_fn = transforms.Normalize(mean=mean_train, std=std_train)
        source = normalize_fn(source)
        target = normalize_fn(target)
        clsname = item["clsname"]
        # image_idx = self.label_to_idx[clsname]
        domain_name, class_name = clsname.split(",")
        image_idx = calc_label_idx(domain_name, class_name)

        return dict(jpg=target, txt=prompt, hint=source, mask=mask, filename=source_filename, clsname=clsname, label=int(image_idx))

