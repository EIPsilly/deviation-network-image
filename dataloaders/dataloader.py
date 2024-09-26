from torch.utils.data import DataLoader
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler
from datasets.PACS import PACS_Data
from datasets.PACS_with_domain_label import PACS_with_domain_label
from datasets.MVTEC import MVTEC_Data
from datasets.MVTEC_with_domain_label import MVTEC_with_domain_label
from datasets.PACS_dataloader import PACSDataset

def build_dataloader(args, **kwargs):

    if args.data_name == "PACS":
        data = PACS_Data(args)
        
    if args.data_name == "PACS_with_domain_label":
        data = PACS_with_domain_label(args)
    
    if args.data_name == "MVTEC":
        data = MVTEC_Data(args)

    if args.data_name == "MVTEC_with_domain_label":
        data = MVTEC_with_domain_label(args)

    if args.data_name == "PACS_dataloader":
        data = PACSDataset(args)

    train_set = data.train_data
    if ("BalancedBatchSampler" in args) and (args.BalancedBatchSampler == 0):
        train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn_seed, shuffle=True, batch_size=args.batch_size, **kwargs)
    else:
        train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn_seed, batch_sampler=BalancedBatchSampler(args, train_set), **kwargs)
    val_data = data.val_data
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, worker_init_fn=worker_init_fn_seed, **kwargs)
    test_loader = {}
    for key in data.test_dict:
        test_loader[key] = DataLoader(data.test_dict[key], batch_size=args.batch_size, shuffle=False, worker_init_fn=worker_init_fn_seed, **kwargs)
    
    unlabeled_data = data.unlabeled_data
    unlabeled_loader = DataLoader(unlabeled_data, worker_init_fn=worker_init_fn_seed, batch_size=args.batch_size, **kwargs)
    
    # elif args.data_name == "MVTec":
    #     train_set = mvtecad.MVTecAD(args, train=True)
    #     test_set = mvtecad.MVTecAD(args, train=False)
    #     train_loader = DataLoader(train_set,
    #                                 worker_init_fn=worker_init_fn_seed,
    #                                 batch_sampler=BalancedBatchSampler(args, train_set),
    #                                 **kwargs)
    #     test_loader = DataLoader(test_set,
    #                                 batch_size=args.batch_size,
    #                                 shuffle=False,
    #                                 worker_init_fn=worker_init_fn_seed,
    #                                 **kwargs)
    return train_loader, val_loader, test_loader, unlabeled_loader