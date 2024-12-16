import torch
from modeling.layers.deviation_loss import DeviationLoss
from modeling.layers.binary_focal_loss import BinaryFocalLoss
from modeling.layers.uniformLoss import UniformLoss
from modeling.layers.mmd_loss import MMDLoss

def build_criterion(criterion, args):
    if criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLoss()
    elif criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "focal":
        print("Loss : Focal")
        return BinaryFocalLoss()
    elif criterion == "uniform":
        print("Loss : Uniform")
        return UniformLoss(args)
    elif criterion == "mmd":
        return MMDLoss()
    else:
        raise NotImplementedError