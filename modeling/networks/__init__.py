from torchvision.models import alexnet
from modeling.networks.resnet18 import feature_resnet18, feature_resnet50, feature_wide_resnet50_2

NET_OUT_DIM = {'alexnet': 256, 'resnet18': 512, 'resnet50': 2048, "wide_resnet50_2": 2048, "DGAD": 2048, "DGAD4": 2048, "DGAD5": 2048, "DGAD6": 2048, "DGAD9": 2048, "DGAD15": 2048}


def build_feature_extractor(args):
    backbone = args.backbone
    if backbone == "alexnet":
        print("Feature extractor: AlexNet")
        return alexnet(pretrained=True).features
    elif backbone == "resnet18":
        print("Feature extractor: ResNet-18")
        return feature_resnet18()
    elif backbone == "resnet50":
        print("Feature extractor: ResNet-50")
        return feature_resnet50()
    elif backbone == "wide_resnet50_2":
        print("Feature extractor: wide_resnet50_2")
        return feature_wide_resnet50_2(args)
    
    elif backbone == "DGAD":
        print("Feature extractor: ResNet-DGAD")
        from .DGAD import wide_resnet50_2
        net = wide_resnet50_2(pretrained=args.pretrained)
        return net[0].to("cuda"), net[1].to("cuda")
    elif backbone == "DGAD4":
        print("Feature extractor: ResNet-DGAD4")
        from .DGAD_method4 import wide_resnet50_2
        net = wide_resnet50_2(pretrained=args.pretrained)
        return net[0].to("cuda"), net[1].to("cuda")
    elif backbone == "DGAD5":
        print("Feature extractor: ResNet-DGAD5")
        from .DGAD_method5 import wide_resnet50_2
        net = wide_resnet50_2(pretrained=args.pretrained)
        return net[0].to("cuda"), net[1].to("cuda")
    elif backbone == "DGAD6":
        print("Feature extractor: ResNet-DGAD6")
        from .DGAD_method6 import wide_resnet50_2
        net = wide_resnet50_2(pretrained=args.pretrained)
        return net[0].to("cuda"), net[1].to("cuda")
    elif backbone == "DGAD9":
        print("Feature extractor: ResNet-DGAD9")
        from .DGAD_method9 import wide_resnet50_2
        net = wide_resnet50_2(pretrained=args.pretrained)
        return net[0].to("cuda"), net[1].to("cuda")
    elif backbone == "EFDM_DGAD":
        print("Feature extractor: EFDM_DGAD")
        from .resnet_TTA import wide_resnet50_2
        net = wide_resnet50_2(pretrained=args.pretrained)
        return net[0].to("cuda"), net[1].to("cuda")
    elif backbone == "VAE_DEVNET":
        print("Feature extractor: VAE_DEVNET")
        from .CVAE import CVAE
        return CVAE(args)
    elif backbone == "VAE":
        print("Feature extractor: VAE")
        from .VAE import VAE
        return VAE(args)
    elif backbone == "VAE_LPIPS_DEVNET":
        print(f"Feature extractor: {backbone}")
        from .VAE_LPIPS_DEVNET import VAE
        return VAE(args)
    elif backbone == "DGAD15":
        print("Feature extractor: ResNet-DGAD15")
        from .DGAD_method15 import wide_resnet50_2
        net = wide_resnet50_2(pretrained=args.pretrained)
        return net[0].to("cuda"), net[1].to("cuda")
    else:
        raise NotImplementedError