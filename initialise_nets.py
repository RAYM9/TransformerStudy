# --------------------------------------------------------
# Credits for original code: https://github.com/PWman/Impossible-Shapes-Paper
# and https://www.sciencedirect.com/science/article/pii/S0042698921002017?via%3Dihub (Heinke et al., 2021)
# ------------------------------------


import config
from torch import nn
from torch import optim
from torch.cuda import init
from torchvision import models
import timm
import torchvision.models as models
from torchvision.models import resnet18, alexnet, vgg11, vgg16, googlenet
from timm.models import swin_base_patch4_window7_224, davit_base, vit_base_patch16_224, tnt_s_patch16_224, deit_base_patch16_224
from timm.models import cait_xxs24_224, coatnet_0_rw_224, convformer_b36, convnext_base


def init_net(model, pretrain=True):
    try:
        if isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, 2)
            if pretrain:
                for param in model.patch_embed.parameters():
                    param.requires_grad = False
                for param in model.head.parameters():
                    param.requires_grad = True
    except AttributeError:
        if pretrain:
            for param in model.parameters():
                param.requires_grad = False
        else:
            try:
                model.aux_head = nn.Linear(model.aux_head.in_features, 2)
            except AttributeError:
                pass
        model.head = nn.Linear(model.head.in_features, 2)
    return model

def init_opt(model, pretrain=True):
    if pretrain:
        try:
            opt = optim.Adam(model.head.parameters(), lr=config.lr)
        except AttributeError:
            opt = optim.Adam(model.parameters(), lr=config.lr)
    else:
        opt = optim.Adam(model.parameters(), lr=config.lr)
    return opt

def get_model(model_name):
    if "pretrain" in model_name:
        pretrain = True
    else:
        pretrain = False
    model_name = model_name.split(" ")[0]
    if "ViT" in model_name:
        out_model = timm.create_model(
            'vit_base_patch16_224.augreg_in1k',
            pretrained=pretrain,
            num_classes=2
        )
    elif "Swin" in model_name:
        out_model = timm.create_model(
            'swin_base_patch4_window7_224.ms_in1k',
            pretrained=pretrain,
            num_classes=2
        )
    elif "DaViT" in model_name:
        out_model = timm.create_model(
            'davit_base.msft_in1k',
            pretrained=pretrain,
            num_classes=2
        )
    elif "DeiT" in model_name:
        out_model = timm.create_model(
            'deit_base_patch16_224.fb_in1k',
            pretrained=pretrain,
            num_classes=2
        )
    elif "CaiT" in model_name:
        out_model = timm.create_model(
            'cait_xxs24_224.fb_dist_in1k',
            pretrained=pretrain,
            num_classes=2
        )
    elif "TNT" in model_name:
        out_model = timm.create_model(
            'tnt_s_patch16_224',
            pretrained=pretrain,
            num_classes=2
        )
    elif "CoATNet0" in model_name:
        out_model = timm.create_model(
            'coatnet_0_rw_224.sw_in1k',
            pretrained=pretrain,
            num_classes=2
        )
    elif "ConvNeXt" in model_name:
        out_model = timm.create_model(
            'convnext_base.fb_in1k',
            pretrained=pretrain,
            num_classes=2
        ) 
    elif "ConvFormer" in model_name:
        out_model = timm.create_model(
            'convformer_b36.sail_in1k',
            pretrained=pretrain,
            num_classes=2
        )          
    elif "TNT" in model_name:
        out_model = timm.create_model(
            'tnt_s_patch16_224',
            pretrained=pretrain,
            num_classes=2
        )                                
        for child in out_model.modules():
            child.track_running_stats = False
    else:
        print("Net name not recognised/supported")
        return
    return out_model

def initialise_DNN(model_name):
    available_nets = config.DNNs
    if type(model_name) == str:
        if model_name in available_nets:
            net = get_model(model_name)
            opt = init_opt(net, pretrain=True)  

            return net, opt
        else:
            print("Net name not recognised/supported")
    else:
        print("Please input DNN name as a string")