# --------------------------------------------------------
# Credits for original code: https://github.com/PWman/Impossible-Shapes-Paper
# and https://www.sciencedirect.com/science/article/pii/S0042698921002017?via%3Dihub (Heinke et al., 2021)
# ------------------------------------

import os
import torch
import torchvision.models as models

def check_make_dir(data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

batch_size = 16
num_seeds = 20
num_epochs = 100
loss_fun = torch.nn.CrossEntropyLoss()
lr = 0.001

target_layers = {
    "ViT": ["head"],
    "ViT (pretrained)": ["head"],
    "Swin": ["head"],
    "Swin (pretrained)": ["head"],
    "DaViT": ["head"],
    "DaViT (pretrained)": ["head"],
    "DeiT": ["head_drop"],
    "DeiT (pretrained)": ["head_drop"],
    "CaiT": ["head"],
    "CaiT (pretrained)": ["head"],
    "CoATNet0": ["head"],
    "CoATNet0 (pretrained)": ["head"],
    "ConvNeXt": ["head"],
    "ConvNeXt (pretrained)": ["head"],
    "ConvFormer": ["head"],
    "ConvFormer (pretrained)": ["head"],
    "TNT": ["head"],
    "TNT (pretrained)": ["head"]
}

DNNs = list(target_layers.keys())

results_basedir = os.path.join(os.getcwd(), "Results")
shapes_basedir = os.path.join(os.getcwd(), "Shapes")
check_make_dir(results_basedir)
check_make_dir(shapes_basedir)

raw_dir = os.path.join(results_basedir, "Raw")
original_dir = os.path.join(shapes_basedir, "Original")
prepro_dir = os.path.join(shapes_basedir, "Preprocessed")
check_make_dir(raw_dir)
check_make_dir(original_dir)
check_make_dir(prepro_dir)

image_checks = os.path.join(shapes_basedir, "Image_Checking")
fully_prepro_dir = os.path.join(image_checks,"Fully_Preprocessed")
bg_segment_dir = os.path.join(image_checks,"Background_Segmentation")
check_make_dir(image_checks)
check_make_dir(fully_prepro_dir)
check_make_dir(bg_segment_dir)

for study_num in range(3):
    check_make_dir(os.path.join(raw_dir, f"Study {study_num}"))
    check_make_dir(os.path.join(results_basedir, f"Study {study_num}"))
DNNs = list(target_layers.keys())

results_basedir = os.path.join(os.getcwd(), "Results")
shapes_basedir = os.path.join(os.getcwd(), "Shapes")
check_make_dir(results_basedir)
check_make_dir(shapes_basedir)

raw_dir = os.path.join(results_basedir, "Raw")
original_dir = os.path.join(shapes_basedir, "Original")
prepro_dir = os.path.join(shapes_basedir, "Preprocessed")
check_make_dir(raw_dir)
check_make_dir(original_dir)
check_make_dir(prepro_dir)

image_checks = os.path.join(shapes_basedir, "Image_Checking")
fully_prepro_dir = os.path.join(image_checks,"Fully_Preprocessed")
bg_segment_dir = os.path.join(image_checks,"Background_Segmentation")
check_make_dir(image_checks)
check_make_dir(fully_prepro_dir)
check_make_dir(bg_segment_dir)

for study_num in range(3):
    check_make_dir(os.path.join(raw_dir, f"Study {study_num}"))
    check_make_dir(os.path.join(results_basedir, f"Study {study_num}"))