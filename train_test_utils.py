# --------------------------------------------------------
# Credits for original code: https://github.com/PWman/Impossible-Shapes-Paper
# and https://www.sciencedirect.com/science/article/pii/S0042698921002017?via%3Dihub (Heinke et al., 2021)
# ------------------------------------

import os
import config
import torch
import numpy as np
import pandas as pd
from preprocessing import Preprocess
from sklearn.metrics import confusion_matrix
from more_utils import set_seed
from initialise_nets import initialise_DNN
from torchvision.models.vision_transformer import _vision_transformer
from torchvision.models import vision_transformer

def feed_net(net, opt, img_batch, lbl_batch, train_net=False, confusion_matrices=False):
    img_batch = img_batch.to(config.device)
    lbl_batch = lbl_batch.to(config.device)
    
    if train_net:
        net.train()
        opt.zero_grad()
    else:
        net.eval()
    
    net_out = net(img_batch)
    loss = config.loss_fun(net_out, lbl_batch)
    
    if train_net:
        loss.backward()
        opt.step()
    else:
        net.eval()
    net_out = net(img_batch)

    if len(net_out[0]) == 2:
        loss = config.loss_fun(net_out, lbl_batch)
    else:
        losses = []
        for i in net_out:
            losses.append(config.loss_fun(i, lbl_batch))
        loss = losses[0] + 0.3 * (sum(losses[1:]))
        net_out = net_out[0]

    if train_net:
        loss.backward()
        opt.step()
        opt.zero_grad()

    lbl_batch = lbl_batch.tolist()
    net_out = net_out.tolist()
    predicts = [i.index(max(i)) for i in net_out]
    matches = [i == j for i, j in zip(predicts, lbl_batch)]

    acc = sum(matches) / len(matches)

    if confusion_matrices:

        cm = confusion_matrix(lbl_batch, predicts, labels=[0, 1])

        return cm
    else:
        return acc, float(loss)

def train_epoch(p, net, opt):
    train_scores = []
    for img_batch, lbl_batch in p.train_loader:
        acc, loss = feed_net(net, opt, img_batch, lbl_batch, train_net=True)
        train_scores.append([acc, loss])

    valid_scores = []
    for x in range(2):
        for vimg_batch, vlbl_batch in p.test_loader:
            vacc, vloss = feed_net(net, opt, vimg_batch, vlbl_batch, train_net=False)
            valid_scores.append([vacc, vloss])

    return np.mean(train_scores, axis=0), np.mean(valid_scores, axis=0)

def train_net(p, net, opt):
    num_epochs = config.num_epochs
    net.to(config.device)
    results = pd.DataFrame(columns=["epoch", "acc", "loss",
                                    "val_acc", "val_loss"])
    for epoch in range(num_epochs):
        [t_acc, t_loss], [v_acc, v_loss] = train_epoch(p, net, opt)

        r = pd.DataFrame([{
            "epoch": epoch,
            "acc": t_acc,
            "val_acc": v_acc,
            "loss": t_loss,
            "val_loss": v_loss,
        }])
        results = results.append(r, sort=True)
        print(f"Epoch {epoch} Complete")
        print(f"Acc = {round(t_acc, 2)} Val Acc = {round(v_acc, 2)}")

    return results

def train_nets_all_seeds(net_name, study_num=2):
    raw_dir = os.path.join(config.raw_dir, f"Study {study_num}", net_name)
    img_dir = os.path.join(config.prepro_dir, f"Study {study_num}")
    config.check_make_dir(raw_dir)
    model_dir = os.path.join(raw_dir, "models")
    config.check_make_dir(model_dir)
    result_dir = os.path.join(raw_dir, "train_results")
    config.check_make_dir(result_dir)

    p = Preprocess(data_dir=img_dir, augment=True, scale_factor=0.9)

    for seed in range(config.num_seeds):
        print(f"Training {net_name} seed {seed}...")
        set_seed(seed)
        print("Initialising network...")
        net, opt = initialise_DNN(net_name)
        print("Training network...")
        result = train_net(p, net, opt)
        result.to_csv(os.path.join(
            result_dir, f"{seed}.csv"
        ))
        torch.save(net.state_dict(), os.path.join(model_dir, f"{seed}.pt"))
    return

def save_all_cmats(net_name, study_num=2):
    def get_cm_result(p, model):
        model.eval()
        cm_tot_t = np.zeros((2, 2))
        for img, lbl in p.train_loader:
            cm_t = feed_net(model, None, img, lbl,
                            train_net=False,
                            confusion_matrices=True)
            cm_tot_t = cm_tot_t + cm_t
        cm_tot_v = np.zeros((2, 2))
        for img, lbl in p.test_loader:
            cm_v = feed_net(model, None, img, lbl,
                            train_net=False,
                            confusion_matrices=True)
            cm_tot_v = cm_tot_v + cm_v
        return cm_tot_t, cm_tot_v

    print("Getting confusion matrices...")

    prepro_dir = os.path.join(config.prepro_dir, f"Study {study_num}")
    model_path = os.path.join(config.raw_dir, f"Study {study_num}", net_name, "models")
    save_dir = os.path.join(config.raw_dir, f"Study {study_num}", net_name, "confusion_matrices")
    train_dir = os.path.join(save_dir, "train_data")
    val_dir = os.path.join(save_dir, "validation_data")
    config.check_make_dir(save_dir)
    config.check_make_dir(train_dir)
    config.check_make_dir(val_dir)

    p = Preprocess(data_dir=prepro_dir, augment=False)
    net, opt = initialise_DNN(net_name)
    net.to(config.device)

    for seed in range(config.num_seeds):
        set_seed(seed)
        net.load_state_dict(torch.load(os.path.join(model_path, f"{seed}.pt")))
        cm_t, cm_v = get_cm_result(p, net)
        np.save(os.path.join(train_dir, f"{seed}"), cm_t)
        np.save(os.path.join(val_dir, f"{seed}"), cm_v)

if __name__ == "__main__":
    train_nets_all_seeds("ViT (pretrained)", study_num= 2)
    train_nets_all_seeds("ViT", study_num= 2)
    train_nets_all_seeds("Swin", study_num= 2)
    train_nets_all_seeds("Swin (pretrained)", study_num= 2)
    train_nets_all_seeds("DaViT", study_num= 2)
    train_nets_all_seeds("DaViT (pretrained)", study_num= 2)
    train_nets_all_seeds("DeiT", study_num= 2)
    train_nets_all_seeds("DeiT (pretrained)", study_num= 2)
    train_nets_all_seeds("CaiT", study_num= 2)
    train_nets_all_seeds("CaiT (pretrained)", study_num= 2)
    train_nets_all_seeds("TNT", study_num= 2)
    train_nets_all_seeds("TNT (pretrained)", study_num= 2)
    train_nets_all_seeds("CoATNet0", study_num= 2)
    train_nets_all_seeds("CoATNet0 (pretrained)", study_num= 2)
    train_nets_all_seeds("ConvNeXt", study_num= 2)
    train_nets_all_seeds("ConvNeXt (pretrained)", study_num= 2)
    train_nets_all_seeds("ConvFormer", study_num= 2)
    train_nets_all_seeds("ConvFormer (pretrained)", study_num= 2)
    save_all_cmats("ViT", study_num= 2)
    save_all_cmats("ViT (pretrained)", study_num= 2)
    save_all_cmats("Swin", study_num= 2)
    save_all_cmats("Swin (pretrained)", study_num= 2)
    save_all_cmats("DaViT", study_num= 2)
    save_all_cmats("DaViT (pretrained)", study_num= 2)
    save_all_cmats("DeiT", study_num= 2)
    save_all_cmats("DeiT (pretrained)", study_num= 2)
    save_all_cmats("CaiT", study_num= 2)
    save_all_cmats("CaiT (pretrained)", study_num= 2)
    save_all_cmats("TNT", study_num= 2)
    save_all_cmats("TNT (pretrained)", study_num= 2)
    save_all_cmats("CoATNet0", study_num= 2)
    save_all_cmats("CoATNet0 (pretrained)", study_num= 2)
    save_all_cmats("ConvNeXt", study_num= 2)
    save_all_cmats("ConvNeXt(pretrained)", study_num= 2)
    save_all_cmats("ConvFormer", study_num= 2)
    save_all_cmats("ConvFormer (pretrained)", study_num= 2)
