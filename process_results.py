# --------------------------------------------------------
# Credits for original code: https://github.com/PWman/Impossible-Shapes-Paper
# and https://www.sciencedirect.com/science/article/pii/S0042698921002017?via%3Dihub (Heinke et al., 2021)
# ------------------------------------


import os
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from more_utils import cm_arr_to_df

plt.style.use("seaborn-v0_8")

def plot_and_save(results, fpath):
    # PLOT ACCURACY
    plt.style.use("seaborn-bright")

    plt.clf()
    plt.plot(results["epoch"], results["acc"], "b")
    plt.plot(results["epoch"], results["val_acc"], "r-")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, config.num_epochs - 1)
    plt.savefig(os.path.join(fpath, "Accuracy.png"))

    # PLOT LOSS
    plt.clf()
    plt.plot(results["epoch"], results["loss"], "b-")
    plt.plot(results["epoch"], results["val_loss"], "r-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.xlim(0, config.num_epochs - 1)
    plt.savefig(os.path.join(fpath, "Loss.png"))
    plt.close()
    return

def avg_train_results(net_name, study_num=2):
    results_path = os.path.join(config.raw_dir, f"Study {study_num}", net_name, "train_results")
    avg_results = []
    for file in os.listdir(results_path):
        result = pd.read_csv(os.path.join(results_path, file), index_col=0)
        avg_results.append(result)

    return sum(avg_results) / len(avg_results)

def get_raw_scores(net_path, average_score=True):
    results_path = os.path.join(net_path, "train_results")
    scores_df = pd.DataFrame(columns=["acc", "val_acc", "loss", "val_loss"])
    for file in os.listdir(results_path):
        result = pd.read_csv(os.path.join(results_path, file), index_col=0)
        end_result = result[result["epoch"] == max(result["epoch"])]
        scores_df = scores_df.append(end_result.drop(columns=["epoch"]))

    if average_score:
        df_av = pd.DataFrame([{
            "acc": np.mean(scores_df["acc"]),
            "acc_std": np.std(scores_df["acc"]),
            "val_acc": np.mean(scores_df["val_acc"]),
            "val_acc_std": np.std(scores_df["val_acc"]),
            "loss": np.mean(scores_df["loss"]),
            "loss_std": np.std(scores_df["loss"]),
            "val_loss": np.mean(scores_df["val_loss"]),
            "val_loss_std": np.std(scores_df["val_loss"])
        }])
        return df_av
    else:
        return scores_df

def total_cmats(net_name, study_num=2):
    cm_dir = os.path.join(config.raw_dir, f"Study {study_num}", net_name, "confusion_matrices")

    train_dir = os.path.join(cm_dir, "train_data")
    val_dir = os.path.join(cm_dir, "validation_data")
    cm_tot_t = np.zeros((2, 2))
    for file in os.listdir(train_dir):
        cm_t = np.load(os.path.join(train_dir, file))
        cm_tot_t = cm_tot_t + cm_t
    cm_tot_v = np.zeros((2, 2))
    for file in os.listdir(val_dir):
        cm_v = np.load(os.path.join(val_dir, file))
        cm_tot_v = cm_tot_v + cm_v
    return cm_tot_t, cm_tot_v

def avg_save_net_results(net_name, study_num=2):
    save_dir = os.path.join(config.results_basedir, f"Study {study_num}", net_name)
    config.check_make_dir(save_dir)
    print("Processing training results...")
    train_results = avg_train_results(net_name, study_num=study_num)
    train_results.to_csv(os.path.join(save_dir, "Train Results.csv"))
    plot_and_save(train_results, save_dir)

    print("Processing confusion matrices...")
    cm_t, cm_v = total_cmats(net_name, study_num=study_num)
    cm_path = os.path.join(save_dir, "Confusion Matrices.xlsx")
    cm_writer = pd.ExcelWriter(cm_path, engine="xlsxwriter")
    cm_t = cm_arr_to_df(cm_t, study_num=study_num)
    cm_v = cm_arr_to_df(cm_v, study_num=study_num)
    cm_t.to_excel(cm_writer, sheet_name="Training")
    cm_v.to_excel(cm_writer, sheet_name="Validation")
    cm_writer.save()

def graph_all_results(study_num=2):
    plt.style.use("seaborn")
    expt_dir = os.path.join(config.results_basedir, f"Study {study_num}")
    acc_ax_ylim = None 

    plt.figure()
    for net_name in config.DNNs:
        net_path = os.path.join(expt_dir, net_name)
        if os.path.isdir(net_path):
            for file in os.listdir(net_path):
                if "Train Results.csv" in file:
                    result = pd.read_csv(os.path.join(expt_dir, net_name, file))
                    plt.plot(result["epoch"], result["val_acc"])

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend(config.DNNs)
    if acc_ax_ylim:
        plt.ylim(acc_ax_ylim)
    plt.savefig(os.path.join(expt_dir, "Validation Accuracies"))
    plt.close("all")

def collate_net_scores_table(study_num=2):
    raw_dir = os.path.join(config.raw_dir, f"Study {study_num}")
    save_dir = os.path.join(config.results_basedir, f"Study {study_num}", "DNN Performance Summary.xlsx")

    df_unformattted = pd.DataFrame([])
    df_full_formatted = pd.DataFrame([])
    for net_name in config.DNNs:
        net_path = os.path.join(raw_dir, net_name)
        df = get_raw_scores(net_path, average_score=True)
        df.insert(0, "net_name", net_name)
        df_unformattted = df_unformattted.append(df)
    xl_writer = pd.ExcelWriter(save_dir, engine="xlsxwriter")
    df_full_formatted.to_excel(xl_writer, sheet_name="Formatted Results")
    df_unformattted.to_excel(xl_writer, sheet_name="Results Unformatted")
    xl_writer.save()

def collate_cmats_xl(study_num=2):

    expt_dir = os.path.join(config.results_basedir, f"Study {study_num}")
    train_cmat_xl_fpath = os.path.join(expt_dir, "Confusion Matrices (train images).xlsx")
    val_cmat_xl_fpath = os.path.join(expt_dir, "Confusion Matrices (validation images).xlsx")
    train_writer = pd.ExcelWriter(train_cmat_xl_fpath, engine="xlsxwriter")
    val_writer = pd.ExcelWriter(val_cmat_xl_fpath, engine="xlsxwriter")
    for net in config.DNNs:
        for file in os.listdir(os.path.join(expt_dir, net)):
            if file == "Confusion Matrices.xlsx":
                rpath = os.path.join(expt_dir, net, file)
                train_result = pd.read_excel(rpath, "Training", index_col=0)
                val_result = pd.read_excel(rpath, "Validation", index_col=0)
                train_result.to_excel(train_writer, sheet_name=net)
                val_result.to_excel(val_writer, sheet_name=net)
    train_writer.save()
    val_writer.save()

if __name__ == "__main__":
    for study_num in range(3):
        for net in config.DNNs:
            avg_save_net_results(net, study_num=2)
