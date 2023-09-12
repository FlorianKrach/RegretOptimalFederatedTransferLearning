"""
author: Florian Krach
"""
import copy

# ==============================================================================
import numpy as np
import os, sys
import pandas as pd
import json
import time
import socket
import matplotlib
import matplotlib.colors
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as mticker
from torch.backends import cudnn
import gc
import ast
from sklearn.linear_model import LinearRegression
from scipy.optimize import fsolve
import scipy.stats as sstats
import itertools


import configs as config

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    SBM = config.SendBotMessage()

# ==============================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True
print(socket.gethostname())
SEND = False
if SERVER:
    SEND = True
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# Global variables
CHAT_ID = config.CHAT_ID
ERROR_CHAT_ID = config.ERROR_CHAT_ID

data_path = config.data_path
saved_models_path = config.saved_models_path


# ==============================================================================
# Functions
def average_duplicated_runs(path):
    """
    Args:
        path: str, path to the csv files
    Returns: pd.DataFrame where the columns were averaged

    """
    files = os.listdir(path)
    files = [f for f in files if (f.endswith('.csv') and f.startswith("run-"))]
    columns = ["comp_time", "path_gen_time", "nb_runs"]
    data = []
    for f in files:
        df = pd.read_csv(os.path.join(path, f), converters={"prices": eval})
        row = df.iloc[-1].values.tolist()
        data.append(row[:2] + row[2])
    nb_datasets = len(data[0]) - 2
    for i in range(nb_datasets):
        columns += ["mean_dataset_{}".format(i), "std_dataset_{}".format(i)]
    data = np.array(data)
    av = [np.median(data[:, 0]), np.median(data[:, 1]), len(data)]
    for i in range(nb_datasets):
        av += [np.mean(data[:, 2 + i]), np.std(data[:, 2 + i])]
    df = pd.DataFrame(data=[av], columns=columns)
    df.to_csv(os.path.join(path, "averaged.csv"), index=False)
    return df, nb_datasets


def get_training_overview(
        path=saved_models_path, ids_from=None,
        ids_to=None, save_file=None,
        params_extract_desc=None,
        get_relative_goodness=True,
        get_paper_table=True,
        sortby="mean_dataset_0",
        send=False,
):
    """
    function to get the important metrics and hyper-params for each model in the
    models_overview.csv file

    Args:
        path: str, where the saved models are
        ids_from: None or int, which model ids to consider start point
        ids_to: None or int, which model ids to consider end point
        params_extract_desc: list of str, names of params to extract from the
            model description dict, special:
                - network_size: gets size of first layer of enc network
                - activation_function_x: gets the activation function of layer x
                    of enc network
                - path_wise_penalty-x-y: extract from x-th path_wise_penalty the
                    value of key y. x has to be convertable to int.
                - dict-x-y: extract from dict x the value of key y
        save_file: str or None
        get_relative_goodness: bool, whether to compute the relative goodness 
            for each dataset, which is computed as 
            ```price(method_i, dataset_j) / price(loc_opt_method_j, dataset_j)```
            for each dataset j and method i
        get_paper_table: bool, whether to compute the table for the paper
        sortby: str or None, sort the output df by this column (ascending)
        send: bool, whether to send the file to the telegram bot

    Returns:
    """
    filename = "{}model_overview.csv".format(path)
    df = pd.read_csv(filename, index_col=0)
    if ids_from:
        df = df.loc[df["id"] >= ids_from]
    if ids_to:
        df = df.loc[df["id"] <= ids_to]
    
    if params_extract_desc is None:
        params_extract_desc = []
    if get_relative_goodness:
        if "dict-opt_method_dict-method" not in params_extract_desc:
            params_extract_desc.append("dict-opt_method_dict-method")
        if "dict-opt_method_dict-which_dataset" not in params_extract_desc:
            params_extract_desc.append("dict-opt_method_dict-which_dataset")

    # extract wanted information
    for param in params_extract_desc:
        df[param] = None

    prices_dfs = []
    index = []
    nb_datasets = None
    for i in df.index:
        desc = df.loc[i, "description"]
        param_dict = json.loads(desc)

        values = []
        for param in params_extract_desc:
            try:
                if param == 'network_size':
                    v = param_dict["enc_nn"][0][0]
                elif 'activation_function' in param:
                    numb = int(param.split('_')[-1])
                    v = param_dict["enc_nn"][numb - 1][1]
                elif "path_wise_penalty" in param:
                    _, num, key = param.split("-")
                    num = int(num)
                    v = param_dict["path_wise_penalty"][num][key]
                elif param.startswith("dict-"):
                    dictname, key = param.split("-")[1:]
                    v = param_dict[dictname][key]
                else:
                    v = param_dict[param]
                values.append(v)
            except Exception:
                values.append(None)
        df.loc[i, params_extract_desc] = values

        id = df.loc[i, "id"]
        _p_df, _nb_datasets = average_duplicated_runs(
            os.path.join(path, "id-{}".format(id)))
        if nb_datasets is None:
            nb_datasets = _nb_datasets
        prices_dfs.append(_p_df)
        index.append(id)
    prices_df = pd.concat(prices_dfs, axis=0,ignore_index=True)
    prices_df["id"] = index
    df = pd.merge(df, prices_df, left_on=["id"], right_on=["id"], how="left")
    
    # compute relative goodness
    if get_relative_goodness:
        # check whether loc_opt_method is in the df for each dataset
        all_loc_opt_ds = df["dict-opt_method_dict-which_dataset"].values.tolist()
        rel_goodness_cols = []
        for i in range(nb_datasets):
            if i not in all_loc_opt_ds:
                raise ValueError(
                    "dataset {} does not have a local_optimizer".format(i))
            which = df["dict-opt_method_dict-which_dataset"] == i
            loc_opt_price = df.loc[which, "mean_dataset_{}".format(i)].values[0]
            df["relative_goodness_dataset_{}".format(i)] = \
                df["mean_dataset_{}".format(i)] / loc_opt_price
            rel_goodness_cols.append("relative_goodness_dataset_{}".format(i))
            z95 = sstats.norm.ppf(0.975)
            df["RG_CI95_diff_dataset_{}".format(i)] = \
                df["std_dataset_{}".format(i)] / loc_opt_price * z95 / np.sqrt(
                    df["nb_runs"])
            df["RG-dataset-{}".format(i)] = df.apply(
                lambda x: "{:.3f}".format(
                    x["relative_goodness_dataset_{}".format(i)]), axis=1)
            df["RG-CI95-dataset-{}".format(i)] = df.apply(
                lambda x: "$[{:.3f}; {:.3f}]$".format(
                    x["relative_goodness_dataset_{}".format(i)] -
                    x["RG_CI95_diff_dataset_{}".format(i)],
                    x["relative_goodness_dataset_{}".format(i)] +
                    x["RG_CI95_diff_dataset_{}".format(i)]), axis=1)
        df["mean_relative_goodness"] = df[rel_goodness_cols].mean(axis=1)
        
    if sortby:
        df.sort_values(axis=0, ascending=True, by=sortby, inplace=True)

    # get paper file
    if get_paper_table:
        dfp = copy.deepcopy(df)
        names_dict = {"local_optimizer": "LO",
                      "mean_local_optimizers": "MLO",
                      "joint_optimizer": "JO",
                      "regret_optimal_algo": "RO",
                      "efficient_regret_optimal_algo": "ERO",
                      "symmetric_regret_optimal_algo": "SRO",}
        def get_name_func(x):
            name = x["dict-opt_method_dict-method"]
            name = names_dict[name]
            if name == "LO":
                name += "-{}".format(1+x["dict-opt_method_dict-which_dataset"])
            elif name in ["RO", "ERO", "SRO"]:
                name += " ($\\eta={}$)".format(
                    x["dict-opt_method_dict-info_sharing_level"])
            return name
        dfp["name"] = dfp.apply(get_name_func, axis=1)
        dfp["RP"] = dfp["RG-dataset-0"]
        dfp["CI"] = dfp["RG-CI95-dataset-0"]
        dfp = dfp[["name", "RP", "CI"]]

    # save
    if save_file is not False:
        if save_file is None:
            save_file = "{}training_overview-ids-{}-{}.csv".format(
                path, ids_from, ids_to)
        if get_paper_table:
            save_file2 = save_file.replace(".csv", "_paper.csv")
            dfp.to_csv(save_file2)
        df.to_csv(save_file)

    if send and save_file is not False:
        files_to_send = [save_file]
        if get_paper_table:
            files_to_send += [save_file2]
        SBM.send_notification(
            text=None,
            text_for_files='training overview',
            files=files_to_send,
            chat_id=CHAT_ID)

    return df


def errorbarplot(ax, t, mean, yerr, label, color, std_color_alpha=0.3,
                 type="fill", ls="-"):
    """
    :param ax: axes object
    :param t: time/x-axis values
    :param mean: mean values y-axis
    :param yerr: error y-axis
    :param label:
    :param color:
    :param std_color_alpha: float, the alpha of the color
    :param type: one of {"fill", "bar"}
    :return:
    """

    if type == "bar":
        ax.errorbar(t, mean, yerr=yerr, label=label, color=color, ls=ls)
    else:
        std_color = list(matplotlib.colors.to_rgb(color)) + [std_color_alpha]
        ax.plot(t, mean, label=label, color=color, ls=ls)
        ax.fill_between(t, mean - yerr, mean + yerr, color=std_color)


def plot_conv_analysis(
        path=saved_models_path, ids=None, run=0, scaling=1,
        iter_from=0, iter_to=None,
        labels=None, linestyles=None, send=False, use_subplots=True,
        save_extras=None, **kwargs,
):
    """
    plot the convergence analysis of the ids

    Args:
        path: str, path to the folder containing the ids
        ids: list of int, the ids to plot
        run: int, run to plot
        scaling: int or list of ints, scaling of the y-axis
        iter_from: int or list of ints, iteration from which to plot
        iter_to: int or list of ints, iteration to which to plot
        labels: list of str, labels for the legend
        linestyles: list of str, linestyles for the plot
        send: bool, whether to send the plot
        use_subplots: bool, whether to use different subplots for loss and
            regret
        save_extras: dict, extra arguments for the fig saving function
    """
    file = "{}id-{}/run-{}.csv"
    if save_extras is None:
        save_extras = {}
    if linestyles is None:
        linestyles = ["-"] * len(ids)
    if isinstance(scaling, int):
        scaling = [scaling] * len(ids)
    if iter_from is None or isinstance(iter_from, int):
        iter_from = [iter_from] * len(ids)
    if iter_to is None or isinstance(iter_to, int):
        iter_to = [iter_to] * len(ids)

    if use_subplots:
        fig, axs = plt.subplots(2)
        for i, id in enumerate(ids):
            df = pd.read_csv(file.format(path, id, run), index_col=None)
            if iter_to[i] is None:
                iter_to[i] = len(df)
            iters = np.arange(iter_from[i], iter_to[i])
            axs[0].plot(
                iters, df["loss"].values[iter_from[i]:iter_to[i]:scaling[i]],
                label=labels[i], ls=linestyles[i])
            axs[1].plot(
                iters, df["regret"].values[iter_from[i]:iter_to[i]:scaling[i]],
                label=labels[i], ls=linestyles[i])
        axs[0].legend()
        axs[0].set_title("Loss")
        axs[1].set_title("$L^{\\star}$")
        axs[0].set_xlabel("iterations")
        axs[1].set_xlabel("iterations")

    else:
        fig = plt.figure()
        for i, id in enumerate(ids):
            df = pd.read_csv(file.format(path, id, run), index_col=None)
            if iter_to[i] is None:
                iter_to[i] = len(df)
            iters = np.arange(iter_from[i], iter_to[i])
            plt.plot(
                iters, df["loss"].values[iter_from[i]:iter_to[i]:scaling[i]],
                label="{} loss".format(labels[i]), ls=linestyles[i])
            plt.plot(
                iters, df["regret"].values[iter_from[i]:iter_to[i]:scaling[i]],
                label="{} ".format(labels[i])+"$L^{\\star}$", ls=linestyles[i])
        plt.legend()
        plt.title("Loss & $L^{\\star}$")
        plt.xlabel("iterations")


    filename = "{}conv_analysis-ids-{}-{}.pdf".format(path, ids[0], ids[-1])
    plt.savefig(filename, **save_extras)
    plt.close(fig)

    if send:
        SBM.send_notification(
            text=None,
            text_for_files='convergence analysis',
            files=[filename],
            chat_id=CHAT_ID)


def plot_conv_analysis_multiruns(
        path=saved_models_path, ids=None, runs=np.arange(0, 10), scaling=1,
        iter_from=0, iter_to=None,
        labels=None, linestyles=None, colors=None, send=False,
        save_extras=None, **kwargs,
):
    """
    plot the convergence analysis of the ids

    Args:
        path: str, path to the folder containing the ids
        ids: list of int, the ids to plot
        runs: list of int, the runs to average over for plotting
        scaling: int or list of ints, scaling of the y-axis
        iter_from: int or list of ints, iteration from which to plot
        iter_to: int or list of ints, iteration to which to plot
        labels: list of str, labels for the legend
        linestyles: list of str, linestyles for the plot
        send: bool, whether to send the plot
        save_extras: dict, extra arguments for the fig saving function
    """
    file = "{}id-{}/run-{}.csv"

    prop_cycle = plt.rcParams['axes.prop_cycle']
    _colors = prop_cycle.by_key()['color']

    if colors is None:
        colors = np.arange(2*len(ids))
    if save_extras is None:
        save_extras = {}
    if linestyles is None:
        linestyles = ["-"] * len(ids)
    if isinstance(scaling, int):
        scaling = [scaling] * len(ids)
    if iter_from is None or isinstance(iter_from, int):
        iter_from = [iter_from] * len(ids)
    if iter_to is None or isinstance(iter_to, int):
        iter_to = [iter_to] * len(ids)

    fig = plt.figure()
    for i, id in enumerate(ids):
        losses = []
        regrets = []
        for run in runs:
            _df = pd.read_csv(file.format(path, id, run), index_col=None)
            losses.append(_df["loss"].values)
            regrets.append(_df["regret"].values)
        loss_mean = np.mean(losses, axis=0)
        loss_std = np.std(losses, axis=0)
        regret_mean = np.mean(regrets, axis=0)
        regret_std = np.std(regrets, axis=0)
        if iter_to[i] is None:
            iter_to[i] = len(loss_mean)
        iters = np.arange(iter_from[i], iter_to[i])
        errorbarplot(
            ax=plt.gca(), t=iters,
            mean=loss_mean[iter_from[i]:iter_to[i]:scaling[i]],
            yerr=loss_std[iter_from[i]:iter_to[i]:scaling[i]],
            label="{} loss".format(labels[i]), color=_colors[colors[2*i]],
            std_color_alpha=0.3, type="fill", ls=linestyles[i])
        errorbarplot(
            ax=plt.gca(), t=iters,
            mean=regret_mean[iter_from[i]:iter_to[i]:scaling[i]],
            yerr=regret_std[iter_from[i]:iter_to[i]:scaling[i]],
            label="{} ".format(labels[i])+"$\\mathcal{L}$",
            color=_colors[colors[2*i+1]], std_color_alpha=0.3, type="fill",
            ls=linestyles[i])
    plt.legend()
    plt.title("Loss & $\\mathcal{L}$")
    plt.xlabel("iterations")


    filename = "{}conv_analysis-ids-{}-{}.pdf".format(path, ids[0], ids[-1])
    plt.savefig(filename, **save_extras)
    plt.close(fig)

    if send:
        SBM.send_notification(
            text=None,
            text_for_files='convergence analysis',
            files=[filename],
            chat_id=CHAT_ID)


if __name__ == "__main__":
    pass


