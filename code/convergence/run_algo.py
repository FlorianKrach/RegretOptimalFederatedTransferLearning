"""
author: Florian Krach
"""
import os, sys, socket

import csv
import random
import time
import matplotlib
import pandas as pd
import numpy as np
import torch

sys.path.append("../")
import convergence.data_generator as data_generator
import optimal_stopping.utils.randomized_neural_networks as \
    randomized_neural_networks
import TLalgo.algorithms as TLalgos
import configs

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    SBM = configs.SendBotMessage()


# ==============================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    N_CPUS = 1
    SEND = False
else:
    SERVER = True
    N_CPUS = 1
    SEND = True
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
print(socket.gethostname())
print('SERVER={}'.format(SERVER))



# ==============================================================================
# Global variables
CHAT_ID = configs.CHAT_ID
ERROR_CHAT_ID = configs.ERROR_CHAT_ID

data_path = configs.data_path
saved_models_path = "{}convergence/saved_models/".format(data_path)

ANOMALY_DETECTION = False
N_DATASET_WORKERS = 0
USE_GPU = False

OPT_METHODS = {
    "regret_optimal_algo": TLalgos.regret_optimal_algo,
    "efficient_regret_optimal_algo": TLalgos.get_stacked_params_wrapper(
        TLalgos.accelerated_regret_optimal_algo),
    "accelerated_regret_optimal_algo": TLalgos.get_stacked_params_wrapper(
        TLalgos.accelerated_regret_optimal_algo),
    "symmetric_regret_optimal_algo": TLalgos.symmetric_regret_optimal_algo,
    "gradient_descent": TLalgos.GD_param_seq_extractor,
}
# ==============================================================================
# Functions
makedirs = configs.makedirs


def get_features(feature_map, X):
    X = torch.from_numpy(X)
    X = X.type(torch.float32)
    features = np.concatenate(
        [feature_map(X).detach().numpy(), np.ones((len(X), 1))], axis=1)
    return features


def get_loss(dataset, params, weights):
    N = len(weights)
    params = np.split(params, N, axis=0)
    weighted_params = np.sum(np.stack(
        [w * p for w, p in zip(weights, params)]), axis=0)
    loss = np.sum([w*np.sum((X@weighted_params - Y)**2)
                   for w, (X, Y) in zip(weights, dataset)])
    return loss


def get_param_regrets(param_seq, regret_coeff, beta=1.):
    param_regrets = [0.]
    for i in range(1, len(param_seq)):
        param_regrets.append(
            regret_coeff*np.sum((param_seq[i] - param_seq[0])**2) +
            beta*np.sum((param_seq[i] - param_seq[i-1])**2))
    return np.array(param_regrets)


def get_loss_and_regret(dataset, params_seq, weights, regret_coeff, beta=1.):
    param_regrets = get_param_regrets(params_seq, regret_coeff, beta=beta)
    losses = np.array(
        [get_loss(dataset, params, weights) for params in params_seq])
    regrets = np.cumsum(param_regrets) + losses
    return losses, regrets



def run_algo(
        dataset_dicts, opt_method_dict,
        input_size, act_fun,
        hidden_size=10, factors=(1.,1.,1.),
        train_eval_split=2, compare_on_train=False,
        model_id=None, run=0, saved_models_path=saved_models_path,
        verbose=0, plot_convergence=False, send=False, **options):
    """
    This function runs one algo for the convergence analysis

    Args:
        dataset_dicts:
        opt_method_dict:
        input_size:
        act_fun:
        hidden_size:
        factors:
        train_eval_split:
        compare_on_train:
        model_id:
        run:
        saved_models_path:
        verbose:
        plot_convergence:
        send:
        **options:

    Returns:

    """

    # get path and check whether exists
    path = os.path.join(saved_models_path, "id-{}".format(model_id))
    makedirs(path)
    fname = os.path.join(path, "run-{}.csv".format(run))
    if os.path.exists(fname):
        print("File {} already exists -> skip".format(fname))
        return

    # set seed according to run number
    np.random.seed(run)
    torch.manual_seed(run)

    # get datasets and features
    N = len(dataset_dicts)
    datasets = [data_generator.DATA_GENERATORS[dsd["name"]](
        input_size=input_size, **dsd).gen_samples() for dsd in dataset_dicts]
    randNN = randomized_neural_networks.Reservoir2(
        state_size=input_size, hidden_size=hidden_size, factors=factors,
        activation=data_generator.ACTIVATION_FUNCTIONS[act_fun],)
    if isinstance(train_eval_split, int):
      train_eval_split = [train_eval_split] * N
    splits = [dsd["n_samples"] // tes for dsd, tes in zip(
      dataset_dicts, train_eval_split)]
    features = [get_features(randNN, X) for X, _ in datasets]
    targets = [Y for _, Y in datasets]
    training_datasets = [(f[:s], t[:s]) for f, t, s in zip(
        features, targets, splits)]
    evaluation_datasets = [(f[s:], t[s:]) for f, t, s in zip(
        features, targets, splits)]

    # get optimizer
    opt_method = OPT_METHODS[opt_method_dict["method"]]
    param_seq, weights = opt_method(
        # we use the identity as feature map, since the features are already
        #   computed and stored in X of the dataset
        datasets=training_datasets, featuremap=lambda x: x, **opt_method_dict,
        verbose=verbose)

    # compute loss and regret
    if compare_on_train:
        loss_dataset = training_datasets
    else:
        loss_dataset = evaluation_datasets
    if "regret_coeff" not in opt_method_dict:
        opt_method_dict["regret_coeff"] = opt_method_dict["ridge_coeff"]
    if "beta" not in opt_method_dict:
        opt_method_dict["beta"] = 1.
    losses, regrets = get_loss_and_regret(
        dataset=loss_dataset, params_seq=param_seq, weights=weights,
        regret_coeff=opt_method_dict["regret_coeff"],
        beta=opt_method_dict["beta"])

    # save results
    df = pd.DataFrame({
        "iter": range(len(losses)), "loss": losses, "regret": regrets,
        "param_seq": param_seq})
    df.to_csv(fname, index=False)

    # plot convergence
    if plot_convergence:
        plot_fname = os.path.join(path, "run-{}.png".format(run))
        plt.figure()
        plt.plot(range(len(losses)), losses, label="loss")
        plt.plot(range(len(losses)), regrets, label="regret")
        plt.legend()
        plt.savefig(plot_fname)
        plt.close()

        if send:
            SBM.send_notification(
                text='convergence analysis plot for model {} run {} with '
                     'opt_method {}'.format(model_id, run, opt_method),
                chat_id=configs.CHAT_ID,
                files=[plot_fname], )

    return




if __name__ == '__main__':
    pass
