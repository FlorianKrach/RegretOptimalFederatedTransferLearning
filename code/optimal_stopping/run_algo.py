# Lint as: python3
"""
Main module to run the optimal stopping algorithms.
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
import optimal_stopping.backward_induction_pricer as BIP
import optimal_stopping.payoff as payoffs
import optimal_stopping.stock_model as stock_model
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
_PAYOFFS = {
    "MaxPut": payoffs.MaxPut,
    "MaxCall": payoffs.MaxCall,
    "GeometricPut": payoffs.GeometricPut,
    "BasketCall": payoffs.BasketCall,
    "Identity": payoffs.Identity,
    "Max": payoffs.Max,
    "Mean": payoffs.Mean,
    "MinPut": payoffs.MinPut,
    "Put1Dim": payoffs.Put1Dim,
    "Call1Dim": payoffs.Call1Dim,
}

_STOCK_MODELS = stock_model.STOCK_MODELS

_ALGOS = {
    "RLSM": BIP.RLSM,
    "RLSMreinit": BIP.RLSMreinit,
    "RRLSM": BIP.RRLSM,
}



# ==============================================================================
# Global variables
CHAT_ID = configs.CHAT_ID
ERROR_CHAT_ID = configs.ERROR_CHAT_ID

data_path = configs.data_path
saved_models_path = configs.saved_models_path

ANOMALY_DETECTION = False
N_DATASET_WORKERS = 0
USE_GPU = False

# ==============================================================================
# Functions
makedirs = configs.makedirs



def run_algo(
        dataset_dicts, algo, opt_method_dict,
        nb_stocks, nb_dates, payoff, strike, maturity, spot, nb_paths=None,
        hidden_size=10, factors=(1.,1.,1.),
        use_path=False, use_payoff_as_input=True, algo_kwargs=None,
        train_eval_split=2, path_gen_seed=None,
        model_id=None, run=0, saved_models_path=saved_models_path,
        verbose=0, plot_convergence=False, send=False, **options):
    """
    This functions runs one algo for option pricing.

    Args:
        dataset_dicts: list of dicts with the dataset parameters
        algo: str, one of _ALGOS, the pricing algorithm
        opt_method_dict: dict, defining the optimizer method, i.e. the method to
            optimize the weights for all datasets (together); in particular
            needs the key 'method' which is one of BIP.OPT_METHODS, additionally
            needs to have all kwargs that the method needs
        nb_paths: int or None, number of paths for each dataset; for each
            dataset the number of paths can be provided in the dataset dict or
            via this argument; if both are provided, the value in the dataset
            dict is used; at least one of them needs to be provided
        nb_stocks: int, number of stocks for each dataset
        nb_dates: int, number of dates for each dataset
        payoff: str, one of _PAYOFFS, the payoff function
        strike: float, the strike of the option
        maturity: float, the maturity of the option
        spot: float, the spot of the option
        hidden_size: int, the hidden size of the neural network
        factors: tuple of floats, the factors for the model
        use_path: bool, whether to use the entire path as input or only data of
            current date; note that RRLSM is the preferred algorithm for
            non-Markovian data (which is used with use_path=False)
        use_payoff_as_input: bool, whether to use the payoff as additional input
            for the NN
        algo_kwargs: dict, additional kwargs for the NN, e.g.
            'act_fun' which should have a value in BIP.ACTIVATION_FUNCTIONS
        train_eval_split: int or list of ints, nb_paths//train_eval_split paths
            are used for training and the rest for evaluation; if list, then
            the list should have the same length as dataset_dicts and each
            element is used for the corresponding split of the dataset
        path_gen_seed: None or int, if not None, the seed for the path generator
        model_id: int, model id; this is used to separate different models
        run: int, run number; this is used to separate several runs of the same
            model configuration
        saved_models_path: str, path where to save the models
        verbose: int, verbosity level
        plot_convergence: bool, whether to plot the convergence of the algorithm
        send: bool, whether to send a telegram message with plot when done
        **options:

    Returns:
    """

    # get path and check whether exists
    path = os.path.join(saved_models_path, "id-{}".format(model_id))
    makedirs(path)
    fname = os.path.join(path, "run-{}.csv".format(run))
    plot_file_name = os.path.join(path, "run-{}-conv_plot.pdf".format(run))
    if os.path.exists(fname):
        print("File {} already exists -> skip".format(fname))
        return

    # set seed according to run number
    np.random.seed(run)
    torch.manual_seed(run)

    payoff_ = _PAYOFFS[payoff](strike)
    for dsd in dataset_dicts:
        if "nb_paths" not in dsd:
            dsd["nb_paths"] = nb_paths
    stockmodels = [_STOCK_MODELS[dsd["name"]](
        nb_stocks=nb_stocks, nb_dates=nb_dates,
        maturity=maturity, spot=spot, **dsd)
        for dsd in dataset_dicts]
    if algo_kwargs is None:
        algo_kwargs = {}
    if "act_fun" in algo_kwargs:
        algo_kwargs["activation_function"] = BIP.ACTIVATION_FUNCTIONS[
            algo_kwargs["act_fun"]]
    pricer = _ALGOS[algo](
        models=stockmodels, payoff=payoff_, use_path=use_path,
        use_payoff_as_input=use_payoff_as_input, hidden_size=hidden_size,
        factors=factors, verbose=verbose, **algo_kwargs)
    t_begin = time.time()
    prices, gen_time = pricer.price(
        train_eval_split=train_eval_split, optimizer_dict=opt_method_dict,
        plot_fname=plot_file_name if (plot_convergence and run == 0) else None,)
    duration = time.time() - t_begin
    comp_time = duration - gen_time

    # save results
    df = pd.DataFrame(data={
        "comp_time": [comp_time], "path_gen_time": [gen_time],
        "prices": [prices]})
    df.to_csv(fname, index=False)
    print_prices = "["+", ".join(["{:.4f}".format(p) for p in prices])+"]"
    opt_method = opt_method_dict["method"]
    if opt_method == "local_optimizer":
        opt_method = "{}-{}".format(
            opt_method, opt_method_dict["which_dataset"])
    print("comp_time: {:.4f}, gen_time: {:.4f}, prices: {}, opt-method: {}".format(
        comp_time, gen_time, print_prices, opt_method))

    if send and os.path.exists(plot_file_name):
        SBM.send_notification(
            text='convergence plot for model {} run {} with algo {} and '
                 'opt_method {}'.format(model_id, run, algo, opt_method),
            chat_id=configs.CHAT_ID,
            files=[plot_file_name],)

    return



