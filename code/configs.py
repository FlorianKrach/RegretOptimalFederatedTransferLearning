"""
author: Florian Krach
"""

# IMPORTS
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid



# GLOBAL CLASSES
class Seed:
    def __init__(self):
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed


class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, *args, **kwargs):
        print(text)


# GLOBAL FUNCTIONS
def get_parameter_array(param_dict):
    """
    helper function to get a list of parameter-list with all combinations of
    parameters specified in a parameter-dict

    :param param_dict: dict with parameters
    :return: 2d-array with the parameter combinations
    """
    param_combs_dict_list = list(ParameterGrid(param_dict))
    return param_combs_dict_list


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


# GLOBAL VARIABLES
path_gen_seed = Seed()
data_path = "data/"
saved_models_path = '{}saved_models/'.format(data_path)

CHAT_ID = -967135163
ERROR_CHAT_ID = -629710899


# ==============================================================================
# PARAMETERS CONFIGURATIONS -- OPTIMAL STOPPING
# ==============================================================================
dataset0 = dict(
    name='BlackScholes',
    drift=0.05, volatility=0.2, dividend=0.1,
)
dataset1 = dict(
    name='BlackScholes',
    drift=0.02, volatility=0.2, dividend=0.1,
)
dataset2 = dict(
    name='BlackScholes',
    drift=0.08, volatility=0.2, dividend=0.1,
)

params_dict0 = dict(
    dataset_dicts=[[
        dataset0,
        dataset1,
        dataset2
    ]],
    algo=["RLSM"],
    opt_method_dict=[
        # dict(method="joint_optimizer", ridge_coeff=0.),
        # dict(method="local_optimizer", ridge_coeff=0., which_dataset=0),
        # dict(method="local_optimizer", ridge_coeff=0., which_dataset=1),
        # dict(method="local_optimizer", ridge_coeff=0., which_dataset=2),
        # dict(method="mean_local_optimizers", ridge_coeff=0.),
        # dict(method="regret_optimal_algo", ridge_coeff=0.,
        #      info_sharing_level=1, T=50,),
        # dict(method="regret_optimal_algo", ridge_coeff=0.,
        #      T=50, init_algo="equal_weights_init"),
        dict(method="efficient_regret_optimal_algo", ridge_coeff=0.,
             info_sharing_level=1, T=50,),
        # dict(method="symmetric_regret_optimal_algo", ridge_coeff=0.,
        #      info_sharing_level=1, T=50,),
    ],
    nb_paths=[100000],
    nb_stocks=[2],
    nb_dates=[9],
    payoff=["MaxCall"],
    strike=[100.],
    maturity=[3.],
    spot=[100.],
    hidden_size=[30],
    factors=[(1.,1.,1.)],
    use_path=[False],
    use_payoff_as_input=[True],
    algo_kwargs=[dict(act_fun="leaky_relu")],
    train_eval_split=[2],
    path_gen_seed=[None],
    plot_convergence=[True],
    saved_models_path=[saved_models_path],
)
params_list0 = get_parameter_array(param_dict=params_dict0)

GTO_0 = dict(
    path=saved_models_path,
    params_extract_desc=[
        "algo", "payoff", "strike", "maturity", "spot", "hidden_size",
        "nb_paths", "nb_stocks", "nb_dates", "opt_method_dict",
        "dict-opt_method_dict-method",
        "dict-opt_method_dict-which_dataset",
    ],
    sortby="dict-opt_method_dict-method",
)


# ------------------------------------------------------------------------------
# -- different nb training samples --> in paper: table 2
path6 = "{}saved_models_RHeston6/".format(data_path)

dataset_6_0 = {
    "correlation": -0.3, "dividend": 0.1, "drift": 0.05, "hurst": 0.1,
    "mean": 0.01, "name": "RoughHeston", "nb_paths": 50100, "nb_steps_mult": 10,
    "speed": 2.0, "volatility": 0.2}
dataset_6_1 = {
    "correlation": -0.3, "dividend": 0.1, "drift": 0.5, "mean": 0.01,
    "name": "Heston", "nb_paths": 100000, "speed": 2.0, "volatility": 0.2}
dataset_6_2 = {
    "correlation": -0.3, "dividend": 0.1, "drift": 0.05,
    "mean": 0.01, "name": "Heston", "nb_paths": 50100,
    "speed": 2.0, "volatility": 0.2}
rc = 2.

params_dict6 = dict(
    dataset_dicts=[[
        dataset_6_0,
        dataset_6_1,
        dataset_6_2,
    ]],
    algo=["RLSM"],
    opt_method_dict=[
        dict(method="local_optimizer", ridge_coeff=rc, which_dataset=0),
        dict(method="local_optimizer", ridge_coeff=rc, which_dataset=1),
        dict(method="local_optimizer", ridge_coeff=rc, which_dataset=2),
        dict(method="joint_optimizer", ridge_coeff=rc),
        dict(method="mean_local_optimizers", ridge_coeff=rc),
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             T=1, init_algo="equal_weights_init"),
        dict(method="efficient_regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=1, T=1, option=2),
        dict(method="symmetric_regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=1, T=1, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=50, T=1, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=100, T=1, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc, regret_coeff=3.,
             beta=0., info_sharing_level=100, T=1, init_algo="wasserstein_init",
             gamma=1e-5, init_regularized=False),
        dict(method="regret_optimal_algo", ridge_coeff=rc, regret_coeff=3.,
             beta=0., info_sharing_level=100, T=1, init_algo="wasserstein_init",
             gamma=1e-2, init_regularized=False),
        dict(method="regret_optimal_algo", ridge_coeff=rc, regret_coeff=3.,
             beta=0., info_sharing_level=100, T=1, init_algo="wasserstein_init",
             gamma=1e-1, init_regularized=False),
    ],
    nb_stocks=[2],
    nb_dates=[9],
    payoff=["MaxCall"],
    strike=[100.],
    maturity=[3.],
    spot=[100.],
    hidden_size=[300],
    factors=[(1.,1.,1.)],
    use_path=[False],
    use_payoff_as_input=[True],
    algo_kwargs=[dict(act_fun="leaky_relu")],
    train_eval_split=[[501, 2, 501]],
    path_gen_seed=[None],
    saved_models_path=[path6],
)
params_list6 = get_parameter_array(param_dict=params_dict6)

GTO_6 = dict(
    path=path6,
    params_extract_desc=[
        "algo", "payoff", "strike", "maturity", "spot", "hidden_size",
        "nb_paths", "nb_stocks", "nb_dates", "opt_method_dict",
        "dict-opt_method_dict-method",
        "dict-opt_method_dict-which_dataset",
        "dict-opt_method_dict-option",
        "dict-opt_method_dict-info_sharing_level",
        "dict-opt_method_dict-gamma",
        "dict-opt_method_dict-init_algo",
    ],
    sortby=["dict-opt_method_dict-method",
            "dict-opt_method_dict-info_sharing_level",
            "dict-opt_method_dict-gamma",
            "dict-opt_method_dict-option", ]
)


# ---------------------
# -- 100 training samples each --> in paper: table 1
path7_1 = "{}saved_models_RHeston7_1/".format(data_path)

rc = 2.
datasets_7_1 = [
    {"correlation": -0.3, "dividend": 0.1, "drift": 0.05,
     "mean": 0.01, "name": "Heston", "nb_paths": 50100,
     "speed": 2.0, "volatility": 0.2}]
for drift in [0.05, 0.5]:
    for vol in [0.15, 0.2, 0.25]:
        for mean in [0.005, 0.015]:
            datasets_7_1.append(
                {"correlation": -0.3, "dividend": 0.1, "drift": drift,
                 "mean": mean, "name": "Heston", "nb_paths": 50100,
                 "speed": 2.0, "volatility": vol})
loc_optims = []
for i in range(len(datasets_7_1)):
    loc_optims.append(
        dict(method="local_optimizer", ridge_coeff=rc, which_dataset=i))


params_dict7_1 = dict(
    dataset_dicts=[datasets_7_1],
    algo=["RLSM"],
    opt_method_dict=
        loc_optims+[
        dict(method="joint_optimizer", ridge_coeff=rc),
        dict(method="mean_local_optimizers", ridge_coeff=rc),
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             T=1, init_algo="equal_weights_init"),
        dict(method="efficient_regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=1, T=1, option=2),
        dict(method="symmetric_regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=1, T=1, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=100, T=1, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=500, T=1, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=100, T=0, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc, regret_coeff=0.,
             info_sharing_level=100, T=10, ),
        dict(method="regret_optimal_algo", ridge_coeff=rc, regret_coeff=3.,
             beta=0., info_sharing_level=100, T=1, init_algo="wasserstein_init",
             gamma=1e-5, init_regularized=False),
        dict(method="regret_optimal_algo", ridge_coeff=rc, regret_coeff=3.,
             beta=0., info_sharing_level=100, T=1, init_algo="wasserstein_init",
             gamma=1e-2, init_regularized=False),
        dict(method="regret_optimal_algo", ridge_coeff=rc, regret_coeff=3.,
             beta=0., info_sharing_level=100, T=1, init_algo="wasserstein_init",
             gamma=1e-1, init_regularized=False),
    ],
    nb_stocks=[2],
    nb_dates=[9],
    payoff=["MaxCall"],
    strike=[100.],
    maturity=[3.],
    spot=[100.],
    hidden_size=[300],
    factors=[(1.,1.,1.)],
    use_path=[False],
    use_payoff_as_input=[True],
    algo_kwargs=[dict(act_fun="leaky_relu")],
    train_eval_split=[[501]*len(datasets_7_1)],
    path_gen_seed=[None],
    plot_convergence=[True],
    saved_models_path=[path7_1],
)
params_list7_1 = get_parameter_array(param_dict=params_dict7_1)

GTO_7_1 = dict(
    path=path7_1,
    params_extract_desc=[
        "algo", "payoff", "strike", "maturity", "spot", "hidden_size",
        "nb_paths", "nb_stocks", "nb_dates", "opt_method_dict",
        "dict-opt_method_dict-method",
        "dict-opt_method_dict-which_dataset",
        "dict-opt_method_dict-option",
        "dict-opt_method_dict-info_sharing_level",
        "dict-opt_method_dict-gamma",
        "dict-opt_method_dict-init_algo",
    ],
    sortby=["dict-opt_method_dict-method",
            "dict-opt_method_dict-info_sharing_level",
            "dict-opt_method_dict-gamma",
            "dict-opt_method_dict-option", ]
)

# ---------------------
# -- reference values for experiments of params_list7_1
path7_2 = "{}saved_models_RHeston7_2/".format(data_path)

rc = 2.
datasets_7_2 = [
    {"correlation": -0.3, "dividend": 0.1, "drift": 0.05,
     "mean": 0.01, "name": "Heston", "nb_paths": 50100,
     "speed": 2.0, "volatility": 0.2},
    {"correlation": -0.3, "dividend": 0.1, "drift": 0.05,
     "mean": 0.01, "name": "Heston", "nb_paths": 100000,
     "speed": 2.0, "volatility": 0.2},
    {"correlation": -0.3, "dividend": 0.1, "drift": 0.05,
     "mean": 0.01, "name": "Heston", "nb_paths": 700*73,
     "speed": 2.0, "volatility": 0.2},
]

params_dict7_2 = dict(
    dataset_dicts=[datasets_7_2],
    algo=["RLSM"],
    opt_method_dict=[
        dict(method="local_optimizer", ridge_coeff=rc, which_dataset=0),
        dict(method="local_optimizer", ridge_coeff=rc, which_dataset=1),
        dict(method="local_optimizer", ridge_coeff=rc, which_dataset=2),
    ],
    nb_stocks=[2],
    nb_dates=[9],
    payoff=["MaxCall"],
    strike=[100.],
    maturity=[3.],
    spot=[100.],
    hidden_size=[300],
    factors=[(1.,1.,1.)],
    use_path=[False],
    use_payoff_as_input=[True],
    algo_kwargs=[dict(act_fun="leaky_relu")],
    train_eval_split=[[501, 2, 73]],
    path_gen_seed=[None],
    saved_models_path=[path7_2],
)
params_list7_2 = get_parameter_array(param_dict=params_dict7_2)

params_dict7_2_1 = dict(
    dataset_dicts=[datasets_7_1[:7]],
    algo=["RLSM"],
    opt_method_dict=[
        dict(method="joint_optimizer", ridge_coeff=rc),
    ],
    nb_stocks=[2],
    nb_dates=[9],
    payoff=["MaxCall"],
    strike=[100.],
    maturity=[3.],
    spot=[100.],
    hidden_size=[300],
    factors=[(1.,1.,1.)],
    use_path=[False],
    use_payoff_as_input=[True],
    algo_kwargs=[dict(act_fun="leaky_relu")],
    train_eval_split=[[501]*7],
    path_gen_seed=[None],
    saved_models_path=[path7_2],
)
params_list7_2 += get_parameter_array(param_dict=params_dict7_2_1)

GTO_7_2 = dict(
    path=path7_2,
    params_extract_desc=[
        "algo", "payoff", "strike", "maturity", "spot", "hidden_size",
        "nb_paths", "nb_stocks", "nb_dates", "opt_method_dict",
        "dict-opt_method_dict-method",
        "dict-opt_method_dict-which_dataset",
        "dict-opt_method_dict-option",
        "dict-opt_method_dict-info_sharing_level",
        "dict-opt_method_dict-gamma",
        "dict-opt_method_dict-init_algo",
    ],
    sortby=["dict-opt_method_dict-method",
            "dict-opt_method_dict-info_sharing_level",
            "dict-opt_method_dict-gamma",
            "dict-opt_method_dict-option", ]
)



# ==============================================================================
# PARAMETER COMBINATIONS -- CONVERGENCE ANALYSIS
# ==============================================================================
# ---------------------
# -- convergence analysis -> in paper: Figure 2
conv_path_2 = "{}convergence/saved_models2/".format(data_path)

rc = 0.
datasets_2 = [
    {"name": "RandomizedNNDataset", "n_samples": 100, "n_features": 10,
     "activation": "relu"}]*5

params_dict = dict(
    experiment=["convergence"],
    dataset_dicts=[datasets_2],
    opt_method_dict=[
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1000, regret_coeff=0.),
        dict(method="efficient_regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1000, option=1, regret_coeff=0.),
        dict(method="gradient_descent", ridge_coeff=rc,
             info_sharing_level=10, T=100000, learning_rate=1e-6, regret_coeff=0.),
        dict(method="gradient_descent", ridge_coeff=rc,
             info_sharing_level=10, T=100000, learning_rate=7e-5, regret_coeff=0.),
    ],
    hidden_size=[500],
    factors=[(1.,1.,1.)],
    input_size=[5],
    act_fun=["relu"],
    train_eval_split=[1],
    compare_on_train=[True],
    plot_convergence=[True],
    saved_models_path=[conv_path_2],
)
conv_params_list_2 = get_parameter_array(param_dict=params_dict)

conv_plot_2 = dict(
    which_plot="convergence_multi_runs",
    path=conv_path_2,
    ids=[1, 2, 4],
    runs=np.arange(0, 10),
    scaling=[1,1,1],
    linestyles=["-",  "-.", "--", ],
    labels=["RO", "ARO", "GD",],
    use_subplots=False,
    save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
)

conv_plot_2_1 = dict(
    which_plot="convergence_multi_runs",
    path=conv_path_2,
    ids=[1, 2, 4],
    runs=np.arange(0, 10),
    iter_to=1000,
    scaling=[1,1,1],
    linestyles=["-", "-.", "--", ],
    labels=["RO", "ARO", "GD", ],
    use_subplots=False,
    save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
)

conv_plot_2_2 = dict(
    which_plot="convergence_multi_runs",
    path=conv_path_2,
    ids=[1, 2,],
    runs=np.arange(0, 10),
    iter_from=500,
    iter_to=1000,
    scaling=[1,1],
    linestyles=["-",  "-.",],
    labels=["RO", "ARO"],
    use_subplots=False,
    save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
)


# -- convergence analysis 2
conv_path_3 = "{}convergence/saved_models3/".format(data_path)

rc = 0.
regcoeff = 1e-4

params_dict3 = dict(
    experiment=["convergence"],
    dataset_dicts=[datasets_2],
    opt_method_dict=[
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1000, regret_coeff=regcoeff),
        dict(method="efficient_regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1000, option=1, regret_coeff=regcoeff),
    ],
    hidden_size=[500],
    factors=[(1.,1.,1.)],
    input_size=[5],
    act_fun=["relu"],
    train_eval_split=[1],
    compare_on_train=[True],
    plot_convergence=[True],
    saved_models_path=[conv_path_3],
)
conv_params_list_3 = get_parameter_array(param_dict=params_dict3)

conv_plot_3 = dict(
    which_plot="convergence_multi_runs",
    path=conv_path_3,
    ids=[1, 2,],
    runs=np.arange(0, 10),
    scaling=[1,1],
    linestyles=["-",  "-.",],
    labels=["RO", "ARO"],
    use_subplots=False,
    save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
)

# -- convergence analysis 3
conv_path_4 = "{}convergence/saved_models4/".format(data_path)

rc = 0.
regcoeff = 2

params_dict4 = dict(
    experiment=["convergence"],
    dataset_dicts=[datasets_2],
    opt_method_dict=[
        dict(method="regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1000, regret_coeff=regcoeff),
        dict(method="efficient_regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1000, option=1, regret_coeff=regcoeff),
        dict(method="efficient_regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1000, option=3, regret_coeff=regcoeff),
        dict(method="efficient_regret_optimal_algo", ridge_coeff=rc,
             info_sharing_level=10, T=1000, option=2, regret_coeff=regcoeff),
    ],
    hidden_size=[500],
    factors=[(1.,1.,1.)],
    input_size=[5],
    act_fun=["relu"],
    train_eval_split=[1],
    compare_on_train=[True],
    plot_convergence=[True],
    saved_models_path=[conv_path_4],
)
conv_params_list_4 = get_parameter_array(param_dict=params_dict4)

conv_plot_4 = dict(
    which_plot="convergence_multi_runs",
    path=conv_path_4,
    ids=[1, 4,],
    runs=np.arange(0, 10),
    scaling=[1,1],
    linestyles=["-",  "-.",],
    labels=["RO", "ARO"],
    use_subplots=False,
    save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
)



