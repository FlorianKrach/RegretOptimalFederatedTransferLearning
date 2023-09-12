"""
author: Florian Krach
"""
import time

import numpy as np
import pandas as pd
import os
import copy
import scipy.optimize as opt
import torch.nn
import tqdm


def get_ridge_reg(featuremap, X, Y, ridge_coeff):
    """
    Calculate the ridge regression for the given data.
    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise, i.e. returns a np.array of shape
            (nb_samples, nb_features)
        X: np.array of shape (nb_samples, dim), the data
        Y: np.array of shape (nb_samples,), the labels
        ridge_coeff: float >=0, regularization parameter for the ridge
            regression

    Returns:
        params: np.array of shape (nb_features,), the optimal parameters
    """
    nb_samples, dim = X.shape
    features = featuremap(X)
    params = np.linalg.lstsq(
        features.T @ features + ridge_coeff * np.eye(dim), features.T @ Y,
        rcond=None)
    return params[0]


def evaluate_fKRR(featuremap, X, params):
    """
    Evaluate the fKRR regression for the given data.
    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise, i.e. returns a np.array of shape
            (nb_samples, nb_features)
        X: np.array of shape (nb_samples, dim), the data
        params: np.array of shape (nb_features, 1), the weights of the fRKR

    Returns:
        output: np.array of shape (nb_samples, 1), the output of the fRKR
    """
    features = featuremap(X)
    if len(params.shape) == 1:
        params = params.reshape(-1,)
    output = features @ params
    return output


def get_MSE(featuremap, X, Y, params):
    """
    Calculate the MSE of the fKRR regression for the given data.
    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise, i.e. returns a np.array of shape
            (nb_samples, nb_features)
        X: np.array of shape (nb_samples, dim), the data
        Y: np.array of shape (nb_samples,), the labels
        params: np.array of shape (nb_features, 1), the weights of the fRKR

    Returns:
        MSE: float >=0, the MSE of the fRKR regression
    """
    output = evaluate_fKRR(featuremap, X, params)
    MSE = np.mean((output - Y)**2)
    return MSE



def initialization_algo(
        featuremap, datasets, ridge_coeff=0., info_sharing_level=0.5):
    """
    Initialization for the regret-optimal transfer learning algorithm.
    corresponds to Algorithm 2 in the paper.

    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise (i.e. returns a np.array of shape
            (nb_samples, nb_features))
        datasets: list of datasets, each dataset is a tuple of X, Y; X is a
            np.array of shape (nb_samples, dim), Y is a np.array of shape
            (nb_samples,); the first dataset, i.e. datasets[0], is the main
            dataset and plays a special role in the algorithm
        ridge_coeff: float >=0, regularization parameter for the ridge
            regression
        info_sharing_level: float >=0, level of information sharing
            between the datasets

    Returns:
        weights: list of optimal weights for the different datasets
        params: list of locally optimal fRKR parameters for the different
            datasets
    """
    nb_datasets = len(datasets)
    eta = info_sharing_level
    scores = []
    params = []
    X0, Y0 = datasets[0]
    for i in range(nb_datasets):
        X, Y = datasets[i]
        params.append(get_ridge_reg(featuremap, X, Y, ridge_coeff))
        scores.append(get_MSE(featuremap, X0, Y0, params[i]))
    s0 = scores[0]
    scores = np.array(scores)
    w = np.maximum(0., s0+eta-scores)
    factors = np.exp(scores/eta)
    w_star = np.nan_to_num(factors*w, nan=0., posinf=0., neginf=0.)
    weights = w_star/np.sum(w_star)
    # print(scores, factors, w, w_star, weights)
    return weights, params


def equal_weights_initialization_algo(
        featuremap, datasets, ridge_coeff=0., **kwargs):
    _, params = initialization_algo(
        featuremap, datasets, ridge_coeff, info_sharing_level=1)
    N = len(datasets)
    weights = np.array([1/N]*N)
    return weights, params



INIT_ALGOS = {
    "standard_init": initialization_algo,
    "equal_weights_init": equal_weights_initialization_algo,
}


def regret_optimal_algo(
        featuremap, datasets, ridge_coeff=0., info_sharing_level=0.5, T=100,
        init_algo="standard_init", regret_coeff=None, beta=1., verbose=0,
        symmetric=False, *args, **kwargs):
    """
    this is the regret-optimal algorithm corresponding to Algorithm 1 and
    Theorem 1 of the paper.

    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise (i.e. returns a np.array of shape
            (nb_samples, nb_features))
        datasets: list of datasets, each dataset is a tuple of X, Y; X is a
            np.array of shape (nb_samples, dim), Y is a np.array of shape
            (nb_samples,); the first dataset, i.e. datasets[0], is the main
            dataset and plays a special role in the algorithm
        ridge_coeff: float >=0, regularization parameter for the ridge
            regression
        info_sharing_level: float >=0, level of information sharing
            between the datasets
        T: int, number of iterations
        init_algo: str, name of the initialization algorithm to use
        regret_coeff: float >=0, regularization parameter for the regret, if
            None defaults to ridge_coeff
        beta: float >=0, regularization parameter for the increments in regret
        symmetric: bool, if True, the algorithm is run in the symmetric version

    Returns:
        param_seq: list of np.arrays of shape (nb_features, 1), the sequence
            of parameters, list has length T+1
    """
    if regret_coeff is None:
        regret_coeff = ridge_coeff
    N = len(datasets)
    weights1, params = INIT_ALGOS[init_algo](
        featuremap=featuremap, datasets=datasets, ridge_coeff=ridge_coeff,
        info_sharing_level=info_sharing_level)
    if symmetric:
        weights = np.array([1 / N] * N)
    else:
        weights = weights1
    param_seq = [np.concatenate(params, axis=0)]
    features = [featuremap(X) for X, _ in datasets]
    feature_mats = [f.T @ f for f in features]
    feature_target_vec = [f.T @ Y for f, (_, Y) in zip(features, datasets)]
    mean_loc_opt = np.mean(params, axis=0)
    mean_loc_opt_N = np.concatenate([mean_loc_opt] * N, axis=0)
    if symmetric:
        ref_param = mean_loc_opt_N
    else:
        ref_param = param_seq[0]

    # get sequences of P and S
    p = params[0].shape[0]
    w_eye = np.concatenate([w*np.eye(p) for w in weights], axis=1)
    weighted_feature_mat = np.sum(
        [w*fm for w, fm in zip(weights, feature_mats)], axis=0)
    weighted_feature_target_vec = np.sum(
        [w*ftv for w, ftv in zip(weights, feature_target_vec)], axis=0)
    P_seq = [w_eye.T @ weighted_feature_mat @ w_eye]
    S_seq = [-w_eye.T @ weighted_feature_target_vec]
    inv_seq = []
    for t in tqdm.tqdm(range(T), disable=(not verbose)):
        inv = np.linalg.inv((regret_coeff+beta)*np.eye(N*p) + P_seq[-1])
        inv_seq.append(inv)
        P_seq.append(beta*np.eye(N*p) - (beta**2)*inv)
        S_seq.append(beta*inv @ (S_seq[-1] - regret_coeff*ref_param))
    inv_seq.append(None)
    P_seq = list(reversed(P_seq))
    S_seq = list(reversed(S_seq))
    inv_seq = list(reversed(inv_seq))

    # get sequence of params
    for t in tqdm.tqdm(range(T), disable=(not verbose)):
        M = (regret_coeff*np.eye(N*p)+P_seq[t+1]) @ param_seq[-1] - \
            regret_coeff*ref_param + S_seq[t+1]
        a = -inv_seq[t+1] @ M
        param_seq.append(param_seq[-1] + a)

    return param_seq, weights1


def symmetric_regret_optimal_algo(
        featuremap, datasets, ridge_coeff=0., info_sharing_level=0.5, T=100,
        init_algo="standard_init", regret_coeff=None, beta=1., verbose=0,
        *args, **kwargs):
    """
    this is the high-dim version of the accelerated regret-optimal algorithm,
    i.e. the high-dim version of Algorithm 3, which corresponds to equ. 68-70
    in the paper.

    same as regret_optimal_algo, but with symmetric=True
    """
    return regret_optimal_algo(
        featuremap=featuremap, datasets=datasets, ridge_coeff=ridge_coeff,
        info_sharing_level=info_sharing_level, T=T, init_algo=init_algo,
        regret_coeff=regret_coeff, beta=beta, verbose=verbose, symmetric=True,
        *args, **kwargs)


def accelerated_regret_optimal_algo(
        featuremap, datasets, ridge_coeff=0., info_sharing_level=0.5, T=100,
        init_algo="standard_init", option=3, regret_coeff=None, beta=1.,
        verbose=0, *args, **kwargs):
    """
    this is the low-dim efficient version of the regret_optimal_algo, i.e. the
    algorithm corresponding to Algorithm 1 of the paper.

    there are 4 options for the algorithm:
        1: use Theta(0)=Theta^star & Theta^star for computation of alpha
        2: use Theta(0)=Theta^star_{(N)} & Theta^star_{(N)} for computation of
            alpha (the fastest option)
        3: use Theta(0)=Theta^star & Theta^star_{(N)} for computation of alpha
            (the default option)
        4: use Theta(0)=Theta^star_{(N)} & Theta^star for computation of alpha
    where Theta^star is the locally optimal parameters and Theta^star_{(N)} is
    the mean of the locally optimal parameters.

    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise (i.e. returns a np.array of shape
            (nb_samples, nb_features))
        datasets: list of datasets, each dataset is a tuple of X, Y; X is a
            np.array of shape (nb_samples, dim), Y is a np.array of shape
            (nb_samples,); the first dataset, i.e. datasets[0], is the main
            dataset and plays a special role in the algorithm
        ridge_coeff: float >=0, regularization parameter for the ridge
            regression
        info_sharing_level: float >=0, level of information sharing
            between the datasets
        T: int, number of iterations
        init_algo: str, name of the initialization algorithm to use
        option: int, option for the algorithm
        regret_coeff: float >=0, regularization parameter for the regret, if
            None defaults to ridge_coeff
        beta: float >=0, regularization parameter for the increments in regret

    Returns:
        param_seq: list of np.arrays of shape (nb_features, 1), the sequence
            of parameters, list has length T+1
    """
    if regret_coeff is None:
        regret_coeff = ridge_coeff
    N = len(datasets)
    weights, params = INIT_ALGOS[init_algo](
        featuremap=featuremap, datasets=datasets, ridge_coeff=ridge_coeff,
        info_sharing_level=info_sharing_level)
    mean_param_loc_opt = np.mean(params, axis=0)
    if option in [1, 3]:
        params_seq = [np.array(params)]
    elif option in [2]:
        params_seq = [mean_param_loc_opt]
    elif option in [4]:
        params_seq = [np.array([mean_param_loc_opt] * N)]
    else:
        raise ValueError("option should be in [1, 2, 3, 4]")
    features = [featuremap(X) for X, _ in datasets]
    feature_mats = [f.T @ f for f in features]
    feature_target_vec = [f.T @ Y for f, (_, Y) in zip(features, datasets)]

    # get sequences of pis
    p = params[0].shape[0]
    piT = np.sum(feature_mats, axis=0)/(N**3)
    pis1 = [piT]
    pis2 = [piT]
    pis3 = [-np.sum(feature_target_vec, axis=0)/(N**2)]
    gammas1 = []
    gammas2 = []

    def gammas(pi1, pi2, pi3, numerically_stabilised=True):
        if numerically_stabilised:
            q1 = (regret_coeff + beta) * np.eye(p) + pi1
            q2 = pi2
            A = np.concatenate([q1, (N-1)*q2], axis=1)
            B = np.concatenate([q2, q1+(N-2)*q2], axis=1)
            C = np.concatenate([A, B], axis=0)
            D = np.concatenate([np.eye(p), np.zeros((p, p))], axis=0)
            G = np.linalg.lstsq(C, D, rcond=None)[0]
            gamma1, gamma2 = np.split(G, 2, axis=0)
            gamma3 = beta*(gamma1.T + (N - 1) * gamma2.T) @ (
                        pi3 - regret_coeff * mean_param_loc_opt)
            gamma1 = gamma1.T
            gamma2 = gamma2.T
        else:
            inv = np.linalg.inv(
                (regret_coeff + beta) * np.eye(p) + pi1 + (N - 2) * pi2)
            gamma1 = np.linalg.inv(
                (regret_coeff + beta) * np.eye(p) + pi1 - (N - 1) * pi2 @ inv @ pi2)
            gamma2 = -gamma1 @ pi2 @ inv
            gamma3 = beta*(gamma1 + (N - 1) * gamma2) @ (
                    pi3 - regret_coeff * mean_param_loc_opt)
        return gamma1, gamma2, gamma3

    for t in tqdm.tqdm(range(T), disable=(not verbose)):
        gamma1, gamma2, gamma3 = gammas(pis1[-1], pis2[-1], pis3[-1],
                                        numerically_stabilised=(t==0))
        pis1.append(beta*np.eye(p) - (beta**2)*gamma1)
        pis2.append(-(beta**2)*gamma2)
        pis3.append(gamma3)
        gammas1.append(gamma1)
        gammas2.append(gamma2)
    gammas1.append(None)
    gammas2.append(None)
    pis3 = list(reversed(pis3))
    pis2 = list(reversed(pis2))
    pis1 = list(reversed(pis1))
    gammas1 = list(reversed(gammas1))
    gammas2 = list(reversed(gammas2))

    def alpha(params, gamma1, gamma2, pi1, pi2, pi3):
        par_sum = np.sum(params, axis=0)
        # # other way to compute alpha
        # if option == 1:
        #     a = (ridge_coeff*np.eye(p)+pi1-pi2)
        #     b = pi2@par_sum
        #     v = [a@pa+b-ridge_coeff*l_op_pa+pi3 for pa, l_op_pa in zip(
        #         params, params_seq[0])]
        #     v_sum = np.sum(v, axis=0)
        #     A = np.array([-(gamma1-gamma2)@vi-gamma2@v_sum for vi in v])
        #     return A
        a = -(gamma1 - gamma2) @ (regret_coeff * np.eye(p) + pi1 - pi2)
        b = -(gamma1 @ pi2 + gamma2 @ (
                    regret_coeff * np.eye(p) + pi1 + (N - 2) * pi2))
        ctilde = -(gamma1 + (N - 1) * gamma2)
        if option in [1, 4]:
            btilde = b @ par_sum
            A = np.array(
                [a @ pa + btilde + ctilde @ (pi3 - regret_coeff * l_op_pa)
                 for pa, l_op_pa in zip(params, params_seq[0])])
            return A
        c = ctilde @ (pi3 - regret_coeff * mean_param_loc_opt)
        if option == 2:
            A = (a + b*N) @ params + c
            return A
        if option == 3:
            btilde = b @ par_sum
            A = np.array([a@pa+btilde+c for pa in params])
            return A

    # get sequence of params
    alphas = []
    for t in tqdm.tqdm(range(T), disable=(not verbose)):
        a = alpha(params_seq[-1], gammas1[t+1], gammas2[t+1],
                  pis1[t+1], pis2[t+1], pis3[t+1])
        alphas.append(a)
        params_seq.append(params_seq[-1] + a)

    return params_seq, weights


def get_final_params_wrapper(param_seq_extractor_func, is_stacked=True):
    """
    wrapper function to extract the final params from the sequence of params and
    the weights return by the param_seq_extractor_func

    Args:
        param_seq_extractor_func: a function
        is_stacked: bool, whether the params are stacked or not (i.e. whether
            for each index t of the sequence of params, the params param_seq[t]
            are a list of N elements of p-dim vectors or a stacked array of dim
            N*p). I.e. if is_stacked is True, then param_seq[t] is of dimension
            (T, N*p), otherwise it is of dimension (T, N, p)

    Returns:
        the final combined params and the sequence of params
    """
    def func(*args, **kwargs):
        param_seq, weights = param_seq_extractor_func(*args, **kwargs)
        last_param = param_seq[-1]
        if is_stacked:
            split_params = np.split(last_param, len(weights), axis=0)
        elif "option" in kwargs and kwargs["option"] == 2:
            return last_param, param_seq
        else:
            split_params = list(last_param)
        return np.sum([w*p for w, p in zip(weights, split_params)], axis=0), \
               param_seq
    return func


def get_stacked_params_wrapper(param_seq_extractor_func):
    """
    wrapper function to get stacked params for each element in the sequence of
    params

    Args:
        param_seq_extractor_func: a function

    Returns:
        the sequence of stacked params and the weights
    """
    def func(*args, **kwargs):
        param_seq, weights = param_seq_extractor_func(*args, **kwargs)
        for i in range(len(param_seq)):
            # if "option" in kwargs and kwargs["option"] in [2, 4]:
            #     raise ValueError(
            #         "option 2 and 4 not supported, since they do not start at "
            #         "locally optimal params")
            # else:
            #     param_seq[i] = np.concatenate(list(param_seq[i]))
            if "option" in kwargs and kwargs["option"] == 2:
                param_seq[i] = np.concatenate([param_seq[i]]*len(weights))
            else:
                param_seq[i] = np.concatenate(list(param_seq[i]))
        return param_seq, weights
    return func


def get_optimal_params_dataset_i(
        featuremap, datasets, ridge_coeff=0., which_dataset=0, *args, **kwargs):
    """
    get the optimal params for the dataset which_dataset

    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise (i.e. returns a np.array of shape
            (nb_samples, nb_features))
        datasets: list of datasets, each dataset is a tuple of X, Y; X is a
            np.array of shape (nb_samples, dim), Y is a np.array of shape
            (nb_samples,); the first dataset, i.e. datasets[0], is the main
            dataset and plays a special role in the algorithm
        ridge_coeff: float >=0, regularization parameter for the ridge
            regression
        which_dataset: int, index of the dataset for which we want to compute
            the optimal params
        *args:
        **kwargs:

    Returns: the optimal params for the dataset which_dataset

    """
    X, Y = datasets[which_dataset]
    return get_ridge_reg(featuremap, X, Y, ridge_coeff), None


def get_jointly_optimal_params(
        featuremap, datasets, ridge_coeff=0., *args, **kwargs):
    """
    get the optimal params for the joint dataset (i.e. the dataset obtained by
    concatenating all the datasets to one big dataset)

    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise (i.e. returns a np.array of shape
            (nb_samples, nb_features))
        datasets: list of datasets, each dataset is a tuple of X, Y; X is a
            np.array of shape (nb_samples, dim), Y is a np.array of shape
            (nb_samples,); the first dataset, i.e. datasets[0], is the main
            dataset and plays a special role in the algorithm
        ridge_coeff: float >=0, regularization parameter for the ridge
            regression
        *args:
        **kwargs:

    Returns: the optimal params for the joint dataset

    """
    return get_ridge_reg(
        featuremap, np.concatenate([X for X, _ in datasets], axis=0),
        np.concatenate([Y for _, Y in datasets], axis=0), ridge_coeff), None


def get_mean_locally_optimal_params(
        featuremap, datasets, ridge_coeff=0., *args, **kwargs):
    """
    get the mean of the locally optimal params for each dataset

    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise (i.e. returns a np.array of shape
            (nb_samples, nb_features))
        datasets: list of datasets, each dataset is a tuple of X, Y; X is a
            np.array of shape (nb_samples, dim), Y is a np.array of shape
            (nb_samples,); the first dataset, i.e. datasets[0], is the main
            dataset and plays a special role in the algorithm
        ridge_coeff: float >=0, regularization parameter for the ridge
            regression
        *args:
        **kwargs:

    Returns: the mean of the locally optimal params for each dataset

    """
    return np.mean(
        [get_ridge_reg(featuremap, X, Y, ridge_coeff)
         for X, Y in datasets], axis=0), None


class GDParamOptModule(torch.nn.Module):
    def __init__(self, featuremap, datasets, ridge_coeff, info_sharing_level,
                 init_algo,):
        super(GDParamOptModule, self).__init__()
        weights, params = INIT_ALGOS[init_algo](
            featuremap=featuremap, datasets=datasets, ridge_coeff=ridge_coeff,
            info_sharing_level=info_sharing_level)
        self.params = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.from_numpy(p)) for p in params])
        self.weights = weights
        self.ridge_coeff = ridge_coeff
        self.features = [torch.from_numpy(featuremap(X)) for X, _ in datasets]
        self.targets = [torch.from_numpy(Y) for _, Y in datasets]

    def forward(self):
        weighted_params = torch.sum(torch.stack(
            [w*p for w, p in zip(self.weights, self.params)]), dim=0)
        loss = torch.sum(torch.cat(
            [w*torch.sum(
                (torch.matmul(X, weighted_params) - Y)**2).reshape(1)
             for w, X, Y in zip(self.weights, self.features, self.targets)]))
        return loss

    def get_params_and_weights(self):
        return np.concatenate(
            [p.detach().numpy() for p in self.params], axis=0), self.weights


def GD_param_seq_extractor(
        featuremap, datasets, ridge_coeff=0., learning_rate=0.1, T=100,
        info_sharing_level=0.5, init_algo="standard_init",
        *args, **kwargs):
    """
    get the sequence of params obtained by running gradient descent on the
    aggregated terminal-time loss function

    Args:
        featuremap: function, mapping the inputs to the features used in
            the regression; the function takes as input the data X and is
            supposed to work batchwise (i.e. returns a np.array of shape
            (nb_samples, nb_features))
        datasets: list of datasets, each dataset is a tuple of X, Y; X is a
            np.array of shape (nb_samples, dim), Y is a np.array of shape
            (nb_samples,); the first dataset, i.e. datasets[0], is the main
            dataset and plays a special role in the algorithm
        ridge_coeff: float >=0, regularization parameter for the ridge
            regression
        learning_rate: float, learning rate for the gradient descent
        T: int, number of iterations of the gradient descent
        info_sharing_level: float >=0, level of information sharing
            between the datasets
        init_algo: str, initialization algorithm for the weights
        *args:
        **kwargs:

    Returns: the sequence of params obtained by running gradient descent on the
        dataset which_dataset
    """
    N = len(datasets)
    module = GDParamOptModule(
        featuremap=featuremap, datasets=datasets, ridge_coeff=ridge_coeff,
        info_sharing_level=info_sharing_level, init_algo=init_algo,)
    params, weights = module.get_params_and_weights()
    param_seq = [params]
    print(module)
    optimizer = torch.optim.SGD(
        module.parameters(), lr=learning_rate, weight_decay=0., momentum=0.,
        nesterov=False, maximize=False)

    for t in tqdm.tqdm(range(T)):
        optimizer.zero_grad()
        loss = module()
        loss.backward()
        optimizer.step()
        params, _ = module.get_params_and_weights()
        param_seq.append(params)

    return param_seq, weights








if __name__ == '__main__':
    pass

