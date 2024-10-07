"""
This code is based on code from https://github.com/HeKrRuTe/OptStopRandNN
(MIT License - Copyright (c) 2021 HeKrRuTe), which was modified to fit the needs
of this project.

author: Florian Krach
"""


# IMPORTS
import numpy as np
import time, os, sys, socket
sys.path.append("../")
import configs
import copy
import TLalgo.algorithms as algos
import optimal_stopping.utils.randomized_neural_networks as \
  randomized_neural_networks
import torch
import matplotlib
if 'ada-' in socket.gethostname():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ------- FUNCTIONS -------
def all_equal(iterator):
  return len(set(iterator)) <= 1


identity = lambda x: x


OPT_METHODS = {
  "local_optimizer": algos.get_optimal_params_dataset_i,
  "joint_optimizer": algos.get_jointly_optimal_params,
  "mean_local_optimizers": algos.get_mean_locally_optimal_params,
  "regret_optimal_algo": algos.get_final_params_wrapper(
    algos.regret_optimal_algo, is_stacked=True),
  "efficient_regret_optimal_algo": algos.get_final_params_wrapper(
    algos.accelerated_regret_optimal_algo, is_stacked=False),
  "accelerated_regret_optimal_algo": algos.get_final_params_wrapper(
    algos.accelerated_regret_optimal_algo, is_stacked=False),
  "symmetric_regret_optimal_algo": algos.get_final_params_wrapper(
    algos.symmetric_regret_optimal_algo, is_stacked=True),
}

ACTIVATION_FUNCTIONS = {
  "relu": torch.nn.ReLU(),
  "sigmoid": torch.nn.Sigmoid(),
  "tanh": torch.nn.Tanh(),
  "identity": identity,
  "leaky_relu": torch.nn.LeakyReLU(),

}




# ------- CLASSES -------
class AmericanOptionPricer:
  """Computes the price of an American Option using backward recursion.
  """
  def __init__(self, models, payoff,
               use_path=False, use_payoff_as_input=False, verbose=0, **kwargs):
    """
    Args:
      models: list of stockmodels that are used for the backward recursion. all
        need to have the same number of dates and number of stocks.
      payoff:
      use_path:
      use_payoff_as_input:
    """

    self.splits = None
    self.models = models
    assert all_equal([model.nb_dates for model in self.models]), \
      "models need to have same number of dates"
    assert all_equal([model.maturity for model in self.models]), \
      "models need to have same maturity"
    assert all_equal([model.nb_stocks for model in self.models]), \
      "models need to have same number of stocks"
    self.nb_dates = self.models[0].nb_dates
    self.nb_datasets = len(self.models)
    self.use_var = False

    self.use_var = [model.return_var for model in self.models]
    assert all_equal(self.use_var), \
      "models need to have same inputs (variance or not)"

    #class payoff: The payoff function of the option (e.g. Max call).
    self.payoff = payoff

    #bool: randomized neural network is replaced by a randomized recurrent NN.
    self.use_rnn = False

    #bool: x_k is replaced by the entire path (x_0, .., x_k) as input of the NN.
    self.use_path = use_path

    #bool: whether to use the payoff as extra input in addition to stocks
    self.use_payoff_as_input = use_payoff_as_input

    #int: used for ONLSM, tells model which weight to use
    self.which_weight = 0

    self.reinit_weights = False

    self.verbose = verbose

  def reset_weights(self):
    """method to reset the weights of the feature-model"""
    pass

  def feature_map(self, X):
    """method to compute the features with the feature-model"""
    raise NotImplementedError

  def compute_hs(self, paths, var_paths=None):
    """method to compute the features with the feature-model"""
    raise NotImplementedError

  def get_stockpaths(self, printtime=0):
    """generates the stock paths for all models."""
    if configs.path_gen_seed.get_seed() is not None:
      np.random.seed(configs.path_gen_seed.get_seed())
    stock_paths = []
    var_paths = []
    payoffs = []
    stock_paths_with_payoffs = []
    hs = []
    for i, model in enumerate(self.models):
      t = time.time()
      _stock_paths, _var_paths = model.generate_paths()
      _payoffs = self.payoff(_stock_paths)
      power = np.arange(0, model.nb_dates + 1)
      disc_factor = np.exp(
        (-model.rate) * model.maturity / model.nb_dates * power)
      disc_factors = np.repeat(
        np.expand_dims(disc_factor, axis=0), repeats=_payoffs.shape[0], axis=0)
      _payoffs = _payoffs * disc_factors
      _stock_paths_with_payoff = np.concatenate(
        [_stock_paths, np.expand_dims(_payoffs, axis=1)], axis=1)
      stock_paths.append(_stock_paths)
      var_paths.append(_var_paths)
      payoffs.append(_payoffs)
      stock_paths_with_payoffs.append(_stock_paths_with_payoff)
      if self.use_rnn:
        if self.use_payoff_as_input:
          _hs = self.compute_hs(_stock_paths_with_payoff, var_paths=_var_paths)
        else:
          _hs = self.compute_hs(_stock_paths, var_paths=_var_paths)
        hs.append(_hs)
      if printtime > 1:
        print("time to generate paths for stockmodel {}: ".format(i),
              time.time() - t)
    return stock_paths, var_paths, payoffs, stock_paths_with_payoffs, hs

  def get_features(
          self, hs, stock_paths, stock_paths_with_payoffs, var_paths, j, date):
    """
    Returns the features of the stock price at a given date for a given
    dataset index.

    Args:
      hs:
      stock_paths:
      stock_paths_with_payoffs:
      var_paths:
      j: dataset index
      date: date index

    Returns: features
    """
    if self.use_rnn:
      features = hs[j][date]
    else:
      if self.use_path:
        if self.use_payoff_as_input:
          finput = stock_paths_with_payoffs[j][:, :, :date + 1]
        else:
          finput = stock_paths[j][:, :, :date + 1]
        if self.use_var[j]:
          varp = var_paths[j][:, :, :date + 1]
          finput = np.concatenate([finput, varp], axis=1)
        # shape [paths, stocks, dates up to now]
        finput = np.flip(finput, axis=2)
        # add zeros to get shape [paths, stocks, dates+1]
        finput = np.concatenate(
          [finput, np.zeros(
            (finput.shape[0], finput.shape[1],
             self.nb_dates + 1 - finput.shape[2]))], axis=-1)
        finput = finput.reshape((finput.shape[0], -1))
      else:
        if self.use_payoff_as_input:
          finput = stock_paths_with_payoffs[j][:, :, date]
        else:
          finput = stock_paths[j][:, :, date]
        if self.use_var[j]:
          varp = var_paths[j][:, :, date]
          finput = np.concatenate([finput, varp], axis=1)
      features = self.feature_map(finput)
    return features

  def price(self, train_eval_split=2, optimizer_dict=None, plot_fname=None):
    """
    Compute the price of an American Option using a backward recursion.
    """
    # generate stock paths
    t1 = time.time()
    stock_paths, var_paths, payoffs, stock_paths_with_payoffs, hs = \
      self.get_stockpaths(printtime=self.verbose)
    if isinstance(train_eval_split, int):
      train_eval_split = [train_eval_split] * len(stock_paths)
    self.splits = [len(sp) // tes for sp, tes in zip(
      stock_paths, train_eval_split)]
    if self.verbose > 2:
      print("splits:", self.splits)
    time_for_path_gen = time.time() - t1
    if plot_fname is not None:
      f = plt.figure()
    plotted = False

    values = [po[:, -1] for po in payoffs]
    for i, date in enumerate(range(self.nb_dates - 1, 0, -1)):
      if self.reinit_weights:
        self.reset_weights()
      self.which_weight = i
      training_datasets = []
      datasets = []
      for j in range(self.nb_datasets):
        # get features
        features = self.get_features(
          hs, stock_paths, stock_paths_with_payoffs, var_paths, j, date)
        # get labels
        labels = values[j]
        training_datasets.append(
          (features[:self.splits[j]], labels[:self.splits[j]]))
        datasets.append((features, labels))
      # get optimal params
      opt_method = OPT_METHODS[optimizer_dict["method"]]
      opt_params, param_seq = opt_method(
        # we use the identity as feature map, since the features are already
        #   computed and stored in X of the dataset
        datasets=training_datasets, featuremap=lambda x: x, **optimizer_dict,
        verbose=self.verbose)
      if plot_fname is not None and param_seq is not None:
        param_seq = np.array(param_seq)
        param_seq = param_seq.reshape((param_seq.shape[0], -1))
        diff_norm = np.sqrt(np.sum((param_seq[1:]-param_seq[:-1])**2, axis=1))
        plt.plot(diff_norm, label="{}".format(date))
        plotted = True
      # get continuation values and stopping decision
      for j in range(self.nb_datasets):
        continuation_values = algos.evaluate_fKRR(
          featuremap=lambda x: x, X=datasets[j][0], params=opt_params)
        immediate_exercise_values = payoffs[j][:, date]
        # stopping_rule = np.zeros_like(immediate_exercise_values)
        which = immediate_exercise_values > continuation_values
        # stopping_rule[which] = 1
        values[j][which] = immediate_exercise_values[which]

    # compute price
    prices = []
    for j in range(self.nb_datasets):
      payoff_0 = payoffs[j][0, 0]
      lower_bound = max(payoff_0, np.mean(values[j][self.splits[j]:]))
      prices.append(lower_bound)

    if plotted:
      plt.legend()
      plt.savefig(plot_fname)
      plt.close(f)

    return prices, time_for_path_gen


class RLSM(AmericanOptionPricer):
  def __init__(self, models, payoff,
               use_path=False, use_payoff_as_input=True,
               hidden_size=10, factors=(1.,),
               activation_function=torch.nn.LeakyReLU(0.5),
               verbose=0, **kwargs):
    super().__init__(models, payoff, use_path, use_payoff_as_input,
                     verbose=verbose)
    self.reinit_weights = False
    if hidden_size < 0:
      hidden_size = 50 + abs(hidden_size) * self.models[0].nb_stocks
    state_size = self.models[0].nb_stocks * (
              1 + self.use_var[0]) + self.use_payoff_as_input * 1
    if self.use_path:
      state_size *= (self.models[0].nb_dates+1)
    self.state_size = state_size
    self.hidden_size = hidden_size
    self.factors = factors

    self.randNN = randomized_neural_networks.Reservoir2(
      state_size=state_size, hidden_size=hidden_size, factors=factors,
      activation=activation_function)

  def feature_map(self, X):
    X = torch.from_numpy(X)
    X = X.type(torch.float32)
    features = np.concatenate(
      [self.randNN(X).detach().numpy(), np.ones((len(X), 1))], axis=1)
    return features

  def reset_weights(self):
    self.randNN.init()


class RLSMreinit(RLSM):
  def __init__(self, models, payoff,
               use_path=False, use_payoff_as_input=True,
               hidden_size=10, factors=(1.,),
               activation_function=torch.nn.LeakyReLU(0.5),
               verbose=0, **kwargs):
    super().__init__(models, payoff, use_path, use_payoff_as_input,
                     hidden_size, factors, activation_function, verbose=verbose)
    self.reinit_weights = True


class RRLSM(RLSM):
  def __init__(self, models, payoff,
               use_path=False, use_payoff_as_input=True,
               hidden_size=10, factors=(1.,),
               activation_function=torch.nn.LeakyReLU(0.5),
               verbose=0, **kwargs):
    super().__init__(models, payoff, use_path, use_payoff_as_input,
                        hidden_size, factors, activation_function,
                     verbose=verbose)
    self.randNN = None
    self.use_rnn = True
    self.RNN = randomized_neural_networks.randomRNN(
      state_size=self.state_size, hidden_size=self.hidden_size,
      factors=self.factors, extend=False)

  def compute_hs(self, stock_paths, var_paths=None):
    """
    Args:
     stock_paths: numpy array, shape [nb_paths, nb_stocks, nb_dates]
     var_paths: None or numpy array of shape [nb_paths, nb_stocks, nb_dates]

    Returns:
     hidden states: numpy array, shape [nb_dates, nb_paths, hidden_size])
    """
    if self.use_var:
        stock_paths = np.concatenate([stock_paths, var_paths], axis=1)
    x = torch.from_numpy(stock_paths).permute(2, 0, 1)
    x = x.type(torch.float32)
    hs = self.RNN(x).detach().numpy()
    return hs


