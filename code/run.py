"""
author: Florian Krach

code for running the experiments
"""

# =====================================================================================================================
import numpy as np
import os
import pandas as pd
import json
import socket
import matplotlib
from joblib import Parallel, delayed
from absl import app
from absl import flags
import multiprocessing

import configs
import optimal_stopping.run_algo as run_algo_optstop
import convergence.run_algo as run_algo_convergence
import utilities.extras as extras

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from configs import SendBotMessage as SBM


# =====================================================================================================================
# FLAGS
FLAGS = flags.FLAGS

flags.DEFINE_string("params", None, "name of the params list (in configs.py) to"
                                    " use for parallel run")
flags.DEFINE_integer("first_id", None, "First id of the given list / "
                                       "to start training of")
flags.DEFINE_integer("DEBUG", 0, "whether to run parallel in debug mode, "
                             ">0 is verbose level")
flags.DEFINE_string("saved_models_path", configs.saved_models_path,
                    "path where the models are saved")
flags.DEFINE_string("get_overview", None,
                    "name of the dict (in config.py) defining input for "
                    "extras.get_training_overview")
flags.DEFINE_string("conv_plot", None,
                    "name of the dict (in config.py) defining input for "
                    "extras.plot_conv_analysis")
flags.DEFINE_integer("nb_runs", 1, "number of runs for each model")

# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    flags.DEFINE_integer("NB_JOBS", 1,
                         "nb of parallel jobs to run  with joblib")
    # flags.DEFINE_integer("NB_CPUS", 1, "nb of CPUs used by each training")
    flags.DEFINE_bool("SEND", False, "whether to send with telegram bot")
else:
    SERVER = True
    flags.DEFINE_integer("NB_JOBS", 24,
                         "nb of parallel jobs to run  with joblib")
    # flags.DEFINE_integer("NB_CPUS", 1, "nb of CPUs used by each training")
    flags.DEFINE_bool("SEND", True, "whether to send with telegram bot")
    matplotlib.use('Agg')

print(socket.gethostname())
print('SERVER={}'.format(SERVER))
NUM_PROCESSORS = multiprocessing.cpu_count()


# =====================================================================================================================
# Functions
def run_switcher(**kwargs):
    if "experiment" in kwargs:
        experiment = kwargs["experiment"]
        if experiment == "convergence":
            return run_algo_convergence.run_algo(**kwargs)
        elif experiment == "optimal_stopping":
            return run_algo_optstop.run_algo(**kwargs)
        else:
            raise ValueError("experiment {} not recognized".format(experiment))
    else:
        return run_algo_optstop.run_algo(**kwargs)


def plot_switcher(**kwargs):
    if "which_plot" in kwargs:
        which_plot = kwargs["which_plot"]
        if which_plot == "convergence_single_run":
            return extras.plot_conv_analysis(**kwargs)
        elif which_plot == "convergence_multi_runs":
            return extras.plot_conv_analysis_multiruns(**kwargs)
        else:
            raise ValueError("which_plot {} not recognized".format(which_plot))
    else:
        return extras.plot_conv_analysis(**kwargs)


def parallel_training(params=None, nb_jobs=1, first_id=None,
                      saved_models_path=configs.saved_models_path,
                      nb_runs=1):
    """
    function for parallel training
    :param params: a list of param_dicts, each dict corresponding to one model
            that should be trained, can be None if model_ids is given
            (then unused)
            all kwargs needed for train.train have to be in each dict
            -> giving the params together with first_id, they can be used to
                restart parallel training (however, the saved params for all
                models where the model_id already existed will be used instead
                of the params in this list, so that no unwanted errors are
                produced by mismatching. whenever a model_id didn't exist yet
                the params of the list are used to make a new one)
            -> giving params without first_id, all param_dicts will be used to
                initiate new models
    :param nb_jobs: int, the number of CPUs to use parallelly
    :param first_id: int or None, the model_id corresponding to the first
            element of params list
    :param saved_models_path: str, path to saved models
    :param nb_runs: int, number of runs for each model
    :return:
    """
    if params is not None and 'saved_models_path' in params[0]:
        saved_models_path = params[0]['saved_models_path']
    model_overview_file_name = '{}model_overview.csv'.format(
        saved_models_path)
    configs.makedirs(saved_models_path)
    if not os.path.exists(model_overview_file_name):
        df_overview = pd.DataFrame(data=None, columns=['id', 'description'])
        max_id = 0
    else:
        df_overview = pd.read_csv(model_overview_file_name, index_col=0)
        max_id = np.max(df_overview['id'].values)

    # get model_id, model params etc. for each param
    if params is None:
        return 0
    if first_id is None:
        model_id = max_id + 1
    else:
        model_id = first_id
    for i, param in enumerate(params):  # iterate through all specified parameter settings
        if model_id in df_overview['id'].values:  # resume training if taken id is specified as first model id
            desc = (df_overview['description'].loc[
                df_overview['id'] == model_id]).values[0]
            params_dict = json.loads(desc)
            params_dict['resume_training'] = True
            params_dict['model_id'] = model_id
        else:  # if new model id, create new training
            desc = json.dumps(param, sort_keys=True)
            df_ov_app = pd.DataFrame([[model_id, desc]],
                                     columns=['id', 'description'])
            df_overview = pd.concat([df_overview, df_ov_app],
                                    ignore_index=True)
            df_overview.to_csv(model_overview_file_name)
            params_dict = json.loads(desc)
            params_dict['resume_training'] = False
            params_dict['model_id'] = model_id
        params[i] = params_dict
        model_id += 1

    for param in params:
        param['parallel'] = True

    if FLAGS.SEND:
        SBM.send_notification(
            text='start parallel training - \nparams:'
                 '\n\n{}'.format(params),
            chat_id=configs.CHAT_ID
        )

    # get list with jobs to run
    jobs = []
    for param in params:
        for run in range(nb_runs):
            jobs.append(delayed(run_switcher)(
                **param, run=run, verbose=FLAGS.DEBUG, send=FLAGS.SEND))
    print(f"Running {len(jobs)} tasks using "
          f"{nb_jobs}/{NUM_PROCESSORS} CPUs...")

    if FLAGS.DEBUG:
        results = Parallel(n_jobs=nb_jobs)(jobs)
        if FLAGS.SEND:
            SBM.send_notification(
                text='finished parallel training - \nparams:'
                     '\n\n{}'.format(params),
                chat_id=configs.CHAT_ID
            )
    else:
        try:
            results = Parallel(n_jobs=nb_jobs)(jobs)
            if FLAGS.SEND:
                SBM.send_notification(
                    text='finished parallel training - \nparams:'
                         '\n\n{}'.format(params),
                    chat_id=configs.CHAT_ID
                )
        except Exception as e:
            if FLAGS.SEND:
                SBM.send_notification(
                    text='error in parallel training - \nerror:'
                         '\n\n{}'.format(e),
                    chat_id=configs.ERROR_CHAT_ID
                )
            else:
                print('error:\n\n{}'.format(e))


def main(arg):
    """
    function to run parallel training with flags from command line
    """
    del arg
    params_list = None
    nb_jobs = FLAGS.NB_JOBS
    if FLAGS.params:
        params_list = eval("configs."+FLAGS.params)
        print('combinations: {}'.format(len(params_list)))
    get_training_overview_dict = None
    if FLAGS.get_overview:
        get_training_overview_dict = eval("configs."+FLAGS.get_overview)
    conv_plot = None
    if FLAGS.conv_plot:
        conv_plot = eval("configs."+FLAGS.conv_plot)
    if params_list is not None:
        parallel_training(
            params=params_list,
            first_id=FLAGS.first_id, nb_jobs=nb_jobs,
            saved_models_path=FLAGS.saved_models_path,
            nb_runs=FLAGS.nb_runs)
    if get_training_overview_dict is not None:
        extras.get_training_overview(
            send=FLAGS.SEND, **get_training_overview_dict)
    if conv_plot is not None:
        plot_switcher(
            send=FLAGS.SEND, **conv_plot)


if __name__ == '__main__':
    app.run(main)

