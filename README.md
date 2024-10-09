# Regret Optimal Federated Transfer Learning

[![DOI](https://zenodo.org/badge/673854185.svg)](https://zenodo.org/badge/latestdoi/673854185)

Official implementation of the paper 
[Regret-Optimal Federated Transfer Learning for Kernel Regression with Applications in American Option Pricing](https://arxiv.org/abs/2309.04557).


## Installation & Requirements

This code was executed using Python 3.8.

To install requirements, download this Repo and cd into it.

Then create a new environment and install all dependencies and this repo.
With [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
 ```sh
conda create --name FederatedTransferLearning python=3.8
conda activate FederatedTransferLearning
pip install -r requirements.txt
 ```

## Usage

Use the run.py script to (parallely) run the experiments.

Example:
```sh
python code/run.py --params=params_list0 --NB_JOBS=1 --nb_runs=10 --DEBUG=1 --first_id=1 --get_overview=GTO_0
```

List of all flags:

- **params**: name of the params list (defined in config.py) to use for parallel run
- **NB_JOBS**: nb of parallel jobs to run with joblib
- **nb_runs**: nb of runs to do for each job
- **first_id**: First id of the given list / to start training of
- **get_overview**: name of the dict (defined in config.py) defining input for extras.get_training_overview
- **conv_plot**: name of the dict (defined in config.py) defining input for extras.plot_conv_analysis(_multiruns)
- **SEND**: whether to send results via telegram
- **DEBUG**: whether to run parallel in debug mode
- **saved_models_path**: path where the models are saved




run Heston / rough Heston experiments in paper:
```sh
python code/run.py --params=params_list6 --NB_JOBS=1 --nb_runs=100 --DEBUG=0 --first_id=1 --get_overview=GTO_6

python code/run.py --params=params_list7_1 --NB_JOBS=1 --nb_runs=100 --DEBUG=0 --first_id=1 --get_overview=GTO_7_1
python code/run.py --params=params_list7_2 --NB_JOBS=1 --nb_runs=100 --DEBUG=0 --first_id=1 --get_overview=GTO_7_2
```

run convergence experiments in paper:
```sh
python code/run.py --params=conv_params_list_2 --NB_JOBS=1 --nb_runs=10 --DEBUG=0 --first_id=1 --conv_plot=conv_plot_2
python code/run.py --conv_plot=conv_plot_2; python code/run.py --conv_plot=conv_plot_2_1; python code/run.py --conv_plot=conv_plot_2_2;  
```

additional convergence experiments:
```sh
python code/run.py --params=conv_params_list_3 --NB_JOBS=1 --nb_runs=10 --DEBUG=0 --first_id=1 --conv_plot=conv_plot_3
python code/run.py --params=conv_params_list_4 --NB_JOBS=1 --nb_runs=10 --DEBUG=0 --first_id=1 --conv_plot=conv_plot_4
```


---

## License

This code can be used in accordance with the [LICENSE](LICENSE).

---

## Citation

If you use this code for your publications, please cite our paper:
[Regret-Optimal Federated Transfer Learning for Kernel Regression with Applications in American Option Pricing](https://arxiv.org/abs/2309.04557).
```
@misc{yang2023regretoptimal,
      title={Regret-Optimal Federated Transfer Learning for Kernel Regression with Applications in American Option Pricing}, 
      author={Xuwei Yang and Anastasis Kratsios and Florian Krach and Matheus Grasselli and Aurelien Lucchi},
      year={2023},
      eprint={2309.04557},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

---

## Acknowledgements

This code uses parts of the code for [Optimal Stopping via Randomized Neural Networks](https://github.com/HeKrRuTe/OptStopRandNN)
(MIT License - Copyright (c) 2021 HeKrRuTe), which were modified to fit the needs
of this project.

---


