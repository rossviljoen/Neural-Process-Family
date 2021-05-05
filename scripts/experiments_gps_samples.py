import logging
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np

import torch

os.chdir("..")
import sys
sys.path.append('./')

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)

N_THREADS = 8
IS_FORCE_CPU = False  # Nota Bene : notebooks don't deallocate GPU memory

if IS_FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch.set_num_threads(N_THREADS)

# %%
from utils.ntbks_helpers import get_all_gp_datasets, get_img_datasets, get_gp_datasets
from sklearn.gaussian_process.kernels import RBF

gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()

# %%
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    GridCntxtTrgtGetter,
    RandomMasker,
    get_all_indcs,
    no_masker,
)
from utils.data import cntxt_trgt_collate, get_test_upscale_factor

# CONTEXT TARGET SPLIT
get_cntxt_trgt_1d = cntxt_trgt_collate(
    CntxtTrgtGetter(
        contexts_getter=GetRandomIndcs(a=0.0, b=50), targets_getter=get_all_indcs,
    )
)

# %%
from functools import partial

from npf import LNP
from npf.architectures import MLP, BayesianMLP, merge_flat_input
from utils.helpers import count_parameters

R_DIM = 128
KWARGS = dict(
    n_z_samples_test=32,  # number of samples when eval
    XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
    r_dim=R_DIM,
    is_q_zCct=True,
)

bayes_decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
    partial(BayesianMLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
)

n_samples_list = [1, 2, 4, 8, 16, 32]

bayes_models = {}
for n in n_samples_list:
    bayes_models[n] = partial(
        LNP,
        x_dim=1,
        y_dim=1,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
        ),
        Decoder=bayes_decoder,
        n_z_samples_train=n,
        **KWARGS
    )

non_bayes_decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
)

non_bayes_models = {}
for n in n_samples_list:
    non_bayes_models[n] = partial(
        LNP,
        x_dim=1,
        y_dim=1,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
        ),
        Decoder=non_bayes_decoder,
        n_z_samples_train=n,
        **KWARGS
    )

# %%
import skorch
from npf import ELBOLossLNPF, PACMLossLNPF, PAC2LossLNPF, PAC2TLossLNPF, PACELBOLossLNPF, NLLLossLNPF
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models

KWARGS = dict(
    test_datasets=gp_test_datasets,
    iterator_train__collate_fn=get_cntxt_trgt_1d,
    iterator_valid__collate_fn=get_cntxt_trgt_1d,
    max_epochs=50,
    is_retrain=True,  # whether to load precomputed model or retrain
    is_reeval=True,
    chckpnt_dirname="results/experiments_05-05-21/",
    device=None,  # use GPU if available
    batch_size=32,
    lr=1e-3,
    decay_lr=10,  # decrease learning rate by 10 during training
    seed=123,
    criterion__eval_use_crossentropy=False,
    # verbose=0
)

beta = 1.

# %%
for n in n_samples_list:
    trainers_pacm = train_models(
        gp_datasets,
        {
            f"LNP_PACM_EncCT_Beta{beta}_nsamples{n}":bayes_models[n],
        },
        criterion=PACMLossLNPF,
        criterion__beta = beta,
        **KWARGS
    )
    
    trainers_elbo = train_models(
        gp_datasets,
        {
            f"LNP_ELBO_EncCT_Beta{beta}_nsamples{n}":non_bayes_models[n],
        },
        criterion=ELBOLossLNPF,
        criterion__beta = beta,
        **KWARGS
    )

    trainers_npml = train_models(
        gp_datasets,
        {
            f"LNP_NPML_EncCT_Beta{beta}_nsamples{n}":non_bayes_models[n],
        },
        criterion=NLLLossLNPF,
        criterion__beta = beta,
        **KWARGS
    )

    trainers_pacelbo = train_models(
        gp_datasets,
        {
            f"LNP_PACELBO_EncCT_Beta{beta}_nsamples{n}":bayes_models[n],
        },
        criterion=PACELBOLossLNPF,
        criterion__beta = beta,
        **KWARGS
    )
