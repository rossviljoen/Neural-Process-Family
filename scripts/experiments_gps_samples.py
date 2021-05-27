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
    get_remaining_indcs,
    no_masker,
)
from utils.data import cntxt_trgt_collate, get_test_upscale_factor

# CONTEXT TARGET SPLIT
get_cntxt_trgt_1d = cntxt_trgt_collate(
    CntxtTrgtGetter(
        contexts_getter=GetRandomIndcs(a=0.0, b=50), targets_getter=get_all_indcs,
    )
)

get_cntxt_trgt_1d_test = cntxt_trgt_collate(
    CntxtTrgtGetter(
        contexts_getter=GetRandomIndcs(a=0.0, b=50), targets_getter=get_remaining_indcs,
    )
)


# %%
from functools import partial

from npf import LNP, PACBayesLNP
from npf.architectures import MLP, BayesianMLP, merge_flat_input
from utils.helpers import count_parameters

n_samples_test=32

R_DIM = 128
MODEL_KWARGS = dict(
    x_dim=1,
    y_dim=1,
    r_dim=R_DIM,
    n_z_samples_test=n_samples_test,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
    ),
)

bayes_decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
    partial(BayesianMLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
)

n_samples_list = [1, 2, 4, 8, 16, 32]

bayes_models = {}
for n in n_samples_list:
    bayes_xencoder = partial(
        BayesianMLP, n_hidden_layers=1, hidden_size=R_DIM, input_sampled=False,
        n_samples_train=n
    )
    bayes_models[n] = partial(
        PACBayesLNP,
        XEncoder=bayes_xencoder,
        Decoder=bayes_decoder,
        n_z_samples_train=n,
        is_q_zCct=True,
        **MODEL_KWARGS
    )

non_bayes_xencoder = partial(MLP, n_hidden_layers=1, hidden_size=R_DIM)

non_bayes_decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
)

non_bayes_models_q_CT = {}
non_bayes_models_q_C = {}

for n in n_samples_list:
    non_bayes_models_q_CT[n] = partial(
        LNP,
        XEncoder=non_bayes_xencoder,
        Decoder=non_bayes_decoder,
        n_z_samples_train=n,
        is_q_zCct=True,
        **MODEL_KWARGS
    )

    non_bayes_models_q_C[n] = partial(
        LNP,
        XEncoder=non_bayes_xencoder,
        Decoder=non_bayes_decoder,
        n_z_samples_train=n,
        is_q_zCct=False,
        **MODEL_KWARGS
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
    max_epochs=30,
    is_retrain=True,  # whether to load precomputed model or retrain
    is_reeval=True,
    chckpnt_dirname="results/experiments_24-05-21/",
    device=None,  # use GPU if available
    batch_size=32,
    lr=1e-3,
    decay_lr=10,  # decrease learning rate by 10 during training
    seed=123,
    criterion__eval_use_crossentropy=False,
    # verbose=0
)

# %%
for n in n_samples_list:
    beta=1.
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
            f"LNP_ELBO_Beta{beta}_nsamples{n}":non_bayes_models_q_CT[n],
        },
        criterion=ELBOLossLNPF,
        criterion__beta = beta,
        **KWARGS
    )

    trainers_npml = train_models(
        gp_datasets,
        {
            f"LNP_NPML_nsamples{n}":non_bayes_models_q_C[n],
        },
        criterion=NLLLossLNPF,
        criterion__beta = beta,
        **KWARGS
    )

    beta=1e-6
    trainers_pacm = train_models(
        gp_datasets,
        {
            f"LNP_PACM_EncCT_Beta{beta}_nsamples{n}":bayes_models[n],
        },
        criterion=PACMLossLNPF,
        criterion__beta = beta,
        **KWARGS
    )
