#!/usr/bin/env python3

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

img_datasets, img_test_datasets = get_img_datasets(["mnist"])

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

get_cntxt_trgt_2d = cntxt_trgt_collate(
    GridCntxtTrgtGetter(
        context_masker=RandomMasker(a=0.0, b=0.3), target_masker=no_masker,
    )
)

from functools import partial

from npf import LNP
from npf.architectures import MLP, BayesianMLP, merge_flat_input
from utils.helpers import count_parameters

R_DIM = 128
KWARGS = dict(
    n_z_samples_train=8,
    n_z_samples_test=32,  # number of samples when eval
    XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
    r_dim=R_DIM,
    is_q_zCct=True  # will use NPVI => posterior sampling
)

bayes_decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
    partial(BayesianMLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
)

non_bayes_decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
)

# image (2D) case
model_2d_bayes = partial(
    LNP,
    x_dim=2,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 3), is_sum_merge=True,
    ),
    Decoder=bayes_decoder,
    **KWARGS
)  # don't add y_dim yet because depends on data

model_2d_non_bayes = partial(
    LNP,
    x_dim=2,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 3), is_sum_merge=True,
    ),
    Decoder=non_bayes_decoder,
    **KWARGS
)  # don't add y_dim yet because depends on data

# model_2d_q_C = partial(
#     LNP,
#     x_dim=2,
#     XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
#         partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 3), is_sum_merge=True,
#     ),
#     is_q_zCct=False,
#     **KWARGS
# )  # don't add y_dim yet because depends on data


import skorch
from npf import ELBOLossLNPF, PACMLossLNPF, PAC2LossLNPF, PAC2TLossLNPF, PACELBOLossLNPF, NLLLossLNPF
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models

KWARGS = dict(
    chckpnt_dirname="results/experiments_05-05-21/",
    is_retrain=True,  # whether to load precomputed model or retrain
    is_reeval=True,
    device=None,  # use GPU if available
    batch_size=16,
    lr=1e-3,
    decay_lr=10,  # decrease learning rate by 10 during training
    seed=123,
    criterion__eval_use_crossentropy=False
)

betas = [1.]

for beta in betas:
    trainers_pacelbo = train_models(
        img_datasets,
        add_y_dim(
            {
                f"LNP_PACELBO_EncCT_Beta{beta}": model_2d_bayes,
            },
            img_datasets),  # y_dim (channels) depend on data
        test_datasets=img_test_datasets,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
        iterator_train__collate_fn=get_cntxt_trgt_2d,
        iterator_valid__collate_fn=get_cntxt_trgt_2d,
        # datasets_kwargs=dict(
        #     zsmms=dict(iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap,)
        # ),  # for zsmm use extrapolation
        max_epochs=50,
        criterion=PACELBOLossLNPF,  # NPVI
        criterion__beta = beta,
        **KWARGS
    )

    trainers_pacm = train_models(
        img_datasets,
        add_y_dim(
            {
                f"LNP_PACM_EncCT_Beta{beta}": model_2d_bayes,
            },
            img_datasets),  # y_dim (channels) depend on data
        test_datasets=img_test_datasets,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
        iterator_train__collate_fn=get_cntxt_trgt_2d,
        iterator_valid__collate_fn=get_cntxt_trgt_2d,
        # datasets_kwargs=dict(
        #     zsmms=dict(iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap,)
        # ),  # for zsmm use extrapolation
        max_epochs=50,
        criterion=PACMLossLNPF,  # NPVI
        criterion__beta = beta,
        **KWARGS
    )
    
    trainers_elbo = train_models(
        img_datasets,
        add_y_dim({f"LNP_ELBO_EncCT_Beta{beta}": model_2d_non_bayes}, img_datasets),  # y_dim (channels) depend on data
        test_datasets=img_test_datasets,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
        iterator_train__collate_fn=get_cntxt_trgt_2d,
        iterator_valid__collate_fn=get_cntxt_trgt_2d,
        # datasets_kwargs=dict(
        #     zsmms=dict(iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap,)
        # ),  # for zsmm use extrapolation
        max_epochs=50,
        criterion=ELBOLossLNPF,  # NPVI
        criterion__beta = beta,
        **KWARGS
    )

    trainers_npml = train_models(
        img_datasets,
        add_y_dim(
            {
                f"LNP_NPML_EncCT_Beta{beta}": model_2d_non_bayes,
            },
            img_datasets),  # y_dim (channels) depend on data
        test_datasets=img_test_datasets,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
        iterator_train__collate_fn=get_cntxt_trgt_2d,
        iterator_valid__collate_fn=get_cntxt_trgt_2d,
        # datasets_kwargs=dict(
        #     zsmms=dict(iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap,)
        # ),  # for zsmm use extrapolation
        max_epochs=50,
        criterion=NLLLossLNPF,  # NPVI
        criterion__beta = beta,
        **KWARGS
    )
