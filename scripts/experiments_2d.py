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

# img_datasets, img_test_datasets = get_img_datasets(["celeba32", "mnist"])
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

from npf import LNP, PACBayesLNP
from npf.architectures import MLP, BayesianMLP, merge_flat_input
from utils.helpers import count_parameters

n_samples_train = 8
n_samples_test = 32

R_DIM = 128
MODEL_KWARGS = dict(
    n_z_samples_train=n_samples_train,
    n_z_samples_test=n_samples_test,  # number of samples when eval
    r_dim=R_DIM,
    x_dim=2,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 3), is_sum_merge=True,
    ),
)

bayes_xencoder = partial(
    BayesianMLP, n_hidden_layers=1, hidden_size=R_DIM, input_sampled=False,
    n_samples_train=n_samples_train, n_samples_test=n_samples_test
)

bayes_decoder=merge_flat_input(
    partial(BayesianMLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
)

non_bayes_xencoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM)

non_bayes_decoder=merge_flat_input(
    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
)

# image (2D) case
model_2d_bayes = partial(
    PACBayesLNP,
    XEncoder=bayes_xencoder,
    Decoder=bayes_decoder,
    is_q_zCct=True,
    **MODEL_KWARGS
)

model_2d_non_bayes = partial(
    LNP,
    XEncoder=non_bayes_xencoder,
    Decoder=non_bayes_decoder,
    is_q_zCct=True,
    **MODEL_KWARGS
)

model_2d_q_C_non_bayes = partial(
    LNP,
    XEncoder=non_bayes_xencoder,
    Decoder=non_bayes_decoder,
    is_q_zCct=False,
    **MODEL_KWARGS
)


import skorch
from npf import ELBOLossLNPF, PACMLossLNPF, PAC2LossLNPF, PAC2TLossLNPF, PACELBOLossLNPF, NLLLossLNPF
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models

TRAINERS_KWARGS = dict(
    chckpnt_dirname="results/experiments_24-05-21/",
    max_epochs=40,
    is_retrain=True,  # whether to load precomputed model or retrain
    is_reeval=True,
    device=None,  # use GPU if available
    batch_size=16,
    lr=1e-3,
    decay_lr=10,  # decrease learning rate by 10 during training
    seed=123,
    criterion__eval_use_crossentropy=False,
    test_datasets=img_test_datasets,
    train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
    iterator_train__collate_fn=get_cntxt_trgt_2d,
    iterator_valid__collate_fn=get_cntxt_trgt_2d,
)


# # %%
# trainers_npml = train_models(
#     img_datasets,
#     add_y_dim(
#         {
#             f"LNP_NPML_EncC": model_2d_q_C_non_bayes,
#             f"LNP_NPML_EncCT": model_2d_non_bayes,
#         },
#         img_datasets),  # y_dim (channels) depend on data
#     criterion=NLLLossLNPF,  # NPVI
#     criterion__beta = 1.,
#     **TRAINERS_KWARGS
# )

# betas = [1e-6, 1e-4, 1e-2, 1., 1e2]
# betas = [1., 1e-2, 1e2, 1e-4]

# for beta in betas:
    # trainers_pacelbo = train_models(
    #     img_datasets,
    #     add_y_dim(
    #         {
    #             f"LNP_PACELBO_EncCT_Beta{beta}": model_2d_bayes,
    #         },
    #         img_datasets),  # y_dim (channels) depend on data
    #     criterion=PACELBOLossLNPF,
    #     criterion__beta = beta,
    #     **TRAINERS_KWARGS
    # )

    # trainers_pacm = train_models(
    #     img_datasets,
    #     add_y_dim(
    #         {
    #             f"LNP_PACM_EncCT_Beta{beta}": model_2d_bayes,
    #         },
    #         img_datasets),  # y_dim (channels) depend on data
    #     criterion=PACMLossLNPF,
    #     criterion__beta = beta,
    #     **TRAINERS_KWARGS
    # )

    # trainers_pacm_joint = train_models(
    #     img_datasets,
    #     add_y_dim(
    #         {
    #             f"LNP_PACM_Joint_EncCT_Beta{beta}": model_2d_bayes,
    #         },
    #         img_datasets),  # y_dim (channels) depend on data
    #     criterion=PACMLossLNPF,
    #     criterion__beta = beta,
    #     **TRAINERS_KWARGS
    # )
    
    # trainers_elbo = train_models(
    #     img_datasets,
    #     add_y_dim({f"LNP_ELBO_EncCT_Beta{beta}": model_2d_non_bayes}, img_datasets),  # y_dim (channels) depend on data
    #     criterion=ELBOLossLNPF,
    #     criterion__beta = beta,
    #     **TRAINERS_KWARGS
    # )


beta = 1e-2

trainers_elbo = train_models(
    img_datasets,
    add_y_dim({f"LNP_ELBO_EncCT_Beta{beta}": model_2d_non_bayes}, img_datasets),  # y_dim (channels) depend on data
    criterion=ELBOLossLNPF,
    criterion__beta = beta,
    **TRAINERS_KWARGS
)

trainers_pacelbo = train_models(
    img_datasets,
    add_y_dim(
        {
            f"LNP_PACELBO_EncCT_Beta{beta}": model_2d_bayes,
        },
        img_datasets),  # y_dim (channels) depend on data
    criterion=PACELBOLossLNPF,
    criterion__beta = beta,
    **TRAINERS_KWARGS
)

beta = 1.
trainers_pacelbo = train_models(
    img_datasets,
    add_y_dim(
        {
            f"LNP_PACELBO_EncCT_Beta{beta}": model_2d_bayes,
        },
        img_datasets),  # y_dim (channels) depend on data
    criterion=PACELBOLossLNPF,
    criterion__beta = beta,
    **TRAINERS_KWARGS
)
