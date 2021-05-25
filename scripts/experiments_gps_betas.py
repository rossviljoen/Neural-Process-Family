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
    get_remaining_indcs,
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

n_samples_train = 8
n_samples_test = 32

R_DIM = 128
MODEL_KWARGS = dict(
    x_dim=1,
    y_dim=1,
    n_z_samples_train=n_samples_train,
    n_z_samples_test=n_samples_test,  # number of samples when eval
    r_dim=R_DIM,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
    ),
)

bayes_xencoder = partial(
    BayesianMLP, n_hidden_layers=1, hidden_size=R_DIM, input_sampled=False,
    n_samples_train=n_samples_train, n_samples_test=n_samples_test
)

bayes_decoder = merge_flat_input(
    partial(BayesianMLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=False,
)

# Use only the context data as an input to the encoder
model_1d_q_C_bayes = partial(
    PACBayesLNP,
    is_q_zCct=False,
    XEncoder=bayes_xencoder,
    Decoder=bayes_decoder,
    **MODEL_KWARGS
)

# Use both the context and target data as inputs to the encoder
model_1d_q_CT_bayes = partial(
    PACBayesLNP,
    is_q_zCct=True,
    XEncoder=bayes_xencoder,
    Decoder=bayes_decoder,
    **MODEL_KWARGS
)

non_bayes_xencoder = partial(MLP, n_hidden_layers=1, hidden_size=R_DIM)

non_bayes_decoder = merge_flat_input(  # MLP takes single input but we give x and R so merge them
    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
)

# Use only the context data as an input to the encoder
model_1d_q_C = partial(
    LNP,
    is_q_zCct=False,
    XEncoder=non_bayes_xencoder,
    Decoder=non_bayes_decoder,
    **MODEL_KWARGS
)

# Use both the context and target data as inputs to the encoder
model_1d_q_CT = partial(
    LNP,
    is_q_zCct=True,
    XEncoder=non_bayes_xencoder,
    Decoder=non_bayes_decoder,
    **MODEL_KWARGS
)

# %%
import skorch
from npf import ELBOLossLNPF, PACMLossLNPF, PAC2LossLNPF, PAC2TLossLNPF, PACELBOLossLNPF, NLLLossLNPF, PACMJointLossLNPF
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models

KWARGS = dict(
    test_datasets=gp_test_datasets,
    iterator_train__collate_fn=get_cntxt_trgt_1d,
    iterator_valid__collate_fn=get_cntxt_trgt_1d_test,
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
# betas = [1e-6, 1e-4, 1e-2, 1., 1e2]

# %%
# trainers_npml = train_models(
#     gp_datasets,
#     {
#         f"LNP_NPML_EncC": model_1d_q_C,
#         f"LNP_NPML_EncCT":model_1d_q_CT,
#     },
#     criterion=NLLLossLNPF,
#     criterion__beta = 1.,
#     **KWARGS
# )

# for beta in betas:
    # trainers_pacelbo = train_models(
    #     gp_datasets,
    #     {
    #         # f"LNP_PACELBO_EncC_Beta{beta}": model_1d_q_C_bayes,
    #         f"LNP_PACELBO_EncCT_Beta{beta}":model_1d_q_CT_bayes,
    #     },
    #     criterion=PACELBOLossLNPF,
    #     criterion__beta = beta,
    #     **KWARGS
    # )
    
    # trainers_pacm = train_models(
    #     gp_datasets,
    #     {
    #         # f"LNP_PACM_EncC_Beta{beta}": model_1d_q_C_bayes,
    #         f"LNP_PACM_EncCT_Beta{beta}":model_1d_q_CT_bayes,
    #     },
    #     criterion=PACMLossLNPF,
    #     criterion__beta = beta,
    #     **KWARGS
    # )

    # trainers_pacm_joint = train_models(
    #     gp_datasets,
    #     {
    #         # f"LNP_PACM_EncC_Beta{beta}": model_1d_q_C_bayes,
    #         f"LNP_PACM_Joint_EncCT_Beta{beta}":model_1d_q_CT_bayes,
    #     },
    #     criterion=PACMJointLossLNPF,
    #     criterion__beta = beta,
    #     **KWARGS
    # )
    
    # trainers_elbo = train_models(
    #     gp_datasets,
    #     {
    #         f"LNP_ELBO_EncCT_Beta{beta}":model_1d_q_CT,
    #     },
    #     criterion=ELBOLossLNPF,
    #     criterion__beta = beta,
    #     **KWARGS
    # )

beta=1e-6
trainers_elbo = train_models(
    gp_datasets,
    {
        f"LNP_ELBO_EncCT_Beta{beta}":model_1d_q_CT,
    },
    criterion=ELBOLossLNPF,
    criterion__beta = beta,
    **KWARGS
)
    
trainers_pacm_joint = train_models(
    gp_datasets,
    {
        # f"LNP_PACM_EncC_Beta{beta}": model_1d_q_C_bayes,
        f"LNP_PACM_Joint_EncCT_Beta{beta}":model_1d_q_CT_bayes,
    },
    criterion=PACMJointLossLNPF,
    criterion__beta = beta,
    **KWARGS
)

beta=1e-2
trainers_elbo = train_models(
    gp_datasets,
    {
        f"LNP_ELBO_EncCT_Beta{beta}":model_1d_q_CT,
    },
    criterion=ELBOLossLNPF,
    criterion__beta = beta,
    **KWARGS
)

trainers_pacelbo = train_models(
    gp_datasets,
    {
        # f"LNP_PACELBO_EncC_Beta{beta}": model_1d_q_C_bayes,
        f"LNP_PACELBO_EncCT_Beta{beta}":model_1d_q_CT_bayes,
    },
    criterion=PACELBOLossLNPF,
    criterion__beta = beta,
    **KWARGS
)

beta=1.
trainers_pacelbo = train_models(
    gp_datasets,
    {
        # f"LNP_PACELBO_EncC_Beta{beta}": model_1d_q_C_bayes,
        f"LNP_PACELBO_EncCT_Beta{beta}":model_1d_q_CT_bayes,
    },
    criterion=PACELBOLossLNPF,
    criterion__beta = beta,
    **KWARGS
)
