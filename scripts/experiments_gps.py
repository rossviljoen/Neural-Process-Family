import logging
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np

import torch

os.chdir("..")

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)

N_THREADS = 4
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
from npf.architectures import MLP, merge_flat_input
from utils.helpers import count_parameters

R_DIM = 128
KWARGS = dict(
    n_z_samples_train=2,
    n_z_samples_test=32,  # number of samples when eval
    XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
    Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
        partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
    ),
    r_dim=R_DIM,
)

# Use only the context data as an input to the encoder
model_1d_q_C = partial(
    LNP,
    x_dim=1,
    y_dim=1,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
    ),
    is_q_zCct=False,
    **KWARGS,
)

# Use both the context and target data as inputs to the encoder
model_1d_q_CT = partial(
    LNP,
    x_dim=1,
    y_dim=1,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
    ),
    is_q_zCct=True,
    **KWARGS,
)

# %%
import skorch
from npf import ELBOLossLNPF, PACMLossLNPF, PAC2LossLNPF, PAC2TLossLNPF, TemperedELBOLossLNPF
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models

KWARGS = dict(
    test_datasets=gp_test_datasets,
    iterator_train__collate_fn=get_cntxt_trgt_1d,
    iterator_valid__collate_fn=get_cntxt_trgt_1d,
    max_epochs=100,
    is_retrain=True,  # whether to load precomputed model or retrain
    is_reeval=True,
    chckpnt_dirname="results/experiments/",
    device=None,  # use GPU if available
    batch_size=32,
    lr=1e-3,
    decay_lr=10,  # decrease learning rate by 10 during training
    seed=123,
)
betas = [0.1, 0.8, 1., 1.2, 2]

# %%
for beta in betas:
    trainers_pac2 = train_models(
        gp_datasets,
        {f"LNP_PAC2_EncC_TrainT_Beta{beta}": model_1d_q_C,
         f"LNP_PAC2_EncC_TrainTC_Beta{beta}": model_1d_q_C,
         f"LNP_PAC2_EncCT_TrainT_Beta{beta}":model_1d_q_CT,
         f"LNP_PAC2_EncCT_TrainTC_Beta{beta}":model_1d_q_CT},
        criterion=PAC2LossLNPF,
        criterion__beta = beta,
        models_kwargs={f"LNP_PAC2_EncC_TrainT_Beta{beta}":dict(criterion__train_all_data=False),
                       f"LNP_PAC2_EncC_TrainTC_Beta{beta}":dict(criterion__train_all_data=True),
                       f"LNP_PAC2_EncCT_TrainT_Beta{beta}":dict(criterion__train_all_data=False),
                       f"LNP_PAC2_EncCT_TrainTC_Beta{beta}":dict(criterion__train_all_data=True)},
        **KWARGS
    )

# %%
for beta in betas:
    trainers_pacm = train_models(
        gp_datasets,
        {f"LNP_PACM_EncC_TrainT_Beta{beta}": model_1d_q_C,
         f"LNP_PACM_EncC_TrainTC_Beta{beta}": model_1d_q_C,
         f"LNP_PACM_EncCT_TrainT_Beta{beta}":model_1d_q_CT,
         f"LNP_PACM_EncCT_TrainTC_Beta{beta}":model_1d_q_CT},
        criterion=PACMLossLNPF,
        criterion__beta = beta,
        models_kwargs={f"LNP_PACM_EncC_TrainT_Beta{beta}":dict(criterion__train_all_data=False),
                       f"LNP_PACM_EncC_TrainTC_Beta{beta}":dict(criterion__train_all_data=True),
                       f"LNP_PACM_EncCT_TrainT_Beta{beta}":dict(criterion__train_all_data=False),
                       f"LNP_PACM_EncCT_TrainTC_Beta{beta}":dict(criterion__train_all_data=True)},
        **KWARGS
    )

# %%
for beta in betas:
    trainers_elbo = train_models(
        gp_datasets,
        {f"LNP_ELBO_EncCT_TrainT_Beta{beta}":model_1d_q_CT,
         f"LNP_ELBO_EncCT_TrainTC_Beta{beta}":model_1d_q_CT},
        criterion=ELBOLossLNPF,
        criterion__beta = beta,
        models_kwargs={f"LNP_ELBO_EncCT_TrainT_Beta{beta}":dict(criterion__train_all_data=False),
                       f"LNP_ELBO_EncCT_TrainTC_Beta{beta}":dict(criterion__train_all_data=True)},
        **KWARGS
    )
