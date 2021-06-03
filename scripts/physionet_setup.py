#!/usr/bin/env python3

# First run this in data/
# wget -r -N -c -np https://physionet.org/files/challenge-2012/1.0.0/

# %%
import numpy as np
import pandas as pd

DATADIR = "../data/"
SETA = DATADIR + "physionet.org/files/challenge-2012/1.0.0/set-a/"
SETB = DATADIR + "physionet.org/files/challenge-2012/1.0.0/set-b/"

# %%
import os

_, _, a_files = next(os.walk(SETA))
_, _, b_files = next(os.walk(SETB))
a_patients = [int(os.path.splitext(f)[0]) for f in a_files if os.path.splitext(f)[1] == ".txt"]
b_patients = [int(os.path.splitext(f)[0]) for f in b_files if os.path.splitext(f)[1] == ".txt"]
data = {}
for p in a_patients:
    df = pd.read_csv(SETA + str(p) + ".txt", index_col=0)
    df.index = df.index.map(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    data[p] = df
for p in b_patients:
    df = pd.read_csv(SETB + str(p) + ".txt", index_col=0)
    df.index = df.index.map(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    data[p] = df
    
# %%
# variables = ["GCS", "HCT"]
datafile = DATADIR + "physionet/data.h5"
with pd.HDFStore(datafile) as hdf_file:
    for p in a_patients + b_patients:
        hdf_file[str(p)] = data[p]


