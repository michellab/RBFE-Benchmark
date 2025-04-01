#!/usr/bin/python

# libraries

# import libraries
import BioSimSpace as BSS
import os
import sys
import glob
import csv
import numpy as np
import networkx as nx
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale, MinMaxScaler
import itertools

import warnings

warnings.filterwarnings("ignore")

import pipeline

from pipeline.prep import *
from pipeline.utils import *
from pipeline.analysis import *

import random
import math
import pandas as pd
import subprocess

import networkanalysis

from cinnabar import wrangle as _wrangle

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from scipy.stats import sem as sem
from scipy.stats import bootstrap, norm
from scipy.stats import spearmanr

import networkx as nx
import numpy as np
import scipy
import sklearn.metrics
from typing import Union


def compute_statistic(
    y_true_sample: np.ndarray, y_pred_sample: np.ndarray, statistic: str
):
    """Compute requested statistic.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values
    statistic : str
        Statistic, one of ['RMSE', 'MUE', 'R2', 'rho','RAE','KTAU']

    """

    def calc_RAE(y_true_sample: np.ndarray, y_pred_sample: np.ndarray):
        MAE = sklearn.metrics.mean_absolute_error(y_true_sample, y_pred_sample)
        mean = np.mean(y_true_sample)
        MAD = np.sum([np.abs(mean - i) for i in y_true_sample]) / float(
            len(y_true_sample)
        )
        return MAE / MAD

    def calc_RRMSE(y_true_sample: np.ndarray, y_pred_sample: np.ndarray):
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true_sample, y_pred_sample))
        mean_exp = np.mean(y_true_sample)
        mds = np.sum([(mean_exp - i) ** 2 for i in y_true_sample]) / float(
            len(y_true_sample)
        )
        rrmse = np.sqrt(rmse**2 / mds)
        return rrmse

    if statistic == "RMSE":
        return np.sqrt(sklearn.metrics.mean_squared_error(y_true_sample, y_pred_sample))
    elif statistic == "MUE":
        return sklearn.metrics.mean_absolute_error(y_true_sample, y_pred_sample)
    elif statistic == "R2":
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            y_true_sample, y_pred_sample
        )
        return r_value**2
    elif statistic == "rho":
        return scipy.stats.pearsonr(y_true_sample, y_pred_sample)[0]
    elif statistic == "RAE":
        return calc_RAE(y_true_sample, y_pred_sample)
    elif statistic == "KTAU":
        return scipy.stats.kendalltau(y_true_sample, y_pred_sample)[0]
    else:
        raise Exception("unknown statistic '{}'".format(statistic))


# dictionary
best_fit_dict = {}

protein = sys.argv[1]
lr = sys.argv[2]  # lomap or rbfenn

exec_folder = f"/home/anna/Documents/benchmark/reruns/{protein}/execution_model"

exper_val_dict = None

repeat_dict = {}
repeat = 0

if lr == "lomap":
    if protein == "tyk2":
        no_perts = 24
    elif protein == "mcl1":
        no_perts = 15
    elif protein == "p38":
        no_perts = 34
    elif protein == "cmet":
        no_perts = 12
    elif protein == "syk":
        no_perts = 35
    elif protein == "hif2a":
        no_perts = 31
elif lr == "rbfenn":
    if protein == "tyk2":
        no_perts = 30
    elif protein == "mcl1":
        no_perts = 20
    elif protein == "p38":
        no_perts = 51
elif lr == "flare":
    if protein == "tyk2":
        no_perts = 24
    elif protein == "mcl1":
        no_perts = 19
    elif protein == "p38":
        no_perts = 62

if lr == "flare":
    extensions = [""]
else:
    if protein == "tyk2" or protein == "mcl1":
        extensions = ["", "-a-optimal", "-d-optimal"]
    else:
        extensions = [""]

for file_ext in extensions:
    repeat_dict[f"{lr}{file_ext}"] = {}

random_ref = True
if random_ref:
    extensions = ["-a-optimal", "-d-optimal"]

while repeat < 5001:
    if random_ref:
        print("running the R script...")
        command = (
            "/usr/bin/Rscript /home/anna/Documents/other_workflows/yang2020_optimal_designs/me/optimal_designs_prenorm.R %s %s %s"
            % (protein, lr, no_perts)
        )

        try:
            result = subprocess.run(command, shell=True, capture_output=True)
            print("done w R")
        except:
            print("failed to run R script.")
            continue

    for file_ext in extensions:
        network_name = f"{lr}{file_ext}"

        print(f"{protein}, {network_name}, {repeat}")

        if random_ref:
            file = f"/home/anna/Documents/other_workflows/yang2020_optimal_designs/me/{protein}/network_{lr}{file_ext}.dat"

            commands = [
                "sed -i 's/LIGAND_1/lig0/g' %s" % (file),
                "sed -i 's/LIGAND_2/lig1/g' %s" % (file),
            ]

            for command in commands:
                try:
                    result = subprocess.run(command, shell=True, capture_output=True)
                except:
                    continue

        else:
            file = f"/home/anna/Documents/benchmark/reruns/{protein}/execution_model/network_{network_name}.dat"

        perts, ligs = pipeline.utils.get_info_network(f"{file}")

        if not exper_val_dict:
            exp_file = (
                f"/home/anna/Documents/benchmark/inputs/experimental/{protein}.yml"
            )
            try:
                exper_val_dict = pipeline.analysis.convert.yml_into_exper_dict(
                    exp_file,
                    temperature=300,
                )
            except Exception as e:
                print(e)
                exper_val_dict = pipeline.analysis.convert._read_yml_kcal(exp_file)

            normalised_exper_val_dict = pipeline.analysis.make_dict.exper_from_ligands(
                exper_val_dict, sorted(ligs), normalise=True
            )
        else:
            pass

        pert_dict = pipeline.analysis.make_dict.exper_from_perturbations(
            exper_val_dict, perts
        )

        # create fep pairwise ddG by randomly adding error to true value, which here is taken to be the experimental
        # sigmna2 fep is 1.0
        variance = 1
        pert_dict_fep = {}
        for pert in pert_dict:
            try:
                true_val = pert_dict[pert][0]
                # make value centred on experimental with variance sigma2fep
                rand_val = random.normalvariate(mu=true_val, sigma=math.sqrt(variance))
                pert_dict_fep[pert] = rand_val
            except:
                pert_dict_fep[pert] = None

        # analyse using cinnabar
        # get the files into cinnabar format for analysis
        cinnabar_file_name = f"{protein}_{network_name}_cinnabar_file.csv"

        df = pd.DataFrame.from_dict(pert_dict_fep, orient="index").reset_index()
        df[["lig_0", "lig_1"]] = df["index"].str.split("~", expand=True)
        df = df.drop(columns="index")
        df = df.rename(columns={0: "freenrg"})
        df["error"] = 0.5
        df["engine"] = "SOMD"
        df = df[["lig_0", "lig_1", "freenrg", "error", "engine"]]

        # write into file for network analysis
        new_file_name = f"{protein}_{network_name}_fwf_file.csv"
        pd.DataFrame.to_csv(df, new_file_name, sep=",", index=False)

        exper_dict = copy.deepcopy(exper_val_dict)
        for key in exper_val_dict.keys():
            if key not in ligs:
                exper_dict.pop(key)
                logging.info(f"removed {key} from the cinnabar dict as no pert values")

        convert.cinnabar_file(
            [new_file_name],
            exper_dict,
            cinnabar_file_name,
            perturbations=perts,
            method=None,
        )

        network = _wrangle.FEMap(f"{cinnabar_file_name}.csv")

        # for self plotting of per ligand
        freenrg_dict = make_dict.from_cinnabar_network_node(network, "calc")
        normalised_exper_val_dict = make_dict.from_cinnabar_network_node(
            network, "exp", normalise=True
        )

        x = [normalised_exper_val_dict[x][0] for x in ligs if "Intermediate" not in x]
        y = [freenrg_dict[x][0] for x in ligs if "Intermediate" not in x]
        dg_error = compute_statistic(x, y, statistic="MUE")
        r2_error = compute_statistic(x, y, statistic="R2")
        ktau_error = compute_statistic(x, y, statistic="KTAU")

        x = [pert_dict[x][0] for x in perts if "Intermediate" not in x]
        y = [pert_dict_fep[x] for x in perts if "Intermediate" not in x]
        ddg_error = compute_statistic(x, y, statistic="MUE")
        ddg_rmse = compute_statistic(x, y, statistic="RMSE")

        # x = [x[0] for x in normalised_exper_val_dict.values()]
        # y = [y[0] for y in freenrg_dict.values()]
        # pipeline.analysis.stats_engines.compute_stats(x=x,y=y, statistic="MUE")
        # dg_error_mse = mse(x, y)

        # x = [x[0] for x in pert_dict.values()]
        # y = [y for y in pert_dict_fep.values()]
        # pipeline.analysis.stats_engines.compute_stats(x=x,y=y, statistic="MUE")
        # ddg_error_mse = mse(x, y)

        # x = [x[0] for x in normalised_exper_val_dict.values()]
        # y = [y[0] for y in freenrg_dict.values()]
        # pipeline.analysis.stats_engines.compute_stats(x=x,y=y, statistic="MUE")
        # coef, p = spearmanr(x, y)

        print((dg_error, ddg_error, ddg_rmse, r2_error, ktau_error))
        repeat_dict[network_name][repeat] = (
            dg_error,
            ddg_error,
            ddg_rmse,
            r2_error,
            ktau_error,
        )

        repeat += 1

for file_ext in extensions:  # ,"-a-optimal","-d-optimal"
    # make a df
    network_name = f"{lr}{file_ext}"

    df = pd.DataFrame.from_dict(
        repeat_dict[f"{network_name}"],
        columns=[
            "mae_dG",
            "mae_ddG",
            "rmse_ddG",
            "r2_dG",
            "ktau_dG",
        ],
        orient="index",
    )
    df.index.name = "repeat"
    if random_ref:
        df.to_csv(f"{exec_folder}/network_{network_name}_stats_random_ref.csv")
    else:
        df.to_csv(f"{exec_folder}/network_{network_name}_stats.csv")
