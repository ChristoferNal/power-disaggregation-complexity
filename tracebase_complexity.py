from __future__ import print_function
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.utils.extmath import cartesian
from nilmtk.feature_detectors.steady_states import cluster

def statesCombinations(meterlist):
    """Returns all possible levels of the aggregated signal, by finding all
    combinations of all possible appliance states

    Args:
        meterlist (list): List of paths to Tracebase directories of each meter

    Returns:
        list: all possible levels
    """
    max_states = 7
    states = [None] * len(meterlist)
    for i, meter in enumerate(meterlist):
        states[i] = cluster(power_series(meter), max_num_clusters=max_states)

    return np.sum(cartesian(states), axis=1)

def power_series(folder):
    """Creates a timeseries from Tracebase files

    Args:
        folder (str): Path to the folder containing the trace files

    Returns:
        Pandas DataFrame: Timeseries of the data
    """
    files = [fn for fn in os.listdir(folder) if fn.startswith('dev')]
    a = np.array([])
    for f in files:
        a = np.append(a,  np.loadtxt(os.path.join(folder, f), delimiter=';', usecols=(2,)))

    df = pd.DataFrame(data=a, index=range(len(a)), columns=['power'])
    return df

def compute(meterpaths):
    """Computes the power disaggregation complexity as described in
    https://arxiv.org/pdf/1501.02954.pdf

    Args:
        meterpaths (list of str): A list of paths to folders in the Tracebase dataset

    Returns:
        (float, float): (max, mean) disaggregation complexity of the given set of meters
    """
    std = 5

    print("Finding appliance states...")
    # All of possible appliance states
    P = statesCombinations(meterpaths)
    Pm = np.max(P)

    print("Computing complexity for each state...")
    # Compute Ck for each state
    C = np.zeros(len(P))
    x1 = np.linspace(0, Pm, 1000)
    for k in range(1,len(P)):
        print(" {} of {}".format(k+1,len(P)), end="\r")
        sys.stdout.flush()
        for j in range(1,len(P)):
            y1 = np.minimum(norm.pdf(x1, P[k], std), norm.pdf(x1, P[j], std))
            C[k] = C[k] + np.trapz(y1,x1)

    return np.max(C[1:]), np.mean(C[1:])
