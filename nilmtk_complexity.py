from __future__ import print_function
import sys
import numpy as np
from sklearn.utils.extmath import cartesian
from scipy.stats import norm
from nilmtk.feature_detectors.steady_states import cluster

def statesCombinations(meterlist):
    """Returns all possible levels of the aggregated signal, by finding all
    combinations of all possible appliance states

    Args:
        meterlist (list): List of ElecMeters

    Returns:
        list: all possible levels
    """
    max_states = 7
    states = [None] * len(meterlist)
    for i, meter in enumerate(meterlist):
        states[i] = cluster(meter.power_series().next(), max_num_clusters=max_states)

    return np.sum(cartesian(states), axis=1)

def compute(metergroup):
    """Computes the power disaggregation complexity as described in
    https://arxiv.org/pdf/1501.02954.pdf

    Args:
        metergroup (MeterGroup): A MeterGroup to compute the complexity on

    Returns:
        (float, float): (max, mean) disaggregation complexity of the given metergroup
    """
    std = 5
    meterlist = metergroup.submeters().all_meters()

    print("Finding appliance states...")
    # All of possible appliance states
    P = statesCombinations(meterlist)
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
