import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.colors as mcolors

from toolbox.exact_barycenter import *
from toolbox.pga import *
from toolbox.tools_WD import wfl, wwl
from toolbox.tools_signal import *
from scipy.spatial.distance import pdist,squareform

import ot


def getcyclelist(df,cycle):
    cyclelist=[]
    n=len(df)
    lo=0
    while lo+cycle<=n:
        cyclelist.append(df[lo:lo+cycle])
        lo+=cycle
    return np.array(cyclelist)


if __name__ == '__main__':



    df = pd.read_csv("Discords_Data/chfdbchf15_2.txt", header=None)
    df = np.array(df).reshape(df.shape[0])

    times = np.linspace(0, 1, 150)
    #frequencies = np.linspace(-100, 100, 2001)

    cycle=150
    sig=getcyclelist(df,cycle)
    n=len(sig)
    frequencies = np.linspace(-100, 100, 100)
    psd = np.zeros((n, len(frequencies)))
    for j in np.arange(n):
        psd[j, :] = Signal2NPSD(frequencies, times, sig[j, :])
    np.savetxt('npsd_chf_100.csv', psd, delimiter=',')

