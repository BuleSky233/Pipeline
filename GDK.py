import math

import numpy as np
from sklearn.kernel_approximation import Nystroem
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tools
from IDK import IDK
from tools import *


def GDK(X, gamma, components):
    nystroem2 = Nystroem(gamma=gamma, n_components=components)
    data_transformed_2 = nystroem2.fit_transform(X)
    mean_map_all = np.mean(data_transformed_2, axis=0)
    gdk2_result = np.dot(data_transformed_2, mean_map_all)
    return gdk2_result


def GDK_square(X, cycle, gamma1, components1, gamma2, components2):
    nystroem = Nystroem(gamma=gamma1, n_components=components1)
    data_transformed = nystroem.fit_transform(X)
    feature_map_subsequence = []

    i = 0
    while i + cycle <= X.shape[0]:
        # tem=data_transformed[i]
        # for j in range(i+1,i+cycle):
        #     tem+=data_transformed[j]

        feature_map_subsequence.append(np.mean(data_transformed[i:i + cycle, :], axis=0))
        i += cycle

    feature_map_subsequence = np.array(feature_map_subsequence)
    return GDK(feature_map_subsequence, gamma2, components2)






if __name__ == '__main__':



    # p = IDK(feature_map_subsequence, t=100, psi=4)
    # print(roc_auc_score(get_label(df, cycle, anomaly_cycles), -p))
    #GDK_square_Exp('Patient_respiration',150,[6,33],'atry')
    pass
