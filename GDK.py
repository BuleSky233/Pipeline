import math

import numpy as np
from sklearn.kernel_approximation import Nystroem
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
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


def GDK_square_Exp(data_name, cycle, ano_cycles, outfolder):
    from os import listdir
    from os.path import isfile, join
    # data
    onlyfiles = [f for f in listdir("Discords_Data") if isfile(join("Discords_Data", f))]
    onlynames=[f.split('.')[0] for f in onlyfiles ]
    if data_name not in onlynames:
        return -2
    ind=onlynames.index(data_name)

    df = np.array(pd.read_csv("Discords_Data/" + onlyfiles[ind], header=None))
    df = np.reshape(df, (-1, 1))
    gamma_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    df = prepocessSubsequence(df, cycle)
    best=-1
    best_para=(-1,-1)
    labels=get_label(df, cycle, ano_cycles)
    for i in range(len(gamma_list)):
        for j in range(len(gamma_list)):
            score=0
            for time in range(10):
                result=GDK_square(X=df, cycle=cycle, gamma1=gamma_list[i], components1=math.ceil(np.sqrt(len(df))),
                           gamma2=gamma_list[j], components2=math.ceil(np.sqrt((int)(len(df) / cycle))))
                score+=roc_auc_score(labels,-result)
            score/=10
            if score>best:
                best=score
                best_para=(i,j)
    best_paraval=(gamma_list[best_para[0]],gamma_list[best_para[1]])
    outputfile=outfolder+"/"+data_name+'.txt'
    with open(outputfile, "w") as f:
        f.write('auc='+str(best)+'\n'+'gamma='+(str)(best_paraval))
    return best



if __name__ == '__main__':



    # p = IDK(feature_map_subsequence, t=100, psi=4)
    # print(roc_auc_score(get_label(df, cycle, anomaly_cycles), -p))
    #GDK_square_Exp('Patient_respiration',150,[6,33],'atry')
    pass
