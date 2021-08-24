import math
from time import time

import numpy as np
from sklearn.kernel_approximation import Nystroem
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tools

from tools import *


def GDK_square_Exp(data_name, cycle, ano_cycles, outfolder):
    from GDK import GDK_square
    from os import listdir
    from os.path import isfile, join
    # data
    onlyfiles = [f for f in listdir("Discords_Data") if isfile(join("Discords_Data", f))]
    onlynames = [f.split('.')[0] for f in onlyfiles]
    if data_name not in onlynames:
        return -2
    ind = onlynames.index(data_name)

    df = np.array(pd.read_csv("Discords_Data/" + onlyfiles[ind], header=None))
    df = np.reshape(df, (-1, 1))
    gamma_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    df = prepocessSubsequence(df, cycle)
    # df=subsequenceMinMaxScale(df,cycle)
    # df=StandardScaler().fit_transform(df)
    # df=MinMaxScaler().fit_transform(df)
    best = -1
    best_para = (-1, -1)
    labels = get_label(df, cycle, ano_cycles)
    for i in range(len(gamma_list)):
        for j in range(len(gamma_list)):
            score = 0
            for time in range(10):
                result = GDK_square(X=df, cycle=cycle, gamma1=gamma_list[i], components1=math.ceil(np.sqrt(len(df))),
                                    gamma2=gamma_list[j], components2=math.ceil(np.sqrt((int)(len(df) / cycle))))
                score += roc_auc_score(labels, -result)
            score /= 10
            if score > best:
                best = score
                best_para = (i, j)
    best_paraval = (gamma_list[best_para[0]], gamma_list[best_para[1]])
    outputfile = outfolder + "/" + data_name + '.txt'
    with open(outputfile, "w") as f:
        f.write('auc=' + str(best) + '\n' + 'gamma=' + (str)(best_paraval))
    return best


def IDK_square_Exp(data_name, cycle, ano_cycles, outfolder):
    from IDK_T import IDK_T
    from os import listdir
    from os.path import isfile, join
    # data
    onlyfiles = [f for f in listdir("Discords_Data") if isfile(join("Discords_Data", f))]
    onlynames = [f.split('.')[0] for f in onlyfiles]
    if data_name not in onlynames:
        return -2
    ind = onlynames.index(data_name)

    df = np.array(pd.read_csv("Discords_Data/" + onlyfiles[ind], header=None))
    df = np.reshape(df, (-1, 1))
    psi1_list = [2, 4, 8, 16, 32, 64, 128, 256]
    psi2_list = [2, 4, 8, 16, 32, 64, 128]
    df = prepocessSubsequence(df, cycle)

    best = -1
    best_para = (-1, -1)

    labels = get_label(df, cycle, ano_cycles)
    flag = False
    for i in range(len(psi1_list)):
        if flag is True or psi1_list[i]>=len(df):
            break
        for j in range(len(psi2_list)):
            if psi2_list[j] >= (int)(len(df) / cycle):
                break
            score = 0
            for time in range(10):
                result = IDK_T(X=df, t=100, width=cycle, psi=psi1_list[i],
                               psi2=psi2_list[j])
                score += roc_auc_score(labels, -result)
            score /= 10
            if score > best:
                best = score
                best_para = (i, j)
            if best == 1:
                flag = True
                break
    best_paraval = (psi1_list[best_para[0]], psi2_list[best_para[1]])
    outputfile = outfolder + "/" + data_name + '.txt'
    with open(outputfile, "w") as f:
        f.write('auc=' + str(best) + '\n' + 'psi=' + (str)(best_paraval))
    return best


def GDK_IDK_exp(data_name, cycle, ano_cycles, outfolder):
    from GDK_IDK import GDK_IDK
    from os import listdir
    from os.path import isfile, join
    # data
    onlyfiles = [f for f in listdir("Discords_Data") if isfile(join("Discords_Data", f))]
    onlynames = [f.split('.')[0] for f in onlyfiles]
    if data_name not in onlynames:
        return -2
    ind = onlynames.index(data_name)

    df = np.array(pd.read_csv("Discords_Data/" + onlyfiles[ind], header=None))
    df = np.reshape(df, (-1, 1))
    gamma_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    psi_list = [2, 4, 8, 16, 32, 64, 128, 256,512,1024]
    df = prepocessSubsequence(df, cycle)

    best = -1
    best_para = (-1, -1)
    labels = get_label(df, cycle, ano_cycles)
    for i in range(len(gamma_list)):
        for j in range(len(psi_list)):
            if psi_list[j] >= (int)(len(df) / cycle):
                break
            score = 0
            for time in range(10):
                result = GDK_IDK(X=df,cycle=cycle,t=100,psi=psi_list[j],components=math.ceil(np.sqrt(len(df))),gamma=gamma_list[i])
                score += roc_auc_score(labels, -result)
            score /= 10
            if score > best:
                best = score
                best_para = (i, j)
    best_paraval = (gamma_list[best_para[0]], psi_list[best_para[1]])
    outputfile = outfolder + "/" + data_name + '.txt'
    with open(outputfile, "w") as f:
        f.write('auc=' + str(best) + '\n' + 'gamma,psi=' + (str)(best_paraval))
    return best

def IDK_GDK_exp(data_name, cycle, ano_cycles, outfolder):
    from IDK_GDK import IDK_GDK
    from os import listdir
    from os.path import isfile, join
    # data
    onlyfiles = [f for f in listdir("Discords_Data") if isfile(join("Discords_Data", f))]
    onlynames = [f.split('.')[0] for f in onlyfiles]
    if data_name not in onlynames:
        return -2
    ind = onlynames.index(data_name)

    df = np.array(pd.read_csv("Discords_Data/" + onlyfiles[ind], header=None))
    df = np.reshape(df, (-1, 1))
    gamma_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    psi_list = [2, 4, 8, 16, 32, 64, 128, 256,512,1024,2048,4096]
    df = prepocessSubsequence(df, cycle)

    best = -1
    best_para = (-1, -1)
    labels = get_label(df, cycle, ano_cycles)
    for i in range(len(gamma_list)):
        for j in range(len(psi_list)):
            if psi_list[j] >= (int)(len(df)):
                break
            score = 0
            for time in range(10):
                result = IDK_GDK(X=df,cycle=cycle,t=100,psi=psi_list[j],components=math.ceil(np.sqrt((int)(len(df) / cycle))),gamma=gamma_list[i])
                score += roc_auc_score(labels, -result)
            score /= 10
            if score > best:
                best = score
                best_para = (i, j)
    best_paraval = (gamma_list[best_para[0]], psi_list[best_para[1]])
    outputfile = outfolder + "/" + data_name + '.txt'
    with open(outputfile, "w") as f:
        f.write('auc=' + str(best) + '\n' + 'gamma,psi=' + (str)(best_paraval))
    return best

if __name__ == '__main__':
   pass
