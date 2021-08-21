import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import norm
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from IDK2 import IDK2
from toolbox.exact_barycenter import *
from toolbox.pga import *
from toolbox.tools_WD import wfl, wwl
from toolbox.tools_signal import *
from scipy.spatial.distance import pdist, squareform
from IDK_sub import IDK_sub
import ot

from ts_IK import ts_IK


def getcyclelist(df, cycle):
    cyclelist = []
    n = len(df)
    lo = 0
    while lo + cycle <= n:
        cyclelist.append(df[lo:lo + cycle])
        lo += cycle
    return np.array(cyclelist)


def prepocessSubsequence(X, cycle):
    lo = 0
    hi = cycle
    scaler = StandardScaler()
    while lo < X.shape[0]:
        hi = min(hi, X.shape[0])
        scaler = scaler.fit(X[lo:hi])
        X[lo:hi] = scaler.transform(X[lo:hi])
        lo = hi
        hi += cycle


def compute2point(u, v):
    count = 0
    for i in range(len(u)):
        if u[i] != -1 and v[i] != -1:
            if u[i] == v[i]:
                count += 1
    return count / 100

def get_label(X, cycle, anomaly_list):
    window_num = (int)(X.shape[0] / cycle)
    label_list = np.zeros(window_num)
    label_list[anomaly_list] = 1
    return label_list

if __name__ == '__main__':
    # n = 186
    #
    # new_data = np.zeros((n, 202))
    # with open("data/Fungi_TEST.tsv") as f:
    #     lines = f.read().split('\n')[:-1]
    #     for i, line in enumerate(lines):
    #         if i == 0:  # header
    #             new_data[i, :] = line.split()
    #         else:
    #             new_data[i, :] = line.split()
    #
    # fungi = new_data[:, 0]
    # sig = np.delete(new_data, 0, 1)
    #
    # times = np.linspace(0, 1, 300)
    # #frequencies = np.linspace(-100, 100, 2001)
    # frequencies = np.linspace(-100, 100, 201)
    # psd = np.zeros((n, len(frequencies)))
    #
    # for j in np.arange(n):
    #     psd[j, :] = Signal2NPSD(frequencies, times, sig[j, :])
    #
    # mapp = matplotlib.cm.get_cmap('rainbow', 18)
    #
    # plt.figure(figsize=(20, 6))
    # for j in np.arange(n):
    #     plt.plot(times, sig[j, :], color=mapp(np.int(fungi[j])), alpha=0.5)
    # plt.title('Melt curves for 18 colour-coded species of fungies')
    # plt.show()
    #
    # bar, values_proj, eigenvectors, mean_matrix = perform_GPCA(psd, frequencies, n)
    # #np.savetxt('gpca_value.csv', values_proj, delimiter=',')
    #
    # plt.figure(figsize=(20, 10))
    # for j in np.arange(n):
    #     plt.scatter(values_proj[j, 0], values_proj[j, 1], color=mapp(np.int(fungi[j])), s=100)
    # plt.xlabel('$t_{1,n}$')
    # plt.ylabel('$t_{2,n}$')
    # plt.title('Values of the projection locations for the first and second geodesic component')
    # plt.show()
    # #similarity_matrix=wfl(list_of_TS=sig,frequencies=frequencies,times=times,gamma=1)
    # #np.savetxt('npsd.csv', psd, delimiter=',')
    # # similarity_matrix=wwl(psd, gamma=0.001, bin=len(frequencies))
    # # for i in range(len(similarity_matrix)):
    # #     similarity_matrix[i][i]=0
    # #     similarity_matrix[i]/=np.sum(similarity_matrix[i])
    # #np.savetxt('WFD_Lap_f501_gamma0.001.csv', similarity_matrix, delimiter=',')
    df = np.array(pd.read_csv("Discords_Data/dutch_power_demand.txt", header=None))

    df = np.reshape(df, (-1, 1))
    cycle = 672
    anomaly_cycles = [4,7,11]
    # 增加数据
    # for i in range(20):
    #     df=np.vstack((df,df[0:cycle*2]))

    redlist = []
    redlist.append((0, 0))
    for it in anomaly_cycles:
        redlist.append((cycle * it, cycle * it + cycle))
    redlist.append((len(df), len(df)))
    plt.plot(df, color='b')
    for it in redlist:
        ls = range(it[0], it[1])
        ly = df[ls]
        plt.plot(ls, ly, color='r')

    plt.show()
    # np.savetxt('noisysine_add30cycle12.txt',df)

    #prepocessSubsequence(df,cycle)
    # sig=getcyclelist(df,cycle)
    # onepoint_matrix=ts_IK(sig,t=200,psi=256)
    # similarity_metirc=squareform(pdist(onepoint_matrix,lambda u, v: np.sum(u==v)))
    # for i in range(len(similarity_metirc)):
    #     similarity_metirc[i][i]=0
    #     similarity_metirc[i]/=np.sum(similarity_metirc[i])
    # np.savetxt('similarity_noisy_200_256.csv', similarity_metirc, delimiter=',')
    psi = 1
    t = 100
    # for time in range(10):
    #     psi*=2
    #     onepoint_matrix=IDK_sub(np.reshape(df,(-1,1)),t=t,psi=psi,width=cycle)
    #     np.savetxt('subsequenceFM_noisy_100_'+str(psi)+'.csv', onepoint_matrix, delimiter=',')
    #     # #similarity_metirc = squareform(pdist(onepoint_matrix, lambda u, v: np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)))
    #     # similarity_metirc = squareform(pdist(onepoint_matrix, lambda u, v: np.dot(u, v)/t))
    #     # for i in range(len(similarity_metirc)):
    #     #     similarity_metirc[i][i]=0
    #     #     similarity_metirc[i]/=np.sum(similarity_metirc[i])
    #     # np.savetxt('similarity_noisyADD_200_'+str(psi)+'.csv', similarity_metirc, delimiter=',')

    psi1 = 256
    psi2 = 8
    p_list = IDK2(np.reshape(df, (-1, 1)), t=t, psi=psi1, psi2=psi2, width=cycle)
    print("auc", roc_auc_score(get_label(df, cycle, anomaly_cycles), -p_list[0]))

    onepoint_matrix_list = IDK_sub(np.reshape(df, (-1, 1)), t=t, psi=psi1, psi2=psi2, width=cycle)


    for onepoint_matrix in onepoint_matrix_list:

        similarity_metirc = squareform(pdist(onepoint_matrix, lambda u, v: compute2point(u, v)))
        for i in range(len(similarity_metirc)):
            if np.sum(similarity_metirc[i])>0:
                similarity_metirc[i] /= np.sum(similarity_metirc[i])
        np.savetxt('IDK_IK_TEK_' + str(psi1) + '_' + str(psi2) + '.csv', similarity_metirc, delimiter=',')
        psi2 *= 2
