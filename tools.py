import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

def get_label(X, cycle, anomaly_list):
    window_num = (int)(X.shape[0] / cycle)
    label_list = np.zeros(window_num)
    label_list[anomaly_list] = 1
    return label_list

def prepocessSubsequence(X, cycle):
    lo = 0
    hi = cycle
    scaler = StandardScaler()
    ans=np.ones(X.shape)
    while lo < X.shape[0]:
        hi = min(hi, X.shape[0])
        scaler = scaler.fit(X[lo:hi])
        ans[lo:hi] = scaler.transform(X[lo:hi])
        lo = hi
        hi += cycle
    return ans

def subsequenceMinMaxScale(X, cycle):
    lo = 0
    hi = cycle
    scaler = preprocessing.MinMaxScaler()
    ans = np.ones(X.shape)
    while lo < X.shape[0]:
        hi = min(hi, X.shape[0])
        ans[lo:hi] = scaler.fit_transform(X[lo:hi])
        lo = hi
        hi += cycle
    return ans

def subsequenceFluc(X, width):

    if X.shape[0] % width != 0: print("不能整除划分")

    window_num = (int)(X.shape[0] / width)

    fluc_list = np.zeros((window_num, width - 1))
    for i in range(0, window_num * width):
        if i % width != 0:
            fluc_list[(int)(i / width)][(i % width) - 1] = X[i] - X[i - 1]
    return fluc_list