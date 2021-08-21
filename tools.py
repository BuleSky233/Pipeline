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

