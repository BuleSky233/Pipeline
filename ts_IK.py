import numpy as np
import random

def ts_IK(ts_list, t, psi):
    featuremap_count = np.zeros(t * psi)
    # onepoint_matrix[i]记录第i个点map到的t*psi维向量里哪些位置为1，onepoint_matrix[i][j]: area number of the i-th point in the j-th partition
    # 如果为初始值-1则表示该点在第time次映射到全0向量
    X=ts_list.reshape((-1,1))
    onepoint_matrix = np.full((X.shape[0], t), -1)
    pre_scores = np.zeros(X.shape[0])
    onepoint_matrix = np.full((X.shape[0], t), -1)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[0]):
    #         if i < j:
    #             distance_matrix[i][j] = np.linalg.norm(X[i] - X[j])
    #         else:
    #             distance_matrix[i][j] = distance_matrix[j][i]
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(X.shape[0])]  # [0, 1, 2, 3]
        sample_list = random.sample(sample_list, sample_num)  # [1, 2]
        sample = X[sample_list, :]  # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi
        min_dist_point2sample = np.argmin(point2sample, axis=1)  # index
        onepoint_matrix[:,time]=min_dist_point2sample

    return onepoint_matrix.reshape((len(ts_list),-1))



