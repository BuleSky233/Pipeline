from GDK import GDK
from IDK import IDK
from sklearn.kernel_approximation import Nystroem
import numpy as np

def GDK_IDK(X,cycle,t,psi,components,gamma):
    nystroem = Nystroem(gamma=gamma, n_components=components)
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
    return IDK(feature_map_subsequence,t=t,psi=psi)
