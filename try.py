import numpy as np
from scipy.spatial.distance import pdist,squareform

if __name__ == '__main__':
    data=np.array([[2,3,4,5],[1,2,3,4]],dtype=float)
    print(np.sum(data[0]))
    for i in range(len(data)):
        tem=np.sum(data[i])
        data[i]/=tem
    print(1j*data)

    a=[[3,4,5],[1,2,3]]
    ans=np.dot(a[0],a[1])

    onepoint_matrix=[[1,2,3],[2,3,3],[2,1,3]]
    similarity_metirc = squareform(
        pdist(onepoint_matrix, lambda u, v: np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
    a=(1e-4, 1e-3)
    print(str(a))
    pass