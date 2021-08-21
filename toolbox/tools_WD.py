import numpy as np
import ot

from toolbox.tools_signal import Signal2NPSD


def Wasserstein_distance(label_sequences, sinkhorn=False, sinkhorn_lambda=1e-2, bin=10):
    """
    Generate the Wasserstein distance matrix for the subsequences
    """
    n = len(label_sequences)
    M = np.zeros((n, n))
    nbins = bin
    xxx = np.arange(nbins, dtype=np.float64)
    costs = ot.dist(xxx.reshape((nbins, 1)), xxx.reshape((nbins, 1)))
    costs /= costs.max()
    for subseque_index_1, subseque_1 in enumerate(label_sequences):
        for subseque_index_2, subseque_2 in enumerate(label_sequences[subseque_index_1:]):
            if sinkhorn:
                mat = ot.sinkhorn(np.ones(len(subseque_1)) / len(subseque_1),
                                  np.ones(len(subseque_2)) / len(subseque_2), costs, sinkhorn_lambda,
                                  numItermax=50)
                M[subseque_index_1, subseque_index_2 + subseque_index_1] = np.sum(np.multiply(mat, costs))
            else:
                M[subseque_index_1, subseque_index_2 + subseque_index_1] = \
                    ot.emd2(subseque_1, subseque_2, costs)
    M = (M + M.T)
    return M

def wwl(list_of_distributions, sinkhorn=False, sinkhorn_lambda=1e-2, gamma=None, bin=10):
    """
    using laplacian_kernel ,,, cost matrix
    return kernel matrix of shape (n_distributions, n_distributions)
    """
    D_W = Wasserstein_distance(list_of_distributions, sinkhorn, sinkhorn_lambda, bin=bin)
    # wwl = laplacian_kernel(D_W, gamma=gamma)
    wwl = np.exp(-D_W / gamma)
    return wwl

def wfl(list_of_TS, frequencies, times, sinkhorn=False, sinkhorn_lambda=1e-2, gamma=1):
    all_TS_pdf = []
    for sign in list_of_TS:
        sign_distri = Signal2NPSD(frequencies, times, sign)
        all_TS_pdf.append(sign_distri)
    similarity_matrix = wwl(all_TS_pdf, sinkhorn=sinkhorn, sinkhorn_lambda=sinkhorn_lambda,gamma=gamma,bin=len(frequencies))
    return similarity_matrix




