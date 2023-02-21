import math
import numpy as np
from tqdm import *
from pomegranate import *

# define how many Gaussians are in the GMM to estimate h(Z)
# doing it for all the datapoints (i.e., Gaussian for each datapoint) is usually very expensive
GMM_MEANS_NUM = 1000
# specify the dropout noise variance that was used for training in order to generate noisy samples
p = 0.01
drp_noise = p/(1-p)

def gaussian_noise_mi(reprs, nonoise_reprs, noise_variance):
    f_dim = reprs.shape[1]
    h_z = repr_entropy(nonoise_reprs, reprs)
    # our innovative way to compute conditional entropy
    h_zGivx = 0
    h_z_part = math.sqrt(2*math.pi*math.e)*noise_variance
    for i in range(f_dim):
        # we have simple multiplication of 1-dim gaussian (noise) by a constant (current value of activations)
        h_zGivx += (1.0/len(nonoise_reprs)) * np.sum(np.log(h_z_part * np.fabs(nonoise_reprs[:,i] + 1e-6)))
    return h_z - h_zGivx

def repr_entropy(nonoise_reprs, reprs, ratio_points=GMM_MEANS_NUM):
    dists = []
    used_points = nonoise_reprs[:ratio_points]
    for i in tqdm(range(len(used_points))):
        dists.append(MultivariateGaussianDistribution(used_points[i], drp_noise*np.diag(abs(used_points[i]))))
    gmm_Z = GeneralMixtureModel(dists, weights=np.full(len(dists), (1.0 / len(dists))))
    log_probs = []
    for i in tqdm(range(int(len(reprs)/10000))):
        log_probs += gmm_Z.log_probability(reprs[i*10000:(i+1)*10000], n_jobs=10).tolist()
    return (-1.0/len(log_probs))*np.array(log_probs).sum()