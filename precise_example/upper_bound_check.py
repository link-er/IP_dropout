import collections
import numpy as np
import math
import argparse
import matplotlib
import matplotlib.pyplot as plt
from pomegranate import *
from tqdm import *
from sklearn.covariance import ledoit_wolf
import pickle

NOISE_SAMPLES = 10
NOISE = 0.4
PLOT_VALUES = {}

parser = argparse.ArgumentParser()
parser.add_argument('--dim', dest='DIM', type=int, help='dimensionality of the input space')
parser.add_argument('--fdim', dest='F_DIM', type=int, help='dimensionality of the representation space')
args = parser.parse_args()

# freq defines on how many parts we split our data for computing entropy
# too many samples at once result in a very slow computation
def repr_entropy(nonoise_reprs, reprs):
    distrs = []
    for i in tqdm(range(len(nonoise_reprs))):
        distrs.append(MultivariateGaussianDistribution(nonoise_reprs[i], NOISE * np.diag(abs(nonoise_reprs[i]))))
    gmm_Z = GeneralMixtureModel(distrs, weights=np.full(len(distrs), (1.0 / len(distrs))))
    if len(reprs) >= 10000:
        log_probs = []
        for i in tqdm(range(int(len(reprs) / 10000))):
            log_probs += gmm_Z.log_probability(reprs[i * 10000:(i + 1) * 10000], n_jobs=10).tolist()
    else:
        log_probs = gmm_Z.log_probability(reprs, n_jobs=10)
    entropy = (-1.0/len(log_probs))*np.array(log_probs).sum()
    return entropy

upper = collections.OrderedDict()
mixture = collections.OrderedDict()
for X_SIZE in [10, 50, 100, 500, 1000, 5000, 10000]:
    Z_SIZE = X_SIZE
    mean = np.zeros(args.DIM)
    cov = np.identity(args.DIM)
    x = np.random.multivariate_normal(mean, cov, X_SIZE)

    mean = np.ones(args.F_DIM)
    cov = np.identity(args.F_DIM) * NOISE
    eps = np.random.multivariate_normal(mean, cov, Z_SIZE*NOISE_SAMPLES)
    fx = 2*x + 0.5
    z = np.repeat(fx, NOISE_SAMPLES, axis=0) * eps

    # Gaussian upper bound
    z_cov = ledoit_wolf(z)[0]
    cov_det = np.linalg.det(z_cov)
    if cov_det == 0:
        upper[X_SIZE] = -1
    else:
        upper_h_z = 0.5 * (args.F_DIM * np.log(2 * math.pi * math.e) + np.log(cov_det))
        upper[X_SIZE] = upper_h_z

    # Gaussian mixture estimate
    mixt_h_z = repr_entropy(fx, z)
    mixture[X_SIZE] = mixt_h_z

###################################
#PLOTTING
###################################
colors = matplotlib.cm.viridis(np.linspace(0.1, 0.9, 2))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Number of samples")
ax1.set_xscale('log')
ax1.set_ylabel("h(Z)")
i = 0
ax1.plot(list(upper.keys()), list(upper.values()), label="Gaussian estimate", color=colors[i])
i = 1
ax1.plot(list(mixture.keys()), list(mixture.values()), label="Mixture estimate", color=colors[i])
ax1.legend(loc='best')
plt.show()

pickle.dump(upper, open("upper_bound_hz/data_upper_dim_" + str(args.DIM) + "noise_" + str(NOISE), "wb"))
pickle.dump(mixture, open("upper_bound_hz/data_mixture_dim_" + str(args.DIM) + "noise_" + str(NOISE), "wb"))

