import collections
import numpy as np
import math
import sys
sys.path.append('../mi_estimators')
from mi_estimators_local.EDGE_4_3_1 import EDGE
from mi_estimators_local.npeet.entropy_estimators import midd
import argparse
import matplotlib
import matplotlib.pyplot as plt
from pomegranate import *
from tqdm import *
from sklearn.covariance import ledoit_wolf
import pickle

CMAP = matplotlib.cm.get_cmap('viridis')
DRAW_HISTS = False
DRAW_BINS = 200
X_SIZE = 1000
Z_SIZE = 1000
NOISE_SAMPLES = 10
NOISE = 0.4
PLOT_VALUES = {}

parser = argparse.ArgumentParser()
parser.add_argument('--dim', dest='DIM', type=int, help='dimensionality of the input space')
parser.add_argument('--fdim', dest='F_DIM', type=int, help='dimensionality of the representation space')
args = parser.parse_args()

def draw_hist(data, title="Histogram", bins_num=DRAW_BINS):
    if not DRAW_HISTS:
        return
    colors_num = data.shape[1]
    colors = CMAP(list(np.arange(0.0, 1.0, 1.0 / colors_num)))
    plt.hist(data, bins=bins_num, color=colors)
    plt.title(title)
    plt.show()

mean = np.zeros(args.DIM)
cov = np.identity(args.DIM)
x = np.random.multivariate_normal(mean, cov, X_SIZE)
draw_hist(x, title = "Input data X")

mean = np.ones(args.F_DIM)
cov = np.identity(args.F_DIM) * NOISE
eps = np.random.multivariate_normal(mean, cov, Z_SIZE*NOISE_SAMPLES)
fx = 2*x + 0.5
z = np.repeat(fx, NOISE_SAMPLES, axis=0) * eps
draw_hist(z, title = "Representation data Z")

def repr_entropy(nonoise_reprs, reprs):
    distrs = []
    for i in tqdm(range(len(nonoise_reprs))):
        distrs.append(MultivariateGaussianDistribution(nonoise_reprs[i], NOISE*np.diag(abs(nonoise_reprs[i])+1e-6)))
    gmm_Z = GeneralMixtureModel(distrs, weights=np.full(len(distrs), (1.0 / len(distrs))))
    if len(reprs) >= 100000:
        log_probs = []
        for i in tqdm(range(int(len(reprs) / 50000))):
            log_probs += gmm_Z.log_probability(reprs[i * 50000:(i + 1) * 50000], n_jobs=10).tolist()
    else:
        log_probs = gmm_Z.log_probability(reprs, n_jobs=10)
    return (-1.0/len(log_probs))*np.array(log_probs).sum()

def gaussian_noise_mi(reprs, nonoise_reprs, noise_variance, method="upper"):
    f_dim = reprs.shape[1]
    if method=="upper":
        # Gaussian upper bound
        z_cov = ledoit_wolf(reprs)[0]
        cov_det = np.linalg.det(z_cov)
        h_z = 0.5*(f_dim*np.log(2*math.pi*math.e)+np.log(cov_det))
    else:
        h_z = repr_entropy(nonoise_reprs, reprs)
    # our innovative way to compute conditional entropy
    h_zGivx = 0
    h_z_part = math.sqrt(2*math.pi*math.e)*noise_variance
    for i in range(f_dim):
        # we have simple multiplication of 1-dim gaussian (noise) by a constant (current value of activations)
        h_zGivx += (1.0/len(nonoise_reprs)) * np.sum(np.log(h_z_part * np.fabs(nonoise_reprs[:,i] + 1e-6)))
    print("h_z", h_z, "h_zGivx", h_zGivx)
    return h_z - h_zGivx

drp_mi_upper = gaussian_noise_mi(z, fx, NOISE)
print(drp_mi_upper)
PLOT_VALUES["our (upper)"] = drp_mi_upper

drp_mi_mixt = gaussian_noise_mi(z, fx, NOISE, method="mixture")
print(drp_mi_mixt)
PLOT_VALUES["our (mixture)"] = drp_mi_mixt

###### DIFFERENT METHODS

inputdata = np.repeat(x, NOISE_SAMPLES, axis=0)
layerdata = z

edge_mi = EDGE(inputdata,layerdata)
print(edge_mi)
PLOT_VALUES["EDGE"] = edge_mi

def create_bins(min_bound, max_bound, num_of_bins=None, bin_size=None):
    if bin_size is not None:
        bins = np.arange(min_bound, max_bound, bin_size, dtype='float32')
    elif num_of_bins is not None:
        bins = np.linspace(min_bound, max_bound, num_of_bins, dtype='float32')
    else:
        print("Computation error; set either bin size or number of bins to a value")
        return None
    return bins

binning_by_num = collections.OrderedDict()
for num_of_bins in [2,5,10,15,20,50,100]:
    bins_inp = create_bins(inputdata.min(), inputdata.max(), num_of_bins=num_of_bins)
    digitized_inp = bins_inp[np.digitize(np.squeeze(inputdata.reshape(1, -1)), bins_inp) - 1].reshape(len(inputdata), -1)
    bins_rep = create_bins(layerdata.min(), layerdata.max(), num_of_bins=num_of_bins)
    digitized_rep = bins_rep[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins_rep) - 1].reshape(len(layerdata), -1)
    bin_mi = midd(digitized_inp,digitized_rep,base=np.exp(1))
    print(bin_mi)
    binning_by_num[num_of_bins] = bin_mi

binning_by_size = collections.OrderedDict()
for bin_size in [0.9, 0.5, 0.2, 0.1, 0.001]:
    bins_inp = create_bins(inputdata.min(), inputdata.max(), bin_size=bin_size)
    digitized_inp = bins_inp[np.digitize(np.squeeze(inputdata.reshape(1, -1)), bins_inp) - 1].reshape(len(inputdata), -1)
    bins_rep = create_bins(layerdata.min(), layerdata.max(), bin_size=bin_size)
    digitized_rep = bins_rep[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins_rep) - 1].reshape(len(layerdata), -1)
    bin_mi = midd(digitized_inp,digitized_rep,base=np.exp(1))
    print(bin_mi)
    binning_by_size[bin_size] = bin_mi

###################################
#PLOTTING
###################################
# the values are produced using the official github for Difference-of-Entropies (DoE) Estimator https://github.com/karlstratos/doe
#PLOT_VALUES["doe"] = 1.61 # noise=0.1, dim=1
#PLOT_VALUES["doe_l"] = 1.7 # noise=0.1, dim=1
#PLOT_VALUES["doe"] = 8.21 # noise=0.1, dim=5
#PLOT_VALUES["doe_l"] = 8.22 # noise=0.1, dim=5
#PLOT_VALUES["doe"] = 15.88 # noise=0.1, dim=10
#PLOT_VALUES["doe_l"] = 15.84 # noise=0.1, dim=10
#PLOT_VALUES["doe"] = 66.94 # noise=0.1, dim=50
#PLOT_VALUES["doe_l"] = 61.6 # noise=0.1, dim=50
#PLOT_VALUES["doe"] = 97.89 # noise=0.1, dim=100
#PLOT_VALUES["doe_l"] = 102.79 # noise=0.1, dim=100
#PLOT_VALUES["doe"] = 58.36 # noise=0.1, dim=200
#PLOT_VALUES["doe_l"] = 40.73 # noise=0.1, dim=200

#PLOT_VALUES["doe"] = 1.26 # noise=0.4, dim=1
#PLOT_VALUES["doe_l"] = 1.31 # noise=0.4, dim=1
#PLOT_VALUES["doe"] = 5.51 # noise=0.4, dim=5
#PLOT_VALUES["doe_l"] = 5.26 # noise=0.4, dim=5
#PLOT_VALUES["doe"] = 9.85 # noise=0.4, dim=10
#PLOT_VALUES["doe_l"] = 9.98 # noise=0.4, dim=10
PLOT_VALUES["doe"] = 40.4 # noise=0.4, dim=50
PLOT_VALUES["doe_l"] = 28.37 # noise=0.4, dim=50
#PLOT_VALUES["doe"] = 57.92 # noise=0.4, dim=100
#PLOT_VALUES["doe_l"] = 62.21 # noise=0.4, dim=100
#PLOT_VALUES["doe"] = 54.17 # noise=0.4, dim=200
#PLOT_VALUES["doe_l"] = 56.9 # noise=0.4, dim=200

colors = matplotlib.cm.viridis(np.linspace(0.1, 0.9, len(PLOT_VALUES)+3))
X_SIZE = 1000

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.set_xlabel("Number of bins")
ax1.set_ylabel("MI")
ax2.set_xlabel("Size of bins")
i = 0
ax1.hlines(np.log(X_SIZE), 0, list(binning_by_num.keys())[-1], label="log(size)", colors=colors[i], ls="--")
i += 1
for k in PLOT_VALUES:
    ax1.hlines(PLOT_VALUES[k], 0, list(binning_by_num.keys())[-1], colors=colors[i], label=k)
    i += 1
ax1.plot(list(binning_by_num.keys()), list(binning_by_num.values()), label="binning by num", color=colors[i], ls=":")

i += 1
ax2.plot(list(binning_by_size.keys()), list(binning_by_size.values()), label="binning by size", color=colors[i], ls=":")

ax1.legend(loc='upper left')
ax2.legend()
plt.show()

pickle.dump(PLOT_VALUES, open("mi_comparison/data_plot_values_dim_" + str(args.DIM) + "noise_" + str(NOISE), "wb"))
pickle.dump(binning_by_num, open("mi_comparison/data_binning_by_num_dim_" + str(args.DIM) + "noise_" + str(NOISE), "wb"))
pickle.dump(binning_by_size, open("mi_comparison/data_binning_by_size_dim_" + str(args.DIM) + "noise_" + str(NOISE), "wb"))

