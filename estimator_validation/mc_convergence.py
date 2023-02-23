import collections
import numpy as np
import math
import argparse
import matplotlib
import matplotlib.pyplot as plt

NOISE_SAMPLES = 10
NOISE = 0.1
PLOT_VALUES = {}

parser = argparse.ArgumentParser()
parser.add_argument('--dim', dest='DIM', type=int, help='dimensionality of the input space')
parser.add_argument('--fdim', dest='F_DIM', type=int, help='dimensionality of the representation space')
args = parser.parse_args()

mc = collections.OrderedDict()
for X_SIZE in [10, 50, 100, 500, 1000, 5000, 10000, 30000, 50000, 100000, 150000]:
    Z_SIZE = X_SIZE
    mean = np.zeros(args.DIM)
    cov = np.identity(args.DIM)
    x = np.random.multivariate_normal(mean, cov, X_SIZE)

    mean = np.ones(args.F_DIM)
    cov = np.identity(args.F_DIM) * NOISE
    eps = np.random.multivariate_normal(mean, cov, Z_SIZE*NOISE_SAMPLES)
    fx = 2*x + 0.5
    z = np.repeat(fx, NOISE_SAMPLES, axis=0) * eps

    # our innovative way to compute conditional entropy
    h_zGivx = 0
    h_z_part = math.sqrt(2*math.pi*math.e)*NOISE
    for i in range(args.F_DIM):
        # we have simple multiplication of 1-dim gaussian (noise) by a constant (current value of activations)
        h_zGivx += (1.0/len(fx)) * np.sum(np.log(h_z_part * np.fabs(fx[:,i] + 1e-6)))
    mc[X_SIZE] = h_zGivx

###################################
#PLOTTING
###################################
colors = matplotlib.cm.viridis(np.linspace(0.1, 0.9, 1))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Number of samples")
#ax1.set_xscale('log')
ax1.set_ylabel("h(Z|X)")
i = 0
ax1.plot(list(mc.keys()), list(mc.values()), label="MC estimate h(Z|X)", color=colors[i])
ax1.legend(loc='best')
plt.show()
