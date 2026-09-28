import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.3

data = np.loadtxt("decay_observed.csv", delimiter=",", skiprows=1)
t = data[:, 0]
observed = data[:, 1]

N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

ax[0].scatter(t, observed)
ax[0].set_title("Observed Data")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Count")

ax[1].plot(t, analytical)
ax[1].set_title("Analytical Decay")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Count")

plt.savefig("figure.png")
