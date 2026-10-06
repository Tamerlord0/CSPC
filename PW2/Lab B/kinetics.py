import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1)
t = data[:, 0]
C = data[:, 1]
C0 = C[0]

def total_error(k):
    return np.sum((C - C0 * np.exp(-k * t))**2)

result = minimize(total_error, x0=0.5, method="SLSQP", bounds=[(0, 5)])
k = result.x[0]

print("Fitted k:", k)

plt.scatter(t, C, label="Measured")
plt.plot(t, C0 * np.exp(-k * t), label="Fitted curve")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.savefig("kinetics.png")