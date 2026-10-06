import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

def k_imbalance(x):
    return (2*x)**2 / ((a-x)*(b-x)) - K

x_newton = newton(k_imbalance, 0.5)
result = minimize(lambda x: k_imbalance(x[0])**2, [0.5],
                  method="SLSQP", bounds=[(0, 0.999)])
x_slsqp = result.x[0]

H2 = a - x_newton
I2 = b - x_newton
HI = 2 * x_newton

print("Newton:", x_newton)
print("SLSQP:", x_slsqp)
print("Agree:", np.isclose(x_newton, x_slsqp))
print("Equilibrium amounts:")
print("H2:", H2)
print("I2:", I2)
print("HI:", HI)

x = np.linspace(0, 0.999, 500)
plt.plot(x, a-x, label="H2")
plt.plot(x, b-x, label="I2")
plt.plot(x, 2*x, label="HI")
plt.axvline(x_newton, linestyle="--", label="Equilibrium")
plt.xlabel("Extent x (mol)")
plt.ylabel("Amount (mol)")
plt.legend()
plt.savefig("equilibrium.png")