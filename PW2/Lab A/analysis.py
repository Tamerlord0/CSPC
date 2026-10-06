import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

data = np.loadtxt("freefall.csv", delimiter=",", skiprows=1)

t = data[:, 0]
y = data[:, 1]

velocity = np.gradient(y, t)
acceleration = np.gradient(velocity, t)

recovered_velocity = cumulative_trapezoid(acceleration, t, initial=0)
recovered_position = cumulative_trapezoid(recovered_velocity, t, initial=0)

difference = np.abs(recovered_position - y)

print("Mean acceleration:", np.mean(acceleration), "m/s^2")
print("Standard deviation:", acceleration.std(), "m/s^2")
print("Largest difference:", np.max(difference), "m")

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

axes[0].plot(t, y)
axes[0].set_ylabel("Position (m)")
axes[0].set_title("Position")
axes[1].plot(t, velocity)
axes[1].set_ylabel("Velocity (m/s)")
axes[1].set_title("Velocity")
axes[2].plot(t, acceleration)
axes[2].axhline(-9.81, linestyle="--", label="-9.81 m/s²")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Acceleration (m/s²)")
axes[2].set_title("Acceleration")
axes[2].legend()

plt.tight_layout()
plt.savefig("motion.png")
