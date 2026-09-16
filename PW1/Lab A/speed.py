import time
from decay import simulate_loop, simulate

N0 = 200000
lam = 0.4
dt = 0.05
steps = 200

start = time.perf_counter()
simulate_loop(N0, lam, dt, steps)
loop_time = time.perf_counter() - start

start = time.perf_counter()
simulate(N0, lam, dt, steps)
numpy_time = time.perf_counter() - start

print()
print("Speed Comparison:")
print("Pure Python :", round(loop_time, 8), "seconds")
print("NumPy       :", round(numpy_time, 8), "seconds")
print("Speed-up    :", round(loop_time / numpy_time, 2), "times faster")