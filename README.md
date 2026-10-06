# CSPC — Computer Science for Physics and Chemistry

## Setup

Create the environment for **PW1 — Lab A**:

```bash
conda env create -f PW1/Lab\ A/environment.yml

conda activate cspc
```

## PW1 — Lab A: Reproducible Foundations

**What I built:**

* Set up a reproducible Python environment for **PW1 — Lab A** using Conda and created a Git repository for the coursework.
* Worked with a radioactive decay simulation, added automated tests with `pytest`, and compared the pure-Python and NumPy implementations.

**Speed comparison (loop vs NumPy):**

* loop : 4.30854502 s
* numpy : 0.00022917 s
* speed-up: 18800.73 x faster

**Tests:** all passing? **yes**

**Tests passed:** 3/3

**Conclusion:**

* **PW1 — Lab A** helped me learn the basics of working with Python environments, Git, automated testing, and NumPy.
* I learned how to test different parts of a program and compare the performance of two implementations. I faced some problems with the Conda environment, file locations, Git setup, and getting the test parameters correct, but I was able to solve eventually.
* AI helped me a little during the lab, mostly by explaining errors and guiding me through some of the setup and testing steps.

## PW1 — Lab B

The observed decay data showed a decrease in the count as time increased. The observed data followed the same general exponential-decay shape as the analytical law N₀e⁻λᵗ with λ = 0.3, so the two matched reasonably well.

The Snakemake pipeline automatically runs plot.py to generate figure.png from decay_observed.csv, and only reruns the step when its input files have changed.

## PW2 — Lab A: Numerical Differentiation and Integration

### Objective

Use numerical differentiation and integration to analyse free fall position data.

### Method

Velocity and acceleration were calculated from the position data using `np.gradient()`.

### Results

* Mean acceleration: **−8.58 m/s²**
* Expected acceleration: **−9.81 m/s²**
* Largest difference after integrating back: **0.785 m**

### Noise and Integration

Acceleration was much noisier than position because **numerical differentiation amplifies measurement noise**, especially when taking two derivatives. Integrating the noisy acceleration back to velocity and then position recovered the original position with a maximum difference of **0.785 m**, showing that integration reduces the effect of noise.

### Output

The results are plotted in `motion.png`, showing position, velocity, and acceleration over time.

## PW2 — Lab B

### Part 2 — Three Routes to a Minimum

Compared gradient descent, Newton's method, and SLSQP.

#### 2A — Easy Convex Function

For f(x) = (x - 3)^2 + 1, starting from x0 = 0:

* Gradient descent: x ≈ 3.0000
* Newton: x = 3.0000
* SLSQP: x = 3.0000

All three methods agree on the global minimum.

#### 2B — Harder Landscape

For g(x) = x^4 - 3x^2 + x + 5:

**Starting from x0 = 0:**

* Gradient descent: x ≈ -1.30084
* Newton: x ≈ 0.16994 — maximum (g'' < 0)
* SLSQP: x ≈ -1.30086

**Starting from x0 = 2:**

* Gradient descent: x ≈ -1.30084
* Newton: x ≈ 1.13090 — minimum (g'' > 0)
* SLSQP: x ≈ -1.30064

Newton can converge to different stationary points depending on the starting point, while gradient descent and SLSQP find the minimum.

### Part 3 — Reaction Rate

Fitted the first-order model C(t) = C0 * exp(-k*t) to the measured kinetics data using SLSQP with 0 <= k <= 5.

The fitted rate constant k was 0.2617613705507395, and the fitted curve followed the measured data.

**Output:** `kinetics.png`

### Part 4 — Chemical Equilibrium

For the reaction:

H2 + I2 <=> 2 HI

with K = 15.6, Newton's method and SLSQP gave the same equilibrium extent:

* Newton: x = 0.66384767
* SLSQP: x = 0.66384743

**Equilibrium amounts:**

* H2 = 0.33615 mol
* I2 = 0.33615 mol
* HI = 1.32770 mol

### Part 5 — Titration

Calculated the pH slope using `np.gradient(pH, V)` and found its maximum.

**Equivalence point:** 50.0 mL
