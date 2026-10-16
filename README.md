# CSPC - Computer Science for Physics and Chemistry

## Setup

Create the environment for PW1 - Lab A:

```bash
conda env create -f PW1/Lab\ A/environment.yml
conda activate cspc
```

## PW1 - Lab A: Reproducible Foundations

**What I did:**
- Set up a reproducible Python environment using Conda and created a Git repository.
- Worked with a radioactive decay simulation and added automated tests using `pytest`.
- Compared pure Python and NumPy implementations.

**Performance comparison:**

| Implementation | Execution time |
|---|---:|
| Python loop | 4.30854502 s |
| NumPy | 0.00022917 s |
| Speed-up | 18,800.73x |

**Tests:** 3/3 passed.

**Conclusion:**

I learned the basics of Python environments, Git, automated testing, and NumPy. I also solved problems with the environment, file locations, Git setup, and test parameters. AI helped me understand errors and work through some setup and testing steps.

## PW1 - Lab B: Radioactive Decay

The observed count decreased over time and followed the general exponential decay pattern described by \(N(t)=N_0e^{-\lambda t}\), with \(\lambda=0.3\).

The Snakemake pipeline runs `plot.py` to generate `figure.png` from `decay_observed.csv` and reruns the step when its inputs change.

## PW2 - Lab A: Numerical Differentiation and Integration

**Objective:** Analyse free-fall position data using numerical differentiation and integration.

Velocity and acceleration were calculated using `np.gradient()`.

| Measurement | Result |
|---|---:|
| Mean acceleration | -8.58 m/s² |
| Expected acceleration | -9.81 m/s² |
| Maximum position difference after integration | 0.785 m |

Differentiation, especially taking two derivatives, amplified measurement noise in the acceleration. Integrating the results back to position gave a maximum difference of 0.785 m.

**Output:** `motion.png`, showing position, velocity, and acceleration over time.

## PW2 - Lab B: Numerical Methods

### Part 2 - Three Routes to a Minimum

I compared gradient descent, Newton's method, and SLSQP.

**2A - Convex function**

For \(f(x)=(x-3)^2+1\), starting from \(x_0=0\):

| Method | Minimum location |
|---|---:|
| Gradient descent | 3.0000 |
| Newton | 3.0000 |
| SLSQP | 3.0000 |

All three methods found the global minimum.

**2B - More complex function**

For \(g(x)=x^4-3x^2+x+5\):

| Starting point | Gradient descent | Newton | SLSQP |
|---|---:|---:|---:|
| \(x_0=0\) | -1.30084 | 0.16994 (maximum) | -1.30086 |
| \(x_0=2\) | -1.30084 | 1.13090 (minimum) | -1.30064 |

Newton's method reached different stationary points depending on the starting point, while gradient descent and SLSQP found the minimum near \(x=-1.30\).

### Part 3 - Reaction Rate

I fitted the first-order model \(C(t)=C_0e^{-kt}\) to the measured data using SLSQP with \(0\leq k\leq5\).

**Fitted rate constant:** \(k=0.26176137\)

**Output:** `kinetics.png`

### Part 4 - Chemical Equilibrium

For the reaction \(H_2+I_2\rightleftharpoons2HI\), with \(K=15.6\):

| Method | Equilibrium extent |
|---|---:|
| Newton | 0.66384767 |
| SLSQP | 0.66384743 |

**Equilibrium amounts:**

| Substance | Amount |
|---|---:|
| H₂ | 0.33615 mol |
| I₂ | 0.33615 mol |
| HI | 1.32770 mol |

Both methods gave nearly identical results.

### Part 5 - Titration

I calculated the pH slope using `np.gradient(pH, V)` and located its maximum.

**Equivalence point:** 50.0 mL

## PW3 - Statistical Analysis of Scientific Data

### Session 1 - Heart Disease Dataset

**Dataset:** `heart.csv` - 303 rows and 14 columns.

I used pandas to inspect the data and plotted the distributions of `age`, `chol`, `trestbps`, and `thalach`.

**Output:** `distributions.png`

**Normality tests**

I used the Shapiro-Wilk test with a significance level of 0.05.

| Variable | W statistic | p-value | Verdict |
|---|---:|---:|---|
| `age` | 0.9865 | 0.006069 | Non-normal |
| `chol` | 0.9473 | 5.912 x 10⁻⁹ | Non-normal |
| `trestbps` | 0.9665 | 1.802 x 10⁻⁶ | Non-normal |
| `thalach` | 0.9765 | 6.996 x 10⁻⁵ | Non-normal |

All p-values were below 0.05, providing evidence against normality.

### Session 2 - Heart Rate and Age

**Maximum heart rate**

I compared healthy patients (`target = 0`) with patients with heart disease (`target = 1`) using the Mann-Whitney U test because the normality tests indicated non-normal data.

| Group | Mean `thalach` | 95% confidence interval |
|---|---:|---:|
| Healthy | 158.38 | 155.42-161.34 |
| Heart disease | 139.26 | 135.47-143.05 |

**U = 5806.5, p = 1.861 x 10⁻¹³**

The difference was statistically significant. The heart-disease group had a mean maximum heart rate about 19.12 beats/min lower.

**Output:** `thalach_comparison.png`

**Age and maximum heart rate**

I used Pearson's correlation to measure their linear relationship.

| Correlation | p-value | Conclusion |
|---:|---:|---|
| -0.394 | 1.109 x 10⁻¹² | Moderate negative relationship |

Older patients tended to have lower maximum heart rates.

**Output:** `age_thalach.png`

### Chemical Exposure and Malignancy

**Dataset:** `chemicals cancer.csv` - 1,000 patients.

**Initial correlations**

I calculated Pearson correlations between each chemical and malignancy without controlling for other variables.

| Chemical | Correlation | p-value |
|---|---:|---:|
| Benzene | 0.385 | 1.276 x 10⁻³⁶ |
| Cadmium | 0.880 | Extremely small |

Initially, cadmium appeared to have the stronger relationship with malignancy.

**Controlling for pollution**

I repeated the analysis for the 201 patients with a pollution index between 40 and 60.

| Chemical | Correlation | p-value |
|---|---:|---:|
| Benzene | 0.698 | 1.154 x 10⁻³⁰ |
| Cadmium | 0.313 | 5.994 x 10⁻⁶ |

Benzene's correlation became stronger, while cadmium's weakened. This suggested that pollution influenced the initial results.

**Multiple regression**

I then fitted a multiple linear regression model including benzene, cadmium, pollution index, and age. This allowed me to examine each variable while accounting for the others.

| Variable | Coefficient | p-value | Result |
|---|---:|---:|---|
| Benzene | +0.5452 | < 0.001 | Significant |
| Cadmium | -0.0677 | 0.224 | Not significant |
| Pollution index | +0.7773 | < 0.001 | Significant |
| Age | +0.1064 | < 0.001 | Significant |

**R² = 0.917** - the model explained about 91.7% of the variation in malignancy.

**Conclusion:** Cadmium initially appeared more strongly related to malignancy, but it was no longer statistically significant in the multiple regression. Benzene remained significant, making it the stronger candidate in this dataset. However, regression alone cannot prove causation; confirming the actual cause requires knowledge of how the synthetic data was generated or other causal evidence.

### Bonus - Category Unpredictability

I calculated the proportions and Shannon entropy of the heart-disease `target` column.

| Measure | Result |
|---|---:|
| Target 0 | 54.13% |
| Target 1 | 45.87% |
| Entropy | 0.995 bits |
| Maximum entropy | 1 bit |

The categories were nearly balanced, so predicting the target without additional information involved considerable uncertainty.

### PW3 Outputs

| File | Contents |
|---|---|
| `distributions.png` | Distributions of four continuous variables |
| `thalach_comparison.png` | Heart-rate group comparison |
| `age_thalach.png` | Age and maximum heart-rate relationship |

### Overall Conclusion

Across these practicals, I worked with reproducible environments, automated testing, numerical methods, optimisation, and statistical analysis. I learned how to compare algorithms, analyse scientific data, and interpret statistical results.

The chemical analysis was especially useful because it showed that a strong correlation can be misleading. After accounting for other variables, benzene remained significantly associated with malignancy while cadmium did not. This demonstrated why confounding variables must be considered before drawing conclusions.