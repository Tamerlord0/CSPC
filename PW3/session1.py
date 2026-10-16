import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#Task 1

table = pd.read_csv("heart.csv")
variables = ["age", "chol", "trestbps", "thalach"]
continuous = table[variables]
print("Dataset shape:", table.shape)
print("\nFirst five rows:")
print(table.head())
print("\nContinuous variables:")
print(continuous.head())

#Task 2

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, variable in zip(axes.flat, variables):
    ax.hist(continuous[variable].dropna(), bins="auto", edgecolor="black")
    ax.set_title(variable)
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("distributions.png", dpi=300)

#Task 3

for variable in variables:
    data = continuous[variable].dropna()
    statistic, p_value = stats.shapiro(data)

    print(f"\n{variable}")
    print(f"Shapiro-Wilk statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.15f}")

    if p_value < 0.05:
        print("Verdict: Evidence against normality.")
    else:
        print("Verdict: Insufficient evidence to reject normality.")

#Task 4

healthy = table.loc[table["target"] == 0, "thalach"].dropna()
disease = table.loc[table["target"] == 1, "thalach"].dropna()

for name, group in [("Healthy", healthy), ("Disease", disease)]:
    ci = stats.t.interval(0.95, len(group)-1, loc=group.mean(),
                          scale=stats.sem(group))
    print(f"{name}: mean={group.mean():.2f}, "
          f"95% CI=({ci[0]:.2f}, {ci[1]:.2f})")

if all(stats.shapiro(g).pvalue >= 0.05 for g in [healthy, disease]):
    statistic, p = stats.ttest_ind(disease, healthy, equal_var=False)
    print("Test: Welch's t-test")
else:
    statistic, p = stats.mannwhitneyu(
        disease, healthy, alternative="two-sided"
    )
    print("Test: Mann-Whitney U")

print(f"Statistic={statistic:.4f}, p-value={p:.4g}")
print("Significant difference." if p < 0.05 else "No significant difference.")

means = [healthy.mean(), disease.mean()]
cis = [stats.t.interval(0.95, len(g)-1, loc=g.mean(), scale=stats.sem(g))
       for g in [healthy, disease]]

plt.figure()
plt.errorbar(["Healthy", "Disease"], means,
             yerr=[[m-c[0] for m, c in zip(means, cis)],
                   [c[1]-m for m, c in zip(means, cis)]],
             fmt="o", capsize=5)
plt.ylabel("Maximum heart rate (thalach)")
plt.tight_layout()
plt.savefig("thalach_comparison.png", dpi=300)
plt.close()

#Task 5

valid = table[["age", "thalach"]].dropna()
r, p = stats.pearsonr(valid["age"], valid["thalach"])
print(f"\nPearson correlation: r={r:.3f}, p-value={p:.4g}")

plt.figure()
plt.scatter(valid["age"], valid["thalach"], alpha=0.7)
plt.xlabel("Age (years)")
plt.ylabel("Maximum heart rate (thalach)")
plt.tight_layout()
plt.savefig("age_thalach.png", dpi=300)
plt.close()
