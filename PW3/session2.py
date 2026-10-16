import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

#Task 6

table = pd.read_csv("chemicals_cancer.csv")
for chemical in ["benzene", "cadmium"]:
    r, p = stats.pearsonr(table[chemical], table["malignancy"])
    print(f"{chemical}: r={r:.3f}, p={p:.30f}")

#Task 7

group = table[table["pollution_index"].between(40, 60)]
print(f"\nPatients in pollution range 40-60: {len(group)}")

for chemical in ["benzene", "cadmium"]:
    r, p = stats.pearsonr(group[chemical], group["malignancy"])
    print(f"{chemical} within pollution range: r={r:.3f}, p={p:.30f}")

#Bonus and malignancy cause

heart = pd.read_csv("heart.csv")
proportions = heart["target"].value_counts(normalize=True)
entropy = -(proportions * np.log2(proportions)).sum()

print("\nTarget proportions:")
print(proportions)
print(f"Entropy: {entropy:.3f} bits")
print(f"Maximum entropy: {np.log2(len(proportions)):.3f} bits")

X = table[["benzene", "cadmium", "pollution_index", "age"]]
X = sm.add_constant(X)
y = table["malignancy"]

model = sm.OLS(y, X).fit()
print(model.summary())