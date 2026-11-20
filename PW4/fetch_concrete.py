import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

os.makedirs("data", exist_ok=True)

concrete = fetch_ucirepo(id=165)
X = concrete.data.features.copy()
y = concrete.data.targets.copy()
print(concrete.metadata)
print(concrete.variables)
X.columns = ["cement", "slag", "ash", "coarse_agg", "fine_agg", "water", "superplastic", "age"]
y = y.iloc[:, 0].rename("strength")
X.assign(strength=y).to_csv("data/concrete.csv", index=False)