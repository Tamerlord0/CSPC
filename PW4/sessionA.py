import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance

os.makedirs("results", exist_ok=True)

# A2
df = pd.read_csv("data/concrete.csv")

print("Duplicate rows:", df.duplicated().sum())
print("Missing values per column:\n", df.isna().sum())

df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["strength"])
y = df["strength"]

print(df.describe())

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(df["cement"], df["strength"])
ax[0].set_xlabel("cement")
ax[0].set_ylabel("strength")
ax[1].scatter(df["age"], df["strength"])
ax[1].set_xlabel("age")
ax[1].set_ylabel("strength")
plt.tight_layout()
plt.savefig("results/eda_plots.png")
plt.close()

# A3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

# A4
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

n = len(X_train)
p = X_train.shape[1]

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_adj_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("Linear Regression Equation:")
print(f"strength = {lr.intercept_:.4f} + " + " + ".join([f"{c:.4f}*{col}" for col, c in zip(X.columns, lr.coef_)]))

a4_table = pd.DataFrame([{
    "train_rmse": train_rmse,
    "test_rmse": test_rmse,
    "test_mae": test_mae,
    "train_r2": train_r2,
    "train_adj_r2": train_adj_r2,
    "test_r2": test_r2
}])
print("\nA4 Metrics Table:\n", a4_table)

residuals = y_test - y_test_pred
fig, ax = plt.subplots()
ax.scatter(X_test["age"], residuals)
ax.set_xscale("log")
ax.set_xlabel("age (log scale)")
ax.set_ylabel("residuals")
plt.savefig("results/residuals.png")
plt.close()

# A5
poly_metrics = []
for d in [1, 2, 3, 4]:
    poly = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    poly.fit(X_train, y_train)
    p_feat = poly[0].n_output_features_
    
    y_tr_p = poly.predict(X_train)
    y_te_p = poly.predict(X_test)
    
    tr_r2 = r2_score(y_train, y_tr_p)
    te_r2 = r2_score(y_test, y_te_p)
    tr_adj_r2 = 1 - (1 - tr_r2) * (n - 1) / (n - p_feat - 1)
    
    poly_metrics.append({
        "degree": d,
        "p": p_feat,
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_tr_p)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_te_p)),
        "train_r2": tr_r2,
        "train_adj_r2": tr_adj_r2,
        "test_r2": te_r2
    })

print("\nA5 Polynomial Table:\n", pd.DataFrame(poly_metrics))

# A6
y_class = (df["strength"] >= 40).astype(int)

Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

k_list = [1, 3, 5, 7, 15, 25, 51, 101]
knn_tr_acc, knn_te_acc = [], []

for K in k_list:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
    knn.fit(Xc_train, c_train)
    knn_tr_acc.append(knn.score(Xc_train, c_train))
    knn_te_acc.append(knn.score(Xc_test, c_test))

fig, ax = plt.subplots()
ax.plot(k_list, knn_tr_acc, label="train")
ax.plot(k_list, knn_te_acc, label="test")
ax.set_xscale("log")
ax.set_xlabel("K")
ax.set_ylabel("accuracy")
ax.legend()
plt.savefig("results/knn_acc.png")
plt.close()

chosen_K = 7
knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=chosen_K))
knn_scaled.fit(Xc_train, c_train)
acc_scaled = knn_scaled.score(Xc_test, c_test)

knn_unscaled = KNeighborsClassifier(n_neighbors=chosen_K)
knn_unscaled.fit(Xc_train, c_train)
acc_unscaled = knn_unscaled.score(Xc_test, c_test)
print(f"\nKNN K={chosen_K} Test Acc - Scaled: {acc_scaled}, Unscaled: {acc_unscaled}")

seed_accs = []
for rs in [0, 1, 2, 3, 4]:
    X_tr_s, X_te_s, c_tr_s, c_te_s = train_test_split(
        X, y_class, test_size=0.2, random_state=rs, stratify=y_class
    )
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=chosen_K))
    m.fit(X_tr_s, c_tr_s)
    seed_accs.append(m.score(X_te_s, c_te_s))
print("KNN 5 seeds test accs:", seed_accs, "spread:", np.ptp(seed_accs))

tree_metrics = []
depths = [1, 2, 3, 4, 5, 6, 8, None]

for d in depths:
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0)
    tree.fit(Xc_train, c_train)
    tree_metrics.append({
        "max_depth": str(d),
        "leaves": tree.get_n_leaves(),
        "train_acc": tree.score(Xc_train, c_train),
        "test_acc": tree.score(Xc_test, c_test)
    })

df_tree = pd.DataFrame(tree_metrics)
print("\nA6 Decision Tree Table:\n", df_tree)

fig, ax = plt.subplots()
ax.plot([str(d) for d in depths], df_tree["train_acc"], label="train")
ax.plot([str(d) for d in depths], df_tree["test_acc"], label="test")
ax.set_xlabel("max_depth")
ax.set_ylabel("accuracy")
ax.legend()
plt.savefig("results/tree_acc.png")
plt.close()

chosen_depth = 4
chosen_tree = DecisionTreeClassifier(criterion="entropy", max_depth=chosen_depth, random_state=0)
chosen_tree.fit(Xc_train, c_train)
print("\nTree Top 2 Levels:\n", export_text(chosen_tree, feature_names=list(X.columns), max_depth=2))

# A7
chosen_knn = knn_scaled

records = []
for name, model in [("KNN", chosen_knn), ("Tree", chosen_tree)]:
    c_pred = model.predict(Xc_test)
    tn, fp, fn, tp = confusion_matrix(c_test, c_pred).ravel()
    records.append({
        "model": name,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": accuracy_score(c_test, c_pred),
        "precision": precision_score(c_test, c_pred),
        "recall": recall_score(c_test, c_pred),
        "f1": f1_score(c_test, c_pred)
    })

metrics_df = pd.DataFrame(records)
print("\nA7 Metrics Table:\n", metrics_df)
metrics_df.to_csv("results/A_metrics.csv", index=False)

knn_proba = chosen_knn.predict_proba(Xc_test)[:, 1]
thresh_records = []
for t in [0.3, 0.5, 0.7, 0.9]:
    c_pred_t = (knn_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(c_test, c_pred_t).ravel()
    thresh_records.append({
        "threshold": t,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "precision": precision_score(c_test, c_pred_t, zero_division=0),
        "recall": recall_score(c_test, c_pred_t, zero_division=0)
    })
print("\nA7 Threshold Sweep (KNN):\n", pd.DataFrame(thresh_records))

# A8
raw_lr = LinearRegression().fit(X_train, y_train)
raw_coef = raw_lr.coef_

std_pipe = make_pipeline(StandardScaler(), LinearRegression())
std_pipe.fit(X_train, y_train)
std_coef = std_pipe[-1].coef_

tree_imp = chosen_tree.feature_importances_

knn_perm = permutation_importance(chosen_knn, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
tree_perm = permutation_importance(chosen_tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)

imp_df = pd.DataFrame({
    "feature": X.columns,
    "raw_coef": raw_coef,
    "std_coef": std_coef,
    "tree_imp": tree_imp,
    "knn_perm_mean": knn_perm.importances_mean,
    "tree_perm_mean": tree_perm.importances_mean
})
print("\nA8 Feature Importance Table:\n", imp_df)

fig, ax = plt.subplots(figsize=(8, 4))
x_idx = np.arange(len(X.columns))
w = 0.25
ax.bar(x_idx - w, imp_df["std_coef"], w, label="std_coef")
ax.bar(x_idx, imp_df["tree_imp"], w, label="tree_imp")
ax.bar(x_idx + w, imp_df["knn_perm_mean"], w, label="knn_perm_mean")
ax.set_xticks(x_idx)
ax.set_xticklabels(X.columns, rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()

# Bonus Physics features
X_tr_phys = X_train.copy()
X_te_phys = X_test.copy()

wb_tr = X_tr_phys["water"] / (X_tr_phys["cement"] + X_tr_phys["slag"] + X_tr_phys["ash"])
wb_te = X_te_phys["water"] / (X_te_phys["cement"] + X_te_phys["slag"] + X_te_phys["ash"])

X_tr_p = pd.DataFrame({"log_age": np.log(X_tr_phys["age"]), "wb_ratio": wb_tr})
X_te_p = pd.DataFrame({"log_age": np.log(X_te_phys["age"]), "wb_ratio": wb_te})

lr_bonus = LinearRegression().fit(X_tr_p, y_train)
bonus_r2 = lr_bonus.score(X_te_p, y_test)
print(f"\nBonus Model - Features: {X_tr_p.shape[1]}, Test R2: {bonus_r2:.4f}")