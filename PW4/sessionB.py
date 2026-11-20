import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("results", exist_ok=True)

wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target
print(wine.DESCR[:500])
print(X.describe())
pca_raw = PCA()
pca_raw.fit(X)
exp_var_raw = pca_raw.explained_variance_ratio_
pc1_loadings_raw = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(ascending=False)

print("PC1 Explained Variance Ratio (Raw):", exp_var_raw[0])
print("PC1 Loadings (Raw):\n", pc1_loadings_raw)

scaler = StandardScaler()
Z = scaler.fit_transform(X)
pca_scaled = PCA()
pca_scaled.fit(Z)
exp_var_scaled = pca_scaled.explained_variance_ratio_
cum_var_scaled = np.cumsum(exp_var_scaled)

plt.figure()
plt.bar(range(1, len(exp_var_scaled) + 1), exp_var_scaled)
plt.plot(range(1, len(cum_var_scaled) + 1), cum_var_scaled, color="red", marker="o")
plt.axhline(y=0.90, color="gray", linestyle="--")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.savefig("results/scree_plot.png")
plt.close()

loadings_df = pd.DataFrame({
    "PC1 Loadings": pca_scaled.components_[0],
    "PC2 Loadings": pca_scaled.components_[1]
}, index=X.columns)

print("Scaled PC1 and PC2 Loadings:\n", loadings_df)

Z_pca = pca_scaled.transform(Z)
plt.figure()
plt.scatter(Z_pca[:, 0], Z_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("results/pca_unlabeled.png")
plt.close()

for seed in range(8):
    km_test = KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=300, random_state=seed)
    km_test.fit(Z)
    print(f"Seed {seed} Inertia:", km_test.inertia_)

inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km.fit(Z)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.savefig("results/kmeans_elbow.png")
plt.close()

chosen_k = 3
km_final = KMeans(n_clusters=chosen_k, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_final.fit(Z)
cluster_counts = np.bincount(km_final.labels_)
centers_real = scaler.inverse_transform(km_final.cluster_centers_)
df_centers = pd.DataFrame(centers_real, columns=X.columns)

print("Cluster Counts:", cluster_counts)
print("Cluster Centers:\n", df_centers)

plt.figure()
plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=km_final.labels_)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("results/pca_clusters.png")
plt.close()

def compute_agreement(y_true, cluster_labels):
    crosstab = pd.crosstab(y_true, cluster_labels)
    return crosstab.max(axis=0).sum() / len(y_true)

km_i = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(Z)
agree_i = compute_agreement(cultivar, km_i.labels_)

km_ii = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(X)
agree_ii = compute_agreement(cultivar, km_ii.labels_)

km_iii = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(Z_pca[:, :2])
agree_iii = compute_agreement(cultivar, km_iii.labels_)

df_agreement = pd.DataFrame([
    {"Version": "Scaled Data", "Agreement": agree_i},
    {"Version": "Raw Data", "Agreement": agree_ii},
    {"Version": "Top 2 PCs", "Agreement": agree_iii}
])

print("Agreement Table:\n", df_agreement)
df_agreement.to_csv("results/B_agreement.csv", index=False)

#Bonus
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
    X, cultivar, test_size=0.25, random_state=42, stratify=cultivar)

pipe_full = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
pipe_pca2 = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(5))
pipe_full.fit(X_w_train, y_w_train)
pipe_pca2.fit(X_w_train, y_w_train)

print("\nSession B Bonus: Supervised Accuracies")
print(f"Scaled KNN (all 13 features) Test Accuracy: {pipe_full.score(X_w_test, y_w_test):.4f}")
print(f"Scaled PCA(2) + KNN Test Accuracy: {pipe_pca2.score(X_w_test, y_w_test):.4f}")