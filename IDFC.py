import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from varclushi import VarClusHi


class IDFC:
    def __init__(self, rho=0.3, maxeigval2=1.0, max_iter=100, tol=1e-5, verbose=False):
        self.rho = rho
        self.maxeigval2 = maxeigval2
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.selected_features_ = None
        self.clusters_ = None
        self.components_ = None

    @staticmethod
    def _compute_cov2(x, c):
        return np.cov(x, c)[0, 1] ** 2

    def _initialize_partition(self, X):
        p_names = X.columns
        varclus = VarClusHi(X, maxeigval2=self.maxeigval2)
        varclus.varclus()
        cluster_assignments = varclus.rsquare.Cluster
        partition = {}
        for label in cluster_assignments.unique():
            features = cluster_assignments[cluster_assignments == label].index.tolist()
            partition[f"cluster_{label}"] = features
        for key, clus in partition.items():
            print(key, clus)
            if len(clus) != 0:
                partition[key] = p_names[clus]
        return partition

    def _refine_partition(self, X, init_partition):
        p_names = X.columns
        clusters = init_partition.copy()
        noise_cluster = "noise"
        clusters[noise_cluster] = []

        prev_assignments = None
        for iteration in range(self.max_iter):
            # Étape 1 : PCA par cluster
            components = {}
            for k, vars_k in clusters.items():
                if k == noise_cluster or len(vars_k) < 2:
                    continue
                pca = PCA(n_components=1)
                components[k] = pca.fit(X[vars_k]).components_[0]

            # Étape 2 : réassignation avec règle K+1
            new_clusters = {k: [] for k in clusters}
            for j in p_names:
                best_k = None
                best_score = -np.inf
                xj = X[j].values
                for k, comp in components.items():
                    c_vec = X[clusters[k]].values @ comp
                    score = self._compute_cov2(xj, c_vec)
                    if score > best_score:
                        best_score = score
                        best_k = k
                var_j = np.var(xj)
                if best_score >= self.rho**2 * var_j:
                    new_clusters[best_k].append(j)
                else:
                    new_clusters[noise_cluster].append(j)

            # Convergence
            current_assignments = [sorted(v) for v in new_clusters.values()]
            if prev_assignments is not None:
                diffs = [set(a) != set(b) for a, b in zip(prev_assignments, current_assignments)]
                if not any(diffs):
                    break
            prev_assignments = current_assignments
            clusters = new_clusters

        # Final : recalcul des composantes
        final_components = {}
        filtered_clusters = {}
        for k, vars_k in clusters.items():
            if len(vars_k) == 0 and k != noise_cluster:
                continue
            if k != noise_cluster and len(vars_k) >= 2:
                pca = PCA(n_components=1)
                final_components[k] = pca.fit(X[vars_k]).components_[0]
            filtered_clusters[k] = vars_k

        return filtered_clusters, final_components

    def _select_interpretable_features(self, X, clusters, components):
        selected = []
        for k, features in clusters.items():
            if k not in components:
                continue  # pas de PCA (e.g. noise ou cluster mono-variable)
            best_var = None
            best_score = -np.inf
            c_vec = X[features].dot(components[k])
            for feat in features:
                r = np.corrcoef(X[feat], c_vec)[0, 1]
                corr = r**2
                if corr > best_score:
                    best_score = corr
                    best_var = feat
            selected.append(best_var)
        return selected

    def fit(self, X):
        """
        Exécute le pipeline complet IDFC :
        1. Initialisation (VARCLUS)
        2. Raffinement (CLV k+1)
        3. Sélection des variables interprétables
        """
        X = (X - X.mean()) / X.std()
        
        if self.verbose:
            print("Étape 1 : Initialisation (VARCLUS)...")
        init_partition = self._initialize_partition(X)           
        
        if self.verbose:
            print("Étape 2 : Raffinement (CLV K+1)...")
        self.clusters_, self.components_ = self._refine_partition(X, init_partition)
        # print('ok')
        if self.verbose:
            print("Étape 3 : Sélection des variables interprétables...")
        self.selected_features_ = self._select_interpretable_features(X, self.clusters_, self.components_)
        

        return self

    def get_selected_features(self):
        return self.selected_features_

    def get_clusters(self):
        return self.clusters_

    def get_components(self):
        return self.components_