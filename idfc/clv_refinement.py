import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def compute_cov2(x, c):
    """Covariance au carré entre une variable x et un vecteur c."""
    return np.cov(x, c)[0, 1] ** 2

def refine_partition(X, init_partition, rho=0.3, max_iter=100, tol=1e-5):
    """
    Raffinement de la partition via CLV avec stratégie K+1.
    
    Paramètres :
    ------------
    X : DataFrame (n échantillons x p variables)
    init_partition : dict[str, list[str]]
        Dictionnaire {cluster_k : [features]} initial (souvent issu de VARCLUS)
    rho : float
        Seuil de corrélation au carré pour affectation dans un vrai cluster. Sinon → bruit.
    max_iter : int
        Nombre maximal d’itérations
    tol : float
        Tolérance pour le critère de convergence
    
    Retourne :
    ----------
    clusters : dict[str, list[str]] avec un cluster spécial 'noise'
    components : dict[str, np.ndarray] : composantes principales normalisées de chaque cluster
    """
    p_names = X.columns
    X = X.copy()
    n_features = len(p_names)
    
    clusters = init_partition.copy()
    noise_cluster = "noise"
    clusters[noise_cluster] = []
    
    prev_assignments = None
    
    for key, clus in clusters.items():
        clusters[key] = p_names[clus]
    # print(p_names[clus])
        
    for iteration in range(max_iter):
        # Étape 1 : calcul des premières composantes pour chaque cluster
        components = {}
        for k, vars_k in clusters.items():
            if k == noise_cluster or len(vars_k) < 2:
                continue
            pca = PCA(n_components=1)
            # print(vars_k)
            components[k] = pca.fit(X[vars_k]).components_[0]
        # Étape 2 : assignation avec règle K+1
        new_clusters = {k: [] for k in clusters}
        for j in p_names:
            best_k = None
            best_score = -np.inf
            xj = X[j].values
            for k, comp in components.items():
                c_vec = X[clusters[k]].values @ comp
                score = compute_cov2(xj, c_vec)
                if score > best_score:
                    best_score = score
                    best_k = k
            var_j = np.var(xj)
            if best_score >= rho**2 * var_j:
                new_clusters[best_k].append(j)
            else:
                new_clusters[noise_cluster].append(j)
        # Vérifier convergence
        current_assignments = [sorted(v) for v in new_clusters.values()]
        if prev_assignments is not None:
            diffs = [set(a) != set(b) for a, b in zip(prev_assignments, current_assignments)]
            if not any(diffs):
                break
        prev_assignments = current_assignments
        clusters = new_clusters

    # Recalcul final des composantes (hors clusters de taille < 2 sauf "noise")
    final_components = {}
    filtered_clusters = {}

    for k, vars_k in clusters.items():
        if len(vars_k) == 0 and k != "noise":
            continue  # Supprimer les clusters vides (sauf "noise")
        if k != "noise" and len(vars_k) >= 2:
            pca = PCA(n_components=1)
            final_components[k] = pca.fit(X[vars_k]).components_[0]
        filtered_clusters[k] = vars_k  # inclut aussi "noise", même vide

    return filtered_clusters, final_components