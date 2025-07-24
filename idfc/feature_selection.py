import numpy as np

def select_interpretable_features(X, clusters, components):
    """
    Pour chaque cluster, sélectionne la variable la plus corrélée à la composante principale.
    """
    selected = []
    for k, features in clusters.items():
        best_var = None
        best_score = -np.inf
        for feat in features:
            corr = np.corrcoef(X[feat], X[features].dot(components[k]))[0, 1] ** 2
            if corr > best_score:
                best_score = corr
                best_var = feat
        selected.append(best_var)
    return selected