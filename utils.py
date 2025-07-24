import numpy as np
import pandas as pd

def compute_cov(x, c):
    """
    Renvoie la covariance entre une variable x et une composante c.
    """
    return np.cov(x, c)[0, 1]


def compute_cov2(x, c):
    """Covariance au carré entre une variable x et un vecteur c."""
    return np.cov(x, c)[0, 1] ** 2

def compute_variable_correlations(X, clusters, components):
    """
    Calcule les corrélations (r²) entre chaque variable d’un cluster et la composante principale du cluster.
    
    Paramètres :
    ------------
    X : pd.DataFrame, (n x p) jeu de données
    clusters : dict[str, list[str]] : dictionnaire des clusters
    components : dict[str, np.ndarray] : composante principale de chaque cluster
    
    Retourne :
    ----------
    result : dict[str, pd.DataFrame] : clé = nom du cluster, valeur = tableau avec corrélations r²
    """
    results = {}
    for k, features in clusters.items():
        if k not in components or len(features) < 2:
            continue
        c_vec = X[features] @ components[k]
        corr_data = {}
        for f in features:
            x = X[f]
            r = np.corrcoef(x, c_vec)[0, 1]
            corr_data[f] = r**2
        df = pd.DataFrame.from_dict(corr_data, orient='index', columns=['r²'])
        df.sort_values(by='r²', ascending=False, inplace=True)
        results[k] = df
    return results