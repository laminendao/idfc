import pandas as pd
from varclushi import VarClusHi

def initialize_partition(X, maxeigval2=1.0):
    """
    Partition initiale des variables par VARCLUS (VarClusHi).
    
    Regroupe les variables en clusters hiérarchiques selon leur corrélation.

    Paramètres :
    ------------
    X : pd.DataFrame
        Matrice de données (n, p)
    maxeigval2 : float
        Seuil d’arrêt de la division (valeur max du 2ème eigenvalue)

    Retour :
    --------
    partition : dict[str, list[str]]
        Dictionnaire des clusters : {nom_cluster : [variables]}
    """
    varclus = VarClusHi(X, maxeigval2=maxeigval2)
    varclus.varclus()

    cluster_assignments = varclus.rsquare.Cluster
    partition = {}
    for label in cluster_assignments.unique():
        features = cluster_assignments[cluster_assignments == label].index.tolist()
        partition[f"cluster_{label}"] = features

    return partition
