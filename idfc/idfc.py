from .varclus_init import initialize_partition
from .clv_refinement import refine_partition
from .feature_selection import select_interpretable_features

def run_idfc(X, rho=0.3, verbose=False):
    """
    Pipeline complet de l'algorithme IDFC :
    - Initialisation avec VARCLUS
    - Raffinement avec CLV k+1
    - Sélection de variables interprétables
    """
    if verbose:
        print("Étape 1 : Initialisation (VARCLUS)...")
    initial_partition = initialize_partition(X)
    
    if verbose:
        print("Étape 2 : Raffinement (CLV k+1)...")
    refined_partition, latent_components = refine_partition(X, initial_partition, rho=rho)

    if verbose:
        print("Étape 3 : Sélection des variables interprétables...")
    selected_features = select_interpretable_features(X, refined_partition, latent_components)

    return selected_features, refined_partition