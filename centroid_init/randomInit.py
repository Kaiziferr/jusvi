"""
Randomly selects n_clusters data points from X without replacement
and uses them as the initial centroids.

Parameters
----------
random_state : int or None
    Seed for reproducibility. Default: None.
"""
from baseInitializer import BaseInitializer
import numpy as np



class RandomInit(BaseInitializer):

    def __init__(self, random_state = None):
        super().__init__(random_state)

    def initialize(self, X:np.ndarray, n_clusters:int) -> np.ndarray:
        """
        Select n_clusters random points from X as initial centroids.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        n_clusters : int

        Returns
        -------
        centroids : np.ndarray, shape (n_clusters, n_features)
        """
        try:
            rng = self._make_rng()
            indices = rng.choice(len(X), size=n_clusters, replace=False)
            return X[indices].copy()
        except:
            pass
