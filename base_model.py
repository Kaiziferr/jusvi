"""
jusvi.base.base_clusterer
--------------------------
Abstract base class for all custom clustering models in jusvi.

Every custom implementation must inherit from BaseClusterer 
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Self
import numpy as np


class BaseClusterer(ABC):
    """
    Base interface for all  clustering models in jusvi.

    Abstract methods (required in every subclass)
    ----------------------------------------------
    fit(X)          -- Train the model.
    predict(X)      -- Assign cluster labels to new data.
    get_params()    -- Return hyperparameters as a dictionary.
    
    Methods with default behavior (overridable)
    --------------------------------------------
    fit_predict(X)  -- fit() + predict() in a single step.
    score(X)        -- Default quality metric (Silhouette Score).
    set_params()    -- Update hyperparameters and reset the model.
    summary()       -- Human-readable summary of the model state.

    Internal helper methods
    -----------------------
    _validate_input(X)  -- Validate and convert X to np.ndarray.
    _check_is_fitted()  -- Verify that fit() has been called.
    _reset()            -- Clear the trained state of the model.

    Attributes available after fit()
    ---------------------------------
    labels_     : np.ndarray -- Cluster label for each sample.
    n_clusters_ : int        -- Number of clusters found.
    """

    def __init__(self):
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None

    pass

    # ------------------------------------------------------------------
    # Abstract methods -- required in every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray) -> Self:
        """
        Train the model on data X.
        Must return self to allow method chaining.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign cluster labels to data X.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
        """
    
    @abstractmethod
    def get_params(self) -> dict:
        """
        Return the model hyperparameters as a dictionary.

        Returns
        -------
        params : dict
        """
        

    # ------------------------------------------------------------------
    # Methods with default behavior -- overridable
    # ------------------------------------------------------------------

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Train the model and return cluster labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
        """
        pass
    
    def score(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
        """
        Compute a clustering quality metric.

        Parameters
        ----------
        X : np.ndarray
        labels : np.ndarray, optional

        Returns
        -------
        score : float
        """
        pass
    
    def summary(self) -> str:
        """
        Return a human-readable summary of the model and its current state.

        Returns
        -------
        summary : str
        """
        pass
