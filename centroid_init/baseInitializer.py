"""
Abstract base class for all centroid initialization strategies.
"""


from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseInitializer(ABC):
    """
    Base interface for centroid initialization strategies.

    Attributes
    ----------
    random_state : int or None
        Seed for reproducibility. 
    """
    def __init__(self, random_state:Optional[int] = None):
        self.random_state = random_state


    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _make_rng(self) -> np.random.Generator:
        """
        Create a numpy random Generator from the stored random_state.

        Returns
        -------
        rng : np.random.Generator
        """
        try:
            return np.random.default_rng(self.random_state)
        except Exception as e:
            pass