from typing import *
from abc import ABC, abstractmethod
import numpy as np

class FactorizedVideo(ABC):
    """
    This captures the numpy array-like functionality for factorized videos in our NMF model.
    """

    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        pass

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    @abstractmethod
    def __getitem__(
            self,
            item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]
    ):
        # Step 1: index the frames (dimension 0)
        pass

