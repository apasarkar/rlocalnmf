from typing import *
import numpy as np
import scipy.sparse
from localnmf.FactorizedVideo import FactorizedVideo


class ACVideo(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    "a" is the matrix of spatial profiles
    "c" is the matrix of temporal profiles
    """

    def __init__(self, fov_shape: tuple[int, int, int], order: str, a: scipy.sparse.coo_matrix, c: np.ndarray):
        """
        Args:
            data_shape (tuple): (Frames, fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            a (scipy.sparse.coo_matrix): Shape (pixels, components)
            c (np.ndarray). Shape (frames, components)
        """
        t = c.shape[0]
        self._shape = (t,) + fov_shape
        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape([self.shape[1], self.shape[2]], order=order)
        self.a = a.tocsr()
        self.c = c
        self.order = order

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def _getitem(
            self,
            item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]
    ):
        # Step 1: index the frames (dimension 0)

        if isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )
            frame_indexer = item[0]
        else:
            frame_indexer = item

        # Step 2: Do some basic error handling for frame_indexer before using it to slice

        if isinstance(frame_indexer, np.ndarray):
            frame_indexer = frame_indexer.tolist()

        if isinstance(frame_indexer, list):
            pass

        elif isinstance(frame_indexer, int):
            pass

        # numpy int scaler
        elif isinstance(frame_indexer, np.integer):
            frame_indexer = frame_indexer.item()

        # treat slice and range the same
        elif isinstance(frame_indexer, (slice, range)):
            start = frame_indexer.start
            stop = frame_indexer.stop
            step = frame_indexer.step

            if start is not None:
                if start > self.shape[0]:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame start index of <{start}> "
                                     f"lies beyond `n_frames` <{self.shape[0]}>")
            if stop is not None:
                if stop > self.shape[0]:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame stop index of <{stop}> "
                                     f"lies beyond `n_frames` <{self.shape[0]}>")

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            frame_indexer = slice(start, stop, step)  # in case it was a range object

        else:
            raise IndexError(
                f"Invalid indexing method, "
                f"you have passed a: <{type(item)}>"
            )

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        c_crop = self.c[frame_indexer, :]
        if c_crop.ndim < self.c.ndim:
            c_crop = c_crop[None, :]

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple):
            if len(item) == 2:
                pixel_space_crop = self.pixel_mat[item[1], :]
                a_indices = pixel_space_crop.reshape((-1), order=self.order)
                implied_fov = pixel_space_crop.shape
            elif len(item) == 3:
                pixel_space_crop = self.pixel_mat[item[1], item[2]]
                a_indices = pixel_space_crop.reshape((-1), order=self.order)
                implied_fov = pixel_space_crop.shape

            a_crop = self.a[a_indices, :]
        else:
            a_crop = self.a
            implied_fov = self.shape[1], self.shape[2]

        product = a_crop.dot(c_crop.T)
        product = product.reshape(implied_fov + (-1,), order=self.order)

        num_dims = product.ndim
        # Create a tuple that represents the new order of dimensions
        # This will put the last dimension first, followed by the remaining dimensions in order
        new_order = (num_dims - 1,) + tuple(range(num_dims - 1))
        product = np.transpose(product, axes=new_order)
        return product.squeeze()

    def __getitem__(
            self,
            item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]
    ):

        output = self._getitem(item)
        return np.array(output)
