from typing import *
import numpy as np
import scipy.sparse
from localnmf.factorized_video import FactorizedVideo


class ACVideo(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    "a" is the matrix of spatial profiles
    "c" is the matrix of temporal profiles
    """

    def __init__(self, fov_shape: tuple[int, int, int], order: str, a: scipy.sparse.coo_matrix, c: np.ndarray):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            a (scipy.sparse.coo_matrix): Shape (pixels, components)
            c (np.ndarray). Shape (frames, components)
        """
        t = c.shape[0]
        self._c = c
        self._shape = (t,) + fov_shape
        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape([self.shape[1], self.shape[2]], order=order)
        self._a = a.tocsr()
        self._order = order

    @property
    def c(self) -> np.ndarray:
        """
        return temporal time courses of all signals, shape (frames, components)
        """
        return self._c

    @property
    def a(self) -> scipy.sparse.csr_matrix:
        """
        return spatial profiles of all signals as sparse matrix, shape (pixels, components)
        """
        return self._a

    @property
    def order(self) -> str:
        """
        The spatial data is "flattened" from 2D into 1D. This specifies the order ("F" for column-major or "C" for row-major) in which reshaping happened. 
        """
        return self._order
        
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

        # Step 4: First do spatial subselection before multiplying by c
        if isinstance(item, tuple):
            pixel_space_crop = self.pixel_mat[item[1:]]
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


class FluctuatingBGVideo(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    "a" is the matrix of spatial profiles
    "c" is the matrix of temporal profiles
    """

    def __init__(self, fov_shape: tuple[int, int, int], order: str, w: scipy.sparse.coo_matrix,
                 b: np.ndarray, u: scipy.sparse.coo_matrix, r: np.ndarray, s: np.ndarray, v: np.ndarray):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            w (scipy.sparse.coo_matrix): shape (pixels, pixels)
            b (np.ndarray): shape (pixels, 1)
            u (scipy.sparse.coo_matrix): shape (pixels, rank1)
            r (scipy.sparse.coo_matrix): shape (rank1, rank2)
            s (np.ndarray): shape (rank2,)
            v (np.ndarray): shape (rank2, frames)
        """
        t = v.shape[1]
        self._shape = (t,) + fov_shape
        self.w = w.tocsr()
        self.b = b
        self.u = u
        self.v = v
        self.r = r
        self.s = s
        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape([self.shape[1], self.shape[2]], order=order)
        self._order = order

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
    def order(self) -> str:
        """
        The spatial data is "flattened" from 2D into 1D. This specifies the order ("F" for column-major or "C" for row-major) in which reshaping happened.
        """
        return self._order

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
            frame_indexer = frame_indexer

        if isinstance(frame_indexer, list):
            pass

        elif isinstance(frame_indexer, int):
            pass

        # numpy int scalar
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
        v_crop = self.v[:, frame_indexer]
        if v_crop.ndim < self.v.ndim:
            v_crop = v_crop[:, None]

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple):
            pixel_space_crop = self.pixel_mat[item[1:]]
            w_indices = pixel_space_crop.reshape((-1), order=self.order)
            
            w_crop = self.w[w_indices, :]
            implied_fov = pixel_space_crop.shape
        else:
            w_crop = self.w
            implied_fov = self.shape[1], self.shape[2]

        # Temporal term is guaranteed to have nonzero "T" dimension below
        if np.prod(implied_fov) <= v_crop.shape[1]:
            wu_crop = w_crop.dot(self.u)
            wur_crop = wu_crop.dot(self.r)
            wurs_crop = wur_crop * self.s[None, :]
            wursv_crop = wurs_crop.dot(v_crop)
            b_term = w_crop.dot(self.b)
            product = wursv_crop - b_term
        else:
            vs_crop = self.s[:, None] * v_crop
            rvs_crop = self.r.dot(vs_crop)
            urvs_crop = self.u.dot(rvs_crop)
            wurvs_crop = w_crop.dot(urvs_crop)
            b_term = w_crop.dot(self.b)
            product = wurvs_crop - b_term

        product = product.reshape(implied_fov + (-1,), order=self.order)

        num_dims = product.ndim
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


class ResidualVideo(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    "a" is the matrix of spatial profiles
    "c" is the matrix of temporal profiles
    """

    def __init__(self, pmd_video: FactorizedVideo, fluctuating_bg_video: FactorizedVideo,
                 ac_video: FactorizedVideo, static_bg: np.ndarray):
        """
        Args:
            pmd_video (FactorizedVideo): a pmd video object (mean zero, noise normalized)
            fluctuating_bg_video (FactorizedVideo): a fluctuating bg video object
            ac_video (FactorizedVideo): spatial * temporal video object (captures the sources we extracted)
            static_bg (np.ndarray): A static background term for the spatial support
        """
        self.pmd_video = pmd_video
        self.fluctuating_bg_video = fluctuating_bg_video
        self.ac_video = ac_video
        self._shape = pmd_video.shape
        self.static_background = static_bg

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return self.pmd_video.dtype

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

    def __getitem__(
            self,
            item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]
    ):
        if isinstance(item, tuple):
            if len(item) > 1:
                return (self.pmd_video[item] - self.fluctuating_bg_video[item]
                        - self.ac_video[item] - self.static_background[item[1:]][None, :])
        return (self.pmd_video[item] - self.fluctuating_bg_video[item] - self.ac_video[item]
                - self.static_background[None, :])


class ColorfulACVideo(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    "a" is the matrix of spatial profiles
    "c" is the matrix of temporal profiles
    """

    def __init__(self, fov_shape: tuple[int, int, int], order: str, a: scipy.sparse.coo_matrix, c: np.ndarray,
                 min_color: int = 30, max_color: int = 255):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            a (scipy.sparse.coo_matrix): Shape (pixels, components)
            c (np.ndarray). Shape (frames, components)
        """
        t = c.shape[0]
        self._shape = (t,) + fov_shape + (3,)
        self.pixel_mat = np.arange(np.prod(self.shape[1:3])).reshape([self.shape[1], self.shape[2]], order=order)
        self.a = a.tocsr()

        ## Establish the coloring scheme
        num_neurons = c.shape[1]
        colors = np.random.uniform(low=min_color, high=max_color, size=num_neurons * 3)
        colors = colors.reshape((num_neurons, 3))
        color_sum = np.sum(colors, axis=1, keepdims=True)
        self.final_colors = colors / color_sum

        updated_c = np.zeros((c.shape[0], c.shape[1], 3))

        for i in range(3):
            updated_c[:, :, i] = c * colors[:, [i]].T

        self.c = updated_c  # Shape (T, K, 3)
        self.order = order

    @property
    def colors(self) -> np.ndarray:
        """
        Colors used for each neuron

        Returns:
            colors (np.ndarray): Shape (number_of_neurons, 3). RGB colors of each neuron
        """
        return self.final_colors

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int, int]:
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
        c_crop = self.c[frame_indexer, :, :]
        if c_crop.ndim < self.c.ndim:
            c_crop = c_crop[None, :, :]

        c_crop = np.transpose(c_crop, axes=(1, 0, 2))

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple):
            #Note the size of the tuple is one larger due to RGB
            if len(item) == 3:
                pixel_space_crop = self.pixel_mat[item[1], :]
                a_indices = pixel_space_crop.reshape((-1), order=self.order)
                implied_fov = pixel_space_crop.shape
            elif len(item) == 4:
                pixel_space_crop = self.pixel_mat[item[1], item[2]]
                a_indices = pixel_space_crop.reshape((-1), order=self.order)
                implied_fov = pixel_space_crop.shape
            else:
                raise ValueError("Too many elements in getitem tuple")

            a_crop = self.a[a_indices, :]
        else:
            a_crop = self.a
            implied_fov = self.shape[1], self.shape[2]

        productR = a_crop.dot(c_crop[:, :, 0])
        productG = a_crop.dot(c_crop[:, :, 1])
        productB = a_crop.dot(c_crop[:, :, 2])
        product = np.stack([productR, productG, productB], axis = 2)
        product = product.reshape(implied_fov + (c_crop.shape[1],) + (3,), order=self.order)

        num_dims = product.ndim
        # Create a tuple that represents the new order of dimensions
        # This will put the last dimension first, followed by the remaining dimensions in order
        new_order = (num_dims - 2,) + tuple(range(num_dims - 2)) + (3,)
        product = np.transpose(product, axes=new_order)
        return product.squeeze()

    def __getitem__(
            self,
            item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]
    ):

        output = self._getitem(item)
        return np.array(output)
