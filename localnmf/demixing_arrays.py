from typing import *
import numpy as np
import scipy.sparse
from localnmf.factorized_video import FactorizedVideo
import torch


class ACArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    Computations happen transparently on GPU, if device = 'cuda' is specified
    """

    def __init__(self, fov_shape: tuple[int, int, int], order: str,
                 a: scipy.sparse.coo_matrix, c: np.ndarray,
                 device: str='cpu'):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            a (scipy.sparse.coo_matrix): Shape (pixels, components)
            c (np.ndarray). Shape (frames, components)
            device (str): The device on which the factrized matrices are processed
        """
        t = c.shape[0]
        self._device = device
        self._shape = (t,) + fov_shape
        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape([self.shape[1], self.shape[2]], order=order)
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(device)
        self._a_orig = a
        self._c_orig = c
        self._a = torch.sparse_coo_tensor(np.array(a.nonzero()), a.data, a.shape).coalesce().float().to(device)
        self._c = torch.from_numpy(c).float().to(device)
        self._order = order


    @property
    def device(self) -> str:
        return self._device

    @property
    def c(self) -> np.ndarray:
        """
        return temporal time courses of all signals, shape (frames, components)
        """
        return self._c_orig

    @property
    def a(self) -> scipy.sparse.coo_matrix:
        """
        return spatial profiles of all signals as sparse matrix, shape (pixels, components)
        """
        return self._a_orig

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

    def __getitem__(
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

        if isinstance(frame_indexer, np.ndarray):
            frame_indexer = torch.from_numpy(frame_indexer).long().to(self.device)

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
        c_crop = self._c[frame_indexer, :]
        if c_crop.ndim < self._c.ndim:
            c_crop = c_crop.unsqueeze(0)

        # Step 4: First do spatial subselection before multiplying by c
        if isinstance(item, tuple):
            pixel_space_crop = self.pixel_mat[item[1:]]
            a_indices = pixel_space_crop.flatten()
            implied_fov = pixel_space_crop.shape
            a_crop = torch.index_select(self._a, 0, a_indices)
        else:
            a_crop = self._a
            implied_fov = self.shape[1], self.shape[2]

        product = torch.sparse.mm(a_crop, c_crop.T)
        product = product.reshape(implied_fov +  (-1,))
        product = product.permute(-1, *range(product.ndim - 1))
        product = product.cpu().numpy().astype(self.dtype)

        return product.squeeze()


class PMDArray(FactorizedVideo):
    """
    Factorized demixing array for PMD movie
    """

    def __init__(self, fov_shape: tuple[int, int], order: str, u: scipy.sparse.coo_matrix,
                 r: np.ndarray, s: np.ndarray, v: np.ndarray, device: str='cpu'):
        """
        The background movie can be factorized as as the matrix product (u)(r)(s)(v),
        where u, r, s, v are the standard matrices from the pmd decomposition
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            u (scipy.sparse.coo_matrix): shape (pixels, rank1)
            r (np.ndarray): shape (rank1, rank2)
            s (np.ndarray): shape (rank 2,)
            v (np.ndarray): shape (rank2, frames)
            device (str): which device the computations take place on (cuda or cpu)
        """
        self._device = device
        t = v.shape[1]
        self._shape = (t,) + fov_shape
        self._u_orig = u
        self._v_orig = v
        self._r_orig = r
        self._s_orig = s

        self._u = torch.sparse_coo_tensor(np.array(u.nonzero()), u.data, u.shape).coalesce().float().to(device)
        self._v = torch.from_numpy(v).float().to(device)
        self._r = torch.from_numpy(r).float().to(device)
        self._s = torch.from_numpy(s).float().to(device)
        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape([self.shape[1], self.shape[2]], order=order)
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(self.device)
        self._order = order

    @property
    def device(self) -> str:
        return self._device

    @property
    def u(self) -> scipy.sparse.coo_matrix:
        return self._u_orig

    @property
    def r(self) -> np.ndarray:
        return self._r_orig

    @property
    def s(self) -> np.ndarray:
        return self._s_orig

    @property
    def v(self) -> np.ndarray:
        return self._v_orig


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

    def __getitem__(
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
            frame_indexer = torch.from_numpy(frame_indexer).long().to(self.device)

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
        v_crop = self._v[:, frame_indexer]
        if v_crop.ndim < self._v.ndim:
            v_crop = v_crop.unsqueeze(1)

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple):
            pixel_space_crop = self.pixel_mat[item[1:]]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self._u, 0, u_indices)
            implied_fov = pixel_space_crop.shape
        else:
            u_crop = self._u
            implied_fov = self.shape[1], self.shape[2]

        # Temporal term is guaranteed to have nonzero "T" dimension below
        if np.prod(implied_fov) <= v_crop.shape[1]:
            product = torch.sparse.mm(u_crop, self._r)
            product *= self._s.unsqueeze(0)
            product  = torch.matmul(product, v_crop)

        else:
            product = self._s.unsqueeze(1) * v_crop
            product = torch.matmul(self._r, product)
            product = torch.matmul(self._u, product)

        product = product.reshape(implied_fov + (-1,))
        product = product.permute(-1, *range(product.ndim - 1))
        product = product.cpu().numpy().astype(self.dtype)
        return product.squeeze()




class FluctuatingBackgroundArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    "a" is the matrix of spatial profiles
    "c" is the matrix of temporal profiles
    """

    def __init__(self, fov_shape: tuple[int, int], order: str, u: scipy.sparse.coo_matrix,
                 r: np.ndarray, q: np.ndarray, v: np.ndarray, device: str='cpu'):
        """
        The background movie can be factorized as as the matrix product (u)(r)(q)(v),
        where u, r, and v are the standard matrices from the pmd decomposition,
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            u (scipy.sparse.coo_matrix): shape (pixels, rank1)
            r (np.ndarray): shape (rank1, rank2)
            q (np.ndarray): shape (rank 2, rank 2)
            v (np.ndarray): shape (rank2, frames)
            device (str): which device the computations take place on (cuda or cpu)
        """
        self._device = device
        t = v.shape[1]
        self._shape = (t,) + fov_shape
        self._u_orig = u
        self._v_orig = v
        self._r_orig = r
        self._q_orig = q

        self._u = torch.sparse_coo_tensor(np.array(u.nonzero()), u.data, u.shape).coalesce().float().to(device)
        self._v = torch.from_numpy(v).float().to(device)
        self._r = torch.from_numpy(r).float().to(device)
        self._q = torch.from_numpy(q).float().to(device)
        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape([self.shape[1], self.shape[2]], order=order)
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(self.device)
        self._order = order

    @property
    def device(self) -> str:
        return self._device

    @property
    def u(self) -> scipy.sparse.coo_matrix:
        return self._u_orig

    @property
    def r(self) -> np.ndarray:
        return self._r_orig

    @property
    def q(self) -> np.ndarray:
        return self._q_orig

    @property
    def v(self) -> np.ndarray:
        return self._v_orig


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

    def __getitem__(
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
            frame_indexer = torch.from_numpy(frame_indexer).long().to(self.device)

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
        v_crop = self._v[:, frame_indexer]
        if v_crop.ndim < self._v.ndim:
            v_crop = v_crop.unsqueeze(1)

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple):
            pixel_space_crop = self.pixel_mat[item[1:]]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self._u, 0, u_indices)
            implied_fov = pixel_space_crop.shape
        else:
            u_crop = self._u
            implied_fov = self.shape[1], self.shape[2]

        # Temporal term is guaranteed to have nonzero "T" dimension below
        if np.prod(implied_fov) <= v_crop.shape[1]:
            product = torch.sparse.mm(u_crop, self._r)
            product = torch.matmul(product, self._q)
            product  = torch.matmul(product, v_crop)

        else:
            product = torch.matmul(self._q, v_crop)
            product = torch.matmul(self._r, product)
            product = torch.matmul(self._u, product)

        product = product.reshape(implied_fov + (-1,))
        product = product.permute(-1, *range(product.ndim - 1))
        product = product.cpu().numpy().astype(self.dtype)
        return product.squeeze()

class ResidualVideo(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data
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
