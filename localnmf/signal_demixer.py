from abc import ABC, abstractmethod
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import ndimage as ndimage
import scipy.ndimage
import scipy.signal
import scipy.sparse
import scipy
from typing import *
import networkx as nx
from scipy.sparse.linalg import factorized
from tqdm import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable

from .demixing_arrays import (
    DemixingResults,
    StandardCorrelationImages,
    ResidualCorrelationImages,
    ResidCorrMode,
)
from localnmf import utils
from localnmf.utils import (
    add_1s_to_rowspan,
    construct_graph_from_sparse_tensor,
    color_and_get_tensors,
    ndarray_to_torch_sparse_coo,
    scipy_sparse_to_torch,
    torch_dense_to_sparse_coo,
)
from localnmf import regression_update
from localnmf.constrained_ring.cnmf_e import RingModel


def make_mask_dynamic(
    corr_img_all_r: np.ndarray,
    corr_percent: np.ndarray,
    mask_a: np.ndarray,
    data_order: str = "C",
) -> np.ndarray:
    """
    update the spatial support: connected region in corr_img(corr(Y,c)) which is connected with previous spatial support
    """
    s = np.ones([3, 3])
    mask_a = (mask_a.reshape(corr_img_all_r.shape, order=data_order)).copy()
    for ii in range(mask_a.shape[2]):
        max_corr_val = np.amax(mask_a[:, :, ii] * corr_img_all_r[:, :, ii])
        corr_thres = corr_percent * max_corr_val
        labeled_array, num_features = scipy.ndimage.measurements.label(
            corr_img_all_r[:, :, ii] > corr_thres, structure=s
        )
        u, indices, counts = np.unique(
            labeled_array * mask_a[:, :, ii], return_inverse=True, return_counts=True
        )

        if len(u) == 1:
            mask_a[:, :, ii] *= 0
        else:
            c = u[1:][np.argmax(counts[1:])]
            labeled_array = labeled_array == c
            mask_a[:, :, ii] = labeled_array

    return mask_a.reshape((-1, mask_a.shape[2]), order=data_order)


def _compute_residual_correlation_image(
    u_sparse: torch.sparse_coo_tensor,
    r: torch.tensor,
    s: torch.tensor,
    v: torch.tensor,
    factorized_ring_term: torch.tensor,
    spatial_comps: torch.sparse_coo_tensor,
    temporal_comps: torch.tensor,
    fov_dims: tuple[int, int],
    blocks: Optional[Union[torch.tensor, list]] = None,
    data_order: str = "F",
    batch_size: int = 1000,
    device: str = "cpu",
) -> ResidualCorrelationImages:
    """
    The residual correlation image for each neuron "i" can be broken up into two parts: the pixels that do
    not currently belong to the spatial support of neuron "i" and the pixels that do. For the former, we can compute
    the correlation image via ((UR(s - q) - AX^T)Vc_norm) and then divide each pixel by the norm of URs - AX^T

    For the latter, we need to compute (UR(s - q) - AX + A_iX_i^T)(Vc_norm) and then divide each pixel by the norm of
    URs - AX^T. Note that here we are only interested in the correlation values on the pixels belonging to support
    of A_i.
    """
    residual_movie_norms = torch.zeros(
        (u_sparse.shape[0], 1), device=device, dtype=torch.float32
    )
    c = temporal_comps - torch.mean(temporal_comps, dim=1, keepdim=True)
    c /= torch.linalg.norm(c, dim=0, keepdim=True)
    c = torch.nan_to_num(c, nan=0, posinf=0, neginf=0)

    v_mean = torch.mean(v, dim=1, keepdim=True)
    c_mean = torch.mean(temporal_comps.T, dim=1, keepdim=True)
    e = torch.ones((1, v.shape[1]), device=device, dtype=torch.float32) @ v.T

    x = temporal_comps.T @ v.T
    c_orth_v = temporal_comps.T - x @ v
    c_U, c_s, Z = torch.linalg.svd(c_orth_v, full_matrices=False)
    y = c_U @ torch.diag_embed(c_s)

    """
    So now ac^T has been divided into two components; one that fits into the V basis, and another that is orth:
    axV and ayZ
    """

    fluctuating_baseline_subtracted_term = torch.diag(s) - factorized_ring_term

    m_baseline = torch.sparse.mm(
        u_sparse, (r @ (fluctuating_baseline_subtracted_term @ v_mean))
    ) - torch.sparse.mm(spatial_comps, c_mean)

    num_iters = math.ceil(r.shape[1] / batch_size)
    for k in range(num_iters):
        start = k * batch_size
        end = min(start + batch_size, r.shape[1])
        temp = (
            torch.sparse.mm(u_sparse, (r @ fluctuating_baseline_subtracted_term[:, start:end]))
            - torch.matmul(spatial_comps, x[:, start:end])
            - m_baseline @ e[:, start:end]
        )
        residual_movie_norms += torch.square(
            torch.linalg.norm(temp, dim=1, keepdim=True)
        )

    num_iters = math.ceil(y.shape[1] / batch_size)
    for k in range(num_iters):
        start = k * batch_size
        end = min(start + batch_size, y.shape[1])
        temp = torch.sparse.mm(spatial_comps, y[:, start:end])
        residual_movie_norms += torch.square(
            torch.linalg.norm(temp, dim=1, keepdim=True)
        )

    residual_movie_norms = torch.sqrt(residual_movie_norms)

    """
    Now we compute a matrix with the same sparsity as "spatial_comps" describing the residual corr image for each 
    neuron on its support
    """
    final_rows = []
    final_cols = []
    final_values = []

    if blocks is None:
        blocks = torch.arange(c.shape[1], device=device).unsqueeze(1)
    for index_select_tensor_net in blocks:
        # print(f"index select tensor net {index_select_tensor_net}")
        num_batches = math.ceil(index_select_tensor_net.shape[0] / batch_size)
        for batch_index in range(num_batches):

            start_batch = batch_index * batch_size
            end_batch = min(start_batch + batch_size, index_select_tensor_net.shape[0])
            index_select_tensor = index_select_tensor_net[start_batch:end_batch]
            # print(f"index select tensor here is {index_select_tensor}")
            a_subset = torch.index_select(spatial_comps, 1, index_select_tensor).coalesce().to_dense()
            c_mean_subset = torch.index_select(c_mean, 0, index_select_tensor)
            c_select = torch.index_select(c, 1, index_select_tensor)
            rows_to_keep = (
                torch.matmul(a_subset, torch.ones((a_subset.shape[1], 1), device=device))
                .squeeze()
                .nonzero()
                .squeeze()
            )
            a_subset_rowcrop = torch.index_select(a_subset, 0, rows_to_keep)
            a_rowcrop = torch.index_select(spatial_comps, 0, rows_to_keep).coalesce()

            x_crop = torch.index_select(x, 0, index_select_tensor)
            y_crop = torch.index_select(y, 0, index_select_tensor)
            u_rowcrop = torch.index_select(u_sparse, 0, rows_to_keep).coalesce()
            m_rowcrop = torch.index_select(m_baseline, 0, rows_to_keep) + torch.mm(
                a_subset_rowcrop, c_mean_subset
            )

            urs_term = torch.sparse.mm(u_rowcrop, r) @ fluctuating_baseline_subtracted_term
            ax_term = torch.sparse.mm(a_rowcrop, x)
            ax_keep_term = torch.matmul(a_subset_rowcrop, x_crop)
            mean_sub_term = m_rowcrop @ e
            net_spatial_basis_V = urs_term - ax_term + ax_keep_term - mean_sub_term
            curr_norm_V = torch.square(
                torch.linalg.norm(net_spatial_basis_V, dim=1, keepdim=True)
            )

            ay_term = torch.sparse.mm(a_rowcrop, y)
            ay_keep_term = torch.mm(a_subset_rowcrop, y_crop)
            net_spatial_basis_Z = -1 * ay_term + ay_keep_term
            curr_norm_Z = torch.square(
                torch.linalg.norm(net_spatial_basis_Z, dim=1, keepdim=True)
            )


            curr_norm = torch.sqrt(curr_norm_V + curr_norm_Z)

            corr_img = (
                net_spatial_basis_V @ v @ c_select + net_spatial_basis_Z @ Z @ c_select
            )
            corr_img /= curr_norm
            corr_img = torch.nan_to_num(corr_img, nan=0.0, posinf=0.0, neginf=0.0)
            local_rows, local_cols = a_subset_rowcrop.nonzero(as_tuple=True)
            local_values = corr_img[(local_rows, local_cols)]
            real_rows = rows_to_keep[local_rows]
            real_cols = index_select_tensor[local_cols]
            final_rows.append(real_rows)
            final_cols.append(real_cols)
            final_values.append(local_values)

    final_rows = torch.cat(final_rows, 0)
    final_cols = torch.cat(final_cols, 0)
    final_values = torch.cat(final_values, 0).to(device)
    resid_corr_indices = torch.stack([final_rows, final_cols]).to(device)
    resid_corr_on_support = torch.sparse_coo_tensor(
        resid_corr_indices, final_values, spatial_comps.size()
    ).coalesce()
    residual_array = ResidualCorrelationImages(
        u_sparse,
        r,
        s,
        v,
        factorized_ring_term,
        spatial_comps,
        temporal_comps,
        resid_corr_on_support,
        m_baseline.squeeze(),
        residual_movie_norms.squeeze(),
        fov_dims,
        mode=ResidCorrMode.DEFAULT,
        order=data_order,
    )
    return residual_array


def _compute_standard_correlation_image(
    u_sparse: torch.sparse_coo_tensor,
    r: torch.tensor,
    s: torch.tensor,
    v: torch.tensor,
    temporal_traces: torch.tensor,
    fov_dims: tuple[int, int],
    data_order: str = "F",
    frame_batch_size: int = 1000,
    device: str = "cpu",
) -> StandardCorrelationImages:
    """
    Correlation image calculation using u, r, s, v

    Args:
        u_sparse (torch.sparse_coo_tensor): dims (d x r), where the FOV has d pixels
        r (torch.tensor) dims (rank1 x rank2), where rank1 is the (overall) rank of the PMD decomposition and rank2 is the
            pruned rank
        s (torch.tensor): dims (rank 2). Singular values of the PMD decomposition
        v (torch.tensor): dims (rank 2, number of frames): The temporal basis (right singular vectors of the PMD
            decomposition.
        temporal_traces (torch.tensor): shape (number of frames, number of neural signals). Temporal traces currently
            extracted
        frame_batch_size (int): The number of frames we expand at any given time (pixels x frames) to avoid OOM errors
        device (str): The device on which computations occur (given to pytorch to move tensors around).

    Returns:
        corr_array (StandardCorrelationImages): A FactorizedVideo object that can lazily compute correlation images.

    """

    # Step 1: Standardize c
    c = temporal_traces - torch.mean(temporal_traces, dim=0, keepdim=True)
    c_norm = torch.sqrt(torch.sum(c * c, dim=0, keepdim=True))
    c /= c_norm

    ##Step 2:
    v_mean = torch.mean(v, dim=1, keepdim=True)
    rv_mean = torch.matmul(r, s.unsqueeze(1) * v_mean)
    m = torch.sparse.mm(u_sparse, rv_mean)  # Dims: d x 1

    ##Step 3:
    e = torch.matmul(torch.ones([1, v.shape[1]], device=device), v.t())

    """
    Step 4: Get the pixelwise norm of (UR - me)V. Since V has orthonormal rows, this amounts to finding the row-wise
    norm of (UR - me)
    """

    ## Step 4c: Get diag(U*R*R^t*U^t)
    data_norms = torch.zeros([u_sparse.shape[0], 1], device=device)

    batch_iters = math.ceil(r.shape[1] / frame_batch_size)
    for k in range(batch_iters):
        start = frame_batch_size * k
        end = min(start + frame_batch_size, u_sparse.shape[0])
        data_chunk = (
            torch.sparse.mm(u_sparse, r[:, start:end]) * s[start:end].unsqueeze(0)
            - m @ e[:, start:end]
        )
        data_norms += torch.square(torch.linalg.norm(data_chunk, dim=1, keepdim=True))
    data_norms = torch.sqrt(data_norms)

    corr_array = StandardCorrelationImages(
        u_sparse, r, s, v, c, m.squeeze(), data_norms.squeeze(), fov_dims, data_order
    )
    return corr_array


def get_mean_data(U_sparse, R, s, V):
    """
    Routine for computing the mean of the dataset.
    Inputs:
        (U_sparse, R, s, V) the SVD-like representation of the PMD data. Each of these are torch.Tensors
            of torch.sparse_coo_tensor
        U_sparse: torch.sparse_coo_tensor. shape (d, K)
        R: torch.Tensor. Shape (K, K)
        s: torch.Tensor. Shape (K)
        V: torch.Tensor. Shape (K, T)
    """
    V_mean = torch.mean(V, dim=1, keepdim=True)
    V_mean = V_mean * s[:, None]
    RsV_mean = torch.matmul(R, V_mean)
    URsV_mean = torch.sparse.mm(U_sparse, RsV_mean)
    return URsV_mean


def get_new_orthonormal_vector(U_sparse, R, tol=1e-6):
    """
    Input:
        U_sparse: torch.sparse_coo_tensor of shape (d, K), where d is the number of pixels in the movie, K is the rank of the PMD decomposition
        R: torch.Tensor of shape (K, K)

    Output: a torch.Tensor with shape (d, 1), and a flag indicating whether the search worked (1) or failed (0)
    """
    device = R.device
    RRt = torch.matmul(R, R.t())

    num_tries = min(U_sparse.shape[0], 500)

    indices_to_try = np.random.choice(U_sparse.shape[0], size=num_tries, replace=False)
    indices_to_try[0] = 0
    indices_to_try[-1] = U_sparse.shape[0] - 1

    for k in indices_to_try:
        curr_index = torch.arange(k, k + 1, device=R.device)
        current_vector = torch.zeros([U_sparse.shape[0], 1], device=device)
        current_vector[k, 0] = 1
        U_sparse_row = torch.index_select(U_sparse.t(), 1, curr_index)

        prod = torch.sparse.mm(U_sparse_row.t(), RRt.t()).t()
        output = current_vector - torch.sparse.mm(U_sparse, prod)

        if not torch.all(torch.abs(output) < 1e-3):
            output = output / torch.linalg.norm(output)
            return output, 1

    return torch.zeros([U_sparse.shape[0], 1], device=device), 0


def PMD_setup_routine(U_sparse, R, s, V):
    """
    Inputs:
        U_sparse: torch.sparse_coo_tensor of shape (d, K), where d is the number of pixels in the movie, K is the rank of the PMD decomposition
        R: torch.Tensor of shape (K, K)
        s: torch.Tensor of shape (K). This describes a diagonal matrix (all other elts 0)
        V: torch.Tensor of shape (K, T) where R1 is the rank of the PMD decomposition

        Key: Conceptually, U_sparse*R*s*V (here, we don't multiply by "s" per se, we multiply by the diagonal matrix which "s" represents) is the expanded PMD decomposition


    Outputs:
        (U_sparse, R, s, V), potentially modified such that the rowspan of V contains the 1's vector. Here is what definitely stays the same --
            - U_sparse * R is a matrix consisting of orthonormal columns (here we operate under the assumption that U_sparse has many more rows than columns)
            - V has orthonormal rows
            - s describes the diagonal of an otherwise empty matrix
    """
    if s.shape[0] == min(U_sparse.shape[0], V.shape[1]):
        print("Full rank SVD provided; no need to modify anything")
        return U_sparse, R, s, V
    device = V.device
    pad_flag, V = add_1s_to_rowspan(V)
    if pad_flag:

        new_vec, attempt_flag = get_new_orthonormal_vector(U_sparse, R)
        if not attempt_flag:
            print(
                "The V row space enhancement step did not find a suitable orthogonal vector"
            )
            pass
        else:
            nonzero_indices = torch.nonzero(new_vec)
            values = new_vec[nonzero_indices[:, 0], nonzero_indices[:, 1]]

            rows = nonzero_indices[:, 0]
            cols = torch.zeros_like(rows) + U_sparse.shape[1]

            # Add new_vec as a column to U_sparse (this should be easy)
            original_values = U_sparse.values()
            original_rows = U_sparse.indices()[0, :]
            original_cols = U_sparse.indices()[1, :]

            new_values = torch.cat((original_values, values))
            new_rows = torch.cat((original_rows, rows))
            new_cols = torch.cat((original_cols, cols))

            U_sparse = torch.sparse_coo_tensor(
                torch.stack([new_rows, new_cols]),
                new_values,
                (U_sparse.shape[0], U_sparse.shape[1] + 1),
            ).coalesce()
            # U_sparse = torch_sparse.tensor.SparseTensor(row=new_rows, col=new_cols, value=new_values, \
            #                                             sparse_sizes=(U_sparse.sparse_sizes()[0],
            #                                                           U_sparse.sparse_sizes()[1] + 1)).coalesce()

            # Add the appropriate padding to R
            R = torch.hstack([R, torch.zeros([R.shape[0], 1], device=device)])
            appendage = torch.zeros([1, R.shape[1]], device=device)
            appendage[0, -1] = 1
            R = torch.vstack([R, appendage])
            s = torch.cat([s, torch.zeros([1], device=device)])

    return U_sparse, R, s, V


def process_custom_signals(
    a: torch.sparse_coo_tensor,
    u_sparse: torch.sparse_coo_tensor,
    r: torch.tensor,
    s: torch.tensor,
    v: torch.tensor,
    c_nonneg: bool = True,
    blocks=None,
) -> tuple[
    torch.sparse_coo_tensor, torch.sparse_coo_tensor, torch.tensor, torch.tensor
]:
    """
    Given spatial footprints matrix "a", prepare a set of initialized signals (spatial footprints, masks,
    temporal matrices, baselines) for running demixing in lieu of superpixelization.

    Function creates a copy of the input "a" tensor to avoid unintended side effects to original input.

    Params:
        a (torch.sparse_coo_tensor): (shape (d1*d2, K) where K is number of neural signals
        u_sparse (torch.sparse_coo_tensor): shape (d1*d2, rank 1) where rank 1 is larger PMD rank
        r (torch.tensor): Shape (rank 1, rank 2) where rank 2 is smaller PMD rank.
        s (torch.tensor): Shape (rank 2). Singular values of PMD decomposition.
        V (torch.tensor): shape (rank 2, num_frames).
        device (str): either 'cpu' or 'cuda'. Passed directly to pytorch "to" function for tensors to place data
            on the correct device.
        order (str): order in which 3d data is reshaped to 2d
    """
    device = v.device

    if not a.is_coalesced():
        a = a.coalesce()  # Coalesce to remove duplicate indices

    new_indices = a.indices().clone().to(device)
    new_values = a.values().clone().to(device)
    dims = (u_sparse.shape[0], a.shape[1])

    a = torch.sparse_coo_tensor(
        indices=new_indices, values=new_values, size=dims
    ).coalesce()

    c = torch.zeros([v.shape[1], a.shape[1]], device=device, dtype=torch.float)

    uv_mean = get_mean_data(u_sparse, r, s, v)

    # Baseline update followed by 'c' update:
    b = regression_update.baseline_update(uv_mean, a, c)
    c = regression_update.temporal_update_hals(
        u_sparse, r, s, v, a, c, b, c_nonneg=c_nonneg, blocks=blocks
    )

    c_norm = torch.linalg.norm(c, dim=0)
    nonzero_dim1 = torch.nonzero(c_norm).squeeze()

    # Only keep the good indices, based on nonzero_dim1
    c_torch = torch.index_select(c, 1, nonzero_dim1)
    a_torch = torch.index_select(a, 1, nonzero_dim1)
    a_mask = a_torch.bool()

    return a_torch, a_mask, c_torch, b


def get_median(tensor, axis):
    max_val = torch.max(tensor, dim=axis, keepdim=True)[0]
    tensor_med_1 = torch.median(
        torch.cat((tensor, max_val), dim=axis), dim=axis, keepdim=True
    )[0]
    tensor_med_2 = torch.median(tensor, dim=axis, keepdim=True)[0]

    tensor_med = torch.mul(tensor_med_1 + tensor_med_2, 0.5)
    return tensor_med


def threshold_data_inplace(movie_chunk, mad_threshold_value: int = 2, dim: int = 2):
    """
    Threshold data: in each pixel, compute the median and median absolute deviation (MAD),
    then zero all bins (x,t) such that Yd(x,t) < med(x) + th * MAD(x).
    Min subtract and return the result.
    Args:
        movie_chunk (torch.tensor). Shape (fov dim 1, fov dim 2, frames)
        mad_threshold_value (int): "th", as described above. The number of median absolute deviations in the threshold.
        dim (int): The axis over which operations are applied.
    Returns:
        Yd: This is an in-place operation
    """

    # Get per-pixel medians
    movie_median = get_median(movie_chunk, axis=dim)
    diff = torch.sub(movie_chunk, movie_median)

    # Calculate MAD values
    torch.abs(diff, out=diff)
    movie_mad_values = get_median(diff, axis=dim)

    # Calculate actual threshold
    torch.mul(movie_mad_values, mad_threshold_value, out=movie_mad_values)
    th_val = movie_median.add(movie_mad_values)

    # Subtract threshold values
    torch.sub(movie_chunk, th_val, out=movie_chunk)
    torch.clamp(movie_chunk, min=0, out=movie_chunk)
    return movie_chunk


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def reshape_c(x, shape):
    return torch.reshape(x, shape)


def get_total_edges(d1, d2):
    assert (
        d1 > 2 and d2 > 2
    ), "At least one dimensions is less than 2 pixels. Not supported"
    overcount = 8 * (d1 - 2) * (d2 - 2) + 2 * (d1 - 2) * 5 + 2 * (d2 - 2) * 5 + 4 * 3
    return math.ceil(overcount / 2)


def get_local_correlation_structure(
    U_sparse: torch.sparse_coo_tensor,
    V: torch.tensor,
    dims: tuple[int, int, int],
    th: int,
    order: str = "C",
    batch_size: int = 10000,
    pseudo: float = 0,
    tol: float = 0.000001,
    a: Optional[torch.sparse_coo_tensor] = None,
    c: torch.tensor = None,
):
    """
    Computes a local correlation data structure, which describes the correlations between all neighboring pairs of pixels

    Context: here,
    d1, d2 are the fov dimensions of the original data (i.e. 512 x 512 pixels or the like)
    T is the number of frames in the video
    R is the rank of the PMD decomposition (so U_sparse has shape (d1*d2, R) and V has shape (R, T))
    K is the number of neural signals identified (in "a" and "c", if they are provided)

    Inputs:
        U_sparse: torch.sparse_coo_tensor object, shape (d1*d2, T)
        V: torch.Tensor, shape (R, T)
        dims: (d1, d2, T)
        th: int (positive integer), describes the MAD threshold. We use this to threshold the pixels for when we compute correlations.
            We compute the median and median absolute deviation (MAD), then zero all bins (x,t) such that Yd(x,t) < med(x) + th * MAD(x).
        order: "C" or "F" Indicates how we reshape the 2D images of the video (d1, d2) into (d1*d2) column vectors. The order here is important for consistency.
        batch_size: int. Maximum number of pixels of the movie that we fully expand out (i.e. we never have more than batch_size * T -sized Tensor in device memory.
            This is useful for GPU memory management, especially on small graphics cards.
        pseudo: float >= 0. a robust correlation parameter, used in the robust correlation calculation between every pair of neighboring pixels.
            In general, a higher value of pseudo will reduce the compute correlation between two pixels.
        tol: float: A tolerance parameter used when normalizing time series (to avoid divide by "close to 0" issues).
        a Optional[torch.sparse_coo_tensor]: A (d1*d2, K)-shaped ndarray whose columns describe the correlation structure of the data.
        c Optional[torch.tensor]: A (T, K)-shaped array whose columns describe the estimated fluorescence time course of each signal.

    Returns:
    The following correlation Data Structure:
    To understand this, recall that we flatten the 2D field of view into a 1 dimensional column vector
        dim1_coordinates: torch.Tensor, 1 dimensional. Describes a list of row coordinates in the field of view
        dim2_coordinates: torch.Tensor, 1 dimensional. Describes a list of row coordinates in the field of view
        correlations: torch.Tensor, 1 dimensional.

    Key: each element at index i of "correlations" describes the computed correlation between the adjacent pixels given by
            dim1_coordinates[i] and dim2_coordinates[i]
    """

    device = V.device
    if a is not None and c is not None:
        resid_flag = True
    else:
        resid_flag = False

    dims = (dims[0], dims[1], V.shape[1])

    ref_mat = torch.arange(np.prod(dims[:-1]), device=device)
    if order == "F":
        ref_mat = reshape_fortran(ref_mat, (dims[0], dims[1]))
    else:
        ref_mat = reshape_c(ref_mat, (dims[0], dims[1]))

    tilesize = math.floor(math.sqrt(batch_size))

    iters_x = math.ceil((dims[0] / (tilesize - 1)))
    iters_y = math.ceil((dims[1] / (tilesize - 1)))

    # Pixel-to-pixel coordinates for highly-correlated neighbors
    total_edges = 2 * get_total_edges(
        dims[0], dims[1]
    )  # Here we multiply by two because when we tile the FOV, some correlations are computed twice
    point1_indices = torch.zeros((total_edges), dtype=torch.int32, device=device)
    point2_indices = torch.zeros((total_edges), dtype=torch.int32, device=device)
    correlation_values = torch.zeros((total_edges), dtype=torch.float32, device=device)

    progress_index = 0
    for tile_x in range(iters_x):
        for tile_y in range(iters_y):
            x_pt = (tilesize - 1) * tile_x
            x_end = x_pt + tilesize
            y_pt = (tilesize - 1) * tile_y
            y_end = y_pt + tilesize

            indices_curr_2d = ref_mat[x_pt:x_end, y_pt:y_end]
            x_interval = indices_curr_2d.shape[0]
            y_interval = indices_curr_2d.shape[1]

            if order == "F":
                indices_curr = reshape_fortran(
                    indices_curr_2d, (x_interval * y_interval,)
                )
            else:
                indices_curr = reshape_c(indices_curr_2d, (x_interval * y_interval,))

            U_sparse_crop = torch.index_select(U_sparse, 0, indices_curr)
            if order == "F":
                Yd = reshape_fortran(
                    torch.sparse.mm(U_sparse_crop, V), (x_interval, y_interval, -1)
                )
            else:
                Yd = reshape_c(
                    torch.sparse.mm(U_sparse_crop, V), (x_interval, y_interval, -1)
                )
            if resid_flag:
                a_sparse_crop = torch.index_select(a, 0, indices_curr)
                if order == "F":
                    ac_mov = reshape_fortran(
                        torch.sparse.mm(a_sparse_crop, c.T),
                        (x_interval, y_interval, -1),
                    )
                else:
                    ac_mov = reshape_c(
                        torch.sparse.mm(a_sparse_crop, c.T),
                        (x_interval, y_interval, -1),
                    )
                Yd = torch.sub(Yd, ac_mov)

            # Get MAD-thresholded movie in-place
            Yd = threshold_data_inplace(Yd, th)

            # Permute the movie
            Yd = Yd.permute(2, 0, 1)

            # Normalize each trace in-place, using robust correlation statistic
            torch.sub(Yd, torch.mean(Yd, dim=0, keepdim=True), out=Yd)
            divisor = torch.std(Yd, dim=0, unbiased=False, keepdim=True)
            final_divisor = torch.sqrt(divisor * divisor + pseudo**2)

            # If divisor is 0, that implies that the std of a 0-mean pixel is 0, which means the
            # pixel is 0 everywhere. In this case, set divisor to 1, so Yd/divisor = 0, as expected
            final_divisor[divisor < tol] = 1  # Temporarily set all small values to 1.
            torch.reciprocal(final_divisor, out=final_divisor)
            final_divisor[divisor < tol] = 0  ##Now set these small values to 0

            torch.mul(Yd, final_divisor, out=Yd)

            # Vertical pixel correlations
            rho = torch.mean(Yd[:, :-1, :] * Yd[:, 1:, :], dim=0)
            point1_curr = indices_curr_2d[:-1, :].flatten()
            point2_curr = indices_curr_2d[1:, :].flatten()
            rho_curr = rho.flatten()
            point1_indices[progress_index : progress_index + point1_curr.shape[0]] = (
                point1_curr
            )
            point2_indices[progress_index : progress_index + point1_curr.shape[0]] = (
                point2_curr
            )
            correlation_values[
                progress_index : progress_index + point1_curr.shape[0]
            ] = rho_curr
            progress_index = progress_index + point1_curr.shape[0]

            # Horizontal pixel correlations
            rho = torch.mean(Yd[:, :, :-1] * Yd[:, :, 1:], dim=0)
            point1_curr = indices_curr_2d[:, :-1].flatten()
            point2_curr = indices_curr_2d[:, 1:].flatten()
            rho_curr = rho.flatten()
            point1_indices[progress_index : progress_index + point1_curr.shape[0]] = (
                point1_curr
            )
            point2_indices[progress_index : progress_index + point1_curr.shape[0]] = (
                point2_curr
            )
            correlation_values[
                progress_index : progress_index + point1_curr.shape[0]
            ] = rho_curr
            progress_index = progress_index + point1_curr.shape[0]

            # Top left and bottom right diagonal correlations
            rho = torch.mean(Yd[:, :-1, :-1] * Yd[:, 1:, 1:], dim=0)
            point1_curr = indices_curr_2d[:-1, :-1].flatten()
            point2_curr = indices_curr_2d[1:, 1:].flatten()
            rho_curr = rho.flatten()
            point1_indices[progress_index : progress_index + point1_curr.shape[0]] = (
                point1_curr
            )
            point2_indices[progress_index : progress_index + point1_curr.shape[0]] = (
                point2_curr
            )
            correlation_values[
                progress_index : progress_index + point1_curr.shape[0]
            ] = rho_curr
            progress_index = progress_index + point1_curr.shape[0]

            # Bottom left and top right diagonal correlations
            rho = torch.mean(Yd[:, 1:, :-1] * Yd[:, :-1, 1:], dim=0)
            point1_curr = indices_curr_2d[1:, :-1].flatten()
            point2_curr = indices_curr_2d[:-1, 1:].flatten()
            rho_curr = rho.flatten()
            point1_indices[progress_index : progress_index + point1_curr.shape[0]] = (
                point1_curr
            )
            point2_indices[progress_index : progress_index + point1_curr.shape[0]] = (
                point2_curr
            )
            correlation_values[
                progress_index : progress_index + point1_curr.shape[0]
            ] = rho_curr
            progress_index = progress_index + point1_curr.shape[0]

    return (
        point1_indices[:progress_index],
        point2_indices[:progress_index],
        correlation_values[:progress_index],
    )


def find_superpixel_UV(
    dims,
    cut_off_point,
    length_cut,
    dim1_coordinates,
    dim2_coordinates,
    correlations,
    order,
):
    """
    Find in the PMD denoised movie. We are given arrays describing the 'local' correlation structure for each pixel of the movie.
    We can threshold this correlation to identify the pairs of neighboring pixels with high correlations. This produces a "graph", whose nodes are the set
    of pixels. The clusters of connected components in this graph are superpixels.


    Context:
        d1, d2: the FOV dimensions
        T: The number of frames
        R: rank of PMD decomposition
    Parameters:
    ----------------
    U_sparse: torch.sparse_coo_tensor object, shape (d1*d2, T)
    V: torch.Tensor, shape (R, T)
    dims: (d1, d2, T)
    cut_off_point: float between 0 and 1. Correlation threshold which we use to determine whether two neighboring pixels are "highly correlated"
    length_cut: int. Minimum size of a connected component required for us to call it a superpixel

    Correlation Data Structure:
    To understand this, note that we flatten the 2D field of view into a 1 dimensional column vector
        dim1_coordinates: torch.Tensor, 1 dimensional. Describes a list of row coordinates in the field of view
        dim2_coordinates: torch.Tensor, 1 dimensional. Describes a list of row coordinates in the field of view
        correlations: torch.Tensor, 1 dimensional. Element at index i of this matrix describes the correlation the pixels given by
            dim1_coordinates[i] and dim2_coordinates[i]

    order: "F" or "C", indicates the order in which we reshape 2D (d1, d2)-shaped images into (d1*d2)-shaped column vectors
    Return:
    ----------------
    connect_mat_1: 2d np.darray, d1 x d2
        illustrate position of each superpixel.
        Each superpixel has all of its pixels labeled the same value.
    comps: list, length = number of superpixels
        comp on comps is also list, its value is position of each superpixel in Yt_r = Yt.reshape(np.prod(dims[:2]),-1,order="F")
    """
    # Here we can apply the threshold:
    good_indices = torch.where(correlations > cut_off_point)[0]
    A = torch.index_select(dim1_coordinates, 0, good_indices).cpu().numpy()
    B = torch.index_select(dim2_coordinates, 0, good_indices).cpu().numpy()

    ########### form connected componnents #########
    G = nx.Graph()
    G.add_edges_from(list(zip(A, B)))
    comps = list(nx.connected_components(G))

    connect_mat = np.zeros(np.prod(dims[:2]))

    ii = 0
    for comp in comps:
        if len(comp) > length_cut:
            connect_mat[list(comp)] = ii + 1  # permute_col[ii]
            ii = ii + 1
    connect_mat_1 = connect_mat.reshape(dims[0], dims[1], order=order)
    return connect_mat_1, comps


def spatial_temporal_ini_UV(
    u_sparse: torch.sparse_coo_tensor,
    r: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    dims: Tuple[int, int, int],
    comps: List[Set[int]],
    length_cut: int,
    a: Optional[torch.sparse_coo_tensor] = None,
    c: Optional[torch.tensor] = None,
) -> Tuple[torch.sparse_coo_tensor, torch.tensor]:
    """
    Apply rank 1 NMF to find spatial and temporal initialization for each superpixel in Yt.

    Args:
        u_sparse (torch.sparse_coo_tensor): Shape (d1*d2, R1) where d1, d2 are field of view dimensions.
        r (torch.Tensor): Shape (R1, R2). PMD spatial mixing matrix. UR describes left singular vectors.
        s (torch.Tensor): Shape (R2,). Singular values of PMD decomposition.
        v (torch.Tensor): Shape (R2, T). T is the number of timepoints.
        dims (tuple): Contains (d1, d2, T). Describes data shape.
        comps (List[Set[int]]): Each set describes a single superpixel. The values in that set are the pixel indices where the superpixel is active.
        length_cut (int): Minimum number of components required for a superpixel to be declared.
        a (Optional[np.ndarray], optional): Shape (d1*d2, K) where K is the number of neurons. Defaults to None.
        c (Optional[np.ndarray], optional): Shape (T, K) where T is the number of time points. Defaults to None.

    Returns:
        a_init (torch.sparse_coo_tensor): Shape (d1*d2, K). Describes initial spatial footprints.
        c_init (torch.tensor): Shape (T, K). Describes temporal initializations.
    """
    device = v.device
    dims = (dims[0], dims[1], v.shape[1])
    t = v.shape[1]

    pre_existing = a is not None and c is not None
    if pre_existing:
        k = c.shape[1]
    else:
        k = 0

    total_length = 0
    good_indices = []
    index_val = 0

    # Step 1: Identify which connected components are large enough to qualify as superpixels
    for comp in comps:
        curr_length = len(list(comp))
        if curr_length > length_cut:
            good_indices.append(index_val)
            total_length += curr_length
        index_val += 1
    comps = [comps[good_indices[i]] for i in range(len(good_indices))]

    # Step 2: Turn the superpixels into "a" and "c" values
    a_row_init = torch.zeros(total_length, dtype=torch.long)
    a_col_init = torch.zeros(total_length, dtype=torch.long)
    a_value_init = torch.zeros(total_length, dtype=v.dtype)

    ref_point = 0
    counter = 0
    for comp in comps:
        curr_length = len(list(comp))
        ##Below line super important: + k allows concatenation
        a_col_init[ref_point : ref_point + curr_length] = counter + k
        a_row_init[ref_point : ref_point + curr_length] = torch.Tensor(list(comp))
        a_value_init[ref_point : ref_point + curr_length] = 1
        ref_point += curr_length
        counter = counter + 1

    a_row_init = a_row_init.to(device)
    a_col_init = a_col_init.to(device)
    a_value_init = a_value_init.to(device)

    if pre_existing:
        c_final = torch.cat([c, torch.zeros(t, len(comps), device=device)], dim=1)
        a_orig_row, a_orig_col = a.indices()
        a_orig_values = a.values()
        final_rows = torch.cat([a_row_init, a_orig_row], dim=0)
        final_cols = torch.cat([a_col_init, a_orig_col], dim=0)
        final_values = torch.cat([a_value_init, a_orig_values], dim=0)
    else:
        c_final = torch.zeros(t, len(comps), device=device)
        final_rows = a_row_init
        final_cols = a_col_init
        final_values = a_value_init

    ## Define a_sparse and compute terms for running 1 set of HALS updates
    a_sparse = (
        torch.sparse_coo_tensor(
            torch.stack([final_rows, final_cols]),
            final_values,
            (dims[0] * dims[1], k + len(comps)),
        )
        .coalesce()
        .to(device)
    )
    uv_mean = get_mean_data(u_sparse, r, s, v)
    mean_ac = torch.sparse.mm(a_sparse, torch.mean(c_final.t(), dim=1, keepdim=True))
    uv_mean -= mean_ac

    for _ in range(1):
        b_torch = regression_update.baseline_update(uv_mean, a_sparse, c_final)
        c_final = regression_update.temporal_update_hals(
            u_sparse, r, s, v, a_sparse, c_final, b_torch
        )

        b_torch = regression_update.baseline_update(
            uv_mean.to(device), a_sparse, c_final
        )
        a_sparse = regression_update.spatial_update_hals(
            u_sparse, r, s, v, a_sparse, c_final, b_torch
        )

    # Now return only the newly initialized components
    col_index_tensor = torch.arange(start=k, end=k + len(comps), step=1, device=device)
    a_sparse = torch.index_select(a_sparse, 1, col_index_tensor)
    c_final = torch.index_select(c_final, 1, col_index_tensor)

    return (
        c_final,
        a_sparse,
    )


def delete_comp(
    spatial_components,
    temporal_components,
    standard_correlation_image: StandardCorrelationImages,
    spatial_masks,
    components_to_delete,
    reasoning_message,
    plot_en,
    order="C",
):
    """
    General routine to delete components in the demixing procedure
    Args:

        spatial_components (torch.sparse_coo_tensor): Dimensions (d, K), d = number of pixels in movie,
            K = number of neurons
        temporal_components (torch.Tensor): Dimensions (T, K), K = number of neurons in movie
        standard_correlation_image (StandardCorrelationImages): Dimensions (d, K). d = number of pixels in movie, K = number of neurons
        spatial_masks (torch.sparse_coo_tensor): Dimensions (d, K). Dtype bool. d = number of pixels in movie, K = number of neurons
        components_to_delete (torch.tensor): 1D tensor indicating which components to delete
        reasoning_message (str): An option to provide a reason for why deletion is happening
        plot_en (bool): Indicates whether plotting is enabled
        order (str): "C" or "F" depending on how we flatten 2D spatial data into 1D vectors (and vice versa)
    Returns:
        Tuple: A tuple containing the following elements:
            - spatial_components (torch.sparse_coo_tensor): Updated sparse tensor of dimensions (d, K')
              containing the spatial components after deletion, where K' is the new number of remaining neurons.
            - temporal_components (torch.Tensor): Updated tensor of dimensions (T, K')
              containing the temporal components after deletion.
            - standard_correlation_image (StandardCorrelationImages): Updated array of dimensions (d, K')
              containing the standard correlation images after deletion.
            - spatial_masks (torch.sparse_coo_tensor): Updated sparse tensor of dimensions (d, K')
              containing the spatial masks after deletion.
    """
    print(reasoning_message)
    pos = torch.nonzero(components_to_delete)[:, 0]
    neg = torch.nonzero(components_to_delete == 0)[:, 0]
    if int(torch.sum(components_to_delete).cpu()) == spatial_components.shape[1]:
        raise ValueError("All Components are slated to be deleted")

    pos_for_cpu = pos.cpu().numpy()

    if plot_en:
        corr_values = standard_correlation_image[pos_for_cpu]
        if corr_values.ndim < standard_correlation_image.ndim:
            corr_values = np.expand_dims(corr_values, 0)
        a_used = spatial_components.cpu().to_dense().numpy()
        spatial_comp_plot(
            a_used[:, pos_for_cpu],
            corr_values,
            ini=False,
            order=order,
        )

    standard_correlation_image.c = torch.index_select(
        standard_correlation_image.c, 1, neg
    )
    spatial_masks = torch.index_select(spatial_masks, 1, neg)
    spatial_components = torch.index_select(spatial_components, 1, neg)
    temporal_components = torch.index_select(temporal_components, 1, neg)
    return (
        spatial_components,
        temporal_components,
        standard_correlation_image,
        spatial_masks,
    )


def order_superpixels(c_mat: torch.tensor) -> np.ndarray:
    """
    Finding an ordering of the components based on most prominent activity (ordered in descending order of brightness)

    Args:
        c_mat (torch.tensor): Shape (T, K) where T is number of frames and K number of neurons

    Returns:
        ordering (np.ndarray): Shape (num_components,). Indices indicating what the brightness rank of
            each component is; brightest component gets rank 1, etc.
    """

    c_mat_norm = c_mat / torch.linalg.norm(c_mat, dim=0, keepdim=True)
    max_values = torch.amax(c_mat_norm, dim=0)
    ordering = torch.argsort(max_values, descending=True).cpu().numpy()
    return ordering


def search_superpixel_in_range(
    connect_mat_cropped: np.ndarray, temporal_mat: torch.tensor
) -> Tuple[np.ndarray, torch.tensor]:
    """
    Given a spatial crop of the superpixel matrix, this routine returns the temporal traces associated with
    the superpixels in this spatial region.

    Args:
        connect_mat_cropped (np.ndarray): Shape (crop_dim1, crop_dim2). Matrix indicating the position of each superpixel.
            If a location has value "i", it belongs to the (i-1)-index superpixel.
        temporal_mat (torch.tensor): Shape (T, num_superpixels). Temporal traces for all superpixels over the full field of view (FOV).

    Returns:
        unique_pix (np.ndarray): Array containing the indices of identified superpixels in this spatial patch.
        temporal_trace_subset (torch.tensor): Shape (T, num_found_superpixels). Temporal traces for all superpixels
            found in this spatial subset of the FOV.

    TODO: Eliminate this function and move all ops end to end to pytorch
    """
    unique_pix = np.asarray(np.sort(np.unique(connect_mat_cropped)), dtype="int")
    unique_pix = unique_pix[np.nonzero(unique_pix)]
    unique_pix = torch.from_numpy(unique_pix).long().to(temporal_mat.device)
    temporal_trace_subset = torch.index_select(temporal_mat, 1, unique_pix - 1)

    return unique_pix.cpu().numpy(), temporal_trace_subset


def successive_projection(
    temporal_traces: torch.tensor,
    max_pure_superpixels: int,
    th: float,
    normalize: int = 1,
    device: str = "cpu",
) -> np.ndarray:
    """
    Find pure superpixels via successive projection algorithm.
    Solve nmf problem M = M(:,K)H, K is a subset of M's columns.

    Parameters:
    ----------------
    temporal_traces (torch.tensor): 2d np.arraynumber of timepoints x number of superpixels
        temporal components of superpixels.
    max_pure_superpixels: int scalar
        maximum number of pure superpixels you want to find.  Usually it's set to idx, which is number of superpixels.
    th: double scalar, correlation threshold
        Won't pick up two pure superpixels, which have correlation higher than th.
    normalize: Boolean.
        Normalize L1 norm of each column to 1 if True.  Default is True.
    Return:
    ----------------
    pure_pixels: 1d np.darray, dimension d x 1. (d is number of pure superpixels)
        pure superpixels for these superpixels, actually column indices of M.
    """
    pure_pixels = []
    if normalize == 1:
        temporal_traces /= torch.linalg.norm(
            temporal_traces, dim=0, ord=1, keepdim=True
        )

    squared_norm_curr = torch.sum(temporal_traces**2, dim=0, keepdim=True)
    norm_curr = torch.sqrt(squared_norm_curr)
    squared_norm_orig = squared_norm_curr.clone()
    norm_orig = torch.sqrt(squared_norm_curr)

    found_components = 0
    u = torch.zeros(
        (temporal_traces.shape[0], max_pure_superpixels),
        device=device,
        dtype=torch.float32,
    )
    while (
        found_components < max_pure_superpixels and (norm_curr / norm_orig).max() > th
    ):
        ## select the column of M with largest relative l2-norm
        relative_norms = squared_norm_curr / squared_norm_orig
        pos = torch.where(relative_norms == relative_norms.max())[1][0]
        ## check ties up to 1e-6 precision
        pos_ties = torch.where(
            (relative_norms.max() - relative_norms) / relative_norms.max() <= 1e-6
        )[1]
        if len(pos_ties) > 1:
            pos = pos_ties[
                torch.where(
                    squared_norm_orig[0, pos_ties]
                    == (squared_norm_orig[0, pos_ties]).max()
                )[0][0]
            ]
        ## update the index set, and extracted column
        pure_pixels.append(pos)
        u[:, found_components] = temporal_traces[:, pos].clone()
        u[:, found_components] = u[:, found_components] - u[:, :found_components] @ (
            u[:, :found_components].T @ u[:, found_components]
        )

        u[:, found_components] /= torch.linalg.norm(u[:, found_components])
        squared_norm_curr = torch.maximum(
            torch.tensor([0.0], device=device),
            squared_norm_curr - (u[:, [found_components]].T @ temporal_traces) ** 2,
        )
        norm_curr = torch.sqrt(squared_norm_curr)
        found_components = found_components + 1
    pure_pixels = torch.tensor(pure_pixels, dtype=torch.int64).cpu().detach().numpy()
    return pure_pixels


def get_mean(U, R, V, a=None, X=None):
    """
    Routine for calculating the mean of the movie in question in terms of the V basis
    Inputs:
        U: torch.sparse_coo_tensor. Dimensions (d1*d2, R) where d1, d2 are the FOV dimensions
        R: torch.Tensor. Dimensions (R, R)
        V: torch.Tensor: Dimensions (R, T), where R is the rank of the matrix

    Returns:
        m: torch.Tensor. Shape (d1*d2, 1)
        s: torch.Tensor. Shape (1, R)

        Idea: msV is the "mean movie"
    """

    V_mean = torch.mean(V, dim=1, keepdim=True)
    RV_mean = torch.matmul(R, V_mean)
    m = torch.sparse.mm(U, RV_mean)
    if a is not None and X is not None:
        XV_mean = torch.matmul(X, V_mean)
        aXV_mean = torch.sparse.mm(a, XV_mean)
        m = m - aXV_mean
    s = torch.matmul(V, torch.ones([V.shape[1], 1], device=R.device)).t()
    return m, s


def get_pixel_normalizer(U, R, V, m, s, pseudo, a=None, X=None, batch_size=200):
    """
    Routine for calculating the pixelwise norm of (UR - ms - aX)V. Due to the orthogonality of V this becomes:
        diag (UR - ms - aX)*(UR - ms - aX)

    Inputs:
        U_sparse: torch.sparse_coo_tensor, shape (d1*d2, R)
        R: torch.Tensor, shape (R, R)
        V: torch.Tensor, shape (R, T)
        m: torch.Tensor. shape (d1*d2, 1)
        s: torch.Tensor. Shape (1, R)
        pseudo: float
        a: torch.sparse_coo_tensor. Shape (d1*d2, K)
        X: torch.Tensor. Shape (K, R).
        batch_size: integer. default. 200.
    """

    num_cols = R.shape[1]
    num_iters = int(math.ceil(num_cols / batch_size))

    cumulator = torch.zeros((U.shape[0], 1), device=R.device)
    for k in range(num_iters):
        start = batch_size * k
        end = min(R.shape[1], start + batch_size)
        R_crop = R[:, start:end]
        s_crop = s[:, start:end]

        total = torch.sparse.mm(U, R_crop) - torch.matmul(m, s_crop)
        if a is not None and X is not None:
            X_crop = X[:, start:end]
            total = total - torch.sparse.mm(a, X_crop)

        cumulator = cumulator + torch.sum(total * total, dim=1, keepdim=True)

    cumulator = cumulator + pseudo**2

    cumulator[cumulator == 0] = 1  # Tactic to avoid division by 0
    return torch.sqrt(cumulator)


def construct_index_mat(d1, d2, order="C", device="cpu"):
    """
    Constructs the convolution matrix (but expresses it in 1D)
    """
    flat_indices = torch.arange(d1 * d2, device=device)
    if order == "F":
        col_indices = torch.floor(flat_indices / d1)
        row_indices = flat_indices - col_indices * d1

    elif order == "C":
        row_indices = torch.floor(flat_indices / d2)
        col_indices = flat_indices - row_indices * d2

    else:
        raise ValueError("Invalid order input")

    addends_dim1 = torch.Tensor([-1, -1, -1, 0, 0, 1, 1, 1]).to(device)[None, :]
    addends_dim2 = torch.LongTensor([-1, 0, 1, -1, 1, -1, 0, 1]).to(device)[None, :]

    row_expanded = row_indices[:, None] + addends_dim1
    col_expanded = col_indices[:, None] + addends_dim2

    values = torch.ones_like(row_expanded, device=device)

    good_components = torch.logical_and(row_expanded >= 0, row_expanded < d1)
    good_components = torch.logical_and(good_components, col_expanded >= 0)
    good_components = torch.logical_and(good_components, col_expanded < d2)

    row_expanded *= good_components
    col_expanded *= good_components
    values *= good_components

    if order == "C":
        col_coordinates = d2 * row_expanded + col_expanded
        row_coordinates = torch.arange(d1 * d2, device=device)[:, None] + torch.zeros(
            (1, col_coordinates.shape[1]), device=device
        )

    elif order == "F":
        col_coordinates = d1 * col_expanded + row_expanded
        row_coordinates = torch.arange(d1 * d2, device=device)[:, None] + torch.zeros(
            (1, col_coordinates.shape[1]), device=device
        )

    col_coordinates = torch.flatten(col_coordinates).long()
    row_coordinates = torch.flatten(row_coordinates).long()
    values = torch.flatten(values).bool()

    good_entries = values > 0
    row_coordinates = row_coordinates[good_entries]
    col_coordinates = col_coordinates[good_entries]
    values = values[good_entries]

    return row_coordinates, col_coordinates, values


def compute_correlation(I, U, R, m, s, norm, a=None, X=None, batch_size=200):
    """
    Computes local correlation matrix given pre-computed quantities:
    Inputs:
        I: torch.sparse_coo_tensor, shape (d1*d2, d1*d2). Extremely sparse (<5 elts per row)
        U: torch.sparse_coo_tensor. Shape (d1*d2, R).
        m: torch.Tensor. Shape (d1*d2, 1)
        s: torch.Tensor. Shape (1, R)
        norm: torch.Tensor. Shape (d1*d2,1)
        a: torch.sparse_coo_tensor. Shape (d1*d2, K)
        X: torch.Tensor. Shape (K, R)
        batch_size: number of columns to process at a time. Default: 200 (to avoid issues with large fov data)
    """
    num_cols = R.shape[1]
    num_iters = int(math.ceil(num_cols / batch_size))

    cumulator = torch.zeros((U.shape[0], 1), device=R.device)

    indicator_vector = torch.ones((U.shape[0], 1), device=R.device)
    for k in range(num_iters):
        start = k * batch_size
        end = min(R.shape[1], start + batch_size)
        R_crop = R[:, start:end]
        s_crop = s[:, start:end]

        total = torch.sparse.mm(U, R_crop) - torch.matmul(m, s_crop)
        if a is not None and X is not None:
            X_crop = X[:, start:end]
            total = total - torch.sparse.mm(a, X_crop)

        total = total / norm

        I_total = torch.sparse.mm(I, total)

        cumulator = cumulator + torch.sum(I_total * total, dim=1, keepdim=True)

    final_I_sum = torch.sparse.mm(I, indicator_vector)
    final_I_sum[final_I_sum == 0] = 1
    return cumulator / final_I_sum


def pure_superpixel_corr_compare_plot(
    connect_mat_1: np.ndarray,
    unique_pix: np.ndarray,
    pure_pix: np.ndarray,
    brightness_rank_sup: np.ndarray,
    brightness_rank: np.ndarray,
    mad_correlation_img: np.ndarray,
    text: bool = False,
    order: str = "C",
) -> tuple[Figure, np.ndarray]:
    """
    General plotting diagnostic for superpixels
    Args:
        connect_mat_1 (np.ndarray): The (d1, d2) shaped superpixel matrix
        unique_pix (np.ndarray): The (N,) shaped array describing the values of the superpixels in the superpix mat
        pure_pix (np.ndarray): The (N,) shaped array describing the values of the pure superpixels

    """

    scale = np.maximum(1, (connect_mat_1.shape[1] / connect_mat_1.shape[0]))
    fig = plt.figure(figsize=(4 * scale, 12))
    ax = plt.subplot(3, 1, 1)

    random_seed = 2
    np.random.seed(random_seed)
    connect_mat_1 = connect_mat_1.astype("int")
    permutation_matrix = np.random.permutation(np.arange(1, len(unique_pix) + 1))
    permutation_matrix = np.concatenate([np.array([0]), permutation_matrix])
    permuted_connect_mat = permutation_matrix[connect_mat_1.flatten()].reshape(
        connect_mat_1.shape
    )
    ax.imshow(permuted_connect_mat, cmap="nipy_spectral_r")

    if text:
        for ii in range(len(unique_pix)):
            pos = np.where(
                permuted_connect_mat[:, :] == permutation_matrix[unique_pix[ii]]
            )
            pos0 = pos[0]
            pos1 = pos[1]
            ax.text(
                (pos1)[np.array(len(pos1) / 3, dtype=int)],
                (pos0)[np.array(len(pos0) / 3, dtype=int)],
                f"{brightness_rank_sup[ii] + 1}",
                verticalalignment="bottom",
                horizontalalignment="right",
                color="black",
                fontsize=15,
            )  # , fontweight="bold")
    ax.set(title="Superpixels")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")

    ax1 = plt.subplot(3, 1, 2)
    dims = connect_mat_1.shape
    connect_mat_1_pure = connect_mat_1.copy()
    connect_mat_1_pure = connect_mat_1_pure.reshape(np.prod(dims), order=order)
    connect_mat_1_pure[~np.in1d(connect_mat_1_pure, pure_pix)] = 0
    connect_mat_1_pure = connect_mat_1_pure.reshape(dims, order=order)

    permuted_connect_mat_1_pure = permutation_matrix[
        connect_mat_1_pure.flatten()
    ].reshape(connect_mat_1_pure.shape)
    ax1.imshow(permuted_connect_mat_1_pure, cmap="nipy_spectral_r")

    if text:
        for ii in range(len(pure_pix)):
            pos = np.where(
                permuted_connect_mat_1_pure == permutation_matrix[pure_pix[ii]]
            )
            pos0 = pos[0]
            pos1 = pos[1]
            ax1.text(
                (pos1)[np.array(len(pos1) / 3, dtype=int)],
                (pos0)[np.array(len(pos0) / 3, dtype=int)],
                f"{brightness_rank[ii] + 1}",
                verticalalignment="bottom",
                horizontalalignment="right",
                color="black",
                fontsize=15,
            )  # , fontweight="bold")
    ax1.set(title="Pure superpixels")
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold")

    ax2 = plt.subplot(3, 1, 3)
    show_img(ax2, mad_correlation_img)
    ax2.set(title="Thresholded Corr Img")
    ax2.title.set_fontsize(15)
    ax2.title.set_fontweight("bold")
    plt.tight_layout()
    plt.show()
    return fig, connect_mat_1_pure


def show_img(ax, img, vmin=None, vmax=None):
    # Visualize local correlation, adapt from kelly's code
    im = ax.imshow(img, cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    if np.abs(img.min()) < 1:
        format_tile = "%.2f"
    else:
        format_tile = "%5d"
    plt.colorbar(im, cax=cax, orientation="vertical", spacing="uniform")


def local_mad_correlation_mat(
    dim1_coordinates: torch.tensor,
    dim2_coordinates: torch.tensor,
    correlations: torch.tensor,
    dims: tuple[int, int, Optional[int]],
    order: str = "C",
) -> np.ndarray:
    """
    We MAD-threshold each pixel and compute correlations between neighboring pixels in the superpixel step
    This routine is compute and memory optimized to manipulate these (on CPU and on GPU)
    and produce a single correlation heatmap. Each pixel's intensity is the average of its correlation with neighboring
    pixels.

    Args:
        dim1_coordinates (torch.tensor): Shape (N). Pixel coordinates in the field of view
        dim2_coordinates (torch.tensor): Shape (N). Pixel coordinates in the field of view
        correlations (torch.tensor): Shape (N). Correlation values for dim1_coordinates[i], dim2_coordinates[i]
        dims (tuple): Integer specifying shape of data (field of view dim1, field of view dim2, and maybe frames).
        order (str): either "F" or "C" indicating how to reshape flattened data

    Returns:
        correlation_image (np.ndarray): Shape (d1, d2).
    """
    coordinate_pairs = torch.stack((dim1_coordinates, dim2_coordinates), dim=1)
    sorted_coordinates, _ = torch.sort(coordinate_pairs, dim=1)

    num_pixels_fov = np.prod(dims[:2])

    multiplicity_tracker = 100
    correlations = (
        correlations + multiplicity_tracker
    )  # Now every correlation is between 99 and 101

    correlations_mat = torch.sparse_coo_tensor(
        sorted_coordinates.T, correlations, (num_pixels_fov, num_pixels_fov)
    ).coalesce()

    rows, cols = correlations_mat.indices()
    correlation_sums = correlations_mat.values()
    multiplicity = torch.round(correlation_sums / multiplicity_tracker)
    correlation_sums = (
        correlation_sums - multiplicity_tracker * multiplicity
    ) / multiplicity

    # Now repeat this algorithm again
    correlation_sums = correlation_sums + multiplicity_tracker

    final_rows = torch.concatenate([rows, cols], dim=0)
    final_cols = torch.zeros_like(final_rows, device=final_rows.device)
    final_values = torch.concatenate([correlation_sums, correlation_sums])

    correlations_mat = torch.sparse_coo_tensor(
        torch.stack([final_rows, final_cols]), final_values, (num_pixels_fov, 1)
    ).coalesce()

    rows, _ = correlations_mat.indices()
    final_values = correlations_mat.values()
    multiplicity = torch.round(final_values / multiplicity_tracker)
    final_values = (final_values - multiplicity_tracker * multiplicity) / multiplicity

    dense_correlation_mat = torch.zeros(num_pixels_fov, device=rows.device)
    dense_correlation_mat[rows] = final_values

    return dense_correlation_mat.cpu().numpy().reshape((dims[0], dims[1]), order=order)


def local_correlation_mat(
    u: torch.sparse_coo_tensor,
    r: torch.tensor,
    v: torch.tensor,
    dims: tuple[int, int, Optional[int]],
    pseudo: float,
    a: Optional[torch.sparse_coo_tensor] = None,
    c: Optional[torch.tensor] = None,
    order: str = "C",
    batch_size: int = 200,
):
    """
    This function uses the PMD representation to compute a "correlation" heatmap.
    Pixels which are highly correlated with their neighbors are brighter, etc.

    Args:
        u (torch.sparse_coo_tensor): Shape (d1*d2, R)
        r (torch.tensor): Shape (R, R)
        v (torch.tensor): Shape (R, T)
        dims (tuple): Integer specifying shape of data (field of view dim1, field of view dim2, and maybe frames).
        pseudo (float): value typically near 0.1. Noise variance parameter used for correlation image calculation
        a (torch.sparse_coo_tensor): Shape (d1*d2, K), where K is # of neural signals
        c (torch.tensor): Shape (T, K).
        order (str): either "F" or "C" indicating how to reshape flattened data
        batch_size (int): Integer specifying batch size for pixelwise norm calcuations.

    Returns:
        correlation_image (np.ndarray): Shape (d1, d2).
    """
    if a is not None and c is not None:
        x = torch.matmul(
            v, c
        ).t()  # Equivalently a linear subspace projection of c onto V...
    else:
        x = None

    m, s = get_mean(u, r, v, a=a, X=x)
    norm = get_pixel_normalizer(u, r, v, m, s, pseudo, a=a, X=x, batch_size=batch_size)

    (rows, columns, values) = construct_index_mat(
        dims[0], dims[1], order=order, device=r.device
    )

    sparse_convolution_matrix = torch.sparse_coo_tensor(
        torch.stack([rows, columns]),
        values.float(),
        (dims[0] * dims[1], dims[0] * dims[1]),
    )

    return (
        compute_correlation(
            sparse_convolution_matrix, u, r, m, s, norm, a=a, X=x, batch_size=batch_size
        )
        .cpu()
        .numpy()
        .reshape((dims[0], dims[1], -1), order=order)
    )


def single_pixel_correlation_image(
    row_index, U_sparse, R, V, a=None, c=None, batch_size=100
):
    device = R.device
    if a is not None and c is not None:
        X = torch.matmul(
            V, c
        ).t()  # Equivalently a linear subspace projection of c onto V...
    else:
        row = torch.Tensor([]).to(device).long()
        col = torch.Tensor([]).to(device).long()
        value = torch.Tensor([]).to(device).bool()
        a = torch.sparse_coo_tensor(
            torch.stack([row, col]), value, (U_sparse.shape[0], 1)
        )
        # a = torch_sparse.tensor.SparseTensor(row=row, col=col, value=value,
        #                                      sparse_sizes=(U_sparse.sparse_sizes()[0], 1))
        X = torch.zeros(1, V.shape[0], device=device)

    m, s = get_mean(U_sparse, R, V, a=a, X=X)
    norm = get_pixel_normalizer(
        U_sparse, R, V, m, s, 0, a=a, X=X, batch_size=batch_size
    )

    # Finally, get the centered row of interest:
    row_index_tensor = torch.Tensor([row_index]).to(device).long()
    U_sparse_row = torch.index_select(U_sparse, 0, row_index_tensor)
    a_sparse_row = torch.index_select(a, 0, row_index_tensor)

    centered_pixel = (
        torch.sparse.mm(U_sparse_row, R)
        - torch.sparse.mm(a_sparse_row, X)
        - torch.matmul(m[[row_index], :], s)
    )
    centered_pixel = centered_pixel.t()

    Rprod = torch.matmul(R, centered_pixel)
    URprod = torch.sparse.mm(U_sparse, Rprod)

    Xprod = torch.matmul(X, centered_pixel)
    aXprod = torch.sparse.mm(a, Xprod)

    sprod = torch.matmul(s, centered_pixel)
    msprod = torch.matmul(m, sprod)

    final_result = URprod - aXprod - msprod

    final_normalizer = norm * norm[row_index]
    final_normalizer[final_normalizer == 0] = 1
    return final_result / final_normalizer


def prepare_iteration_uv(
    pure_pix: np.ndarray, a_mat: torch.sparse_coo_tensor, c_mat: torch.tensor
) -> Tuple[torch.sparse_coo_tensor, torch.tensor, np.ndarray]:
    """
    Extract pure superpixels and order the components by brightness

    Args:
        pure_pix (numpy.ndarray): Shape (number_of_pure_superpixels,). A value of "i" indicates the superpixel at index
            i - 1 is a pure superpixel
        a_mat (torch.sparse_coo_tensro): Shape (d1*d2, K) where K is the total number of superpixels
        c_mat (torch.tensor): Shape (T, K) where T is the number of frames.

    Returns:
        a_mat_pure (torch.sparse_coo_tensor): The brightness-ordered spatial matrix containing only pure superpixels
        c_mat_pure (torch.tensor): The brightness ordered temporal matrix containing only pure superpixels
        brightness_rank (np.ndarray): Shape (number of pure superpixels,). The brightness ranks involved in reordering
    """

    # Extract the pure superpixels
    pure_pix_indices = pure_pix - np.array([1]).astype("int")
    pure_pix_indices = torch.from_numpy(pure_pix_indices).long().to(a_mat.device)
    a_mat = torch.index_select(a_mat, 1, pure_pix_indices).coalesce()
    c_mat = torch.index_select(c_mat, 1, pure_pix_indices)

    c_mat_norm = c_mat / torch.linalg.norm(c_mat, dim=0, keepdim=True)
    max_values = torch.amax(c_mat_norm, dim=0)
    ordering = torch.argsort(max_values, descending=True)

    a_mat = torch.index_select(a_mat, 1, ordering).coalesce()
    c_mat = torch.index_select(c_mat, 1, ordering)
    return a_mat, c_mat, ordering.cpu().numpy()


def fit_large_spatial_support(
    comp, c_init, U_sparse_torch, V_torch, th, a_sparse=None, c=None, batch_size=500
):
    """
    Routine for estimating
    """
    print("Fitting larger spatial support")
    comp = list(comp)
    num_iters = math.ceil(len(comp) / batch_size)
    final_values = torch.zeros(0, device=V_torch.device)

    for k in range(num_iters):
        start_pt = batch_size * k
        end_pt = min(len(comp), batch_size * (k + 1))
        components = comp[start_pt:end_pt]
        comp_tensor = torch.LongTensor(components).to(V_torch.device)
        U_subset = torch.index_select(U_sparse_torch, 0, comp_tensor)
        y_temp = torch.sparse.mm(U_subset, V_torch)

        if a_sparse is not None and c is not None:
            a_subset = torch.index_select(a_sparse, 0, comp_tensor)
            ac_prod = torch.sparse.mm(a_subset, c)
            y_temp = torch.sub(y_temp, ac_prod)

        y_temp = threshold_data_inplace(y_temp, th, dim=1)

        normalizer = torch.sum(c_init * c_init)
        elt_product = torch.sum(c_init[None, :] * y_temp, dim=1)

        curr_values = elt_product / normalizer
        threshold_function = torch.nn.ReLU()
        curr_values_thr = threshold_function(curr_values)

        final_values = torch.cat(
            (final_values, curr_values_thr.type(final_values.dtype)), dim=0
        )

    return final_values


def superpixel_init(
    u_sparse: torch.sparse_coo_tensor,
    r: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    patch_size: Tuple[int, int],
    data_order: str,
    dims: Tuple[int, int, int],
    cut_off_point: float,
    residual_cut: float,
    length_cut: int,
    device: str,
    dim1_coordinates: torch.Tensor,
    dim2_coordinates: torch.Tensor,
    correlations: torch.Tensor,
    text: bool = True,
    plot_en: bool = False,
    a: Optional[torch.sparse_coo_tensor] = None,
    c: Optional[torch.tensor] = None,
) -> Tuple[
    torch.sparse_coo_tensor,
    Optional[torch.sparse_coo_tensor],
    torch.Tensor,
    torch.Tensor,
    Dict[str, torch.Tensor],
    np.ndarray,
]:
    """
    Args:
        u_sparse (torch.sparse_coo_tensor): Shape (d1*d2, R)
        r (torch.Tensor): Shape (R1, R2). Mixing matrix in the PMD decomposition
        s (torch.Tensor): Shape (R2,). Singular values of PMD decomposition.
        v (torch.Tensor): dims (R2, T). PMD temporal basis.
        patch_size (tuple): Patch size that we use to partition the FOV when computing pure superpixels
        data_order (str): "F" or "C" depending on how the field of view "collapsed" into 1D vectors
        dims (tuple): containing (d1, d2, T), the dimensions of the data
        cut_off_point (float): between 0 and 1. Correlation thresholds used in superpixel calculations
        residual_cut (float): between 0 and 1. Threshold used in successive projection to find pure superpixels
        length_cut (int): Minimum allowed sizes of superpixels
        device (string): string used by pytorch to move and construct objects on cpu or gpu
        dim1_coordinates (torch.tensor): shape number_correlations
        dim2_coordinates (torch.tensor): shape number_correlations
        correlations (torch.tensor):
        text (bool): Whether or not to overlay text onto correlation plots (when plotting is enabled)
        plot_en (bool) : Whether or not plotting is enabled (for diagnostic purposes)
        a (torch.sparse_coo_tensor): shape (d1*d2, K) where K is the number of neurons
        c (torch.tensor): shape (T, K) where T is the number of time points, K is number of neurons

    Returns:
        a (torch.sparse_coo_tensor): Shape (d1*d2, K) where d1, d2 are the FOV dimensions and K is the number of signals identified
        mask_ab (torch.sparse_coo_tensor): None or torch.sparse_coo_tensor of shape same as "a"
        c (torch.tensor): Temporal data, shape (T,  K)
        b (torch.Tensor): Pixelwise baseline estimate, shape(d1*d2)
        superpixel_dictionary (dict): Dictionary of key superpixel matrices for this round of initialization
        superpixel_img (np.ndarray): Shape (d1, d2): Plotted superpixel image

    TODO: Make the second pass "a" also a sparse tensor.
    """

    if a is None and c is None:
        first_init_flag = True
    elif a is not None and c is not None:
        first_init_flag = False
    else:
        raise ValueError("Invalid configuration of c and a values were provided")

    print("find superpixels!")
    connect_mat_1, comps = find_superpixel_UV(
        dims,
        cut_off_point,
        length_cut,
        dim1_coordinates,
        dim2_coordinates,
        correlations,
        data_order,
    )

    c_ini, a_ini = spatial_temporal_ini_UV(
        u_sparse, r, s, v, dims, comps, length_cut, a=a, c=c
    )

    print("find pure superpixels!")
    ## cut image into small parts to find pure superpixels ##
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    height_num = int(np.ceil(dims[0] / patch_height))
    width_num = int(np.ceil(dims[1] / patch_width))
    num_patch = height_num * width_num
    patch_ref_mat = np.array(range(num_patch)).reshape(
        height_num, width_num, order=data_order
    )

    unique_pix = np.asarray(np.sort(np.unique(connect_mat_1)), dtype="int")
    unique_pix = unique_pix[np.nonzero(unique_pix)]
    brightness_rank_sup = order_superpixels(c_ini)
    pure_pix = []

    connect_mat_2d = connect_mat_1.reshape(dims[0], dims[1], order=data_order)
    for kk in range(num_patch):
        pos = np.where(patch_ref_mat == kk)
        up = pos[0][0] * patch_height
        down = min(up + patch_height, dims[0])
        left = pos[1][0] * patch_width
        right = min(left + patch_width, dims[1])
        unique_pix_temp, m = search_superpixel_in_range(
            connect_mat_2d[up:down, left:right],
            c_ini,
        )
        pure_pix_temp = successive_projection(
            m, m.shape[1], residual_cut, device=device
        )
        if len(pure_pix_temp) > 0:
            pure_pix.append(unique_pix_temp[pure_pix_temp])
    pure_pix = np.hstack(pure_pix)
    pure_pix = np.unique(pure_pix)

    print("prepare iteration!")
    if not first_init_flag:
        a_newpass, c_newpass, brightness_rank = prepare_iteration_uv(
            pure_pix,
            a_ini,
            c_ini,
        )

        ## Boilerplate for concatenating two sparse tensors along dim 1:
        a_dims = (a.shape[0], a.shape[1] + a_newpass.shape[1])
        a_row, a_col = a.indices()
        a_vals = a.values()
        a_new_row, a_new_col = a_newpass.indices()
        a_new_vals = a_newpass.values()

        new_rows = torch.concatenate([a_row, a_new_row])
        new_col = torch.concatenate([a_col, a_new_col + a.shape[1]])
        new_vals = torch.concatenate([a_vals, a_new_vals])
        a = torch.sparse_coo_tensor(torch.stack([new_rows, new_col]), new_vals, a_dims)
        c = torch.concatenate([c, c_newpass], dim=1)
        uv_mean = get_mean_data(u_sparse, r, s, v)
        b = regression_update.baseline_update(uv_mean, a, c)
    else:
        a, c, brightness_rank = prepare_iteration_uv(
            pure_pix,
            a_ini,
            c_ini,
        )
        uv_mean = get_mean_data(u_sparse, r, s, v)
        b = regression_update.baseline_update(uv_mean, a, c)

    assert a.shape[1] > 0, (
        "Superpixels did not identify any components, re-run "
        "with different parameters before proceeding"
    )

    # Plot superpixel correlation image
    if plot_en:
        mad_correlation_img = local_mad_correlation_mat(
            dim1_coordinates, dim2_coordinates, correlations, dims, data_order
        )
        _, superpixel_img = pure_superpixel_corr_compare_plot(
            connect_mat_1,
            unique_pix,
            pure_pix,
            brightness_rank_sup,
            brightness_rank,
            mad_correlation_img,
            text,
            order=data_order,
        )
    else:
        superpixel_img = None

    superpixel_dict = {
        "connect_mat_1": connect_mat_1,
        "pure_pix": pure_pix,
        "unique_pix": unique_pix,
        "brightness_rank": brightness_rank,
        "brightness_rank_sup": brightness_rank_sup,
    }

    return a, a.bool(), c, b, superpixel_dict, superpixel_img


def merge_components(
    a: torch.sparse_coo_tensor,
    c: torch.tensor,
    standard_correlation_image: StandardCorrelationImages,
    merge_corr_thr=0.6,
    merge_overlap_thr=0.6,
    plot_en=False,
    data_order="C",
) -> tuple[
    torch.sparse_coo_tensor,
    torch.tensor,
    torch.sparse_coo_tensor,
    StandardCorrelationImages,
]:
    """
    We want to merge components whose correlation images are highly overlapped,
    and update a and c after merge with region constraint

    Args:
        a: torch.sparse_coo_tensor
             sparse matrix describing the spatial supports of all signals. Shape (d, K) where d is the number of pixels in the movie and K is the number of neural signals
        c: torch.Tensor
             torch Tensor describing the temporal profiles of all signals. Shape (T, K), where T is the number of frames in the movie
        standard_correlation_image (StandardCorrelationImages): The object which stores the standard correlation images as
            a (number of neurons, fov dim 1, fov fim 2) array-like object.
        merge_corr_thr (float): scalar between 0 and 1
            temporal correlation threshold for truncating corr image (corr(Y,c)) (default 0.6)
        merge_overlap_thr (float): :scalar between 0 and 1
            overlap ratio threshold for two corr images (default 0.6)
        plot_en (bool) Whether or not to plot the results. This is useful for development, not production (TODO: Check what things need to be moved to CPU for this)
        data_order: string. Either "C" or "F".
    Returns:
        a (torch.sparse_coo_tensor). Shape (pixels, number of signals). New spatial components for demixing
        c (torch.tensor): Shape (frames, number of signals). New temporal components for demixing
        a_bool (torch.sparse_coo_tensor): The updated masks of the spatial components
        standard_correlation_image (StandardCorrelationImages): Updated correlation images
    """
    device = c.device
    num_corr_signals = standard_correlation_image.shape[0]
    standard_correlation_image_full = standard_correlation_image.getitem_tensor(
        slice(0, num_corr_signals, 1)
    )
    ############ calculate overlap area ###########

    a_corr = torch.sparse.mm(a.t(), a).to_dense()
    a_corr = torch.triu(a_corr, diagonal=1)
    cor = ((standard_correlation_image_full > merge_corr_thr) * 1).float()
    temp = torch.sum(cor, dim=[1, 2])
    temp[temp == 0] = 1  # For division safety
    cor_corr = torch.tensordot(
        cor,
        cor,
        dims=([1, 2], [1, 2]),
    )
    cor_corr = torch.triu(cor_corr, diagonal=1)

    # Test to see for each pair of neurons (a, b) whether overlap(a, b) / support_size(corr_img(a)) > merge_overlap_thres
    condition1 = (cor_corr / temp.unsqueeze(1)) > merge_overlap_thr

    # Test to see for each pair of neurons (a, b) whether overlap(a, b) / support_size(corr_img(b)) > merge_overlap_thres
    condition2 = (cor_corr / temp.unsqueeze(0)) > merge_overlap_thr

    # Test to make sure the two cells actually overlap
    condition3 = a_corr > 0
    cri = condition1 * condition2 * condition3

    connect_comps = torch.argwhere(cri)

    if torch.numel(connect_comps) > 0:
        merge_graph = nx.Graph()
        merge_graph.add_edges_from(
            list(
                zip(
                    connect_comps[:, 0].cpu().numpy(), connect_comps[:, 1].cpu().numpy()
                )
            )
        )
        comps = list(nx.connected_components(merge_graph))
        remove_indices = torch.unique(torch.flatten(connect_comps))
        all_indices = torch.ones([c.shape[1]], device=device)
        all_indices[remove_indices] = 0
        all_indices = all_indices.bool()

        indices_arange = torch.arange(a.shape[1], device=device)[all_indices]

        a_preserved = torch.index_select(a, 1, indices_arange).coalesce()
        c_preserved = torch.index_select(c, 1, indices_arange)

        c_append_list = [c_preserved]

        row_indices = [a_preserved.indices()[0, :]]
        col_indices = [a_preserved.indices()[1, :]]
        values_indices = [a_preserved.values()]
        num_preserved_comps = a_preserved.shape[1]
        added_counter = 0
        for comp in comps:
            print(f"merging {comp}")
            comp = list(comp)
            good_comps = torch.Tensor(comp).to(device).long()

            a_merge = torch.index_select(a, 1, good_comps).coalesce()
            c_merge = torch.index_select(c, 1, good_comps)

            a_rank1, c_rank1 = rank_1_NMF_fit(a_merge, c_merge)

            if plot_en:
                spatial_comp_plot(
                    a_merge.cpu().to_dense().numpy(),
                    standard_correlation_image_full[comp].cpu().numpy(),
                    ini=False,
                    order=data_order,
                )

            nonzero_indices = torch.nonzero(a_rank1)
            row_temp = nonzero_indices[:, 0]
            col_temp = nonzero_indices[:, 1]

            nonzero_values = a_rank1[row_temp, col_temp]

            row_indices.append(row_temp)
            col_indices.append(col_temp + num_preserved_comps + added_counter)
            added_counter += 1
            values_indices.append(nonzero_values)

            c_append_list.append(c_rank1)

        row_indices_net = torch.cat(row_indices, dim=0)
        col_indices_net = torch.cat(col_indices, dim=0)
        value_indices_net = torch.cat(values_indices, dim=0)
        c = torch.cat(c_append_list, dim=1)
        a = torch.sparse_coo_tensor(
            torch.stack([row_indices_net, col_indices_net]),
            value_indices_net,
            (a.shape[0], c.shape[1]),
        ).coalesce()

        standard_correlation_image.c = c
    return a, c, a.bool(), standard_correlation_image


def rank_1_NMF_fit(a_merge, c_merge):
    """
    Fast HALS_based routine to perform a rank-1 NMF fit, constrained by the support of a_merge
    Inputs:
        a_merge: torch.sparse_coo_tensor. Shape (d, K), where d is the number of pixels and K is the number of neural
            signals to be merged
        c_merge: torch.Tensor. Shape (T, K), where T is the number of frames in the movie

    Returns:
        spatial_component: torch.Tensor. Shape (d, 1)
        temporal_component: Torch.Tensor. Shape (T, 1). These two tensors, when multiplied like so:
                    torch.matmul(spatial_component, temporal_component.t()), give the rank-1 constrained NMF
                    approximation to the movie given by torch.matmul(a_merge, c_merge.t())
    """
    device = c_merge.device

    # Step 1: Figure out how to initialize the first and second components of the rank-1 factorization.
    # We init the first component to the mean:
    summand = torch.ones([a_merge.shape[1], 1], device=device)
    summand /= a_merge.shape[1]
    spatial_component = torch.sparse.mm(a_merge, summand)
    mask = spatial_component > 0

    temporal_component = torch.zeros([c_merge.shape[0], 1], device=device)

    my_relu_obj = torch.nn.ReLU()

    num_iters = 5

    for k in range(num_iters):
        temporal_component = my_relu_obj(
            _temporal_fit_routine(a_merge, c_merge, spatial_component)
        )
        spatial_component = my_relu_obj(
            _spatial_fit_routine(a_merge, c_merge, temporal_component, mask)
        )

    return spatial_component, temporal_component


def _spatial_fit_routine(a_merge, c_merge, temporal_component, mask):
    """
    Fits a spatial component in the rank-1 nonnegative merging fit via standard least squares
    Inputs:
        a_merge: torch.sparse_coo_tensor of shape (d, K) where K is the number of signals slated to be merged
        c_merge: torch.Tensor of shape (T, K) where T is the number of frames
        temporal_component: torch.Tensor of shape (T, 1)
        mask: torch.Tensor of shape (d, 1). Has values 1 where the support is defined and 0 elsewhere
    Output:
        spatial_component: torch.Tensor of shape (d, 1)
    """

    temporal_dot_product = torch.matmul(temporal_component.t(), temporal_component)
    temporal_dot_product[temporal_dot_product == 0] = 1  # Avoid division by zero issues

    merge_dots = torch.matmul(c_merge.t(), temporal_component)
    row_dots = torch.sparse.mm(a_merge, merge_dots)

    least_squares_fits = row_dots / temporal_dot_product
    return least_squares_fits * mask  # Set other elts outside of support to 0


def _temporal_fit_routine(a_merge, c_merge, spatial_component):
    """
    Fits a nonnegative temporal component in the rank-1 nonnegative merging fit
    Inputs:
        a_merge: torch.sparse_coo_tensor of shape (d, K) where K is the number of signals slated to be merged
        c_merge: torch.Tensor of shape (T, K) where T is the number of frames
        spatial_component: torch.Tensor of shape (d, 1)
    Output:
        temporal_component: torch.Tensor of shape (T, 1)
    """

    aA = torch.sparse.mm(a_merge.t(), spatial_component).t()
    aAC = torch.matmul(aA, c_merge.t())

    spatial_norm = torch.matmul(spatial_component.t(), spatial_component)
    spatial_norm[spatial_norm == 0] = 1

    least_squares_fits = aAC / spatial_norm

    return least_squares_fits.T


def spatial_comp_plot(
    a: np.ndarray,
    standard_correlation_image: np.ndarray,
    ini: bool = False,
    order: str = "C",
):
    print("DISPLAYING SOME OF THE COMPONENTS")
    max_neurons = 5
    num = min(max_neurons, a.shape[1])
    patch_size = standard_correlation_image.shape[1:]
    scale = np.maximum(
        1, (standard_correlation_image.shape[2] / standard_correlation_image.shape[1])
    )
    fig = plt.figure(figsize=(8 * scale, 4 * num))
    neuron_numbering = np.arange(num)
    for ii in range(num):
        plt.subplot(num, 2, 2 * ii + 1)
        plt.imshow(a[:, ii].reshape(patch_size, order=order), cmap="nipy_spectral_r")
        plt.ylabel(str(neuron_numbering[ii] + 1), fontsize=15, fontweight="bold")
        if ii == 0:
            if ini:
                plt.title("Spatial components ini", fontweight="bold", fontsize=15)
            else:
                plt.title("Spatial components", fontweight="bold", fontsize=15)
        ax1 = plt.subplot(num, 2, 2 * (ii + 1))
        show_img(ax1, standard_correlation_image[ii, :, :])
        if ii == 0:
            ax1.set(title="corr image")
            ax1.title.set_fontsize(15)
            ax1.title.set_fontweight("bold")
    plt.tight_layout()
    plt.show()
    return fig


class SignalProcessingState(ABC):
    def __init__(self, pixel_batch_size: int, frame_batch_size: int):
        """Constructor to initialize pixel_batch_size and frame_batch_size."""
        if not isinstance(pixel_batch_size, int) or pixel_batch_size <= 0:
            raise ValueError("pixel_batch_size must be a positive integer.")
        if not isinstance(frame_batch_size, int) or frame_batch_size <= 0:
            raise ValueError("frame_batch_size must be a positive integer.")

        self._pixel_batch_size = pixel_batch_size
        self._frame_batch_size = frame_batch_size

    @property
    def pixel_batch_size(self):
        """Get the pixel batch size."""
        return self._pixel_batch_size

    @pixel_batch_size.setter
    def pixel_batch_size(self, value):
        """Set the pixel batch size."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("pixel_batch_size must be a positive integer.")
        self._pixel_batch_size = value

    @property
    def frame_batch_size(self):
        """Get the frame batch size."""
        return self._frame_batch_size

    @frame_batch_size.setter
    def frame_batch_size(self, value):
        """Set the frame batch size."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("frame_batch_size must be a positive integer.")
        self._frame_batch_size = value

    def initialize_signals(self, **kwargs):
        """Initialize signals based on provided parameters."""
        raise NotImplementedError(
            "This method is not implemented for the current state."
        )

    @property
    def state_description(self):
        """Return a description of the current state."""
        raise NotImplementedError("This is not implemented for the current state.")

    def demix(self, **kwargs):
        """Perform the demixing process based on provided parameters."""
        raise NotImplementedError(
            "This method is not implemented for the current state."
        )

    @property
    def results(self):
        """Return the results from any given state."""
        raise NotImplementedError("This is not implemented for the current state.")

    def lock_results_and_continue(self, context, carry_background: bool):
        """Lock in the current results and transition context object to new state."""
        raise NotImplementedError(
            "This method is not implemented for the current state."
        )


class SignalDemixer:

    def __init__(
        self,
        u_sparse: scipy.sparse.coo_matrix,
        r: np.ndarray,
        s: np.ndarray,
        v: np.ndarray,
        dimensions: tuple[int, int, int],
        data_order: str = "F",
        device: str = "cpu",
        frame_batch_size: int = 5000,
        pixel_batch_size: int = 10000,
    ):
        """
        A class to manage the state and execution of the maskNMF demixing pipeline

        Provides methods to run, update, and manage the iterative process of signal demixing from imaging data.
        It allows for the addition of new signals, tracks the current state of the demixing process,
        and provides access to the unmixed signals. The class is designed to
        facilitate interactive usage, enabling users to iteratively refine the demixing results.

        Args:
            u_sparse (scipy.sparse.coo_matrix): Shape (pixels, PMD Rank1).
            r (numpy.ndarray): Shape (PMD rank1, PMD rank 2). Together, (u_sparse)(r) give left singular vectors for pmd.
            s (numpy.ndarray): Shape (PMD rank 2). Singular values for the pmd decomposition
            v (numpy.ndarray): Shape (pmd rank2, frames). Orthogonal temporal basis vectors for PMD rank.
            dimensions (tuple): (frames, fov dimension 1, fov dimension 2).
            data_order (str): The order in which n-dimensional vectors are flattened to 1D
            device (str): Indicator for pytorch for which device to use ("cpu" or "cuda")
            frame_batch_size (int): Number of full frames of data we load onto the GPU at a time
            pixel_batch_size (int): Number of full pixels of data we load onto the GPU at a time
        """
        self.device = device
        self.data_order = data_order
        self.shape = dimensions
        self.r = torch.Tensor(r).float().to(self.device)
        self.s = torch.from_numpy(s).float().to(self.device)
        self.u_sparse = (
            torch.sparse_coo_tensor(
                np.array(u_sparse.nonzero()), u_sparse.data, u_sparse.shape
            )
            .coalesce()
            .float()
            .to(self.device)
        )
        self.v = torch.Tensor(v).float().to(self.device)
        self.d1 = dimensions[0]
        self.d2 = dimensions[1]
        self.T = dimensions[2]

        self.u_sparse, self.r, self.s, self.v = PMD_setup_routine(
            self.u_sparse, self.r, self.s, self.v
        )

        self._num_transitions = 0

        # Start with an initialization state
        self._state = InitializingState(
            self.u_sparse,
            self.r,
            self.s,
            self.v,
            (self.d1, self.d2, self.T),
            data_order=self.data_order,
            device=self.device,
            a=None,
            c=None,
            frame_batch_size=frame_batch_size,
            pixel_batch_size=pixel_batch_size,
        )

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state: SignalProcessingState):
        self._state = new_state

    @property
    def results(self):
        return self.state.results

    def initialize_signals(self, **kwargs):
        return self._state.initialize_signals(**kwargs)

    def demix(self, **kwargs):
        self._state.demix(**kwargs)

    def lock_results_and_continue(self, carry_background: bool = False):
        """
        The state initiates the transition to a new state, updating this object via its state setter
        """
        self._state.lock_results_and_continue(self, carry_background = carry_background)


class InitializingState(SignalProcessingState):

    def __init__(
        self,
        u_sparse: torch.sparse_coo_tensor,
        r: torch.tensor,
        s: torch.tensor,
        v: torch.tensor,
        dimensions: tuple[int, int, int],
        data_order: str = "F",
        device: str = "cpu",
        a: Optional[torch.sparse_coo_tensor] = None,
        c: Optional[torch.tensor] = None,
        pixel_batch_size: int = 40000,
        frame_batch_size: int = 2000,
        factorized_ring_term: Optional[torch.tensor] = None,
    ):
        super().__init__(pixel_batch_size, frame_batch_size)
        """
        Class for initializing the signals
        """
        self.d1, self.d2, self.T = dimensions
        self.shape = (self.d1, self.d2, self.T)
        self.data_order = data_order
        self.device = device

        self.u_sparse = u_sparse.to(self.device)
        self.r = r.to(self.device)
        self.s = s.to(self.device)
        self.v = v.to(self.device)
        self.factorized_ring_term = factorized_ring_term

        self.a_init = None
        self.mask_a_init = None
        self.c_init = None
        self.b_init = None
        self.diagnostic_image = None

        if a is not None:
            self.a = a.to(self.device).coalesce()
        else:
            self.a = None

        if c is not None:
            self.c = c.to(self.device)
        else:
            self.c = None

        self._results = None

        # Superpixel-specific initializers, move to new class
        self._th = None
        self._robust_corr_term = None
        self.dim1_coordinates = None
        self.dim2_coordinates = None
        self.correlations = None

    @property
    def frame_batch_size(self):
        return self._frame_batch_size

    @frame_batch_size.setter
    def frame_batch_size(self, new_batch_size: int):
        self._frame_batch_size = new_batch_size

    @property
    def pixel_batch_size(self):
        return self._pixel_batch_size

    @pixel_batch_size.setter
    def pixel_batch_size(self, new_batch_size: int):
        self._pixel_batch_size = new_batch_size

    @property
    def results(self):
        return self.a_init, self.mask_a_init, self.c_init, self.b_init

    def lock_results_and_continue(self, context: SignalDemixer, carry_background: bool = True):
        if any(element is None for element in self.results):
            raise ValueError("Results do not exist. Run initialize signals first.")
        else:  # Initiate state transition
            if carry_background:
                background_term = self.factorized_ring_term
            else:
                background_term = None
            context.state = DemixingState(
                self.u_sparse,
                self.r,
                self.s,
                self.v,
                self.a_init,
                self.b_init,
                self.c_init,
                self.mask_a_init,
                (self.d1, self.d2, self.T),
                factorized_ring_term = background_term,
                data_order = self.data_order,
                device = self.device,
                frame_batch_size = self.frame_batch_size
            )
            print("Now in demixing state")

    @property
    def state_description(self):
        return "Initialization state: identify initial estimates of the signals present in the data"

    def _initialize_signals_superpixels(
        self,
        mad_threshold: int = 1,
        mad_correlation_threshold: float = 0.9,
        min_superpixel_size: int = 3,
        residual_threshold: float = 0.3,
        patch_size: tuple[int, int] = (100, 100),
        robust_corr_term: float = 0.03,
        text: bool = True,
        plot_en: bool = False,
    ):
        """
        Args:
            mad_threshold (int): the threshold, th, which is used to run a maximum absolute deviation (MAD) threshold on
                each time series y(t). We set all bins to 0 where y(t) < median(y(t)) + th(MAD(y(t)).
            mad_correlation_threshold (float). Between 0 and 1. We compute temporal correlations between the
                MAD thresholded versions of neighboring pixels.

            min_superpixel_size (int): Minimum cluster size of highly correlated pixels which can form a "superpixel".
            residual_threshold (float): Between 0 and 1. Each superpixel has a temporal estimate. "Pure" superpixels have
                the property that their temporal estimates are "far away" from the linear subspace spanned by other
                (spatially closeby) superpixel temporal estimates. This parameter quantifies this distance.
                By setting a high residual_threshold, fewer superpixels qualify as "pure" superpixels.
            patch_size (tuple): When determining pure superpixels, we compare superpixels within local patches; this
                is the patch size (height, width).
            robust_corr_term (float): A robust correlation parameter used in the superpixel calculations
            text (bool): If plotting the superpixel image, this determines whether we assign numbers to the superpixels.
            plot_en (bool): Used for debugging; whether to plot the superpixels and pure superpixel plots.
        """
        if mad_threshold != self._th or self._robust_corr_term != robust_corr_term:
            print(
                f"Computing correlation data structure with MAD threshold  {mad_threshold}"
                f"and the robust corr term is {robust_corr_term}"
            )
            # This indicates that it is the first time we are running the superpixel init with this set of
            # pre-existing self.a and self.c values, so we need to compute the local correlation data
            if self.factorized_ring_term is not None:
                bg_subtract_term = torch.diag(self.s) - self.factorized_ring_term
            else:
                bg_subtract_term = torch.diag(self.s)
            self.dim1_coordinates, self.dim2_coordinates, self.correlations = (
                get_local_correlation_structure(
                    self.u_sparse,
                    torch.matmul(self.r @ bg_subtract_term, self.v),
                    self.shape,
                    mad_threshold,
                    order=self.data_order,
                    batch_size=self.pixel_batch_size,
                    pseudo=robust_corr_term,
                    a=self.a,
                    c=self.c,
                )
            )
            self._th = mad_threshold
            self._robust_corr_term = robust_corr_term

        (
            self.a_init,
            self.mask_a_init,
            self.c_init,
            self.b_init,
            output_dictionary,
            self.diagnostic_image,
        ) = superpixel_init(
            self.u_sparse,
            self.r,
            self.s,
            self.v,
            patch_size,
            self.data_order,
            self.shape,
            mad_correlation_threshold,
            residual_threshold,
            min_superpixel_size,
            self.device,
            self.dim1_coordinates,
            self.dim2_coordinates,
            self.correlations,
            text=text,
            plot_en=plot_en,
            a=self.a,
            c=self.c,
        )

    def _initialize_signals_custom(
        self, spatial_footprints: Union[torch.sparse_coo_tensor, np.ndarray]
    ):
        """
        Given a set of spatial footprints, initialize all of the signals.
        Args:
            spatial_footprints (Union[torch.sparse_coo_tensor, torch.tensor, np.ndarray, scipy.sparse.spmatrix]):
                A set of footprints, either 2D (fov dim 1 * fov dim 2, number of neurons) or 3D (fov dim 1, fov dim 2,
                number of neurons). If it is 2D, the assumption is that the 2D frames have been flattened into
                1D vectors in the same "order" (i.e. "C" or "F" ordering) in which the input video has been reordered.
        """
        if isinstance(spatial_footprints, np.ndarray):
            if spatial_footprints.ndim == 3:
                # Shape is (fov dim 1, fov dim 2, number of neurons)
                spatial_2d = spatial_footprints.reshape(
                    (self.d1 * self.d2, -1), order=self.data_order
                )

            elif spatial_footprints.ndim == 2:
                spatial_2d = spatial_footprints
            else:
                raise ValueError(
                    f"Spatial footprint array should have shape (fov dim 1, fov dim 2, "
                    f"number of neurons) or (fov dim 1 * fov dim 2, number of neurons. "
                    f"Input array here had shape {spatial_footprints.shape}"
                )
            processed_spatial_tensor = ndarray_to_torch_sparse_coo(spatial_2d).to(
                self.device
            )

        elif isinstance(spatial_footprints, torch.Tensor):
            if spatial_footprints.is_sparse:
                processed_spatial_tensor = spatial_footprints.to(self.device)
            else:
                if spatial_footprints.ndim == 3:
                    print(
                        f"Passed in 3D dense torch.tensor for custom initialization. This "
                        f"will be slower because the code will convert to numpy, reshape to 2D, and then "
                        f"construct the sparse torch tensor. For faster processing pass in a torch.sparse_coo_tensor"
                        f"of shape (fov dim 1 * fov dim 2, number of neurons)"
                    )
                    spatial_2d = spatial_footprints.cpu().detach().numpy()
                    spatial_2d = spatial_2d.reshape(
                        (self.d1 * self.d2, -1), order=self.data_order
                    )
                    processed_spatial_tensor = ndarray_to_torch_sparse_coo(
                        spatial_2d
                    ).to(self.device)

                elif spatial_footprints.ndim == 2:
                    processed_spatial_tensor = torch_dense_to_sparse_coo(
                        spatial_footprints
                    ).to(self.device)

                else:
                    raise ValueError(
                        f"Passed in a {spatial_footprints.ndim}D dense tensor."
                        f"Initialization routine only accepts 2D and 3D tensors."
                    )

        elif isinstance(spatial_footprints, scipy.sparse.spmatrix):
            processed_spatial_tensor = scipy_sparse_to_torch(spatial_footprints).to(
                self.device
            )

        else:
            raise ValueError(
                f"Provided input array of type {type(spatial_footprints)},"
                f"which is not supported"
            )

        self.a_init, self.mask_a_init, self.c_init, self.b_init = (
            process_custom_signals(
                processed_spatial_tensor,
                self.u_sparse,
                self.r,
                self.s,
                self.v,
            )
        )
        self.diagnostic_image = None

    def initialize_signals(
        self,
        is_custom: bool = False,
        **init_kwargs: dict,
    ):
        """
        Runs an initialization algorithm to get initial signal estimates.

        Args:
            is_custom (bool): Indicates whether custom init or regular init is used
            init_kwargs (dict): Dictionary of method-specific parameter values used in superpixel init


        See the functions _initialize_signals_superpixels and _initialize_signals_custom for documentation

        Generates the following:
            tuple consisting of
            - spatial footprints (torch.sparse_coo_tensor) of shape (pixels, signals)
            - spatial masks (torch.sparse_coo_tensor) of shape (pixels, signals). The binary masks corresponding to the
                spatial footprints
            - temporal footprints (torch.tensor) of shape (timepoints, signals)
            - baseline (torch.tensor) of shape (pixels, 1)
            - diagnostic_image (Optional, np.ndarray): Diagnostic reference image
        """
        if is_custom:
            self._initialize_signals_custom(**init_kwargs)
        else:
            self._initialize_signals_superpixels(**init_kwargs)


class DemixingState(SignalProcessingState):

    def __init__(
        self,
        u_sparse: torch.sparse_coo_tensor,
        r: torch.tensor,
        s: torch.tensor,
        v: torch.tensor,
        a_init,
        b_init,
        c_init,
        mask_init,
        dimensions: tuple[int, int, int],
        factorized_ring_term: Optional[torch.tensor] = None,
        data_order: str = "F",
        device: str = "cpu",
        pixel_batch_size: int = 10000,
        frame_batch_size: int = 10000,
    ):
        super().__init__(pixel_batch_size, frame_batch_size)
        # Define the data dimensions, data ordering scheme, and device
        self.d1, self.d2, self.T = dimensions
        self.shape = (self.d1, self.d2, self.T)
        self.data_order = data_order
        self.device = device
        self._results = None

        self.u_sparse = u_sparse.to(device)
        self.r = r.to(device)
        self.s = s.to(device)
        self.v = v.to(device)

        self._mask_a_init = mask_init
        self._a_init = a_init.to(device).coalesce()
        self._b_init = b_init.to(device)
        self._c_init = c_init.to(device)
        self.a = None
        self.b = None
        self.c = None
        self.mask_ab = None
        self.standard_correlation_image = None
        self.residual_correlation_image = None
        self.uv_mean = get_mean_data(self.u_sparse, self.r, self.s, self.v)

        if factorized_ring_term is None:
            self._factorized_ring_term_init = torch.zeros(
                [self.r.shape[1], self.v.shape[0]], device=self.device
            )
        else:
            self._factorized_ring_term_init = factorized_ring_term.to(self.device)
            self._validate_factorized_ring_term()
        self.factorized_ring_term = None

        self.W = None

        self.a_summand = torch.ones((self.d1 * self.d2, 1)).to(self.device)
        self.blocks = None

    @property
    def state_description(self):
        return (
            "Demixing state: Given initial estimates of the signals, this state is designed to run the "
            "NMF demixing algorithm to get refined source extractions"
        )

    @property
    def results(self):
        return self._results

    def lock_results_and_continue(self, context: SignalDemixer, carry_background: bool = True):
        """
        Args:
            context (SignalDemixer): The context that manages and delegates work to all states.
                As per the state model, the state constructs the next object and updates the context's state.
            carry_background (bool): Whether to carry the fluctuating background term to the next state or not.
        """
        if self.results is None:
            raise ValueError(
                "Results do not exist. Run demixing signals before moving to next step."
            )
        else:
            if carry_background:
                background_term = self.factorized_ring_term
            else:
                background_term = None
            context.state = InitializingState(
                self.u_sparse,
                self.r,
                self.s,
                self.v,
                (self.d1, self.d2, self.T),
                self.data_order,
                self.device,
                self.a,
                self.c,
                pixel_batch_size=self.pixel_batch_size,
                frame_batch_size=self.frame_batch_size,
                factorized_ring_term=background_term,
            )
            print("Now in the initialization state")

    def precompute_quantities(self):
        """
        Move relevant data to the GPU
        """

        a_indices = self._a_init.indices().clone()
        a_values = self._a_init.values().clone()
        self.a = (
            torch.sparse_coo_tensor(a_indices, a_values, self._a_init.size())
            .to(self.device)
            .coalesce()
        )
        self.b = self._b_init.clone()
        self.c = self._c_init.clone()
        self.mask_ab = self._mask_a_init.clone().coalesce()
        if self.mask_ab is None:
            self.mask_ab = self.a.bool().coalesce()

        self.factorized_ring_term = self._factorized_ring_term_init.clone()

    def _validate_factorized_ring_term(self):
        """Checks that the factorized ring term at the initialization is valid"""
        expected_dimensions = self.r.shape[1], self.v.shape[0]
        if not isinstance(self._factorized_ring_term_init, torch.Tensor):
            raise ValueError(f"Expected data of type {torch.Tensor} for factorized ring term but got type"
                             f"{type(self._factorized_ring_term_init)}")
        if self.r.shape[1] != self._factorized_ring_term_init.shape[0] or self.v.shape[0] != self._factorized_ring_term_init.shape[1]:
            raise ValueError(f"Shape of factorized_background_term is {self._factorized_ring_term_init.shape}"
                             f"expected shape is {expected_dimensions}")

    def initialize_standard_correlation_image(self):
        self.standard_correlation_image = _compute_standard_correlation_image(
            self.u_sparse,
            self.r,
            self.s,
            self.v,
            self.c,
            (self.shape[0], self.shape[1]),
            self.data_order,
            frame_batch_size=self.frame_batch_size,
            device=self.device,
        )

    def compute_residual_correlation_image(self):
        self.residual_correlation_image = _compute_residual_correlation_image(
            self.u_sparse,
            self.r,
            self.s,
            self.v,
            self.factorized_ring_term,
            self.a,
            self.c,
            (self.shape[0], self.shape[1]),
            blocks=self.blocks,
            data_order=self.data_order,
            batch_size=self.frame_batch_size,
            device=self.device,
        )

    def update_hals_scheduler(self):
        """
        Lots of HALS updates can be done in parallel because the underlying signals don't overlap
        """
        adjacency_mat = torch.sparse.mm(self.a.t(), self.a)
        graph = construct_graph_from_sparse_tensor(adjacency_mat)
        self.blocks = color_and_get_tensors(graph, self.device)

    def update_ring_model_support(self):
        ones_vec = torch.ones((self.a.shape[1], 1), device=self.a.device)
        indicator = (torch.sparse.mm(self.a, ones_vec).squeeze() == 0).to(torch.float32)
        self.W.support = indicator

    def ring_model_weight_update(self):
        self.W.weights = torch.ones(
            (self.shape[0] * self.shape[1]), device=self.device
        ).float()
        ur = torch.sparse.mm(self.u_sparse, self.r)
        e = torch.matmul(
            torch.ones([1, self.v.shape[1]], device=self.device), self.v.t()
        )
        x = torch.matmul(self.c.t(), self.v.t())

        spatial_term = ur * self.s.unsqueeze(0)
        spatial_term -= self.b @ e
        full_residual = spatial_term - torch.sparse.mm(self.a, x)
        ring_output = self.W.forward(spatial_term)

        numerator = torch.sum(full_residual * ring_output, dim=1)
        denominator = torch.square(torch.linalg.norm(ring_output, dim=1))

        weights = torch.nan_to_num(
            numerator / denominator, nan=0.0, posinf=0.0, neginf=0.0
        )

        threshold_function = torch.nn.ReLU()
        weights = threshold_function(weights)
        # Now export the ring model to its factorized form:
        self.factorized_ring_term = ur.T @ (weights.unsqueeze(1) * ring_output)

    def static_baseline_update(self):
        self.b = regression_update.baseline_update(self.uv_mean, self.a, self.c)

    def fluctuating_baseline_update(self):
        """
        Performs a fluctuating baseline update
        """
        self.update_ring_model_support()
        self.ring_model_weight_update()

    def spatial_update(self, plot_en=False):
        self.a = regression_update.spatial_update_hals(
            self.u_sparse,
            self.r,
            self.s,
            self.v,
            self.a,
            self.c,
            self.b,
            q=self.factorized_ring_term,
            mask_ab=self.mask_ab,
            blocks=self.blocks,
        )

        ## Delete Bad Components
        temp = torch.squeeze(
            torch.sparse.mm(self.a.t(), self.a_summand) == 0
        ).long()  # Identify which columns of 'a' are all zeros
        if torch.sum(temp):
            (
                self.a,
                self.c,
                self.standard_correlation_image,
                self.mask_ab,
            ) = delete_comp(
                self.a,
                self.c,
                self.standard_correlation_image,
                self.mask_ab,
                temp,
                "zero a!",
                plot_en,
                order=self.data_order,
            )
            print(f"new shape of a is {self.a.shape}")
            self.update_hals_scheduler()

    def temporal_update(self, denoise=False, plot_en=False, c_nonneg=True):
        self.c = regression_update.temporal_update_hals(
            self.u_sparse,
            self.r,
            self.s,
            self.v,
            self.a,
            self.c,
            self.b,
            q=self.factorized_ring_term,
            c_nonneg=c_nonneg,
            blocks=self.blocks,
        )

        # Denoise 'c' components if desired
        if denoise:
            c = self.c.cpu().numpy()
            c = utils.denoise(
                c
            )  # We now use OASIS denoising to improve improve signals
            c = np.nan_to_num(
                c, posinf=0, neginf=0, nan=0
            )  # Gracefully handle invalid values
            self.c = torch.from_numpy(c).float().to(self.device)

        # Delete bad components
        temp = torch.squeeze(torch.sum(self.c, dim=0) == 0).long()
        if torch.sum(temp):
            (
                self.a,
                self.c,
                self.standard_correlation_image,
                self.mask_ab,
            ) = delete_comp(
                self.a,
                self.c,
                self.standard_correlation_image,
                self.mask_ab,
                temp,
                "zero c!",
                plot_en,
                order=self.data_order,
            )
            self.update_hals_scheduler()

    def _flag_components_for_deletion(self, deletion_threshold: float):
        """
        For each neuron, we check that its residual correlation image over its spatial support contains
        at least one pixel whose correlation value is above a specified threshold.

        Otherwise we tag this component for deletion

        Args:
            deletion_threshold (float): The threshold for deciding whether a component should be deleted or not

        Returns:
            indices_to_keep (torch.tensor): The indices of the neural signals we should keep.
        """
        if self.residual_correlation_image is None:
            raise ValueError(
                "Deletion Routine requires that a residual correlation image was calculated"
            )

        support_data = self.residual_correlation_image.support_correlation_values
        rows, columns = support_data.indices()
        values = (support_data.values() > deletion_threshold).long()

        new_vector = torch.sparse_coo_tensor(
            torch.stack([rows * 0, columns]), values, (1, support_data.shape[1])
        ).coalesce()
        boolean_indices = new_vector.to_dense().squeeze().bool()
        indices_to_keep = torch.arange(support_data.shape[1], device=self.device)[
            boolean_indices
        ]

        return indices_to_keep

    def connected_comps(
        self, thresholded_images: torch.tensor, masks: torch.tensor, num_iters: int = 30
    ):
        """
        Args:
            thresholded_images (torch.tensor): Shape (images, fov dim 1, fov dim 2). All binary
            masks (torch.tensor): Shape (images, fov dim 1, fov dim 2). All binary
        Returns:
            updated_masks (torch.tensor): Shape (images, fov dim 1, fov dim 2)
        """

        for k in range(num_iters):
            masks = torch.nn.functional.max_pool2d(
                masks, kernel_size=3, stride=1, padding=1
            )
            masks = masks * thresholded_images
        return masks

    def _mask_expansion_routine(
        self,
        relative_correlation_fraction: float,
        mask: torch.sparse_coo_tensor,
        spatial_comps: torch.sparse_coo_tensor,
        residual_correlation_data: ResidualCorrelationImages,
    ) -> tuple[torch.sparse_coo_tensor, torch.sparse_coo_tensor]:

        num_iters = math.ceil(spatial_comps.shape[1] / self.frame_batch_size)

        final_spatial_rows = []
        final_spatial_cols = []
        final_spatial_values = []

        final_mask_rows = []
        final_mask_cols = []

        max_correlation_values = torch.zeros(
            spatial_comps.shape[1], device=self.device
        ).float()
        _, correlation_cols = (
            residual_correlation_data.support_correlation_values.indices()
        )
        correlation_values = (
            residual_correlation_data.support_correlation_values.values()
        )

        max_correlation_values.scatter_reduce_(
            0, correlation_cols, correlation_values, "amax", include_self=False
        )
        max_correlation_thresholds = (
            max_correlation_values * relative_correlation_fraction
        )

        for k in range(num_iters):
            start = k * self.frame_batch_size
            end = min(spatial_comps.shape[1], start + self.frame_batch_size)
            neuron_indices = torch.arange(start, end, device=self.device).long()

            curr_thresholds = max_correlation_thresholds[start:end]

            curr_residual_images = residual_correlation_data.getitem_tensor(
                slice(start, end)
            )  # Images x fov dim 1 x fov dim 2

            curr_thresholded_residual_images = (
                curr_thresholds[:, None, None] < curr_residual_images
            ).float()
            curr_masks = torch.index_select(mask, 1, neuron_indices).to_dense().float()

            if (
                self.data_order == "F"
            ):  # Torch uses reshape C, so we need to modify here
                curr_masks = curr_masks.reshape((self.shape[1], self.shape[0], -1))
                curr_masks = curr_masks.permute(1, 0, 2)
            elif self.data_order == "C":
                curr_masks = curr_masks.reshape((self.shape[0], self.shape[1], -1))
            else:
                raise ValueError(f"Error with data order")

            curr_masks = curr_masks.permute(2, 0, 1)

            new_masks = self.connected_comps(
                curr_thresholded_residual_images, curr_masks
            )

            if self.data_order == "F":
                new_masks = new_masks.permute(
                    2, 1, 0
                )  # This is now d2 x d1 x frames to account for C vs F reshape

            new_masks = new_masks.reshape((self.shape[0] * self.shape[1], -1))

            a_crop = torch.index_select(spatial_comps, 1, neuron_indices).coalesce()
            curr_a_row, curr_a_col = a_crop.indices()
            curr_a_new_values = a_crop.values() * new_masks[(curr_a_row, curr_a_col)]

            final_spatial_rows.append(curr_a_row)
            final_spatial_cols.append(start + curr_a_col)
            final_spatial_values.append(curr_a_new_values)

            curr_mask_row, curr_mask_col = torch.nonzero(new_masks, as_tuple=True)

            final_mask_rows.append(curr_mask_row)
            final_mask_cols.append(curr_mask_col)

        # Construct the new mask
        final_mask_rows = torch.cat(final_mask_rows, 0)
        final_mask_cols = torch.cat(final_mask_cols, 0)
        final_mask_values = torch.ones_like(final_mask_cols).float()
        final_mask = torch.sparse_coo_tensor(
            torch.stack([final_mask_rows, final_mask_cols]),
            final_mask_values,
            spatial_comps.shape,
        ).coalesce()

        final_spatial_rows = torch.cat(final_spatial_rows, 0)
        final_spatial_cols = torch.cat(final_spatial_cols, 0)
        final_spatial_values = torch.cat(final_spatial_values, 0)
        final_spatial = torch.sparse_coo_tensor(
            torch.stack([final_spatial_rows, final_spatial_cols]),
            final_spatial_values,
            spatial_comps.shape,
        ).coalesce()

        return final_mask, final_spatial

    def support_update_routine(
        self, relative_correlation_fraction: float, corr_th_del: float, plot_en
    ):
        self.compute_residual_correlation_image()
        indices_to_keep = self._flag_components_for_deletion(corr_th_del)
        if indices_to_keep.shape[0] < self.a.shape[1]:
            self.a = torch.index_select(self.a, 1, indices_to_keep).coalesce()
            self.c = torch.index_select(self.c, 1, indices_to_keep)
            self.standard_correlation_image.c = self.c
            # Need to update the residual correlation image since the A/C terms changed
            self.update_hals_scheduler()
            self.compute_residual_correlation_image()

        # Currently using rigid mask
        self.mask_ab = self.a.bool()
        self.mask_ab, self.a = self._mask_expansion_routine(
            relative_correlation_fraction,
            self.mask_ab,
            self.a,
            self.residual_correlation_image,
        )

        self.residual_correlation_image = None

    def merge_signals(self, merge_corr_thr, merge_overlap_thr, plot_en):
        self.a, self.c, self.mask_ab, self.standard_correlation_image = (
            merge_components(
                self.a,
                self.c,
                self.standard_correlation_image,
                merge_corr_thr=merge_corr_thr,
                merge_overlap_thr=merge_overlap_thr,
                plot_en=plot_en,
                data_order=self.data_order,
            )
        )

    def demix(
        self,
        maxiter: int = 25,
        support_threshold: Union[list, float] = 0.9,
        deletion_threshold: float = 0.2,
        ring_model_start_pt: int = 5,
        ring_radius: int = 10,
        merge_threshold: float = 0.8,
        merge_overlap_threshold: float = 0.4,
        update_frequency: int = 4,
        c_nonneg: bool = True,
        denoise: Union[list, bool] = None,
        plot_en: bool = False,
    ):
        """
        Function for computing background, spatial and temporal components of neurons. Uses HALS updates to iteratively
        refine spatial and temporal estimates.

        Args:
            maxiter (int): Number of HALS iterations to be performed
            support_threshold (Union[list, float]): Value between 0 and 1.
                For each neuron, we take the max value of its correlation image and multiply it by this parameter.
                This gives us a correlation cutoff used to set the new support.
            deletion_threshold (float): We delete neurons whose residual correlation image over its support is below
                this value.
            ring_model_start_pt (int): How many HALS iterations to wait before fitting the ring model. To disable
                the ring model set this to be greater than maxiter.
            ring_radius (int): The radius of the ring model (if it is used)
            merge_threshold (float): Between 0 and 1. We merge two signals based on the degree of overlap between their thresholded
                correlation images. This parameter is the cutoff for computing those thresholded correlation images.
            merge_overlap_threshold (float): Between 0 and 1. Specifies what fraction of a neuron's thresholded correlation
                image must overlap with another candidate neuron's thresholded correlation image for the two signals
                to be merged.
            update_frequency (int): Determines how frequently we perform  spatial support updates and delete and merge
                neural signal estimates. For example, the default value of 4 means every 4 HALS iterations, we perform this step.
            c_nonneg (bool): Indicates whether the temporal estimates are allowed to be negative;
                this is useful for e.g. in voltage imaging.
            denoise (Union[list, bool]): Indicates whether to run a denoiser at each HALS iteration on the temporal traces.
            plot_en (bool): Indicates whether plotting is enabled; this is only used for debugging purposes.
        """
        # Key: precompute_quantities is a setup function which must be run first in this routine
        self.precompute_quantities()
        self.W = RingModel(
            self.shape[0], self.shape[1], ring_radius, self.device, self.data_order
        )
        self.update_hals_scheduler()
        self.initialize_standard_correlation_image()
        self.compute_residual_correlation_image()

        if isinstance(support_threshold, list):
            if len(support_threshold) != maxiter:
                raise ValueError(
                    f"Length of list ``support_threshold`` is not equal to maxiter, which is {maxiter}"
                )
        elif isinstance(support_threshold, float):
            support_threshold = [support_threshold] * maxiter
        else:
            raise ValueError(
                f"support_threshold has invalid type: {type(support_threshold)}"
            )

        if denoise is None:
            denoise = [False for i in range(maxiter)]
        elif isinstance(denoise, bool):
            denoise = [denoise for i in range(maxiter)]
        elif len(denoise) != maxiter:
            print(
                "Length of denoise list is not consistent, setting all denoise values to false for this pass of NMF"
            )
            denoise = [False for i in range(maxiter)]

        for iters in tqdm(range(maxiter)):
            self.static_baseline_update()

            if iters >= ring_model_start_pt:
                self.fluctuating_baseline_update()
            else:
                pass

            self.spatial_update(plot_en=plot_en)
            self.static_baseline_update()

            denoise_flag = denoise[iters]
            self.temporal_update(
                denoise=denoise_flag, plot_en=plot_en, c_nonneg=c_nonneg
            )

            if update_frequency and ((iters + 1) % update_frequency == 0):
                ##First: Compute correlation images
                self.standard_correlation_image.c = self.c

                # Merge signals as needed and update the scheduler
                original_shape = self.a.shape[1]
                self.merge_signals(merge_threshold, merge_overlap_threshold, plot_en)
                if self.a.shape[1] < original_shape:
                    self.update_hals_scheduler()

                self.support_update_routine(
                    support_threshold[iters], deletion_threshold, plot_en
                )

                self.update_hals_scheduler()

        self.standard_correlation_image.c = self.c
        self.compute_residual_correlation_image()
        self._results = DemixingResults(
            self.u_sparse,
            self.r,
            self.s,
            self.factorized_ring_term,
            self.v,
            self.a,
            self.c,
            self.b.squeeze(),
            self.residual_correlation_image,
            self.standard_correlation_image,
            self.data_order,
            (self.T, self.d1, self.d2),
            "cpu",
        )

        return self.results
