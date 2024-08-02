import scipy.sparse
import torch
from typing import *


def fast_matmul(a, b, device="cuda"):
    a_torch = torch.from_numpy(a).float().to(device)
    b_torch = torch.from_numpy(b).float().to(device)
    return torch.mm(a_torch, b_torch).cpu().numpy()


def baseline_update(uv_mean, a, c, to_torch=False):
    """
    Calculates baseline. Inputs:
        uv_mean. torch.Tensor of shape (d, 1), where d is the number of pixels in the FOV
        a: torch.sparse_coo_tensor. tensor of shape (d, k) where k is the number of neurons in the FOV
        c: torch.Tensor of shape (T, k) where T is number of frames in video
        to_torch: indicates whether the inputs are np.ndarrays that need to be converted to torch objects. Also implies that the result will be returned as a np.ndarray (the same format as the inputs)
    Output:
        b. torch.Tensor of shape (d, 1). Describes new static baseline
    """
    if to_torch:
        a_sp = scipy.sparse.csr_matrix(a)
        torch.sparse_coo_tensor(a_sp.nonzero(), a_sp.data, a_sp.shape).coalesce()
        c = torch.from_numpy(c).float()
        uv_mean = torch.from_numpy(uv_mean).float()
    mean_c = torch.mean(c, dim=0, keepdim=True).t()
    b = uv_mean - torch.sparse.mm(a, mean_c)

    if to_torch:
        return b.numpy()
    else:
        return b


def project_U_HALS(
    U_sparse, R, W, vector, a_sparse, selected_indices: Optional[torch.tensor] = None
):
    """
    Commonly used routine to project background term (W) onto U basis in HALS calculations.
    We exploit the fact that the product UR has orthonormal columns
    Note that we multiple W by v first (before doing any other matmul) because that collapses to a single vector. Then
    all subsequent matmuls are very fast
    Inputs:
        U_sparse: torch.sparse_coo_tensor. object. Dimensions d x r where d is number of pixels, r is rank
        R: torch.Tensor object. Shape r x r' (where r may be equal to r'). UR is the orthonormal left basis term for the SVD of the PMD movie (recall the SVD decomposition is URsV where UR
            is the left set of orthonormal col vectors, is the diagonal matrix, and V contains the right orthonormal row vectors.
        W: ring model object. Represents a (d, d)-shaped sparse tensor
        vector: torch.Tensor. Dimensions (d, k) for some value k
        a_sparse. torch.sparse_coo_tensor object, with shape (d, k)
    """
    Wv = W.apply_model_right(vector, a_sparse)
    UtWv = torch.sparse.mm(U_sparse.t(), Wv)
    RRt = torch.matmul(R, R.t())
    RRtUtWv = torch.matmul(RRt, UtWv)

    if selected_indices is None:
        final_projection = torch.sparse.mm(U_sparse, RRtUtWv)
    else:
        u_subset = torch.index_select(U_sparse, 0, selected_indices)
        final_projection = torch.sparse.mm(u_subset, RRtUtWv)

    return final_projection


def spatial_update_HALS(
    U_sparse, R, s, V, W, a_sparse, c, b, blocks=None, mask_ab=None
):
    """
    Computes a spatial HALS updates:
    Params:

        Note: The first four parameters are the "PMD" representation of the data: it is given in a traditional SVD form: URsV, where UR is the left orthogonal basis, 's' represents the diagonal matrix, and V is the right orthogonal basis.
        U_sparse: torch.sparse_coo_tensor. Sparse matrix, with dimensions (d x R)
        R: torch.Tensor. Dimensions (R, R') where R' is roughly equal to R (it may be equal to R+1)
        s: torch.Tensor. This is the diagonal of a R' x R' matrix (so it is represented by a R' shaped tensor)
        V: torch.Tensor. Dimensions R' x T. Dimensions R' x T, where T is the number of frames, where all rows are orthonormal.
            Note:  V must contain the 1 x T vector of all 1's in its rowspan.
        W: ring_model object. Describes a sparse matrix with dimensions (d x d)
        a_sparse: torch.sparse_coo_tensor. dimensions d x k, where k represents the number of neural signals.
        c: torch.Tensor. Dimensions T x k
        b: torch.Tensor. Dimensions d x 1. Represents static background
        mask_ab: torch.sparse_coo_tensor. Dimensions (d x k). For each neuron, indicates the allowed support of neuron

    Returns:
        a_sparse: torch.sparse_coo_tensor. Dimensions d x k, containing updated spatial matrix

    TODO: Make 'a' input a sparse matrix
    """
    import pdb

    # Load all values onto device in torch
    device = V.device

    if mask_ab is None:
        mask_ab = a_sparse.bool()

    mask_ab = mask_ab.long().to_dense()
    nonzero_row_indices = torch.squeeze(torch.sum(mask_ab, dim=1).nonzero())
    mask_ab = torch.index_select(mask_ab, 0, nonzero_row_indices)

    Vc = torch.matmul(V, c)
    a_dense = torch.index_select(a_sparse, 0, nonzero_row_indices).to_dense()

    # Find the tensor, e, (a 1 x R' shaped tensor) such that eV gives a 1 x T tensor consisting of all 1's
    e = torch.matmul(torch.ones([1, V.shape[1]], device=device), V.t())

    # Find the tensor, X (a N x K shaped tensor) such that XV closely approximates c.T
    X = (
        Vc.t()
    )  # This is just a linear subspace projection of c.T onto the rowspace of V; can reuse above computation for this

    C_prime = torch.matmul(c.t(), c)
    C_prime_diag = torch.diag(C_prime)
    C_prime_diag[C_prime_diag == 0] = 1  # For division safety
    """
    We will now compute the TRANSPOSE of the following expression: 
    
    [URs - be - Proj(W)(URS -be - aX)]Vc
    
    The result (after the transpose) is a N x d sized matrix, where N is the number of neural signals
    """

    ### TODO: Optimize this later to avoid computing D x r matrices...?
    URsVc = torch.sparse.mm(U_sparse, torch.matmul(R, s[:, None] * Vc))
    beVc = torch.matmul(b, torch.matmul(e, Vc))
    aXVc = torch.sparse.mm(a_sparse, torch.matmul(X, Vc))

    ring_term = project_U_HALS(
        U_sparse,
        R,
        W,
        URsVc - beVc - aXVc,
        a_sparse,
        selected_indices=nonzero_row_indices,
    )
    cumulator = (
        torch.index_select(URsVc, 0, nonzero_row_indices)
        - torch.index_select(beVc, 0, nonzero_row_indices)
        - ring_term
    )

    threshold_func = torch.nn.ReLU(0)
    if blocks is None:
        blocks = torch.arange(c.shape[1], device=device)[None, :]
    for index_select_tensor in blocks:
        mask_apply = torch.squeeze(torch.index_select(mask_ab, 1, index_select_tensor))

        c_prime_i = C_prime.index_select(0, index_select_tensor).t()
        cumulator_i = cumulator.index_select(1, index_select_tensor)
        acc = torch.matmul(a_dense, c_prime_i)
        final_vec = (cumulator_i - acc) / C_prime_diag[None, index_select_tensor]
        curr_frame = torch.squeeze(torch.index_select(a_dense, 1, index_select_tensor))
        curr_frame += torch.squeeze(final_vec)
        curr_frame *= mask_apply
        curr_frame = threshold_func(curr_frame)
        a_dense[:, index_select_tensor] = curr_frame

    pruned_indices = a_dense.nonzero()
    pruned_row, pruned_col = [pruned_indices[:, i] for i in range(2)]
    final_values = a_dense[pruned_row, pruned_col]
    real_row = nonzero_row_indices[pruned_row]

    a_sparse = torch.sparse_coo_tensor(
        torch.stack([real_row, pruned_col]), final_values, a_sparse.shape
    ).coalesce()
    return a_sparse


##Compute the projection matrix:
def get_projection_matrix_temporal_HALS_routine(U_sparse, R, W, a_dense, a_sparse):
    aU = torch.sparse.mm(U_sparse.t(), a_dense).t()
    aUR = torch.matmul(aU, R)
    aURRt = torch.matmul(aUR, R.t())
    aURRtUt = torch.sparse.mm(U_sparse, aURRt.t()).t()  # Shape here is N x d
    projector = W.apply_model_left(aURRtUt, a_sparse)

    return projector


def temporal_update_HALS(U_sparse, R, s, V, W, a_sparse, c, b, c_nonneg=True):
    """
    Inputs:
         Note: The first four parameters are the "PMD" representation of the data: it is given in a traditional SVD form: URsV, where UR is the left orthogonal basis, 's' represents the diagonal matrix, and V is the right orthogonal basis.
        U_sparse: torch.sparse_coo_tensor. Sparse matrix, with dimensions (d x R)
        R: torch.Tensor. Dimensions (R, R') where R' is roughly equal to R (it may be equal to R+1)
        s: torch.Tensor. This is the diagonal of a R' x R' matrix (so it is represented by a R' shaped tensor)
        V: torch.Tensor. Dimensions R' x T. Dimensions R' x T, where T is the number of frames, where all rows are orthonormal.
            Note:  V must contain the 1 x T vector of all 1's in its rowspan.
        W: (d1*d2, d1*d2)-shaped ring model object
        a: (d1*d2, k)-shaped torch.sparse_coo_tensor
        c: (T, k)-shaped torch.Tensor
        b: (d1*d2, 1)-shaped torch.Tensor
        c_nonneg (bool): Indicates whether "c" should be nonnegative or fully unconstrained. For voltage data, it should be unconstrained; for calcium it should be constrained.

    Returns:
        c: (T, k)-shaped np.ndarray. Updated temporal components
    """
    device = V.device

    ##Precompute quantities used throughout all iterations
    a_dense = a_sparse.to_dense()

    # Find the tensor, e, (a 1 x R' shaped tensor) such that eV gives a 1 x T tensor consisting of all 1's
    e = torch.matmul(torch.ones([1, V.shape[1]], device=device), V.t())

    # Find the tensor, X (a N x K shaped tensor) such that XV closely approximates c.T
    X = torch.matmul(
        V, c
    ).t()  # This is just a linear subspace projection of c.T onto the rowspace of V; can reuse above computation for this

    # Step 1: Get aTURs
    aTU = torch.sparse.mm(U_sparse.t(), a_dense).t()
    aTUR = torch.matmul(aTU, R)
    aTURs = aTUR * s[None, :]

    # Step 2: Get aTbe
    aTb = torch.matmul(a_dense.t(), b)
    aTbe = torch.matmul(aTb, e)

    # Step 3:
    projector = get_projection_matrix_temporal_HALS_routine(
        U_sparse, R, W, a_dense, a_sparse
    )
    PU = torch.sparse.mm(U_sparse.t(), projector.t()).t()
    PUR = torch.matmul(PU, R)
    PURs = PUR * s[None, :]

    PA = torch.matmul(projector, a_dense)
    PAX = torch.matmul(PA, X)

    Pb = torch.matmul(projector, b)
    Pbe = torch.matmul(Pb, e)

    cumulator = aTURs - aTbe - (PURs - PAX - Pbe)
    cumulator = torch.matmul(cumulator, V)

    ata = torch.matmul(a_dense.t(), a_dense)

    ata_d = torch.diag(ata)
    ata_d[ata_d == 0] = 1  # For division-safety

    if c_nonneg:
        threshold_function = torch.nn.ReLU()
    else:
        threshold_function = lambda x: x

    for i in range(c.shape[1]):
        curr_index = torch.arange(i, i + 1, device=device)
        a_ia = torch.index_select(ata, 0, curr_index)
        a_iaC = torch.matmul(a_ia, c.t())

        curr_trace = torch.index_select(c, 1, curr_index)
        curr_trace += (
            (torch.index_select(cumulator, 0, curr_index) - a_iaC) / ata_d[i]
        ).t()
        curr_trace = threshold_function(curr_trace)
        c[:, [i]] = curr_trace

    return c
