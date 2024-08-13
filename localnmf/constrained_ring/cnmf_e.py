import math

import torch
import numpy as np
from localnmf import ca_utils
import logging
import math

class RingModel:

    def __init__(self, d1, d2, radius, empty=False, device='cpu', order="F", batchsize=200000):
        self._shape = (d1, d2)
        self._empty=empty
        self._radius = radius
        self._device = device
        self._order = order
        if self.empty:
            row = torch.Tensor([]).to(device).long()
            col = torch.Tensor([]).to(device).long()
            value = torch.Tensor([]).to(device).float()
            self.W_mat = torch.sparse_coo_tensor(torch.stack([row, col]), value, (d1 * d2, d1 * d2)).coalesce()
            self._weights = torch.zeros((d1 * d2, 1), device=device)
        else:
            rowcol_stacked, values = self._construct_init_values(batchsize=batchsize)
            torch.cuda.empty_cache()
            self.W_mat = torch.sparse_coo_tensor(rowcol_stacked, values, (d1 * d2, d1 * d2)).coalesce()
            self._weights = torch.ones((d1 * d2, 1), device=device)

    @property
    def shape(self):
        return self._shape

    @property
    def order(self):
        return self._order

    @property
    def device(self):
        return self._device

    @property
    def radius(self):
        return self._radius

    @property
    def empty(self):
        return self._empty

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, tensor):
        self._weights = tensor

    def _construct_init_values(self, batchsize = 100000):
        num_rows = self.shape[0]*self.shape[1]
        num_iterations = math.ceil(num_rows / batchsize)
        net_rowcol = [] #Store the results of torch.stack([curr_rows, curr_columns])
        net_values = []
        for ind in range(num_iterations):
            start_value = batchsize * ind
            end_value = min(num_rows, start_value + batchsize)
            rows_to_process = torch.arange(start_value, end_value, device=self.device, dtype=torch.long)
            curr_rowcol, curr_values = self._construct_at_indices(rows_to_process)
            net_rowcol.append(curr_rowcol)
            net_values.append(curr_values)

        print(curr_rowcol.shape)

        return torch.cat(net_rowcol, dim=1), torch.cat(net_values)

    def _construct_at_indices(self, rows_to_process):
        d1, d2 = self.shape
        dim1_spread = torch.arange(-(self.radius + 1), (self.radius + 2), device=self.device)
        dim2_spread = torch.arange(-(self.radius + 1), (self.radius + 2), device=self.device)
        spr1, spr2 = torch.meshgrid((dim1_spread, dim2_spread), indexing='ij')
        norms = torch.sqrt(spr1 * spr1 + spr2 * spr2)
        outputs = torch.logical_and(norms >= self.radius, norms < self.radius + 1).to(self.device)

        dim1_ring = spr1[outputs].flatten().squeeze().long()
        dim2_ring = spr2[outputs].flatten().squeeze().long()

        #flatten the 2D ring representation to 1D for efficiency
        if self.order == "C":
            ring_indices = dim1_ring*d2 + dim2_ring
        elif self.order == "F":
            ring_indices = dim1_ring + d1*dim2_ring
        else:
            raise ValueError("Not a valid ordering")
        logging.debug("number of elts in ring is {}".format(outputs.shape[0]))

        #Define the "column indices" of the (d1*d2, d1*d2) sparse matrix (every row is a ring) in 1D representation.
        column_indices = rows_to_process.unsqueeze(1) + ring_indices.unsqueeze(0)
        row_indices = rows_to_process.unsqueeze(1) + torch.zeros((1, ring_indices.shape[0]),
                                                                 device=self.device, dtype=torch.long)

        #preliminary filter
        good_components = torch.logical_and(column_indices >= 0, column_indices < d1 * d2)
        if self.order == "C":
            '''
            We get rid of values that are out of bounds
            '''
            twod_column_indices = (rows_to_process % d2).unsqueeze(1) + dim2_ring.unsqueeze(0)
            good_components = torch.logical_and(good_components, twod_column_indices >= 0)
            good_components = torch.logical_and(good_components, twod_column_indices < d2)

            twod_row_indices = (rows_to_process // d2).unsqueeze(1) + dim1_ring.unsqueeze(0)
            good_components = torch.logical_and(good_components, twod_row_indices >= 0)
            good_components = torch.logical_and(good_components, twod_row_indices < d1)
        elif self.order == "F": #Make sure the vertical indices do not shift by columns
            '''
            good_components, as defined above, will filter out columns that are out of bounds, now we filter out rows
            '''
            twod_column_indices = (rows_to_process // d1).unsqueeze(1) + dim2_ring.unsqueeze(0)
            good_components = torch.logical_and(good_components, twod_column_indices >= 0)
            good_components = torch.logical_and(good_components, twod_column_indices < d2)

            twod_row_indices = (rows_to_process % d1).unsqueeze(1) + dim1_ring.unsqueeze(0)
            good_components = torch.logical_and(good_components, twod_row_indices >= 0)
            good_components = torch.logical_and(good_components, twod_row_indices < d1)

        column_indices = column_indices[good_components]
        row_indices = row_indices[good_components]
        values = torch.ones(row_indices.shape, device=self.device, dtype=torch.float32)

        return torch.stack([row_indices, column_indices]), values

    def generate_coo_ring_indices(self, rows: torch.tensor):
        """
        Given the row indices, this function makes row, column, value indices to construct a subset of the ring matrix
        """

    def apply_model_right(self, tensor, a_sparse):
        """
        Computes ring_matrix times tensor. Inputs: tensor: torch.Tensor of dimensions (d1*d2, X) for some X a_sparse:
        torch.sparse_coo_tensor. Dimensions (d1*d2, K) where K is the number of neurons described by a_sparse,  and d1,
        d2 are the FOV dimensions of the imaging video.
        """
        device = tensor.device
        a_sum_vec = torch.ones((a_sparse.shape[1], 1), device=device)
        a_sum = torch.sparse.mm(a_sparse, a_sum_vec)
        good_indices = (a_sum == 0).bool()

        return self.apply_weighted_ring_right(tensor, good_indices=good_indices)

    def apply_model_left(self, tensor, a_sparse):
        """
        Computes tensor times ring_matrix.
        Inputs:
            tensor: torch.Tensor of dimensions (X, d1*d2) for some X
            a_sparse: torch.sparse_coo_tensor. Dimensions (d1*d2, K) where K is the number of neurons described by a_sparse, and
                d1, d2 are the FOV dimensions of the imaging video.
        """
        device = tensor.device
        a_sum_vec = torch.ones((a_sparse.shape[1], 1), device=device)
        a_sum = torch.sparse.mm(a_sparse, a_sum_vec)
        good_indices = (a_sum == 0).bool()

        return self.apply_weighted_ring_left(tensor, good_indices=good_indices)

    def apply_weighted_ring_right(self, tensor, good_indices=None):
        """
        Multiplies weighted_ring_matrix by tensor. Note that tensor must have (d1*d2) rows (since ring_matrix is (d1*d2, d1*d2)
        Inputs:
            tensor. torch.Tensor. Shape (d1*d2, X) for some X
            good_indices. torch.Tensor, dtype bool. Shape (d1*d2, 1). Describes which columns
             of the ring matrix must be zero'd out.
        """
        if good_indices is not None:
            tensor = good_indices * tensor
        return self.weights * torch.sparse.mm(self.W_mat, tensor)

    def apply_weighted_ring_left(self, tensor, good_indices=None):
        tensor_t = self.weights * tensor.t()
        product = torch.sparse.mm(self.W_mat.t(), tensor_t)

        if good_indices is not None:
            return (good_indices * product).t()
        else:
            return product.t()

    def compute_fluctuating_background_row(self, U_sparse, R, s, V, active_weights, b, row_index):
        """
        Computes a specific row of W(URsV - ac - b), which is the fluctuating background.
        Recall  we enforce that column i of W contains all zeros whenever row 'i' of a contains nonzero entries. So W*a will always be zero, and the above expression becomes:
        W(URV - b)
        Parameters:
            U_sparse. torch.sparse_coo_tensor. Tensor. Dimensions (d, K)
            R: torch.Tensor. Dimensions (K x K)
            s: torch.Tensor of shape (K)
            V: torch.Tensor. Dimensions (K x T)
            active_weights: np.ndarray of shape (d, 1). Describes, for each pixel, whether its ring weight should be active or not.
            b: np.ndarray. Dimensions (d,1). Describes, for each pixel, its static background estimate.

        TODO: When the entire demixing procedure manages all objects directly on the device, we can avoid the awkward convention where some inputs are on "device" and others are definitely on cpu.
        """
        device = R.device
        b = torch.Tensor(b).float().to(device)
        active_weights = torch.Tensor(active_weights).float().to(device)

        row_index_tensor = torch.arange(row_index, row_index + 1, device=device)
        W_selected = torch.index_select(self.W_mat, 0, row_index_tensor).coalesce().to_dense().float()

        W_selected = W_selected * active_weights.t()

        # Apply unweighted ring model to W(URV - b)
        WU = torch.sparse.mm(U_sparse.t(), W_selected.t()).t()
        WUR = torch.matmul(WU, R)
        WURs = WUR * s[None, :]
        WURsV = torch.matmul(WURs, V)
        Wb = torch.matmul(W_selected, b)

        difference = WURsV - Wb

        # Apply weight at end
        weighted_row = self.weights[row_index_tensor] * difference
        return weighted_row

    def zero_weights(self):
        self.weights = self.weights * 0

    def reset_weights(self):
        self.weights = torch.ones_like(self.weights)

    def create_complete_ring_matrix(self, a):
        """
        Constructs a complete W matrix, combining the ring weights and the zero'd out columns into a single scipy sparse csr matrix.
        Input:
            a: np.ndarray. Dimensions (d, K) where d is the number of pixels in FOV and K is number of neurons identified
        Output:
            W_plain: The ring matrix with the weights (and zero'd out columns) applied
        """
        W_plain = ca_utils.torch_sparse_to_scipy_coo(self.W_mat).tocsr()
        W_plain = W_plain.multiply(self.weights.cpu().numpy())

        a_sum = (np.sum(a, axis=1, keepdims=True) == 0).T
        W_plain = W_plain.multiply(a_sum)

        return W_plain


def get_sampled_indices(num_frames, num_samples, device='cuda'):
    num_samples = min(num_samples, num_frames)
    tensor_to_sample = torch.arange(num_frames, device=device)
    weights = torch.ones_like(tensor_to_sample, device=device) * (1 / num_frames)
    new_values = tensor_to_sample[torch.multinomial(weights, num_samples=num_samples, replacement=False)]
    return new_values


def ring_model_update(U_sparse, R, V, W, c, b, a, num_samples=1000):
    device = V.device
    batches = math.ceil(R.shape[1] / num_samples)
    denominator = 0
    numerator = 0
    W.reset_weights()

    X = torch.matmul(c.t(), V.t())
    sV = torch.ones([1, V.shape[1]], device=device)
    s = torch.matmul(sV, V.t())

    for k in range(batches):
        start = num_samples * k
        end = min(R.shape[1], start + num_samples)
        indices = torch.arange(start, end, device=device)
        R_crop = torch.index_select(R, 1, indices)
        X_crop = torch.index_select(X, 1, indices)
        s_crop = torch.index_select(s, 1, indices)

        resid_V_basis = torch.sparse.mm(U_sparse, R_crop) - torch.sparse.mm(a, X_crop) - torch.matmul(b, s_crop)

        W_residual = W.apply_model_right(resid_V_basis, a)

        denominator += torch.sum(W_residual * W_residual, dim=1)
        numerator += torch.sum(W_residual * resid_V_basis, dim=1)

    values = torch.nan_to_num(numerator / denominator, nan=0.0, posinf=0.0, neginf=0.0)
    threshold_function = torch.nn.ReLU()
    values = threshold_function(values)
    W.weights = values[:, None]