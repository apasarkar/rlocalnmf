import torch
import scipy
from localnmf import ca_utils
import logging
import math
from typing import *

class RingModel:

    def __init__(self, d1: int, d2: int, radius: int, empty: bool=False, device: str='cpu',
                 order: str="F", batchsize: int=200000):
        """
        Ring Model object manages the state of the ring model during the model fit phase.

        Args:
            d1 (int): the 0th dimension of the FOV (in python indexing)
            d2 (int): the 1st dimension of the FOV (in python indexing)
            radius (int): the ring radius
            empty (bool): if we want an empty ring model (all elements 0)
            device (str): which device the pytorch data lies on
            order (str): the order used to reshape from 1D to 2D (and vice versa)
            batchsize (int): the batch size for constructing the ring model (to save memory)
        """
        self._shape = (d1, d2)
        self._empty=empty
        self._radius = radius
        self._device = device
        self._order = order
        if self.empty:
            row = torch.Tensor([]).to(device).long()
            col = torch.Tensor([]).to(device).long()
            value = torch.Tensor([]).to(device).float()
            self.ring_mat = torch.sparse_coo_tensor(torch.stack([row, col]), value, (d1 * d2, d1 * d2)).coalesce()
            self.weights = torch.zeros((d1 * d2), device=device)
        else:
            rowcol_stacked, values = self._construct_init_values(batchsize=batchsize)
            torch.cuda.empty_cache()
            self.ring_mat = torch.sparse_coo_tensor(rowcol_stacked, values, (d1 * d2, d1 * d2)).coalesce()
            self.weights = torch.ones((d1 * d2), device=device)

        self.support = torch.ones((self.shape[0]*self.shape[1]), device=self.device, dtype=torch.float32)

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
        """
        The ring model uses a constant weight assumption: that pixel "i" of the data can be explained as a scaled
        average of the pixels in a ring surrounding pixel "i". This is enforced by a diagonal weight matrix: d_weight.

        Returns:
            d_weights (torch.sparse_coo_tensor): (d1*d2, d1*d2) diagonal matrix
        """
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        """
        Sets the weights
        Args:
            new_weights (torch.tensor): Shape (d1*d2)
        """
        net_pixels = (self.shape[0] * self.shape[1])
        index_values = torch.arange(net_pixels, device=self.device, dtype=torch.long)
        self._weights = torch.sparse_coo_tensor(torch.stack([index_values, index_values]), new_weights,
                                             (net_pixels, net_pixels)).coalesce()


    @property
    def support(self):
        """
        The ring model only operates on pixels that do not contain spatial footprints. This is enforced by a diagonal
        mask matrix, D_{mask}. The i-th entry  is 0 if pixel i contains neural footprints, otherwise it is 1

        Returns:
            d_mask (torch.sparse_coo_tensor): (d1*d2, d1*d2) diagonal matrix represe
        """
        return self._support

    @support.setter
    def support(self, new_mask):
        """
        Args:
            new_mask (torch.tensor): Shape (d1*d2), index i is 0 if pixel i contains neural signal, otherwise it is 0
        """
        net_pixels = (self.shape[0]*self.shape[1])
        index_values = torch.arange(net_pixels, device=self.device, dtype=torch.long)
        self._support = torch.sparse_coo_tensor(torch.stack([index_values, index_values]), new_mask,
                                                (net_pixels, net_pixels)).coalesce()

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


    def apply_model_right(self, tensor: Union[torch.tensor, torch.sparse_coo_tensor]) -> Union[torch.tensor, torch.sparse_coo_tensor]:
        """
        Applies the model, d_weights @ Ring @ d_mask @ tensor

        Returns sparse or dense tensor
        """
        output = torch.sparse.mm(self.support, tensor)
        output = torch.sparse.mm(self.ring_mat, output)
        return  torch.sparse.mm(self.weights, output)

    def apply_model_left(self, tensor):
        """
        Applies the model, tensor @ d_weights @ Ring @ d_mask

        Returns sparse or dense tensor
        """
        output = torch.sparse.mm(self.weights, tensor.T)
        output = torch.sparse.mm(self.ring_mat.T, output)
        output = torch.sparse.mm(self.support, output)
        return output.T

    def compute_fluctuating_background_row(self, U_sparse, R, s, V, active_weights, b, row_index):
        """
        TODO: Remove this
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
        W_selected = torch.index_select(self.ring_mat, 0, row_index_tensor).coalesce().to_dense().float()

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
        self.weights = torch.zeros((self.shape[0]*self.shape[1]), device=self.device)

    def reset_weights(self):
        self.weights = torch.ones((self.shape[0]*self.shape[1]), device=self.device)

    def export_to_scipy(self) -> scipy.sparse.coo_matrix:
        """
        Temporary code to export the ring model to a scipy sparse matrix
        """
        output = torch.sparse.mm(self.weights, self.ring_mat)
        output = torch.sparse.mm(output, self.support)
        return ca_utils.torch_sparse_to_scipy_coo(output)

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

        W_residual = W.apply_model_right(resid_V_basis)

        denominator += torch.sum(W_residual * W_residual, dim=1)
        numerator += torch.sum(W_residual * resid_V_basis, dim=1)

    values = torch.nan_to_num(numerator / denominator, nan=0.0, posinf=0.0, neginf=0.0)
    threshold_function = torch.nn.ReLU()
    values = threshold_function(values)
    W.weights = values