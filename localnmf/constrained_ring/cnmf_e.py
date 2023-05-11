import copy
import math
import time
import torch

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import os

import torch_sparse
######
###
######


 
class ring_model:
    
    def __init__(self, d1, d2, r, empty=False, device='cpu', order="F"):
        #If empty, construct an empty W matrix
        if empty:
            row = torch.Tensor([]).to(device).long()
            col = torch.Tensor([]).to(device).long()
            value = torch.Tensor([]).to(device).bool()
            self.W_mat = torch_sparse.tensor.SparseTensor(row = row, col = col, value=value, sparse_sizes=(d1*d2, d1*d2))
            self.weights = torch.zeros((d1*d2, 1), device=device)
            self.empty=True
        else:
            row_coordinates, column_coordinates, values = self._construct_init_values(d1, d2, r, device=device, order=order)
            torch.cuda.empty_cache()
            self.W_mat = torch_sparse.tensor.SparseTensor(row=row_coordinates, col=column_coordinates, value=values, sparse_sizes = (d1*d2, d1*d2))
            self.weights = torch.ones((d1*d2, 1), device=device)
            self.empty=False

    


    def _construct_init_values(self, d1, d2, r, device='cuda', order="F"):
        a, b = torch.meshgrid((torch.arange(d1, device=device), torch.arange(d2, device=device)), indexing='ij')
        dim1_spread = torch.arange(-(r+1), (r+2), device=device)
        dim2_spread = torch.arange(-(r+1), (r+2), device=device)

        spr1, spr2 = torch.meshgrid((dim1_spread, dim2_spread ), indexing='ij')
        norms = torch.sqrt(spr1*spr1 + spr2 * spr2)
        outputs = torch.argwhere(torch.logical_and(norms >= r, norms < r+1)).to(device)
        print("number of elts in ring is {}".format(outputs.shape[0]))

        ring_dim1 = dim1_spread[outputs[:, 0]].to(device)
        ring_dim2 = dim2_spread[outputs[:, 1]].to(device)

        dim1_full = a[:, :, None] + ring_dim1[None, None, :]
        dim2_full = b[:, :, None] + ring_dim2[None, None, :]

        values = torch.ones_like(dim1_full, device=device).bool()#float()#.long()

        good_components = torch.logical_and(dim1_full >= 0, dim1_full < d1)
        good_components = torch.logical_and(good_components, dim2_full >= 0)
        good_components = torch.logical_and(good_components, dim2_full < d2)
        print("the max of good components is {}".format(good_components.max()))

        dim1_full *= good_components
        dim2_full *= good_components
        values *= good_components

        if order == "C":
            column_coordinates = d2*dim1_full + dim2_full
            del dim1_full
            del dim2_full
            row_coordinates = torch.flatten((d2*a+b)[:, :, None] + torch.zeros([1, 1, column_coordinates.shape[2]], device=device)).long()

            column_coordinates = torch.flatten(column_coordinates).long()
            values = torch.flatten(values).bool()

        elif order == "F":
            column_coordinates = dim1_full + d1*dim2_full
            del dim1_full
            del dim2_full
            row_coordinates = torch.flatten((a + d1*b)[:, :, None]+ torch.zeros([1, 1, column_coordinates.shape[2]], device=device)).long()

            column_coordinates = torch.flatten(column_coordinates).long()
            values = torch.flatten(values).bool()

        else:
            raise ValueError("Order must be row-major (C) or column-major (F)")

        good_entries = values>0
        row_coordinates = row_coordinates[good_entries]
        column_coordinates = column_coordinates[good_entries]
        values = values[good_entries]

        return row_coordinates, column_coordinates, values
    
    
    def set_weights(self, tensor):
        '''
        Tensor should have shape (d1*d2, 1)
        '''
        self.weights = tensor
        
    def apply_model_right(self, tensor, a_sparse):
        '''
        Computes ring_matrix times tensor. 
        Inputs:
            tensor: torch.Tensor of dimensions (d1*d2, X) for some X
            a_sparse: torch_sparse.tensor. Dimensions (d1*d2, K) where K is the number of neurons described by a_sparse,  and 
                d1, d2 are the FOV dimensions of the imaging video. 
        '''
        device=tensor.device
        a_sum_vec = torch.ones((a_sparse.sparse_sizes()[1], 1), device=device)
        a_sum = torch_sparse.matmul(a_sparse, a_sum_vec)
        good_indices = (a_sum == 0).bool()
        
        return self.apply_weighted_ring_right(tensor, good_indices=good_indices)
    
    def apply_model_left(self, tensor, a_sparse):
        '''
        Computes tensor times ring_matrix.
        Inputs: 
            tensor: torch.Tensor of dimensions (X, d1*d2) for some X
            a_sparse: torch_sparse.tensor. Dimensions (d1*d2, K) where K is the number of neurons described by a_sparse, and 
                d1, d2 are the FOV dimensions of the imaging video. 
        '''
        device=tensor.device
        a_sum_vec = torch.ones((a_sparse.sparse_sizes()[1], 1), device=device)
        a_sum = torch_sparse.matmul(a_sparse, a_sum_vec)
        good_indices = (a_sum == 0).bool()

        return self.apply_weighted_ring_left(tensor, good_indices=good_indices)
    
    def apply_weighted_ring_right(self, tensor, good_indices=None):
        '''
        Multiplies weighted_ring_matrix by tensor. Note that tensor must have (d1*d2) rows (since ring_matrix is (d1*d2, d1*d2)
        Inputs: 
            tensor. torch.Tensor. Shape (d1*d2, X) for some X
            good_indices. torch.Tensor, dtype bool. Shape (d1*d2, 1). Describes which columns of the ring matrix must be zero'd out.
        
        '''
        if good_indices is not None:
            tensor = good_indices * tensor
        return self.weights * torch_sparse.matmul(self.W_mat, tensor)
    
    def apply_weighted_ring_left(self, tensor, good_indices=None):
        tensor_t = self.weights*tensor.t()
        product = torch_sparse.matmul(self.W_mat.t(), tensor_t)
        
        if good_indices is not None:
            return (good_indices * product).t()
        else:
            return product.t()
        
    def compute_fluctuating_background_row(self, U_sparse, R, V, a, b, row_index):
        '''
        Computes a specific row of W(URV - ac - b), which is the fluctuating background. 
        Recall  we enforce that column i of W contains all zeros whenever row 'i' of a contains nonzero entries. So Wa will always be zero, and the above expression becomes: 
        W(URV - b)
        Parameters: 
            U_sparse. torch_sparse Tensor. Dimensions (d1*d2, d1*d2)
            R: torch.Tensor. Dimensions (r x r)
            V: torch.Tensor. Dimensions (r x T) 
            a_sparse: numpy.ndarray. Dimensions (d1*d2, K) where K is the number of neural shapes
            b: numpy.ndarray. Dimensions (d1*d2, 1)
            
        TODO: Splitting between torch and numpy here is bad design, long term fix this
        '''
        device = R.device
        b_torch = torch.Tensor(b).float().to(device)
        
        a_support = np.sum(a, axis = 1) == 0
        row_index_tensor = torch.Tensor([row_index]).to(device).long()
        W_selected = torch_sparse.index_select(self.W_mat, 0, row_index_tensor).float()
        
        #Apply unweighted ring model to W(URV - b)
        sparsity_selector_matrix = scipy.sparse.diags(a_support.astype("float32"), shape = (U_sparse.sparse_sizes()[0], U_sparse.sparse_sizes()[0]))
        sparsity_selector_matrix = torch_sparse.tensor.from_scipy(sparsity_selector_matrix).to(device)
        sparser_U = torch_sparse.matmul(sparsity_selector_matrix, U_sparse)
        sparse_WU = torch_sparse.matmul(W_selected, sparser_U)
        sparse_WUR = torch_sparse.matmul(sparse_WU, R)
        WURV = torch.matmul(sparse_WUR, V)
        
        sparser_b = torch_sparse.matmul(sparsity_selector_matrix, b_torch)
        Wb = torch_sparse.matmul(W_selected, sparser_b)
        
        difference = WURV - Wb
        
        #Apply weight at end
        weighted_row = self.weights[row_index_tensor] * difference
        return weighted_row
        
        
    
    def zero_weights(self):
        self.weights = self.weights * 0
        
    def reset_weights(self):
        self.weights = torch.ones_like(self.weights)
        
    def create_complete_ring_matrix(self, a):
        '''
        Constructs a complete W matrix, combining the ring weights and the zero'd out columns into a single scipy sparse csr matrix.
        Input: 
            a: np.ndarray. Dimensions (d, K) where d is the number of pixels in FOV and K is number of neurons identified
        Output: 
            W_plain: The ring matrix with the weights (and zero'd out columns) applied
        '''
        W_plain = self.W_mat.to_scipy(layout='csr')
        W_plain = W_plain.multiply(self.weights.cpu().numpy())
        
        a_sum = (np.sum(a, axis = 1, keepdims=True) == 0).T
        W_plain = W_plain.multiply(a_sum)
        
        return W_plain
        
        
        
        

def get_sampled_indices(num_frames, num_samples, device='cuda'):
    num_samples = min(num_samples, num_frames)
    tensor_to_sample = torch.arange(num_frames, device=device)
    weights = torch.ones_like(tensor_to_sample, device=device) * (1/num_frames)
    new_values = tensor_to_sample[torch.multinomial(weights, num_samples=num_samples, replacement=False)]
    
    return new_values

    
    
def ring_model_update_sampling(U_sparse, V, W, c, b, a, d1, d2, num_samples=1000, device='cuda'):
    
    sampled_indices = get_sampled_indices(V.shape[1], num_samples, device=device)
    V_crop = torch.index_select(V, 1, sampled_indices)
    c_crop = torch.index_select(c, 0, sampled_indices).t()
    
    
    residual = torch_sparse.matmul(U_sparse, V_crop) - torch_sparse.matmul(a, c_crop) - b
    
    W.reset_weights()
    W_residual = W.apply_model_right(residual, a)
    
    denominator = torch.sum(W_residual * W_residual, dim=1)
    numerator = torch.sum(W_residual * residual, dim=1)
    
    values = torch.nan_to_num(numerator/denominator, nan=0.0, posinf=0.0, neginf=0.0)
    threshold_function = torch.nn.ReLU()
    values = threshold_function(values)
    
    W.set_weights(values[:, None])
   

def ring_model_update(U_sparse, R, V, W, X, b, a, d1, d2, num_samples=1000, device='cuda'):
    
    batches = math.ceil(R.shape[1] / num_samples)
    denominator = 0
    numerator = 0
    W.reset_weights()
    
    sV = torch.ones([1, V.shape[1]], device=device)
    s = torch.matmul(sV, V.t())
    
    for k in range(batches):

        start = num_samples * k
        end = min(R.shape[1], start + num_samples)
        R_crop = R[:, start:end]
        X_crop = X[:, start:end]
        s_crop = s[:, start:end]


        resid_V_basis = torch_sparse.matmul(U_sparse, R_crop) - torch_sparse.matmul(a, X_crop) - torch.matmul(b, s_crop)

        
        W_residual = W.apply_model_right(resid_V_basis, a)

        denominator += torch.sum(W_residual * W_residual, dim=1)
        numerator += torch.sum(W_residual * resid_V_basis, dim=1)

    values = torch.nan_to_num(numerator/denominator, nan=0.0, posinf=0.0, neginf=0.0)
    threshold_function = torch.nn.ReLU()
    values = threshold_function(values)
    W.set_weights(values[:, None])