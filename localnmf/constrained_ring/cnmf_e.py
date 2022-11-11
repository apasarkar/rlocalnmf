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
        else:
            row_coordinates, column_coordinates, values = self._construct_init_values(d1, d2, r, device=device, order=order)
            torch.cuda.empty_cache()
            self.W_mat = torch_sparse.tensor.SparseTensor(row=row_coordinates, col=column_coordinates, value=values, sparse_sizes = (d1*d2, d1*d2))
            self.weights = torch.ones((d1*d2, 1), device=device)

    


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
        W_plain = self.W_mat.to_scipy().tocsr()
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

    
    
def ring_model_update(U_sparse, V, W, c, b, a, d1, d2, num_samples=1000, device='cuda'):
    
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
    
    W.set_weights(values)
    
    

    
    



        
# def update_ring_model_w_const(U_sparse, R, V, A, X, b, W, d1, d2, T, r, mask_a=None, device = 'cpu', batch_size = 10000):
#     """Update W matrix using ring model
#     :param U: spatial component matrix from denoiser, R(d=d1xd2, N)
#     :param V: temporal component matrix from denoiser, R(N, T). 
#         MUST BE ORTHOGONAL: V(V^t) = I
#     :param A: spatial component matrix, R(d, K)
#     :param X: temporal component matrix (the actual temporal matrix C = XV),  R(K, N)
#     :param b: constant baseline, dimension d x 1
#     :param W: weighting matrix, scipy.sparse.csr_matrix. Dimensions (d, d)
#     :param d1: x axis frame size
#     :param d2: y axis frame size
#     :param T: number to time steps along time axis
#     :param r: ring radius of ring model
#     :return:
#         W: updated weighting matrix
#         b0: constant baseline of background image
#     """
#     print("DEVICE USED ON CONST W UPDATE IS {}".format(device))
    
#     start_time = time.time()
#     if W is None:
#         W = init_w(d1, d2, r)
#     A_sparse = scipy.sparse.coo_matrix(A)
    
#     if mask_a is None:
#         A_sum = np.sum(A, axis=1)
#     else:
#         A_sum = np.sum(mask_a, axis=1)
    
#     print("A_sum at {}".format(time.time() - start_time))
    
#     W = update_w_1p_const(U_sparse.tocoo(), R, V, W, X, b, A_sparse, A_sum, d1, d2, batch_size = batch_size, device=device)
#     return W
 
    
    
    
# def update_w_1p_const(U_sparse, R, V, W, X, b, A_sparse, A_sum, d1, d2, batch_size = 10000, num_samples=1000, device='cpu'):
#     """Constant Ring Model codebase 
#     params:
#         U_sparse: scipy.sparse.csr_matrix. Dimensions d x r
#         R: np.ndarray. Dimensions r x r
#         V: np.ndarray. Dimensions r x T
#         W: scipy.sparse.csr_matrix. Dimensions d x d
#         A_sparse: scipy.sparse.csr_matrix. Dimensions d x k (k neurons)
#         A_sum: the "summed" A onto 1 plane
#         d1: x axis frame size
#         d2: y axis frame size
#         batch_size: number of ring indices to simultaneously update
#     return:
#         W: Sparse matrix (d x d) representing ring weights
    
#     This is the CONST pipeline
#     """
#     first_time = time.time()
#     d = d1 * d2
#     weights = np.zeros((d, 1))

#     iters = math.ceil(d / batch_size)
    
#     start_time = time.time()
#     #Preprocess W
#     rows, cols, values = (W.row, W.col, W.data)
#     indices = (values > 0)
#     rows = rows[indices]
#     cols = cols[indices]
#     values[indices] = 1
#     values = values[indices]
    
#     #Turn off pixels in columns in which neurons are active
#     support = (A_sum > 0)
#     intersection = support[cols]
#     inter_keep = (intersection == 0)
    
#     rows = rows[inter_keep]
#     cols = cols[inter_keep]
#     values = values[inter_keep]
    
#     W = coo_matrix((values, (rows, cols)), shape = (d,d))
    
#     sampled_indices = np.random.choice(V.shape[1], size=num_samples, replace=False)
#     V_crop = torch.from_numpy(V[:, sampled_indices]).float().to(device)
#     X_torch = torch.from_numpy(X).float().to(device)
#     U_sparse_torch = torch_sparse.tensor.from_scipy(U_sparse).float().to(device)
#     A_sparse_torch = torch_sparse.tensor.from_scipy(A_sparse).float().to(device)
#     R = torch.from_numpy(R).float().to(device)
#     b_torch = torch.from_numpy(b).float().to(device)
    
#     RV = torch.matmul(R, V_crop)
#     URV = torch_sparse.matmul(U_sparse_torch, RV)
    
#     XV = torch.matmul(X_torch, V_crop)
#     AXV = torch_sparse.matmul(A_sparse_torch, XV)
    
#     R_movie = URV - AXV - b_torch
    
#     W_torch = torch_sparse.tensor.from_scipy(W).float().to(device)
#     WR_movie = torch_sparse.matmul(W_torch, R_movie)
    
#     denominator = torch.sum(WR_movie * WR_movie, dim=1)
#     numerator = torch.sum(WR_movie * R_movie, dim=1)
    
#     values = torch.nan_to_num(numerator/denominator, nan=0.0, posinf=0.0, neginf=0.0)
#     threshold_function = torch.nn.ReLU()
#     values = threshold_function(values)
    
#     weights = values.cpu().numpy()
    
#     #Create new CSR matrix   
#     pos = W.nonzero()
#     rows = pos[0]
#     values = weights[rows]
#     W = csr_matrix((values.squeeze(), (pos[0], pos[1])), shape = (d, d))
    
#     print("we are done with W update. took {}".format(time.time() - first_time))
#     return W.tocoo()



# def update_w_1p_const_orig(U_sparse, R, V, W, X, b, A_sparse, A_sum, d1, d2, batch_size = 10000):
#     """Constant Ring Model codebase 
#     params:
#         U_sparse: scipy.sparse.csr_matrix. Dimensions d x r
#         R: np.ndarray. Dimensions r x r
#         V: np.ndarray. Dimensions r x T
#         W: scipy.sparse.csr_matrix. Dimensions d x d
#         A_sparse: scipy.sparse.csr_matrix. Dimensions d x k (k neurons)
#         A_sum: the "summed" A onto 1 plane
#         d1: x axis frame size
#         d2: y axis frame size
#         batch_size: number of ring indices to simultaneously update
#     return:
#         W: Sparse matrix (d x d) representing ring weights
    
#     This is the CONST pipeline
#     """
#     first_time = time.time()
#     d = d1 * d2
#     weights = np.zeros((d, 1))

#     iters = math.ceil(d / batch_size)
    
#     start_time = time.time()
#     #Preprocess W
#     print("WE ARE IN THE SPATIAL W")
#     W = W.tocoo()
#     rows, cols, values = (W.row, W.col, W.data)
#     indices = (values > 0)
#     rows = rows[indices]
#     cols = cols[indices]
#     values[indices] = 1
#     values = values[indices]
    
#     #Turn off pixels in columns in which neurons are active
#     support = (A_sum > 0)
#     intersection = support[cols]
#     inter_keep = (intersection == 0)
    
#     rows = rows[inter_keep]
#     cols = cols[inter_keep]
#     values = values[inter_keep]
    
#     W = coo_matrix((values, (rows, cols)), shape = (d,d))
#     W = W.tocsr() #Convert to csr for quick multiply
#     W.eliminate_zeros()
    
#     #Concisely represent 'b' using the V basis. We want to say static_bg_movie = bsV for some row vector s. 
#     # s should satisfy sV = 1 (the row vector of 1's)
#     s = np.ones((1, V.shape[1])).dot(V.T)

    
#     ##We want to calculate two diagonals: 
#     ## (1) diag(W*(UR - AX - bs)*(UR - AX - bs)^t)
#     ## (2) diag(W*(UR - AX - bs)*(UR - AX - bs)^t*W^t)
    
#     #Note that WA is 0 (because we exclude ring weights over pixels of support of A). So we really want to find: 
    
#     ## (1) T = diag(W*(UR - bs)*(UR - AX - bs)^t)
#     ## (2) P = diag(W*(UR - bs)*(UR - bs)^t*W^t)
    
#     ####NOTATION: rowsum(A, B) refers to the rowsum of the elementwise product of A and B
    
#     ##We start with objective 1, and abbreviate diag by d() 
#     ## diag(W*(UR - bs)*(UR - AX - bs)^t)
#     ## = d(W(UR)(UR)^t) - d(W(UR)(AX)^t) - d(W(UR)(bs)^t) - d(W(bs)(UR)^t) + d(W(bs)(AX)^t) - d(W(bs)(bs)^t)
    
#     T = np.zeros((d, 1))
#     #We calculate the first three terms: d(W(UR)(UR)^t) - d(W(UR)(AX)^t) - d(W(UR)(bs)^t)
    
#     #Precompute WU, which is sparse assuming W has ring structure, U is overlapping block-sparse
#     WU = W.dot(U_sparse)

    
#     ##Step 1.1: Find
#     #d(W(UR)(UR)^t) = rowsum(WU, U*R*R^t)
#     #d(W(UR)(AX)^t) = rowsum(WU, AXR^t)
    
#     RRt = R.dot(R.T)
#     XRt = X.dot(R.T)
#     diag_WURRU = np.zeros((d,1))
#     diag_WURAX = np.zeros((d,1))
#     for k in range(iters):
#         iter_time = time.time()
#         start = batch_size*(k)
#         end = start + batch_size
#         WU_crop = WU[start:end, :]
        
                
#         U_sparse_crop = U_sparse[start:end, :]
#         URRtt_crop = U_sparse_crop.dot(RRt)

#         WU_URRt = (WU_crop.multiply(URRtt_crop)).tocsr()


#         diag_WURRU[start:end, :] = WU_URRt.sum(1)  

        
#         AXRt_crop = A_sparse[start:end, :].dot(XRt)
#         WU_AXRt = (WU_crop.multiply(AXRt_crop)).tocsr()
#         diag_WURAX[start:end, :] = WU_AXRt.sum(1)
    
#     T += diag_WURRU
#     T -= diag_WURAX
    

    
#     #Step 1.2: Find d(W(UR)(bs)^t)
#     Rst = R.dot(s.T) #Dims r x 1
#     URst = U_sparse.dot(Rst) #Dims d x 1
#     WURst = W.dot(URst)
#     diag_WURbs = WURst * b
#     T -= diag_WURbs
    

    
#     #Step 1.3: Get d(W(bs)(UR)^t) and d(W(bs)(AX)^t)
#     ## d(W(bs)(UR)^t) = rowsum (Wb, URs^t)
#     ## d(W(bs)(AX)^t) = rowsum(Wb, AXs^t)
    
#     #Precompute Wb: 
#     Wb = W.dot(b) 
#     diag_WbsUR = Wb * URst
    
#     T -= diag_WbsUR
    
#     Xst = X.dot(s.T)
#     AXst = A_sparse.dot(Xst)
    
#     diag_WbsAX = Wb * AXst
    
#     T += diag_WbsAX
    

    
#     #Step 1.4: Add diag((bs)(bs)^t)
#     T += Wb * ((s.dot(s.T))*b)
    

    

    
#     ###Second objective: compute P = diag(W*(UR - bs)*(UR - bs)^t*W^t)
#     P = np.zeros((d,1))
#     ##Four terms to calculate..
    
#     ##Step 2.1: Find diag(W(UR)(UR)^tW^t) = rowsum(WU, WURR^t)
#     diag_WURURW = np.zeros((d,1))
#     for k in range(iters):
#         start = batch_size*(k)
#         end = start + batch_size
        
#         WU_crop = WU[start:end, :]
#         WURRt_crop = WU_crop.dot(RRt)
#         WURUR_prod = WU_crop.multiply(WURRt_crop)
#         diag_WURURW[start:end, :] = WURUR_prod.sum(1)
    
#     P += diag_WURURW
    

    
#     ##Step 2.2: Find diag(W(UR)(bs)^tW^t) = diag(W(bs)(UR)^tW^t) and subtract these from P
#     ## diag(W(UR)(bs)^tW^t) = diag(WURs^tb^tW^t) = rowsum(Wb, WURs^t)
#     diag_WbsURW = Wb*WURst
#     P -= 2*diag_WbsURW
    

    
#     ## Step 2.3: Find diag(W(bs)(bs)^tW^t)
#     ## = (ss^t)*rowsum(Wb, Wb)
#     diag_WbsbsW = s.dot(s.T) * Wb * Wb
#     P += diag_WbsbsW
    

    

    
    
#     values = T / P
#     values[values < 0] = 0
#     values = np.nan_to_num(values, nan = 0, posinf = 0)
        
        
           
#     #Add weights
#     weights = values
    
#     #Create new CSR matrix   
#     pos = W.nonzero()
#     rows = pos[0]
#     values = weights[rows]
#     W = csr_matrix((values.squeeze(), (pos[0], pos[1])), shape = (d, d))
    
#     print("we are done. took {}".format(time.time() - first_time))
#     return W.tocoo()
