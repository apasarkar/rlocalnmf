import scipy.sparse
import numpy as np
import torch
import torch_sparse
import time


def fast_matmul(a, b, device='cuda'):
    a_torch = torch.from_numpy(a).float().to(device)
    b_torch = torch.from_numpy(b).float().to(device)
    return torch.mm(a_torch, b_torch).cpu().numpy()


def baseline_update(uv_mean, a, c, to_torch=False):
    '''
    Calculates baseline. Inputs: 
        uv_mean. torch.Tensor of shape (d, 1), where d is the number of pixels in the FOV
        a: torch_sparse tensor of shape (d, k) where k is the number of neurons in the FOV
        c: torch.Tensor of shape (T, k) where T is number of frames in video
        to_torch: indicates whether the inputs are np.ndarrays that need to be converted to torch objects. Also implies that the result will be returned as a np.ndarray (the same format as the inputs)
    Output: 
        b. torch.Tensor of shape (d, 1). Describes new static baseline
    '''
    if to_torch:
        a = torch_sparse.tensor.from_scipy(scipy.sparse.csr_matrix(a)).float()
        c = torch.from_numpy(c).float()
        uv_mean = torch.from_numpy(uv_mean).float()
    mean_c = torch.mean(c, dim=0, keepdim=True).t()
    b = uv_mean - torch_sparse.matmul(a, mean_c)
    
    if to_torch:
        return b.numpy()
    else:
        return b         
    
def project_U_HALS(U_sparse, R, W, vector, a_sparse):
    '''
    Commonly used routine to project background term (W) onto U basis in HALS calculations. 
    We exploit the fact that the product UR has orthonormal columns
    Note that we multiple W by v first (before doing any other matmul) because that collapses to a single vector. Then 
    all subsequent matmuls are very fast
    Inputs: 
        U_sparse: torch_sparse.SparseTensor object. Dimensions d x r where d is number of pixels, r is rank
        R: torch.Tensor object. Shape r x r' (where r may be equal to r'). UR is the orthonormal left basis term for the SVD of the PMD movie (recall the SVD decomposition is URsV where UR 
            is the left set of orthonormal col vectors, is the diagonal matrix, and V contains the right orthonormal row vectors. 
        W: ring model object. Represents a (d, d)-shaped sparse tensor
        vector: torch.Tensor. Dimensions (d, k) for some value k
        a_sparse. torch_sparse.Tensor object, with shape (d, k)
    '''
    Wv = W.apply_model_right(vector, a_sparse)
    UtWv = torch_sparse.matmul(U_sparse.t(), Wv)
    RRt = torch.matmul(R, R.t())
    RRtUtWv = torch.matmul(RRt, UtWv)
    final_projection = torch_sparse.matmul(U_sparse, RRtUtWv)
    
    return final_projection
    
def spatial_update_HALS(U_sparse, R, s, V, W, a_sparse, c, b, mask_ab = None):
    '''
    Computes a spatial HALS updates: 
    Params: 
        
        Note: The first four parameters are the "PMD" representation of the data: it is given in a traditional SVD form: URsV, where UR is the left orthogonal basis, 's' represents the diagonal matrix, and V is the right orthogonal basis. 
        U_sparse: torch_sparse.tensor. Sparse matrix, with dimensions (d x R)
        R: torch.Tensor. Dimensions (R, R') where R' is roughly equal to R (it may be equal to R+1)
        s: torch.Tensor. This is the diagonal of a R' x R' matrix (so it is represented by a R' shaped tensor) 
        V: torch.Tensor. Dimensions R' x T. Dimensions R' x T, where T is the number of frames, where all rows are orthonormal. 
            Note:  V must contain the 1 x T vector of all 1's in its rowspan. 
        W: ring_model object. Describes a sparse matrix with dimensions (d x d)
        a_sparse: torch_sparse.tensor. dimensions d x k, where k represents the number of neural signals. 
        c: torch.Tensor. Dimensions T x k
        b: torch.Tensor. Dimensions d x 1. Represents static background
        mask_ab: torch_sparse.tensor. Dimensions (k x d). For each neuron, indicates the allowed support of neuron
        
    Returns: 
        a_sparse: torch_sparse.tensor. Dimensions d x k, containing updated spatial matrix
        
    TODO: Make 'a' input a sparse matrix
    '''
    #Load all values onto device in torch
    device = V.device
    
    if mask_ab is None: 
        mask_ab = a_sparse.bool().t()
        
    Vc = torch.matmul(V, c) 
    a_dense = a_sparse.to_dense()
        
    #Find the tensor, e, (a 1 x R' shaped tensor) such that eV gives a 1 x T tensor consisting of all 1's
    e = torch.matmul(torch.ones([1, V.shape[1]], device=device), V.t())
    
    #Find the tensor, X (a N x K shaped tensor) such that XV closely approximates c.T
    X = Vc.t() #This is just a linear subspace projection of c.T onto the rowspace of V; can reuse above computation for this
    
    C_prime = torch.matmul(c.t(), c)
    C_prime_diag = torch.diag(C_prime)
    C_prime_diag[C_prime_diag == 0] = 1 # For division safety
    
    ctVtUt_net = torch.zeros((c.shape[1], U_sparse.sparse_sizes()[0]), device=device)
    
    '''
    We will now compute the TRANSPOSE of the following expression: 
    
    [URs - be - Proj(W)(URS -be - aX)]Vc
    
    The result (after the transpose) is a N x d sized matrix, where N is the number of neural signals
    '''
    cumulator = torch.zeros([U_sparse.sparse_sizes()[0], c.shape[1]], device=device)
    
    URsVc = torch_sparse.matmul(U_sparse, torch.matmul(R, s[:, None] * Vc))
    beVc = torch.matmul(b, torch.matmul(e, Vc))
    aXVc = torch.matmul(a_dense, torch.matmul(X, Vc))
    
    ring_term = project_U_HALS(U_sparse, R, W, URsVc - beVc - aXVc, a_sparse)
    cumulator = URsVc - beVc - ring_term
    
    cumulator = cumulator.t()

    threshold_func = torch.nn.ReLU(0)
    for i in range(c.shape[1]):
              
    
        index_select_tensor = torch.arange(i, i+1, device=device)
        mask_ab_torchsparse_sub = torch_sparse.index_select(mask_ab, 0,\
                                                            index_select_tensor)
        ind_torch = mask_ab_torchsparse_sub.storage.col()
        mask_apply = torch.zeros([U_sparse.sparse_sizes()[0]], device=device)
        mask_apply[ind_torch] = 1

        C_prime_i = C_prime.index_select(0, index_select_tensor).t()
        cumulator_i = cumulator.index_select(0, index_select_tensor)
        cca = torch.matmul(a_dense, C_prime_i).t()
        final_vec = (cumulator_i - cca)/C_prime_diag[i]
        curr_frame = a_dense[:, i]
        curr_frame += torch.squeeze(final_vec)
        curr_frame *= mask_apply
        curr_frame = threshold_func(curr_frame)
        a_dense[:, i] = curr_frame

        
    nonzero_indices = torch.nonzero(a_dense)
    rows = nonzero_indices[:, 0]
    cols = nonzero_indices[:, 1]
    values = a_dense[rows,cols]
    
    a_sparse = torch_sparse.tensor.SparseTensor(row=rows, col=cols, value=values, \
                                                    sparse_sizes = a_sparse.storage.sparse_sizes()).coalesce()
        
        
    return a_sparse   
 

    
# def left_project_U_HALS(a_sparse, a_dense, U_sparse, U_sparse_inverse, W):
#     atU = torch_sparse.matmul(U_sparse.t(), a_dense).t()
#     atU_Uinv = torch.matmul(atU, U_sparse_inverse)
#     atU_UinvUt = torch_sparse.matmul(U_sparse, atU_Uinv.t()).t()
#     final = W.apply_model_left(atU_UinvUt, a_sparse)
#     return final
    
    
##Compute the projection matrix:
def get_projection_matrix_temporal_HALS_routine(U_sparse, R, W, a_dense, a_sparse):
    aU = torch_sparse.matmul(U_sparse.t(), a_dense).t()
    aUR = torch.matmul(aU, R)
    aURRt = torch.matmul(aUR, R.t())
    aURRtUt = torch_sparse.matmul(U_sparse, aURRt.t()).t() #Shape here is N x d
    projector = W.apply_model_left(aURRtUt, a_sparse)
    
    return projector
    
    
def temporal_update_HALS(U_sparse, R, s, V,  W, a_sparse, c, b):
    '''
    Inputs: 
         Note: The first four parameters are the "PMD" representation of the data: it is given in a traditional SVD form: URsV, where UR is the left orthogonal basis, 's' represents the diagonal matrix, and V is the right orthogonal basis. 
        U_sparse: torch_sparse.tensor. Sparse matrix, with dimensions (d x R)
        R: torch.Tensor. Dimensions (R, R') where R' is roughly equal to R (it may be equal to R+1)
        s: torch.Tensor. This is the diagonal of a R' x R' matrix (so it is represented by a R' shaped tensor) 
        V: torch.Tensor. Dimensions R' x T. Dimensions R' x T, where T is the number of frames, where all rows are orthonormal. 
            Note:  V must contain the 1 x T vector of all 1's in its rowspan. 
        W: (d1*d2, d1*d2)-shaped ring model object
        a: (d1*d2, k)-shaped torch_sparse.tensor
        c: (T, k)-shaped torch.Tensor
        b: (d1*d2, 1)-shaped torch.Tensor
        
    Returns: 
        c: (T, k)-shaped np.ndarray. Updated temporal components
    '''
    device = V.device
    
    
    ##Precompute quantities used throughout all iterations
    atU_net = torch.zeros((a_sparse.sparse_sizes()[1], V.shape[0]), device=device)
    a_dense = a_sparse.to_dense()
    
    #Find the tensor, e, (a 1 x R' shaped tensor) such that eV gives a 1 x T tensor consisting of all 1's
    e = torch.matmul(torch.ones([1, V.shape[1]], device=device), V.t())
    
    #Find the tensor, X (a N x K shaped tensor) such that XV closely approximates c.T
    X = torch.matmul(V, c).t() #This is just a linear subspace projection of c.T onto the rowspace of V; can reuse above computation for this
    
    #Step 1: Get aTURs
    aTU = torch_sparse.matmul(U_sparse.t(), a_dense).t()
    aTUR = torch.matmul(aTU, R)
    aTURs = aTUR * s[None, :]
    
    #Step 2: Get aTbe
    aTb = torch.matmul(a_dense.t(), b)
    aTbe = torch.matmul(aTb, e)
    
    #Step 3:
    projector = get_projection_matrix_temporal_HALS_routine(U_sparse, R, W, a_dense, a_sparse)
    PU = torch_sparse.matmul(U_sparse.t(), projector.t()).t()
    PUR = torch.matmul(PU, R)
    PURs = PUR * s[None, :]
    
    PA = torch.matmul(projector, a_dense)
    PAX = torch.matmul(PA, X)
    
    Pb = torch.matmul(projector, b)
    Pbe = torch.matmul(Pb, e)
    
    cumulator = aTURs - aTbe - (PURs - PAX - Pbe)
    cumulator = torch.matmul(cumulator, V)
    
    ata = torch.matmul(a_dense.t(), a_dense) #This is faster than a sparse-sparse matrix product (not well-parallelized as of May 2023 on GPU)
    
    ata_d = torch.diag(ata)
    ata_d[ata_d == 0] = 1 #For division-safety
    
    threshold_function = torch.nn.ReLU()
    for i in range(c.shape[1]):
        curr_index = torch.arange(i, i+1, device=device)
        a_ia = torch.index_select(ata, 0, curr_index)
        a_iaC = torch.matmul(a_ia, c.t())
        
        curr_trace = torch.index_select(c, 1, curr_index)
        curr_trace += ((torch.index_select(cumulator, 0, curr_index) - a_iaC)/ata_d[i]).t()
        curr_trace = threshold_function(curr_trace)
        c[:, [i]] = curr_trace
        
    return c