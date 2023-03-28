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

def estimate_X(c, V, VVt):
    '''
    Function for finding an approximate solution to c^t = XV
    Params:
        c: torch.Tensor, dimensions (T, k)
        V: torch.Tensor, dimensions (R, T)
        VVt: torch.Tensor, dimensions (R, R)
    Returns:
        X: torch.Tensor, dimensions (k x R)
    '''
    cV_T = torch.matmul(V, c)
    output = torch.linalg.lstsq(VVt, cV_T).solution
    
    return output.t()


    
def project_U_HALS(U_sparse, U_sparse_inverse, W, vector, a_sparse):
    '''
    Commonly used routine to project background term (W) onto W basis in HALS calculations
    Note that we multiple W by v first (before doing any other matmul) because that collapses to a single vector. Then 
    all subsequent matmuls are very fast
    Inputs: 
        U_sparse: torch_sparse.SparseTensor object. Dimensions d x r where d is number of pixels, r is rank
        U_sparse_inverse: torch.Tensor object. Equal to finding the inverse of U_sparse^t times U_sparse (precomputed)
        W: ring model object. Represents a (d, d)-shaped sparse tensor
        vector: torch.Tensor. Dimensions (d, k) for some value k.
    '''
    Wv = W.apply_model_right(vector, a_sparse)
    UtWv = torch_sparse.matmul(U_sparse.t(), Wv)
    U_invUtWv = torch.matmul(U_sparse_inverse, UtWv)
    UU_invUtWv = torch_sparse.matmul(U_sparse, U_invUtWv)
    
    return UU_invUtWv
    
def spatial_update_HALS(U_sparse, V, W, X, a_sparse, c, b, s, U_sparse_inner=None, mask_ab = None):
    '''
    Computes a spatial HALS updates: 
    Params: 
        U_sparse: torch_sparse.tensor. Sparse matrix, with dimensions (d x R)
        V: torch.Tensor. Dimensions R x T. Dimensions R x T
            V has as its last row a vector of all -1's
        W: ring_model object. Describes a sparse matrix with dimensions (d x d)
        X: torch.Tensor. Dimensions (k x R), with k neurons in a and c. Approximate solution to c^t = XV
        a_sparse: torch_sparse.tensor. dimensions d x k
        c: torch.Tensor. Dimensions T x k
        b: torch.Tensor. Dimensions d x 1. Represents static background
        U_sparse_inner: A pre-computed R x R matrix. It is the inverse of U_sparse^T times U_sparse. Used for linear subspace projection
        mask_ab: torch_sparse.tensor. Dimensions (k x d). For each neuron, indicates the allowed support of neuron
        
    Returns: 
        a_sparse: torch_sparse.tensor. Dimensions d x k, containing updated a_i elements
        
    TODO: Make 'a' input a sparse matrix
    '''
    #Load all values onto device in torch
    device = V.device
    
    if mask_ab is None: 
        mask_ab = a_sparse.bool().t()
        
    if U_sparse_inner is None:
        U_sparse_inner = torch.inverse(torch_sparse.matmul(U_sparse.t(), U_sparse).to_dense())
    
    #Init s such that bsV = static background
    # s = torch.zeros([1, V.shape[0]], device=device)
    # s[:, -1] = -1
    
    #Init s such that bsV = static background
    C_prime = torch.matmul(c.t(), c)
    
    ctVt = torch.matmul(V, c).t()
    
    ctVtUt_net = torch.zeros((c.shape[1], U_sparse.sparse_sizes()[0]), device=device)
    
    #Step 1: Compute ctVtU_PMD^t
    ctVtU_PMDt = torch_sparse.matmul(U_sparse, ctVt.t()).t()
    ctVtUt_net = ctVtUt_net + ctVtU_PMDt
    
    #Step 2: Compute ctVtU_PMD^t(Proj)^t
    # where Proj = U(U^tU)^-1U^tW, the linear projection of W onto U
    ctVtU_PMD_tWt = project_U_HALS(U_sparse, U_sparse_inner, W, ctVtU_PMDt.t(), a_sparse).t()
    ctVtUt_net -= ctVtU_PMD_tWt
    
    #Step 3: Compute ctVtXtat(Proj)^t
    # where Proj = U(U^tU)^-1U^tW, the linear projection of W onto U
    ctVtXt = torch.matmul(ctVt, X.t())
    ctVtXtat = torch_sparse.matmul(a_sparse, ctVtXt.t()).t()
    ctVtXtatWt = project_U_HALS(U_sparse, U_sparse_inner, W, ctVtXtat.t(), a_sparse).t()
    ctVtUt_net += ctVtXtatWt
    
    #Step 4: ctVtstbt(Proj)^t
    # where Proj = U(U^tU)^-1U^tW, the linear projection of W onto U
    Proj_Wb = project_U_HALS(U_sparse, U_sparse_inner, W, b, a_sparse).t()
    # btWt = torch_sparse.matmul(W, b).t()
    ctVtst = torch.matmul(ctVt, s.t())
    ctVtstbtWt = torch.matmul(ctVtst, Proj_Wb)
    ctVtUt_net += ctVtstbtWt
    
    #Step 5: ctVtstbt
    ctVtstbt = torch.matmul(ctVtst, b.t())
    ctVtUt_net -= ctVtstbt

    
    index_select_tensor = torch.LongTensor([0]).to(device)
    for i in range(c.shape[1]):
              
            
        index_select_tensor[0] = i
        mask_ab_torchsparse_sub = torch_sparse.index_select(mask_ab, 0,\
                                                            index_select_tensor)
        ind_torch = mask_ab_torchsparse_sub.storage.col()

        
        cca = torch_sparse.matmul(a_sparse, C_prime[[i], :].t()).t()
        final_vec = (ctVtUt_net[[i], :] - cca)/C_prime[i, i]

        
        #Crop final_vec
        final_vec = torch.squeeze(final_vec.t())
        final_vec = final_vec[ind_torch]

        
        values = final_vec # final_vec[ind]
        rows = ind_torch
        col = torch.ones_like(rows)*i
        
        original_values = a_sparse.storage.value()
        original_rows = a_sparse.storage.row()
        original_cols = a_sparse.storage.col()
        
        
        new_values = torch.cat((original_values, values))
        
        threshold_func = torch.nn.ReLU(0)
        new_values = threshold_func(new_values)
        
        new_rows = torch.cat((original_rows, rows))
        new_cols = torch.cat((original_cols, col))
        
        a_sparse = torch_sparse.tensor.SparseTensor(row=new_rows, col=new_cols, value=new_values, \
                                                    sparse_sizes = a_sparse.storage.sparse_sizes()).coalesce()
        
        
        
        
    return a_sparse   
 
    
def left_project_U_HALS(a_sparse, U_sparse, U_sparse_inverse, W):
    atU = torch_sparse.matmul(a_sparse.t(), U_sparse)
    atU_Uinv = torch_sparse.matmul(atU, U_sparse_inverse)
    atU_UinvUt = torch_sparse.matmul(U_sparse, atU_Uinv.t()).t()
    final = W.apply_model_left(atU_UinvUt, a_sparse)
    return final
    
    
def temporal_update_HALS(U_sparse, V, W, X, a_sparse, c, b, s, U_sparse_inner=None):
    '''
    Inputs: 
        U_sparse: (d1*d2, R)-shaped torch_sparse.tensor
        V: (R, T)-shaped torch.Tensor
        W: (d1*d2, d1*d2)-shaped ring model object
        X: (k, R)-shaped torch.Tensor
        a: (d1*d2, k)-shaped torch_sparse.tensor
        c: (T, k)-shaped torch.Tensor
        b: (d1*d2, 1)-shaped torch.Tensor
        s: (1, R)-shaped torch.Tensor. Has the property that sV is the 1 x T vector of all 1's
        U_sparse_inner: (R, R)-shaped torch.Tensor
        
        
    Returns: 
        c: (T, k)-shaped np.ndarray. Updated temporal components
        
        KEY ASSUMPTION: V has, as its last row, a vector where each component has value -1
    '''
    device = V.device
    
    if U_sparse_inner is None:
        U_sparse_inner = torch.inverse(torch_sparse.matmul(U_sparse.t(), U_sparse).to_dense())

    
    #Init s such that bsV = static background
    # s = torch.zeros([1, V.shape[0]], device=device)
    # s[:, -1] = -1
    
    
    ##Precompute quantities used throughout all iterations
    atU_net = torch.zeros((a_sparse.sparse_sizes()[1], V.shape[0]), device=device)
    
    #Step 1: Add atU
    atU_net += torch_sparse.matmul(a_sparse.t(), U_sparse).to_dense()
    
    #Step 2: Subtract atWU
    # atW = torch_sparse.matmul(a_sparse.t(), W).to_dense()
    atW = left_project_U_HALS(a_sparse, U_sparse, U_sparse_inner, W) 
    atWU = torch_sparse.matmul(U_sparse.t(), atW.t()).t()
    atU_net -= atWU
    
    #Step 3: Add atWaX
    aX = torch_sparse.matmul(a_sparse, X)
    atWaX = torch.matmul(atW, aX)
    atU_net += atWaX
    
    #Step 4: Add atWbs
    atWb = torch.matmul(atW, b)
    atU_net += torch.matmul(atWb, s)
    
    #Step 5: Subtract atbs
    atb = torch_sparse.matmul(a_sparse.t(), b)
    atbs = torch.matmul(atb, s)
    atU_net -= atbs
    
    atU_net_V = torch.matmul(atU_net, V)
    
    
    ata = torch_sparse.matmul(a_sparse.t(), a_sparse).to_dense()
    
    threshold_function = torch.nn.ReLU()
    for i in range(c.shape[1]):
        a_ia = ata[[i], :]
        a_iaC = torch.matmul(a_ia, c.t())
        
        c[:, [i]] += ((atU_net_V[[i], :] - a_iaC)/ata[i, i]).t()
        
        c[:, [i]] = threshold_function(c[:, [i]])
        
    return c