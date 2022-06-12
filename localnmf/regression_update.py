import scipy.sparse
import numpy as np
import torch
import torch_sparse
import time


def fast_matmul(a, b, device='cuda'):
    a_torch = torch.from_numpy(a).float().to(device)
    b_torch = torch.from_numpy(b).float().to(device)
    return torch.mm(a_torch, b_torch).cpu().numpy()


def baseline_update(uv_mean, a, c):
    '''
    Function for performing baseline updates
    
    '''
    
    b = uv_mean-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)
    return b
          



def estimate_X(c, V, VVt, device='cpu'):
    '''
    Function for finding an approximate solution to c^t = XV
    Params:
        c: np.ndarray, dimensions (T, k)
        V: np.ndarray, dimensions (R, T)
        VVt: np.ndarrray, dimensions (R, R)
        device: device on which computations occur ('cuda', 'cpu', etc.) 
    Returns:
        X: np.ndarray, dimensionis (k x R)
    '''

    c_torch = torch.from_numpy(c).float().to(device)
    V_torch = torch.from_numpy(V).float().to(device)
    cV_T = torch.matmul(V_torch, c_torch)
    
    VVt_torch = torch.from_numpy(VVt).float().to(device)
    output = torch.linalg.lstsq(VVt_torch, cV_T).solution
    
    return output.cpu().numpy().T

def sparse_coo_threshold(a, threshold=0):
    data = a.data
    rows = a.row
    cols = a.col
    good_indices = data >= threshold
    
    data_new = data[good_indices]
    rows_new = a.row[good_indices]
    cols_new = a.col[good_indices]
    
    a.data = data_new
    a.row = rows_new
    a.col = cols_new
    return a

# def spatial_update_HALS(U_sparse, V, W, X, a, c, b, device='cpu', mask_ab = None):
#     '''
#     Computes a temporal HALS updates: 
#     Params: 
#         U_sparse: scipy.sparse.coo matrix. Sparse U matrix, dimensions d x R 
#         V: PMD V matrix, dimensions R x T
#             V has as its last row a vector of all -1's
#         X: Approximate solution to c^t = XV. Dimensions k x R (k neurons in a/c)
#         a_sparse: scipy.sparse.csr_matrix. dimensions d x k
#         c: np.ndarray. dimensions T x k
#         b: np.ndarray. dimensions d x 1. represents static background
#         mask_a: np.ndarray. dimensions (k x d). For each neuron, indicates the allowed support of neuron
        
#     TODO: Make 'a' input a sparse matrix
#     '''
#     ##TODO for further speedup: do not repeatedly construct a sparse array
    
#     #Precompute relevant quantities
#     start_time = time.time()
#     # C_prime = c.T.dot(c) #Ouput: k x k matrix (low-rank)
#     C_prime = fast_matmul(c.T, c, device=device) #Output: k x k matrix (low-rank)
#     cV = fast_matmul(c.T, V.T, device=device)
#     # cV = c.T.dot(V.T) #Output: k x R matrix (low-rank)
#     cVX = fast_matmul(cV, X.T, device=device)
#     # cVX = cV.dot(X.T)
    
#     a_sparse = scipy.sparse.csr_matrix(a)
    
#     if mask_ab is None:
#         mask_ab = (a > 0).T
#     print("mask_ab done at {}".format(time.time() - start_time))
#     #Init s such that bsV = static background
#     s = np.zeros((1, V.shape[0]))
#     s[:, -1] = -1 

    
#     for i in range(c.shape[1]):
        
#         #Identify positive support of a_i
#         ind = (mask_ab[i, :] > 0)
#         in_time = time.time()
#         #In this notation, * refers to matrix multiplication
        
#         '''
#         Step 1: Find (c^t)_i * V^t * U^t, where
#         U = U_PMD - W*(U_PMD - a*X - b*s) - b*s
#         '''
#         #(1) First compute (c^t)_i * V^t * (U_PMD)^t
#         cVU = U_sparse.dot(cV[i, ].T).T #1 x d vector
# #         if print_text:
# #             print("1 done at {}".format(time.time() - in_time))
        
#         #(2) Get static bg component: (c^t)_i * V^t * (s^t * b^t)
#         bg = cV[i, :].dot(s.T)
#         bg = bg.dot(b.T)  #Output: 1 x d vector
        
        
#         '''
#         Step (3) Calculate (c^t)_i * V^t * (W * (U_PMD - a * X - b * s))^t
#         This is equal to (c^t)_i * V^t * (U_PMD - a * X - b * s)^t * W^t
#         we refer to (c^t)_i * V^t as h_i.
#         We need to calculate
#         (a) h_i * U_PMD^t
#         (b) h_i * X^t * a^t
#         (c) h_i * s^t * b^t
        
#         Note that (a) has already been computed above (cVU)
#         Also note (c) has been computed above (bg)
        
#         So all we need to compute is (b) 
        
#         These are all 1 x d vectors, so we add them, and then multiply by W^t to get our 
#         final result
#         '''
        
#         #Get (b)
# #         cVX = cV[i, :].dot(X.T)
#         cVXa = (a_sparse.dot(cVX[i, :].T)).T
        
#         #Add (a) - (b) - (c) 
#         W_sum = cVU - cVXa - bg
#         W_temp = W[ind, :]
#         W_term = (W_temp.dot(W_sum.T)).T
        
#         #Final step: get (c^t)_i * c * a^t
#         cca = (a_sparse.dot(C_prime[i, :].T)).T        
#         final_vec = (cVU - bg - cca)/C_prime[i, i]
                
#         #Crop final_vec
#         final_vec = final_vec.T
#         final_vec = final_vec[ind]
#         final_vec -= W_term/C_prime[i,i]
       
       
        
#         a_sparse = a_sparse.tocoo()
#         values = final_vec # final_vec[ind]
#         rows = np.argwhere(ind>0)
#         col = [i for k in range(len(values))]
#         a_sparse = a_sparse.tocoo()
        
#         a_sparse.data = np.append(a_sparse.data, values)
#         a_sparse.row = np.append(a_sparse.row, rows)
#         a_sparse.col = np.append(a_sparse.col, col)
        
#         a_sparse = sparse_coo_threshold(a_sparse, 0)
#         a_sparse = a_sparse.tocsr()
        
#         ##TODO: make this faster: 
#         # a_sparse[a_sparse < 0] = 0
        
        
        
#     return a_sparse
    
    
def scipy_coo_to_torchsparse_coo(scipy_coo_mat):
    values = scipy_coo_mat.data
    row = torch.LongTensor(scipy_coo_mat.row)
    col = torch.LongTensor(scipy_coo_mat.col)
    value = torch.FloatTensor(scipy_coo_mat.data)

    return torch_sparse.tensor.SparseTensor(row=row, col=col, value=value, sparse_sizes = scipy_coo_mat.shape)
    # return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    
def spatial_update_HALS(U_sparse, V, W, X, a, c, b, device='cpu', mask_ab = None):
    '''
    Computes a temporal HALS updates: 
    Params: 
        U_sparse: scipy.sparse.coo matrix. Sparse U matrix, dimensions d x R 
        V: PMD V matrix, dimensions R x T
            V has as its last row a vector of all -1's
        X: Approximate solution to c^t = XV. Dimensions k x R (k neurons in a/c)
        a_sparse: scipy.sparse.csr_matrix. dimensions d x k
        c: np.ndarray. dimensions T x k
        b: np.ndarray. dimensions d x 1. represents static background
        mask_a: np.ndarray. dimensions (k x d). For each neuron, indicates the allowed support of neuron
        
    TODO: Make 'a' input a sparse matrix
    '''
    #Load all values onto device in torch
    U_sparse = scipy_coo_to_torchsparse_coo(U_sparse.tocoo()).to(device)
    V = torch.from_numpy(V).float().to(device)
    W = scipy_coo_to_torchsparse_coo(W.tocoo()).to(device)
    X = torch.from_numpy(X).float().to(device)
    c = torch.from_numpy(c).float().to(device)
    b = torch.from_numpy(b).float().to(device)
    
    
    if mask_ab is None: 
        mask_ab = (a > 0).T
        mask_ab = scipy_coo_to_torchsparse_coo(mask_ab).to(device)
    else: 
        mask_ab = scipy.sparse.coo_matrix(mask_ab)
        mask_ab = scipy_coo_to_torchsparse_coo(mask_ab).to(device) 
        
        

    a = scipy.sparse.coo_matrix(a)
    a_sparse = scipy_coo_to_torchsparse_coo(a).to(device)
    
    #Init s such that bsV = static background
    s = torch.zeros([1, V.shape[0]], device=device)
    s[:, -1] = -1
    
    #Init s such that bsV = static background
    C_prime = torch.matmul(c.t(), c)
    cV = torch.matmul(V, c).t()
    cVX = torch.matmul(cV, X.t())
    
    index_select_tensor = torch.LongTensor([0]).to(device)
    
    for i in range(c.shape[1]):
        
        index_select_tensor[0] = i
        mask_ab_torchsparse_sub = torch_sparse.index_select(mask_ab, 0,\
                                                            index_select_tensor)
        ind_torch = mask_ab_torchsparse_sub.storage.col()

        
        #(1) First compute (c^t)_i * V^t * (U_PMD)^t
        cVU = torch_sparse.matmul(U_sparse, cV[[i], ].t()).t()
        bg = torch.matmul(cV[[i], :], s.t())
        bg = torch.matmul(bg, b.t())
        
        #(2) Get static bg component: (c^t)_i * V^t * (s^t * b^t)
        # bg = cV[[i], :].dot(s.t())
        # bg = bg.dot(b.T)  #Output: 1 x d vector

        '''
        Step (3) Calculate (c^t)_i * V^t * (W * (U_PMD - a * X - b * s))^t
        This is equal to (c^t)_i * V^t * (U_PMD - a * X - b * s)^t * W^t
        we refer to (c^t)_i * V^t as h_i.
        We need to calculate
        (a) h_i * U_PMD^t
        (b) h_i * X^t * a^t
        (c) h_i * s^t * b^t
        
        Note that (a) has already been computed above (cVU)
        Also note (c) has been computed above (bg)
        
        So all we need to compute is (b) 
        
        These are all 1 x d vectors, so we add them, and then multiply by W^t to get our 
        final result
        '''
        
        #Get (b)
        cVXa = torch_sparse.matmul(a_sparse, cVX[[i], :].t()).t()
        # cVXa = (a_sparse.dot(cVX[i, :].T)).T
        
        #Add (a) - (b) - (c) 
        W_sum = cVU - cVXa - bg
        W_temp = torch_sparse.index_select(W, 0, ind_torch) ##
        W_term = torch_sparse.matmul(W_temp, W_sum.t()).t()
        
        cca = torch_sparse.matmul(a_sparse, C_prime[[i], :].t()).t()
        final_vec = (cVU - bg - cca)/C_prime[i, i]

        
        #Crop final_vec
        final_vec = torch.squeeze(final_vec.t())
        final_vec = final_vec[ind_torch]

        final_vec = torch.sub(final_vec, torch.squeeze(W_term/C_prime[i,i]))

        
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
        
        
        
        
    return a_sparse.cpu()
    
def temporal_update_HALS(U_sparse, V, W, X, a, c, b, device='cpu'):
    '''
    Inputs: 
        U_sparse: (d1*d2, R)-shaped scipy.sparse.csr.csr_matrix.
        V: (R, T)-shaped np.ndarray
        W: (d1*d2, d1*d2)-shaped scipy.sparse.csr.csr_matrix.
        X: (k, R)-shaped np.ndarray.
        a: (d1*d2, k)-shaped np.ndarray
        c: (T, k)-shaped np.ndarray
        b: (d1*d2, 1)-shaped np.ndarray
        
    Returns: 
        c: (T, k)-shaped np.ndarray. Updated temporal components
        
        KEY ASSUMPTION: V has, as its last row, a vector where each component has value -1
    '''
    
    #Convert all inputs to torch and move to device
    U_sparse = torch_sparse.tensor.from_scipy(U_sparse).to(device)
    V = torch.from_numpy(V).float().to(device)
    W = torch_sparse.tensor.from_scipy(W).to(device)
    X = torch.from_numpy(X).float().to(device)
    c = torch.from_numpy(c).float().to(device)
    b = torch.from_numpy(b).float().to(device)

    a = scipy.sparse.coo_matrix(a)
    a_sparse = torch_sparse.tensor.from_scipy(a).to(device)
    
    #Init s such that bsV = static background
    s = torch.zeros([1, V.shape[0]], device=device)
    s[:, -1] = -1
    
    index_select_tensor = torch.LongTensor([0]).to(device)
    
    threshold_function = torch.nn.ReLU()
    for i in range(c.shape[1]):
        
        index_select_tensor[0] = i
        a_i = torch_sparse.index_select(a_sparse, 1, index_select_tensor)
        a_i_t = a_i.t()

        denominator = torch_sparse.matmul(a_i_t, a_i).to_dense()  
        denominator = denominator[0, 0]
        
        
        ##Step 1: Calculate a_i^t * U. NOTE: This is consistent
        a_iU = torch_sparse.matmul(a_i_t, U_sparse).to_dense()  
        
        ##Step 2: Calculate a_i^t*W*U
        
        a_iW = torch_sparse.matmul(a_i_t, W)
        a_iWU = torch_sparse.matmul(a_iW, U_sparse).to_dense()
        
        ##Step 3: Calculate a_i^tWaX
        a_iWa = torch_sparse.matmul(a_iW,a_sparse)
        a_iWaX = torch_sparse.matmul(a_iWa, X)
        
        ##Step 4: Calculate a_i^tWbs
        a_iWb = torch_sparse.matmul(a_iW, b)
        a_iWbs = torch.matmul(a_iWb, s)
        
        ##Step 5: Calculate a_i^t * b * s. NOTE: THIS WORKS
        a_ibs = torch.matmul(torch_sparse.matmul(a_i_t, b), s)
        
        ##Step 6: Calculate a_i^t * (U_PMD) * V
        a_i_t_U_V= torch.matmul(a_iU - (a_iWU - a_iWaX - a_iWbs) - a_ibs, V)

        
        ##Step 7: Calculate a_i^t * a * c^t. NOTE: THIS WORKS
        a_ia = torch_sparse.matmul(a_i_t, a_sparse)
        a_iaC = torch_sparse.matmul(a_ia, c.t())
        
        c[:, [i]] += ((a_i_t_U_V - a_iaC)/denominator).t()
        
        c[:, [i]] = threshold_function(c[:, [i]])
        
    return c.cpu().numpy()
        
        

        
    
# def temporal_update_HALS(U_sparse, V, W, X, a, c, b):
#     '''
#     Computes a temporal HALS updates: 
#     Params: 
#         U_sparse: scipy.sparse.coo matrix. Sparse U matrix, dimensions d x R 
#         V: PMD V matrix, dimensions R x T
#             V has as its last row a vector of all -1's
#         X: Approximate solution to c^t = XV. Dimensions k x R (k neurons in a/c)
#         a_sparse: scipy.sparse.csr_matrix. dimensions d x k
#         c: np.ndarray. dimensions T x k
#         b: np.ndarray. dimensions d x 1. represents static background 
#     '''
#     #Initialize c
#     c_new = c
    
#     #Initialize a_sparse
#     a_sparse = scipy.sparse.csr_matrix(a)
    
#     #Precompute some transposes to avoid repeatedly taking transposes
#     U_sparse_t = U_sparse.transpose().tocsr()
#     W_t = W.transpose().tocsr()
#     a_sparse_t = a_sparse.transpose().tocsr()
    
#     aW_prod = a_sparse_t * W_t
#     aW_prod = aW_prod.tocsr()
    
#     aU_prod = a_sparse_t * U_sparse
#     aU_prod = aU_prod.tocsr()
#     #Precompute a^t * a
#     A_prime = (a_sparse_t * a_sparse).toarray() 
    
#     #Init s such that bsV = static background
#     s = np.zeros((1, V.shape[0]))
#     s[:, -1] = -1 
#     for i in range(c.shape[1]):
        
#         #For all this notation, * denotes matrix multiplication
        
        
        
#         #This is (a_i)^t * a  * c^t
#         aac = A_prime[i, ].dot(c.T)
        
#         #Now we calculate (a_i)^t * U, where
#         # U = U_PMD - W * (U_PMD - a * X - b * s) - b * s
        
#         #(1) First find (a_i)^t * U_PMD
# #         aU = (U_sparse_t.dot(a_sparse_t[i, :].T)).T
#         aU = aU_prod[i, :]
        
#         #(2) Find (a_i)^t * b * s
#         ab = a_sparse_t[i, :].dot(b)
#         ab_s = ab.dot(s)
        
#         ##Now find (a_i)^t * W(U_PMD - a * X - b * s)
        
#         #For each term, we always need to compute (a_i)^t * W 
#         h_i = aW_prod[i, :]
        
#         #(3a) Find (a_i)^t * W * U_PMD
#         aWU = ((U_sparse_t.dot(h_i.T)).T).toarray()
# #         print("type of aWU is {}".format(type(aWU)))
        
#         #(3b) Find (a_i)^t * W * a * X
#         #Find (a_i)^t * W
#         aWa = (a_sparse_t.dot(h_i.T)).transpose()
#         aWaX = aWa.dot(X) 
        
#         #(3c) (a_i)^t * W * b * s
#         aWb = h_i.dot(b)
#         aWbs = aWb.dot(s)
        
        
#         final_aU = (aU - ab_s - (aWU - aWaX - aWbs))
#         final_aUV = final_aU.dot(V)
        
        
#         c_new[:, [i]] += ((final_aUV - aac)/A_prime[i,i]).T
        
#         out = c_new[:, [i]]
#         out[out<0] = 0
#         c_new[:, [i]] = out
        

#     return c_new