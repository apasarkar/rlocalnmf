import scipy.sparse
import numpy as np
import time


def baseline_update(uv_mean, a, c):
    '''
    Function for performing baseline updates
    
    '''
    
    b = uv_mean-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)
    return b
          


def estimate_X(c, V, VVt):
    '''
    Function for finding an approximate solution to c^t = XV
    Params:
        c: np.ndarray, dimensions (T, k)
        V: np.ndarray, dimensions (R, T)
        VVt: np.ndarrray, dimensions (R, R)
    Returns:
        X: np.ndarray, dimensionis (k x R)
    '''
    cV = c.T.dot(V.T) #Output: k x R matrix
    x, _, _, _ = np.linalg.lstsq(VVt.T,cV.T)
    return x.T



def spatial_update_HALS(U_sparse, V, W, X, a, c, b, mask_ab = None):
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
    ##TODO for further speedup: do not repeatedly construct a sparse array
    
    #Precompute relevant quantities
    start_time = time.time()
    C_prime = c.T.dot(c) #Ouput: k x k matrix (low-rank)
    cV = c.T.dot(V.T) #Output: k x R matrix (low-rank)
    cVX = cV.dot(X.T)
    
    a_sparse = scipy.sparse.csr_matrix(a)
    
    if mask_ab is None:
        mask_ab = (a > 0).T
    print("mask_ab done at {}".format(time.time() - start_time))
    #Init s such that bsV = static background
    s = np.zeros((1, V.shape[0]))
    s[:, -1] = -1 

    
    for i in range(c.shape[1]):
        
        #Identify positive support of a_i
        ind = (mask_ab[i, :] > 0)
        in_time = time.time()
        #In this notation, * refers to matrix multiplication
        
        '''
        Step 1: Find (c^t)_i * V^t * U^t, where
        U = U_PMD - W*(U_PMD - a*X - b*s) - b*s
        '''
        #(1) First compute (c^t)_i * V^t * (U_PMD)^t
        cVU = U_sparse.dot(cV[i, ].T).T #1 x d vector
#         if print_text:
#             print("1 done at {}".format(time.time() - in_time))
        
        #(2) Get static bg component: (c^t)_i * V^t * (s^t * b^t)
        bg = cV[i, :].dot(s.T)
        bg = bg.dot(b.T)  #Output: 1 x d vector
        
        
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
#         cVX = cV[i, :].dot(X.T)
        cVXa = (a_sparse.dot(cVX[i, :].T)).T
        
        #Add (a) - (b) - (c) 
        W_sum = cVU - cVXa - bg
        W_temp = W[ind, :]
        W_term = (W_temp.dot(W_sum.T)).T
        
        #Final step: get (c^t)_i * c * a^t
        cca = (a_sparse.dot(C_prime[i, :].T)).T        
        final_vec = (cVU - bg - cca)/C_prime[i, i]
                
        #Crop final_vec
        final_vec = final_vec.T
        final_vec = final_vec[ind]
        final_vec -= W_term/C_prime[i,i]
       
       
        
        a_sparse = a_sparse.tocoo()
        values = final_vec # final_vec[ind]
        rows = np.argwhere(ind>0)
        col = [i for k in range(len(values))]
        a_sparse = a_sparse.tocoo()
        a_sparse.data = np.append(a_sparse.data, values)
        a_sparse.row = np.append(a_sparse.row, rows)
        a_sparse.col = np.append(a_sparse.col, col)
        a_sparse = a_sparse.tocsr()
        
        ##TODO: make this faster: 
        a_sparse[a_sparse < 0] = 0
        
        
        
    return a_sparse
    
    
def temporal_update_HALS(U_sparse, V, W, X, a, c, b):
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
    '''
    
    
    #Initialize c
    c_new = c
    
    #Initialize a_sparse
    a_sparse = scipy.sparse.csr_matrix(a)
    
    #Precompute some transposes to avoid repeatedly taking transposes
    U_sparse_t = U_sparse.transpose().tocsr()
    W_t = W.transpose().tocsr()
    a_sparse_t = a_sparse.transpose().tocsr()
    
    aW_prod = a_sparse_t * W_t
    aW_prod = aW_prod.tocsr()
    
    aU_prod = a_sparse_t * U_sparse
    aU_prod = aU_prod.tocsr()
    #Precompute a^t * a
    A_prime = (a_sparse_t * a_sparse).toarray() 
    
    #Init s such that bsV = static background
    s = np.zeros((1, V.shape[0]))
    s[:, -1] = -1 
    for i in range(c.shape[1]):
        
        #For all this notation, * denotes matrix multiplication
        
        
        
        #This is (a_i)^t * a  * c^t
        aac = A_prime[i, ].dot(c.T)
        
        #Now we calculate (a_i)^t * U, where
        # U = U_PMD - W * (U_PMD - a * X - b * s) - b * s
        
        #(1) First find (a_i)^t * U_PMD
#         aU = (U_sparse_t.dot(a_sparse_t[i, :].T)).T
        aU = aU_prod[i, :]
        
        #(2) Find (a_i)^t * b * s
        ab = a_sparse_t[i, :].dot(b)
        ab_s = ab.dot(s)
        
        ##Now find (a_i)^t * W(U_PMD - a * X - b * s)
        
        #For each term, we always need to compute (a_i)^t * W 
        h_i = aW_prod[i, :]
        
        #(3a) Find (a_i)^t * W * U_PMD
        aWU = ((U_sparse_t.dot(h_i.T)).T).toarray()
#         print("type of aWU is {}".format(type(aWU)))
        
        #(3b) Find (a_i)^t * W * a * X
        #Find (a_i)^t * W
        aWa = (a_sparse_t.dot(h_i.T)).transpose()
        aWaX = aWa.dot(X) 
        
        #(3c) (a_i)^t * W * b * s
        aWb = h_i.dot(b)
        aWbs = aWb.dot(s)
        
        
        final_aU = (aU - ab_s - (aWU - aWaX - aWbs))
        final_aUV = final_aU.dot(V)
        
        
        c_new[:, [i]] += ((final_aUV - aac)/A_prime[i,i]).T
        
        out = c_new[:, [i]]
        out[out<0] = 0
        c_new[:, [i]] = out
        

    return c_new