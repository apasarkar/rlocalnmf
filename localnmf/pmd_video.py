import torch
import torch_sparse
import torchnmf
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from scipy import ndimage as ndimage
import scipy.stats as ss
import scipy.ndimage
import scipy.signal
import scipy.sparse
import scipy


from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import NMF
from sklearn import linear_model
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import distance_transform_bf
from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD
from matplotlib import ticker


from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import distance_transform_bf
from scipy.sparse import csc_matrix
from localnmf import ca_utils
from localnmf.ca_utils import add_1s_to_rowspan, denoise 
from localnmf.constrained_ring.cnmf_e import ring_model, ring_model_update, ring_model_update_sampling
from localnmf import regression_update
import time

### TODO: Create an actual API around this
# from localnmf.video_objects import FunctionalVideo

from localnmf.constrained_ring.cnmf_e import ring_model, ring_model_update, ring_model_update_sampling

def make_mask_dynamic(corr_img_all_r, corr_percent, mask_a, data_order = "C"):
    """
    update the spatial support: connected region in corr_img(corr(Y,c)) which is connected with previous spatial support
    """
    s = np.ones([3,3]);
    mask_a = (mask_a.reshape(corr_img_all_r.shape,order=data_order)).copy()
    print("entering thresholding..")
    for ii in range(mask_a.shape[2]):
        max_corr_val = np.amax(mask_a[:,:,ii]*corr_img_all_r[:, :, ii])
        corr_thres = corr_percent * max_corr_val
        labeled_array, num_features = scipy.ndimage.measurements.label(corr_img_all_r[:,:,ii] > corr_thres,structure=s);
        u, indices, counts = np.unique(labeled_array*mask_a[:,:,ii], return_inverse=True, return_counts=True);
        
        if len(u)==1:
            mask_a[:, :, ii] *= 0
        else:
            c = u[1:][np.argmax(counts[1:])];
            labeled_array = (labeled_array==c);
            mask_a[:,:,ii] = labeled_array;

    return mask_a.reshape((-1, mask_a.shape[2]), order = data_order)


def vcorrcoef_resid(U_sparse, R, V, a_sparse, c_orig, batch_size = 10000, tol = 0.000001):
    '''
    Residual correlation image calculation. Expectation is that there are at least two neurons (otherwise residual corr image is not meaningful)
    Params:
        U_sparse: torch_sparse.Tensor. Dimensions (d x K)
        R: torch.Tensor. Dimensions (K x K)
        V: numpy.ndarray. Dimensions (K x T). V has orthonormal rows; i.e. V.dot(V.T) is identity
            The row of 1's must belong in the row space of V
        a_sparse: torch_sparse.Tensor. Dimensions (d, N) (N = number of neurons)
        c_orig: numpy.ndaray. Dimensions (T x k)
        batch_size: number of pixels to process at once. Limits matrix sizes to O((batch_size+T)*r)
    '''
    assert c_orig.shape[1] > 1, "Need at least 2 components to meaningfully calculate residual corr image"
    
    device=R.device
    T = V.shape[1]
    d = U_sparse.sparse_sizes()[0]
    
    corr_img = torch.zeros((d, a_sparse.sparse_sizes()[1]), device=device)
    X = torch.matmul(V, c_orig).float().t()
    
    #Step 1: Standardize c
    c = c_orig - torch.mean(c_orig, dim=0, keepdim=True)
    c_norm = torch.sqrt(torch.sum(c*c, dim=0, keepdim=True))
    c /= c_norm
    
    a_dense = a_sparse.to_dense()
    
    #Define number of iterations so we only process batch_size pixels at a time
    batch_iters = math.ceil(d / batch_size)

    V_mean = torch.mean(V, dim=1, keepdim=True)
    
    ##Step 2: For each iteration below, we will express the 'consant'-mean movie in terms of V basis: Mean_Movie = m*s*V for some (1 x r) vector s. We know sV should be a row vector of 1's. So we solve sV = 1; since V is orthogonal:
    s = torch.matmul(V, torch.ones([V.shape[1], 1], device=device)).t()
    

    diag_URRtUt = torch.zeros([U_sparse.sparse_sizes()[0], 1], device=device)
    diag_AXXtAt = torch.zeros((d, 1), device=device)
    diag_AXUR = torch.zeros((d,1), device=device)
    
    batch_iters = math.ceil(R.shape[1] / batch_size)
    for k in range(batch_iters):
        start = batch_size * k
        end = min(batch_size * (k+1), R.shape[1])
        indices = torch.arange(start, end, device=device)
        R_crop = torch.index_select(R, 1, indices)
        X_crop = torch.index_select(X, 1, indices)
        UR_crop = torch_sparse.matmul(U_sparse, R_crop)
        AX_crop = torch.matmul(a_dense, X_crop)
        
        diag_URRtUt += torch.sum(UR_crop * UR_crop, dim=1, keepdim=True)
        diag_AXXtAt += torch.sum(AX_crop * AX_crop, dim=1, keepdim=True)
        diag_AXUR += torch.sum(AX_crop * UR_crop, dim=1, keepdim=True)
       
    ##Get mean of the movie: (UR - A_k * X_k)V
    RV_mean = torch.matmul(R, V_mean)
    m_UR = torch_sparse.matmul(U_sparse, RV_mean)
    
    # neuron_ind_torch = torch.arange(0, a_sparse.sparse_sizes()[1], step=1, device=device)
    threshold_func = torch.nn.ReLU()
    for k in range(c.shape[1]):
        current_ind = torch.arange(k, k+1, device=device)
        c_curr = torch.index_select(c, 1, current_ind)
        a_curr = torch.index_select(a_dense, 1, current_ind)
        X_curr = torch.index_select(X, 0, current_ind)

        #Step 1: Update the mean 
        m_AX = torch.matmul(a_dense, torch.matmul(X, V_mean))
        m_AX_curr = torch.matmul(a_curr, torch.matmul(X_curr, V_mean))
        m = m_UR - m_AX + m_AX_curr

        ##Step 2: Get square of norm of mean-subtracted movie. Let A_k denote k-th neuron of A and X_k denote k-th row of X
        ## We have: Y_res = (UR - AX + A_k * X_k - ms)V. 
        ## Square of norm is equivalent to diag(Y_res * Y_res^t). Abbreviate diag() by d()
        ## = d((UR)(UR)^t) - d((AX)(UR)^t) + d((A_k*X_k)(UR)^t) - d(ms(UR)^t)
        ##     - d(UR(AX)^t) + d(AX(AX)^t) - d((A_kX_k)(A_kX_k)^t) + d((ms)(AX)^t)
        ##     + d((UR)(A_kX_k)^t) - d((AX)((A_kX_k)^t)) + d(((A_kX_k)(A_kX_k)^t) - d((ms)(A_kX_k)^t)
        ##     - d(UR(ms)^t) + d((AX)(ms)^t) - d((A_kX_k)(ms)^t) + d((ms)(ms)^t)
        
        ##We find the diagonal of each term individually, and add the results.
        
        final_diagonal = torch.zeros((d, 1), device=device)
        
        ## Step 2.a Add/subtract precomputed values, d((UR)(UR)^t) and  d(AX(AX)^t) to final_diagonal
        final_diagonal += diag_URRtUt
        final_diagonal += diag_AXXtAt
        final_diagonal -= 2*diag_AXUR
        
        ## Step 2.b. Add diag(A_k * X_k (UR)^t) + diag(UR(A_k * X_k)^t) = 2*diag(URX_k^t * A_k^t) to final_diagonal
        RX_k = torch.matmul(R, X_curr.t())
        URX_k = torch_sparse.matmul(U_sparse, RX_k)
        diag_URAkXk = a_curr * URX_k
        final_diagonal += 2*diag_URAkXk

        
        
        ## Step 2.c. Subtract diag((AX)(A_k*X_k)^t) + diag((A_k*X_k)*(AX)^t) = 2*diag((A_k*X_k)*(AX)^t) from final diagonal
        XX_k = torch.matmul(X, X_curr.t())
        AXX_k = torch_sparse.matmul(a_sparse, XX_k) 
        diag_AXAkXk = a_curr * AXX_k
        final_diagonal -= 2*diag_AXAkXk

        
        ## 2.d. Subtract d(UR(ms)^t) + d(ms(UR)^t) = 2*d(URs^tm^t)
        
        Rst = torch.matmul(R, s.t())
        URst = torch_sparse.matmul(U_sparse, Rst)
        diag_URstm = URst*m
        final_diagonal -= 2*diag_URstm

        
        ## 2.e. Add d(AX(ms)^t) + d(ms(AX)^t) = 2*d(AXs^tm^t) to final_diagonal
        Xst = torch.matmul(X, s.t())
        AXst = torch_sparse.matmul(a_sparse, Xst)
        diag_AXstm = AXst * m
        final_diagonal += 2*diag_AXstm

        
        ## 2.f. Subtract d((A_kX_k)(ms)^t) + d((ms)(A_kX_k)^t) = 2*d(A_kX_ks^tm^t)
        Xkst = torch.matmul(X_curr, s.t())
        diag_akXkms = Xkst * (a_curr * m)
        final_diagonal -= 2* diag_akXkms

        ## 2.g. Add d((ms)(ms)^2)
        sst = torch.matmul(s, s.t())
        diag_msms = m*m*sst
        final_diagonal += diag_msms
        
        
        ## 2.h. Add d(((A_kX_k)(A_kX_k)^t)
        XkXk = torch.matmul(X_curr, X_curr.t())
        a_norm = a_curr * a_curr
        diag_axxa = (a_norm) * XkXk
        final_diagonal += diag_axxa

    
        norm = torch.sqrt(final_diagonal)
        norm = threshold_func(norm)

        #Find the unnormalized pixel-wise product, and normalize after..
        Vc = torch.matmul(V, c_curr)
        corr_fin = torch.zeros((d, 1), device=device)
        
        sVc = torch.matmul(s, Vc)
        msVc = torch.matmul(m, sVc)
        corr_fin -= msVc
        
        X_currVc = torch.matmul(X_curr, Vc)
        AX_currVc = torch.matmul(a_curr, X_currVc)
        corr_fin += AX_currVc
        
        XVc = torch.matmul(X, Vc)
        AXVc = torch.matmul(a_dense, XVc)
        corr_fin -= AXVc
        
        RVc = torch.matmul(R, Vc)
        URVc = torch_sparse.matmul(U_sparse, RVc)
        corr_fin += URVc
        
        corr_fin /= norm
        corr_fin = torch.nan_to_num(corr_fin, nan = 0, posinf = 0, neginf = 0)
        
        corr_img[:, [k]] = corr_fin
       
    return corr_img.cpu().numpy()

   


def vcorrcoef_UV_noise(U_sparse, R, V, c_orig, pseudo = 0, batch_size = 1000, tol = 0.000001, device='cpu'):
    '''
    New standard correlation calculation. Finds the correlation image of each neuron in 'c' 
    with the denoised movie URV
    
    TODO: Add robust statistic support (i.e. make pseudo do something) 
    Params:
        U_sparse: scipy.sparse.coo_matrix. dims (d x r), where the FOV has d pixels
        R: np.ndarray. dims (r x r), where r is the rank of the PMD decomposition
        V: np.ndarray. dims (r x T), where T is the number of frames in the movie
        c_orig: np.ndarray. dims (T x k), where k is the number of neurons
        pseudo: nonnegative integer
        batch_size: maximum number of pixels to process at a time. (batch_size x R)-sized matrices will be constructed
    '''
    T = V.shape[1]
    d = U_sparse.sparse_sizes()[0]
    
    #Load pytorch objects
    
    #Step 1: Standardize c
    c = c_orig - torch.mean(c_orig, dim=0, keepdim=True)
    c_norm = torch.sqrt(torch.sum(c*c, dim=0, keepdim=True))
    c /= c_norm 
    
    ##Step 2: 
    V_mean = torch.mean(V, dim=1, keepdim=True)
    RV_mean = torch.matmul(R, V_mean)
    m = torch_sparse.matmul(U_sparse, RV_mean) #Dims: d x 1
    
    ##Step 3: 
    s = torch.matmul(torch.ones([1, V.shape[1]], device=device), V.t())

    ##Step 4: Find the pixelwise norm: sqrt(diag((U*R - m*s)*V*V^t*(U*R - m*s)^t)) 
    ## diag((U*R - mov_mean*S)*V*V^t*(U*R - mov_mean*S)^t) = diag((U*R - m*s)*(U*R - m*s)^t) since V orthogonal
    ## diag((U*R - m*s)*(U*R - m*s)^t) = diag(U*R*R^t*U^t - U*R*s^t*m^t - m*s*R^t*U^t + m*s*s^t*m^t)
    ## diag(U*R*R^t*U^t - U*R*s^t*m^t - m*s*R^t*U^t + m*s*s^t*m^t) = diag(U*R*R^t*U^t) - diag(U*R*s^t*m^t) - diag(m*s*R^t*U^t) + diag(m*s*s^t*m^t)
    
    ##Step 4a: Get diag(U*R*s^t*m^t) and diag(m*s*R^t*U^t)
    #These are easy because U*R*s^t and s*R^t*U^t are 1-dimensional and transposes of each other: 

    Rst = torch.matmul(R, s.t())
    URst = torch_sparse.matmul(U_sparse, Rst)
    
    #Now diag(U*R*s^t*m^t) is easy:
    diag_URstmt = URst*m #Element-wise product

    #Now diag(m*s*R^t*U^t) is easy: 
    diag_msRtUt = m*URst

    ##Step 4b: Get diag(m*s*s^t*m^t)
    #Note that s*s^t just a dot product
    s_dot = torch.matmul(s, s.t())
    diag_msstmt = s_dot * (m*m)
    
    ## Step 4c: Get diag(U*R*R^t*U^t)
    diag_URRtUt = torch.zeros([U_sparse.sparse_sizes()[0], 1], device=device)

    batch_iters = math.ceil(d / batch_size)
    for k in range(batch_iters):
        start = batch_size * k
        end = min(batch_size * (k+1), U_sparse.sparse_sizes()[0])
        ind_torch = torch.arange(start, end, step=1, device=device)
        U_crop = torch_sparse.index_select(U_sparse, 0, ind_torch)
        UR_crop = torch_sparse.matmul(U_crop, R)
        UR_crop = UR_crop * UR_crop
        UR_crop = torch.sum(UR_crop, dim=1)
        diag_URRtUt[start:end, 0] =UR_crop   
           
    norm_sqrd = diag_URRtUt - diag_msRtUt - diag_URstmt + diag_msstmt
    norm = torch.sqrt(norm_sqrd)
    threshold_func = torch.nn.ReLU()
    norm = threshold_func(norm)

    
    #First precompute Vc: 
    Vc = torch.matmul(V, c)
    
    #Find (UR - ms)V*c
    RVc = torch.matmul(R, Vc)
    URVc = torch_sparse.matmul(U_sparse, RVc)
    
    sVc = torch.matmul(s, Vc)
    msVc = torch.matmul(m, sVc)
    
    fin_corr = URVc - msVc
    
    #Step 6: Divide by pixelwise norm from step 4
    fin_corr /= norm

    
    fin_corr = torch.nan_to_num(fin_corr, nan = 0, posinf = 0, neginf = 0)
    return fin_corr.cpu().numpy()    




def get_mean_data(U_sparse, R, s, V):
    '''
    Routine for computing the mean of the dataset. 
    Inputs: 
        (U_sparse, R, s, V) the SVD-like representation of the PMD data. Each of these are torch.Tensors of torch_sparse.Tensors. 
        U_sparse: torch_sparse.Tensor. shape (d, K)
        R: torch.Tensor. Shape (K, K)
        s: torch.Tensor. Shape (K)
        V: torch.Tensor. Shape (K, T)
    '''
    V_mean = torch.mean(V, dim=1, keepdim=True)
    V_mean = V_mean * s[:, None] 
    RsV_mean = torch.matmul(R, V_mean) 
    URsV_mean = torch_sparse.matmul(U_sparse, RsV_mean) 
    return URsV_mean

def get_new_orthonormal_vector(U_sparse, R, tol = 1e-6):
    '''
    Input: 
        U_sparse: torch_sparse.Tensor of shape (d, K), where d is the number of pixels in the movie, K is the rank of the PMD decomposition
        R: torch.Tensor of shape (K, K)
    
    Output: a torch.Tensor with shape (d, 1), and a flag indicating whether the search worked (1) or failed (0)
    '''
    device = R.device
    RRt = torch.matmul(R, R.t())
    
    num_tries = min(U_sparse.sparse_sizes()[0], 500)
    
    indices_to_try = np.random.choice(U_sparse.sparse_sizes()[0], size = num_tries, replace=False)
    indices_to_try[0] = 0
    indices_to_try[-1] = U_sparse.sparse_sizes()[0] - 1

    for k in indices_to_try:
        curr_index = torch.arange(k, k+1, device=R.device)
        current_vector = torch.zeros([U_sparse.sparse_sizes()[0], 1], device=device)
        current_vector[k, 0] = 1
        U_sparse_row = torch_sparse.index_select(U_sparse.t(), 1, curr_index)
        
        prod = torch_sparse.matmul(U_sparse_row.t(), RRt.t()).t()
        output = current_vector - torch_sparse.matmul(U_sparse, prod) 
        
        
        if not torch.all(torch.abs(output) < 1e-3):
            output = output / torch.linalg.norm(output)
            return output, 1
            
    
    return torch.zeros([U_sparse.sparse_sizes()[0], 1], device=device), 0



def PMD_setup_routine(U_sparse, R, s, V):
    '''
    Inputs: 
        U_sparse: torch_sparse.Tensor of shape (d, K), where d is the number of pixels in the movie, K is the rank of the PMD decomposition
        R: torch.Tensor of shape (K, K)
        s: torch.Tensor of shape (K). This describes a diagonal matrix (all other elts 0)
        V: torch.Tensor of shape (K, T) where R1 is the rank of the PMD decomposition
        
        Key: Conceptually, U_sparse*R*s*V (here, we don't multiply by "s" per se, we multiply by the diagonal matrix which "s" represents) is the expanded PMD decomposition
        
            
    Outputs:
        (U_sparse, R, s, V), potentially modified such that the rowspan of V contains the 1's vector. Here is what definitely stays the same --
            - U_sparse * R is a matrix consisting of orthonormal columns (here we operate under the assumption that U_sparse has many more rows than columns)
            - V has orthonormal rows
            - s describes the diagonal of an otherwise empty matrix 
    '''
    device = V.device
    pad_flag, V = add_1s_to_rowspan(V)
    if pad_flag: 
        
        new_vec, attempt_flag = get_new_orthonormal_vector(U_sparse, R)
        if not attempt_flag:
            print("The V row space enhancement step did not find a suitable orthogonal vector")
            pass
        else:
            nonzero_indices = torch.nonzero(new_vec)
            values = new_vec[nonzero_indices[:, 0], nonzero_indices[:, 1]]

            rows = nonzero_indices[:, 0]
            cols = torch.zeros_like(rows) + U_sparse.sparse_sizes()[1]


            #Add new_vec as a column to U_sparse (this should be easy)
            original_values = U_sparse.storage.value()
            original_rows = U_sparse.storage.row()
            original_cols = U_sparse.storage.col()

            new_values = torch.cat((original_values, values))
            new_rows = torch.cat((original_rows, rows))
            new_cols = torch.cat((original_cols, cols))

            U_sparse = torch_sparse.tensor.SparseTensor(row=new_rows, col=new_cols, value=new_values, \
                                                        sparse_sizes = (U_sparse.sparse_sizes()[0],U_sparse.sparse_sizes()[1]+1)).coalesce()

            #Add the appropriate padding to R
            R = torch.hstack([R, torch.zeros([R.shape[0], 1], device=device)])
            appendage = torch.zeros([1, R.shape[1]], device=device)
            appendage[0, -1] = 1
            R = torch.vstack([R, appendage])
            s = torch.cat([s, torch.zeros([1], device=device)])

        
    return U_sparse, R, s, V

def process_custom_signals(a_init, U_sparse, R, s, V, order="C"):
    '''
    Custom initialization: Given a set of neuron spatial footprints ('a'), this provides initial estimates to the other component (temporal traces, baseline, fluctuating background)
    Terms:
        d1, d2: the dimensions of the FOV
        K: number of neurons identified
        R: rank of the PMD decomposition
    
    Params:
        a_init: np.ndarray, dimensions (d1, d2, K)
        U_sparse_torch: torch_sparse.Tensor, shape (d1*d2, R)
        V_torch: torch.Tensor, shape (R, T)
        device: string; either 'cpu' or 'cuda'
        order: order in which 3d data is reshaped to 2d
    
    
    TODO: Eliminate awkward naming issues in 'process custom signals'
    '''
    device = V.device
    dims = (a_init.shape[0], a_init.shape[1], V_torch.shape[1])
    
    a = a_init.reshape(dims[0]*dims[1],-1, order=order)
    a_mask = (a_init>0).reshape(dims[0]*dims[1],-1,order=order)

    c = np.zeros((dims[2], a_init.shape[2]))
    b = np.zeros((dims[0]*dims[1], 1))

    #Cast the data to torch tensors 
    a = torch_sparse.tensor.from_scipy(scipy.sparse.coo_matrix(a)).float().to(device)
    c = torch.zeros([dims[2], a_init.shape[2]], device=device, dtype=torch.float)
    b = torch.zeros([dims[0]*dims[1], 1], device=device, dtype=torch.float)
    W = ring_model(dims[0], dims[1], 1, device=device, order=order, empty=True)
    
    uv_mean = get_mean_data(U_sparse_torch, R, s, V)

    #Baseline update followed by 'c' update:
    b_torch = regression_update.baseline_update(uv_mean, a_torch, c_torch)
    c_torch = regression_update.temporal_update_HALS(U_sparse, V, W, a, c, b)
    
    c_norm = torch.linalg.norm(c_torch, dim = 0)
    nonzero_dim1 = torch.nonzero(c_norm).squeeze()
    
    #Only keep the good indices, based on nonzero_dim1
    c_torch = torch.index_select(c_torch, 1, nonzero_dim1)
    a_torch = torch_sparse.index_select(a_torch, 1, nonzero_dim1)
    a_mask = a_torch.bool()
    
    return a_torch, a_mask, c_torch, b_torch,




def get_median(tensor, axis):
    max_val = torch.max(tensor, dim=axis, keepdim=True)[0]
    tensor_med_1 = torch.median(torch.cat((tensor, max_val), dim = axis), dim = axis, keepdim=True)[0]
    tensor_med_2 = torch.median(tensor, dim = axis, keepdim = True)[0]
    
    tensor_med = torch.mul(tensor_med_1 + tensor_med_2, 0.5)
    return tensor_med

def threshold_data_inplace(Yd, th = 2, axisVal = 2):
    '''
    Threshold data: in each pixel, compute the median and median absolute deviation (MAD),
    then zero all bins (x,t) such that Yd(x,t) < med(x) + th * MAD(x).  Default value of th is 2.
    Inputs: 
        Yd: torch.Tensor, shape (d1, d2, T)
    Outputs:
        Yd: This is an in-place operation 
    '''
    
    #Get per-pixel medians
    Yd_med = get_median(Yd, axis = axisVal)
    diff = torch.sub(Yd, Yd_med)
    
    #Calculate MAD values
    torch.abs(diff, out=diff)
    MAD = get_median(diff, axis = axisVal)
    
    #Calculate actual threshold
    torch.mul(MAD, th, out=MAD)
    th_val = Yd_med.add(MAD)  
    
    #Subtract threshold values
    torch.sub(Yd,th_val, out = Yd)
    torch.clamp(Yd, min = 0, out = Yd)
    return Yd


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))   

def reshape_c(x, shape):
    return torch.reshape(x, shape)
    

def get_total_edges(d1, d2):
    assert d1 > 2 and d2 > 2, "At least one dimensions is less than 2 pixels. Not supported" 
    overcount = 8*(d1-2)*(d2-2) + 2*(d1-2)*5 + 2*(d2-2)*5 + 4*3
    return (math.ceil(overcount / 2))

def get_local_correlation_structure(U_sparse, V, dims, th, order = "C", batch_size = 10000, pseudo=0, tol = 0.000001, a = None, c = None):
    '''
    Computes a local correlation data structure, which describes the correlations between all neighboring pairs of pixels
    
    Context: here, 
    d1, d2 are the fov dimensions of the original data (i.e. 512 x 512 pixels or the like) 
    T is the number of frames in the video
    R is the rank of the PMD decomposition (so U_sparse has shape (d1*d2, R) and V has shape (R, T)) 
    K is the number of neural signals identified (in "a" and "c", if they are provided)
    
    Inputs: 
        U_sparse: torch_sparse.Tensor object, shape (d1*d2, T)
        V: torch.Tensor, shape (R, T)
        dims: (d1, d2, T)
        th: int (positive integer), describes the MAD threshold. We use this to threshold the pixels for when we compute correlations. 
            We compute the median and median absolute deviation (MAD), then zero all bins (x,t) such that Yd(x,t) < med(x) + th * MAD(x). 
        order: "C" or "F" Indicates how we reshape the 2D images of the video (d1, d2) into (d1*d2) column vectors. The order here is important for consistency.
        batch_size: int. Maximum number of pixels of the movie that we fully expand out (i.e. we never have more than batch_size * T -sized Tensor in device memory.
            This is useful for GPU memory management, especially on small graphics cards.
        pseudo: float >= 0. a robust correlation parameter, used in the robust correlation calculation between every pair of neighboring pixels.
            In general, a higher value of pseudo will reduce the compute correlation between two pixels.
        tol: float. A tolerance parameter used when normalizing time series (to avoid divide by "close to 0" issues).
        (Optional) a: numpy.ndarray. A (d1*d2, K)-shaped ndarray whose columns describe the correlation structure of the data. 
        (Optional) c: numpy.ndarray. A (T, K)-shaped array whose columns describe the estimated fluorescence time course of each signal.
        
    Returns:
    The following correlation Data Structure: 
    To understand this, recall that we flatten the 2D field of view into a 1 dimensional column vector
        dim1_coordinates: torch.Tensor, 1 dimensional. Describes a list of row coordinates in the field of view 
        dim2_coordinates: torch.Tensor, 1 dimensional. Describes a list of row coordinates in the field of view
        correlations: torch.Tensor, 1 dimensional. 
        
    Key: each element at index i of "correlations" describes the computed correlation between the adjacent pixels given by 
            dim1_coordinates[i] and dim2_coordinates[i]
    '''
    
    device = V.device
    if a is not None and c is not None: 
        resid_flag = True
    else:
        resid_flag = False

    if resid_flag: 
        c = torch.Tensor(c).t().to(device)
        a_sparse = torch_sparse.tensor.from_scipy(scipy.sparse.coo_matrix(a)).to(device)
    

    dims = (dims[0], dims[1], V.shape[1])
    T = V.shape[1]
    
    ref_mat = torch.arange(np.prod(dims[:-1]), device=device)
    if order == "F":
        ref_mat = reshape_fortran(ref_mat, (dims[0], dims[1]))
    else:
        ref_mat = reshape_c(ref_mat, (dims[0], dims[1]))
    
        
    
    tiles = math.floor(math.sqrt(batch_size))
    
    iters_x = math.ceil((dims[0]/(tiles-1)))
    iters_y = math.ceil((dims[1]/(tiles-1)))
        
    
    
    #Pixel-to-pixel coordinates for highly-correlated neighbors
    total_edges = 2*get_total_edges(dims[0], dims[1])  #Here we multiply by two because when we tile the FOV, some correlations are computed twice
    point1_indices = torch.zeros((total_edges), dtype=torch.int32, device=device)
    point2_indices = torch.zeros((total_edges), dtype=torch.int32, device=device)
    correlation_values = torch.zeros((total_edges), dtype=torch.float32, device=device)
    
    
    progress_index = 0
    for tile_x in range(iters_x):
        for tile_y in range(iters_y):
            x_pt = (tiles-1)*tile_x
            x_end = x_pt + tiles
            y_pt = (tiles - 1)*tile_y
            y_end = y_pt + tiles
            
            indices_curr_2d = ref_mat[x_pt:x_end, y_pt:y_end]
            x_interval = indices_curr_2d.shape[0]
            y_interval = indices_curr_2d.shape[1]
            
            
            if order == "F":
                indices_curr = reshape_fortran(indices_curr_2d, (x_interval*y_interval,))
            else:
                indices_curr = reshape_c(indices_curr_2d, (x_interval*y_interval,))
                
            
            
            U_sparse_crop = torch_sparse.index_select(U_sparse, 0, indices_curr)
            if order == "F":
                Yd = reshape_fortran(torch_sparse.matmul(U_sparse_crop, V), (x_interval, y_interval, -1))
            else:
                Yd = reshape_c(torch_sparse.matmul(U_sparse_crop, V), (x_interval, y_interval, -1))
            if resid_flag:
                a_sparse_crop = torch_sparse.index_select(a_sparse, 0, indices_curr)
                if order == "F":
                    ac_mov = reshape_fortran(torch_sparse.matmul(a_sparse_crop, c), (x_interval, y_interval, -1))
                else:
                    ac_mov = reshape_c(torch_sparse.matmul(a_sparse_crop, c), (x_interval, y_interval, -1))
                Yd = torch.sub(Yd, ac_mov)
            
            #Get MAD-thresholded movie in-place
            Yd = threshold_data_inplace(Yd, th)
            
            #Permute the movie
            Yd = Yd.permute(2,0,1)
            
            #Normalize each trace in-place, using robust correlation statistic
            torch.sub(Yd, torch.mean(Yd, dim=0, keepdim = True), out = Yd)
            divisor = torch.std(Yd, dim = 0, unbiased = False, keepdim = True)
            final_divisor = torch.sqrt(divisor*divisor + pseudo**2)
            
            #If divisor is 0, that implies that the std of a 0-mean pixel is 0, whcih means the 
            #pixel is 0 everywhere. In this case, set divisor to 1, so Yd/divisor = 0, as expected
            final_divisor[divisor < tol] = 1  #Temporarily set all small values to 1..
            torch.reciprocal(final_divisor, out=final_divisor)
            final_divisor[divisor < tol] = 0  ##Now set these small values to 0
            
            torch.mul(Yd, final_divisor, out = Yd)
           
            #Vertical pixel correlations
            rho = torch.mean(Yd[:, :-1, :] * Yd[:, 1:, :], dim = 0)
            point1_curr = indices_curr_2d[:-1, :].flatten()
            point2_curr = indices_curr_2d[1:, :].flatten()
            rho_curr = rho.flatten()
            point1_indices[progress_index:progress_index+point1_curr.shape[0]] = point1_curr
            point2_indices[progress_index:progress_index+point1_curr.shape[0]] = point2_curr
            correlation_values[progress_index:progress_index+point1_curr.shape[0]] = rho_curr
            progress_index = progress_index + point1_curr.shape[0]
                        
            
            #Horizonal pixel correlations
            rho = torch.mean(Yd[:, :, :-1] * Yd[:, :, 1:], dim = 0)
            point1_curr = indices_curr_2d[:, :-1].flatten()
            point2_curr = indices_curr_2d[:, 1:].flatten()
            rho_curr = rho.flatten()
            point1_indices[progress_index:progress_index+point1_curr.shape[0]] = point1_curr
            point2_indices[progress_index:progress_index+point1_curr.shape[0]] = point2_curr
            correlation_values[progress_index:progress_index+point1_curr.shape[0]] = rho_curr
            progress_index = progress_index + point1_curr.shape[0]
            
            #Top left and bottom right diagonal correlations
            rho = torch.mean(Yd[:, :-1, :-1]*Yd[:, 1:, 1:], dim=0)
            point1_curr = indices_curr_2d[:-1, :-1].flatten()
            point2_curr = indices_curr_2d[1:, 1:].flatten()
            rho_curr = rho.flatten()
            point1_indices[progress_index:progress_index+point1_curr.shape[0]] = point1_curr
            point2_indices[progress_index:progress_index+point1_curr.shape[0]] = point2_curr
            correlation_values[progress_index:progress_index+point1_curr.shape[0]] = rho_curr
            progress_index = progress_index + point1_curr.shape[0]

            #Bottom left and top right diagonal correlations
            rho = torch.mean(Yd[:, 1:, :-1]*Yd[:, :-1, 1:], dim=0)
            point1_curr = indices_curr_2d[1:, :-1].flatten()
            point2_curr = indices_curr_2d[:-1, 1:].flatten()
            rho_curr = rho.flatten()
            point1_indices[progress_index:progress_index+point1_curr.shape[0]] = point1_curr
            point2_indices[progress_index:progress_index+point1_curr.shape[0]] = point2_curr
            correlation_values[progress_index:progress_index+point1_curr.shape[0]] = rho_curr
            progress_index = progress_index + point1_curr.shape[0]
                
    return point1_indices[:progress_index], point2_indices[:progress_index], correlation_values[:progress_index]

    

def find_superpixel_UV(dims, cut_off_point, length_cut, dim1_coordinates, dim2_coordinates, correlations, order):
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
    U_sparse: torch_sparse.Tensor object, shape (d1*d2, T)
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
        Each superpixel has a random number "indicator".  Same number means same superpixel.
    idx: double scalar
        number of superpixels
    comps: list, length = number of superpixels
        comp on comps is also list, its value is position of each superpixel in Yt_r = Yt.reshape(np.prod(dims[:2]),-1,order="F")
    permute_col: list, length = number of superpixels
        all the random numbers used to idicate superpixels in connect_mat_1
    """
    
    #Here we can apply the threshold: 
    good_indices = torch.where(correlations > cut_off_point)[0]
    A = torch.index_select(dim1_coordinates, 0, good_indices).cpu().numpy()
    B = torch.index_select(dim2_coordinates, 0, good_indices).cpu().numpy()
    values = torch.index_select(correlations, 0, good_indices).cpu().numpy()

    ########### form connected componnents #########
    G = nx.Graph();
    G.add_edges_from(list(zip(A, B)))
    comps=list(nx.connected_components(G))

    connect_mat=np.zeros(np.prod(dims[:2]));
    idx=0;
    for comp in comps:
        if(len(comp) > length_cut):
            idx = idx+1;

    np.random.seed(2) #Reproducibility of superpixels image
    permute_col = np.random.permutation(idx)+1;

    ii=0;
    for comp in comps:
        if(len(comp) > length_cut):
            connect_mat[list(comp)] = permute_col[ii];
            ii = ii+1;
    connect_mat_1 = connect_mat.reshape(dims[0],dims[1],order=order);
    return connect_mat_1, idx, comps, permute_col
  
    
##TODO: (1) Use URsV representation
def spatial_temporal_ini_UV(U_sparse, R, s, V, dims, th, comps, idx, length_cut, a = None, c = None):
    """
    Apply rank 1 NMF to find spatial and temporal initialization for each superpixel in Yt.
    Params:
        - U_sparse_torch: torch_sparse.Tensor, shape (d1*d2, R)
        - V_torch: torch.Tensor. Shape (R, T)
        - dims: tuple, contains (d1, d2, T) )
        - comps: list of sets. The i-th set here describes the spatial support of the i-th neural component
        - idx: int. number of components (in comps)
        - length_cut: int. Min size of superpixel
        Optional inputs: 
            - a: numpy.ndarray, dimensions (d1*d2, K)
            - c: numpy.ndarray, dimensions (T, K)
    Outputs: 
        - a_init: np.ndarray. Shape (d1*d2, K) where K is the total number of neurons AFTER this round of initialization
        - c_init: np.ndarray. Shape (T, K). Describes temporal initializations
    """
    device = V.device
    dims = (dims[0], dims[1], V.shape[1])
    T = V.shape[1]
    
    pre_existing = a is not None and c is not None
    if pre_existing:
        K = c.shape[1]
    else:
        K = 0

    total_length = 0
    good_indices = []
    index_val = 0
    for comp in comps:
        curr_length = len(list(comp))
        if curr_length > length_cut: 
            good_indices.append(index_val)
            total_length += curr_length 
        index_val += 1
    comps = [comps[good_indices[i]] for i in range(len(good_indices))]
    
    UV_mean = get_mean_data(U_sparse, R, s, V)
    if pre_existing: 
        c_orig = torch.from_numpy(c).to(device)
        c_final = torch.cat([c_orig, torch.zeros(T, len(comps), device=device)], dim=1)
        a_orig = torch_sparse.tensor.from_scipy(scipy.sparse.coo_matrix(a)).to(device)
        mean_ac = torch_sparse.matmul(a_orig, torch.mean(c_orig.t(), dim=1, keepdim=True))
        UV_mean = UV_mean - mean_ac 
    else:
        c_final = torch.zeros(T, len(comps), device=device)
    
    a_row_init = torch.zeros(total_length, dtype=torch.long)
    a_col_init = torch.zeros(total_length, dtype=torch.long)
    a_value_init = torch.zeros(total_length, dtype=V.dtype)
    
    ref_point = 0
    counter = 0
    for comp in comps:
        curr_length = len(list(comp))
        a_col_init[ref_point:ref_point+curr_length] = counter + K
        comp_tensor = torch.Tensor(list(comp))
        a_row_init[ref_point:ref_point+curr_length] = torch.Tensor(list(comp))
        a_value_init[ref_point:ref_point+curr_length] = 1 #UV_mean[list(comp), :].squeeze()
        ref_point += curr_length
        counter = counter + 1
    
    #Finally, combine a_orig and new a: 
    
    a_row_init = a_row_init.to(device)
    a_col_init = a_col_init.to(device)
    a_value_init = a_value_init.to(device) 
    
    if pre_existing:
        a_orig_row = a_orig.storage.row()
        a_orig_col = a_orig.storage.col()
        a_orig_values = a_orig.storage.value()

        final_rows = torch.cat([a_row_init, a_orig_row], dim=0)
        final_cols = torch.cat([a_col_init, a_orig_col], dim=0)
        final_values = torch.cat([a_value_init, a_orig_values], dim=0)
    else:
        final_rows =  a_row_init
        final_cols = a_col_init
        final_values = a_value_init
        
    a_sparse = torch_sparse.tensor.SparseTensor(row=final_rows, col=final_cols, value=final_values, \
                                                    sparse_sizes = (dims[0]*dims[1], K+len(comps))).coalesce().to(device)
    
    T = V.shape[1]
    W = ring_model(dims[0], dims[1], 1, device=device, empty=True)
    
    a_summand = torch.ones((dims[0]*dims[1], 1)).to(device)
    for k in range(1):
        
        b_torch = regression_update.baseline_update(UV_mean, a_sparse, c_final)
        c_final = regression_update.temporal_update_HALS(U_sparse, R, s, V, W,a_sparse, c_final, b_torch)


        b_torch = regression_update.baseline_update(UV_mean.to(device), a_sparse, c_final)      
        a_sparse = regression_update.spatial_update_HALS(U_sparse, R, s, V, W, a_sparse, c_final, b_torch)
        

    
    

    #Now return only the newly initialized components
    col_index_tensor = torch.arange(start = K, end = K + len(comps), step = 1, device=device)
    a_sparse = torch_sparse.index_select(a_sparse, 1, col_index_tensor)
    c_final = torch.index_select(c_final, 1, col_index_tensor)
    
    return  c_final.cpu().numpy(), a_sparse.cpu().to_dense().numpy(),

def delete_comp(a, c, corr_img_all_reg, corr_img_all, mask_a, num_list, temp, word, plot_en, fov_dims, order="C"):
    """
    Delete zero components, specified by "temp". 
    Inputs: 
        a: torch_sparse.tensor. Dimensions (d, K), d = number of pixels in movie, K = number of neurons
        c: torch.Tensor. Dimensions (T, K), K = number of neurons in movie
        corr_img_all_reg. np.ndarray. Dimensions (d, K). d = number of pixels in movie, K = number of neurons
        corr_img_all. np.ndarray. Dimensions (d, K). d = number of pixels in movie, K = number of neurons
        mask_a. torch_sparse.tensor. Dimensions (d, K). Dtype bool. d = number of pixels in movie, K = number of neurons
        num_list. np.ndarray. 
        temp: torch.Tensor. Dimensions (K). K= number of neurons
    Returns: 
        Updated a, c, corr_img_all_reg, corr_img_all, mask_a, num_list after getting rid of deleted comps
        
    Notes: 
    As of now, the correlation images are confined to the CPU as numpy ndarrays. Soon, these will be ported over to pytorch once an appropriate memory efficient implementation is ready. 
    """
    print(word);
    pos = torch.nonzero(temp)[:, 0]
    neg = torch.nonzero(temp == 0)[:, 0]
    if int(torch.sum(temp).cpu()) == a.sparse_sizes()[1]:
        raise ValueError("All Components are slated to be deleted")
    
    pos_for_cpu = pos.cpu().numpy()
    neg_for_cpu = neg.cpu().numpy()
    print("delete components" + str(num_list[pos_for_cpu]+1));
    corr_img_all_reg_r = corr_img_all_reg.reshape((fov_dims[0], fov_dims[1], -1), order = order)
    if plot_en:
        a_used = a.cpu().to_dense().numpy()
        spatial_comp_plot(a_used[:,pos_for_cpu],\
                          corr_img_all_reg_r[:,:,pos_for_cpu],\
                          num_list=num_list[pos_for_cpu], ini=False, order=order);
    corr_img_all_reg = np.delete(corr_img_all_reg, pos_for_cpu, axis=1);
    corr_img_all = np.delete(corr_img_all, pos_for_cpu, axis = 1);
    mask_a = torch_sparse.index_select(mask_a, 1, neg)
    a = torch_sparse.index_select(a, 1, neg)
    c = torch.index_select(c, 1, neg)
    num_list = np.delete(num_list, pos_for_cpu);
    return a, c, corr_img_all_reg, corr_img_all, mask_a, num_list


def order_superpixels(permute_col, unique_pix, U_mat, V_mat):
    """
    order superpixels according to brightness
    """
    ####################### pull out all the superpixels ################################
    permute_col = list(permute_col);
    pos = [permute_col.index(x) for x in unique_pix];
    U_mat = U_mat[:,pos];
    V_mat = V_mat[:,pos];
    ####################### order pure superpixel according to brightness ############################
    brightness = np.zeros(len(unique_pix));

    u_max = U_mat.max(axis=0);
    v_max = V_mat.max(axis=0);
    brightness = u_max * v_max;
    brightness_arg = np.argsort(-brightness); #
    brightness_rank = U_mat.shape[1] - ss.rankdata(brightness,method="ordinal");
    return brightness_rank


def search_superpixel_in_range(connect_mat, permute_col, V_mat):
    """
    Search all the superpixels within connect_mat
    Parameters:
    ----------------
    connect_mat_1: 2d np.darray, d1 x d2
        illustrate position of each superpixel, same value means same superpixel
    permute_col: list, length = number of superpixels
        random number used to idicate superpixels in connect_mat_1
    V_mat: 2d np.darray, dimension T x number of superpixel
        temporal initilization
    Return:
    ----------------
    unique_pix: list, length idx (number of superpixels)
        random numbers for superpixels in this patch
    M: 2d np.array, dimension T x idx
        temporal components for superpixels in this patch
    """

    unique_pix = np.asarray(np.sort(np.unique(connect_mat)),dtype="int");
    unique_pix = unique_pix[np.nonzero(unique_pix)];

    M = np.zeros([V_mat.shape[0], len(unique_pix)]);
    for ii in range(len(unique_pix)):
        M[:,ii] =  V_mat[:,int(np.where(permute_col==unique_pix[ii])[0])];

    return unique_pix, M


def fast_sep_nmf(M, r, th, normalize=1):
    """
    Find pure superpixels. solve nmf problem M = M(:,K)H, K is a subset of M's columns.
    Parameters:
    ----------------
    M: 2d np.array, dimension T x idx
        temporal components of superpixels.
    r: int scalar
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

    pure_pixels = [];
    if normalize == 1:
        M = M/np.sum(M, axis=0,keepdims=True);

    normM = np.sum(M**2, axis=0,keepdims=True);
    normM_orig = normM.copy();
    normM_sqrt = np.sqrt(normM);
    nM = np.sqrt(normM);
    ii = 0;
    U = np.zeros([M.shape[0], r]);
    while ii < r and (normM_sqrt/nM).max() > th:
        ## select the column of M with largest relative l2-norm
        temp = normM/normM_orig;
        pos = np.where(temp == temp.max())[1][0];
        ## check ties up to 1e-6 precision
        pos_ties = np.where((temp.max() - temp)/temp.max() <= 1e-6)[1];

        if len(pos_ties) > 1:
            pos = pos_ties[np.where(normM_orig[0,pos_ties] == (normM_orig[0,pos_ties]).max())[0][0]];
        ## update the index set, and extracted column
        pure_pixels.append(pos);
        U[:,ii] = M[:,pos].copy();
        for jj in range(ii):
            U[:,ii] = U[:,ii] - U[:,jj]*sum(U[:,jj]*U[:,ii])

        U[:,ii] = U[:,ii]/np.sqrt(sum(U[:,ii]**2));
        normM = np.maximum(0, normM - np.matmul(U[:,[ii]].T, M)**2);
        normM_sqrt = np.sqrt(normM);
        ii = ii+1;
    pure_pixels = np.array(pure_pixels);
    return pure_pixels #, coef, coef_rank




def get_mean(U,R, V, a=None, X=None):
    '''
    Routine for calculating the mean of the movie in question in terms of the V basis
    Inputs: 
        U: torch_sparse.Tensor. Dimensions (d1*d2, R) where d1, d2 are the FOV dimensions
        R: torch.Tensor. Dimensions (R, R)
        V: torch.Tensor: Dimensions (R, T), where R is the rank of the matrix
        
    Returns:
        m: torch.Tensor. Shape (d1*d2, 1) 
        s: torch.Tensor. Shape (1, R)
        
        Idea: msV is the "mean movie"
        
    '''
    
    V_mean = torch.mean(V, dim=1, keepdim=True)
    RV_mean = torch.matmul(R, V_mean)
    m = torch_sparse.matmul(U, RV_mean)
    if a is not None and X is not None:
        XV_mean = torch.matmul(X, V_mean)
        aXV_mean = torch_sparse.matmul(a, XV_mean)
        m = m - aXV_mean
    s = torch.matmul(V, torch.ones([V.shape[1], 1], device=R.device)).t()
    return m, s

def get_pixel_normalizer(U, R, V, m, s, pseudo, a=None, X=None, batch_size = 200):
    '''
    Routine for calculating the pixelwise norm of (UR - ms - aX)V. Due to the orthogonality of V this becomes: 
        diag (UR - ms - aX)*(UR - ms - aX)
        
    Inputs: 
        U_sparse: torch_sparse.Tensor, shape (d1*d2, R)
        R: torch.Tensor, shape (R, R)
        V: torch.Tensor, shape (R, T)
        m: torch.Tensor. shape (d1*d2, 1)
        s: torch.Tensor. Shape (1, R)
        pseudo: float
        a: torch_sparse.Tensor. Shape (d1*d2, K) 
        X: torch.Tensor. Shape (K, R). 
        batch_size: integer. default. 200. 
    '''
    
    num_cols = R.shape[1]
    num_iters = int(math.ceil(num_cols / batch_size))
    
    cumulator = torch.zeros((U.sparse_sizes()[0], 1), device=R.device)
    for k in range(num_iters):
        start = batch_size * k
        end = min(R.shape[1], start + batch_size)
        R_crop = R[:, start:end]
        s_crop = s[:, start:end]
        
        total = torch_sparse.matmul(U, R_crop) - torch.matmul(m, s_crop)
        if a is not None and X is not None:
            X_crop = X[:, start:end]
            total = total - torch_sparse.matmul(a, X_crop)
            
        cumulator = cumulator +torch.sum(total*total, dim=1, keepdim=True)
        
    cumulator = cumulator + pseudo**2
    
    cumulator[cumulator == 0] = 1 #Tactic to avoid division by 0
    return torch.sqrt(cumulator)

def construct_index_mat(d1, d2, order="C", device="cpu"):
    '''
    Constructs the convolution matrix (but expresses it in 1D)
    '''
    flat_indices = torch.arange(d1*d2, device=device)
    if order == "F":
        col_indices = torch.floor(flat_indices / d1)
        row_indices = flat_indices - col_indices * d1

    elif order == "C":
        row_indices = torch.floor(flat_indices / d2)
        col_indices = flat_indices - row_indices * d2
        
    else:
        raise ValueError("Invalid order input")
        
    addends_dim1 = torch.Tensor([-1,-1,-1,0,0,1,1,1]).to(device)[None, :]
    addends_dim2 = torch.LongTensor([-1, 0,1,-1,1,-1,0,1]).to(device)[None, :]
    
    row_expanded = row_indices[:,  None] + addends_dim1
    col_expanded = col_indices[:, None] + addends_dim2
    
    values = torch.ones_like(row_expanded, device=device)
    
    good_components = torch.logical_and(row_expanded >= 0, row_expanded < d1)
    good_components = torch.logical_and(good_components, col_expanded >= 0)
    good_components = torch.logical_and(good_components, col_expanded < d2)
    
    row_expanded*=good_components
    col_expanded*=good_components
    values*=good_components
    
    if order == "C":
        col_coordinates = d2*row_expanded + col_expanded
        row_coordinates = torch.arange(d1*d2, device=device)[:, None] + torch.zeros((1,col_coordinates.shape[1]), device=device)
        
        
    elif order == "F":
        col_coordinates = d1*col_expanded + row_expanded
        row_coordinates = torch.arange(d1*d2, device=device)[:, None] + torch.zeros((1,col_coordinates.shape[1]), device=device)
        
    col_coordinates = torch.flatten(col_coordinates).long()
    row_coordinates = torch.flatten(row_coordinates).long()
    values = torch.flatten(values).bool()
    
    
    good_entries = values>0
    row_coordinates = row_coordinates[good_entries]
    col_coordinates = col_coordinates[good_entries]
    values = values[good_entries]

    return row_coordinates, col_coordinates, values
   
        
        
def compute_correlation(I, U, R, m, s, norm, a=None, X=None, batch_size=200):
    '''
    Computes local correlation matrix given pre-computed quantities:
    Inputs: 
        I: torch_sparse.Tensor, shape (d1*d2, d1*d2). Extremely sparse (<5 elts per row)
        U: torch_sparse.Tensor. Shape (d1*d2, R). 
        m: torch.Tensor. Shape (d1*d2, 1)
        s: torch.Tensor. Shape (1, R)
        norm: torch.Tensor. Shape (d1*d2,1)
        a: torch_sparse.Tensor. Shape (d1*d2, K)
        X: torch.Tensor. Shape (K, R)
        batch_size: number of columns to process at a time. Default: 200 (to avoid issues with large fov data)
    
    '''
    num_cols = R.shape[1]
    num_iters = int(math.ceil(num_cols / batch_size))
    
    cumulator = torch.zeros((U.sparse_sizes()[0], 1), device=R.device)
    
    indicator_vector = torch.ones((U.sparse_sizes()[0], 1), device=R.device)
    for k in range(num_iters):
        start = k*batch_size
        end = min(R.shape[1], start + batch_size)
        R_crop = R[:, start:end]
        s_crop = s[:, start:end]
        
        total = torch_sparse.matmul(U, R_crop)  - torch.matmul(m, s_crop)
        if a is not None and X is not None:
            X_crop = X[:, start:end]
            total = total - torch_sparse.matmul(a, X_crop)
            
        total = total / norm
        
        I_total = torch_sparse.matmul(I, total)
        
        cumulator = cumulator + torch.sum(I_total*total, dim=1, keepdim=True)
    
    final_I_sum = torch_sparse.matmul(I, indicator_vector)
    final_I_sum[final_I_sum == 0] = 1
    return cumulator / final_I_sum
        
 

def pure_superpixel_corr_compare_plot(connect_mat_1, unique_pix, pure_pix, brightness_rank_sup, brightness_rank, Cnt, text=False, order="C"):
    scale = np.maximum(1, (connect_mat_1.shape[1]/connect_mat_1.shape[0]));
    fig = plt.figure(figsize=(4*scale,12));
    ax = plt.subplot(3,1,1);
    ax.imshow(connect_mat_1,cmap="nipy_spectral_r");

    if text:
        for ii in range(len(unique_pix)):
            pos = np.where(connect_mat_1[:,:] == unique_pix[ii]);
            pos0 = pos[0];
            pos1 = pos[1];
            ax.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{brightness_rank_sup[ii]+1}",
                verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)#, fontweight="bold")
    ax.set(title="Superpixels")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")

    ax1 = plt.subplot(3,1,2);
    dims = connect_mat_1.shape;
    connect_mat_1_pure = connect_mat_1.copy();
    connect_mat_1_pure = connect_mat_1_pure.reshape(np.prod(dims),order=order);
    connect_mat_1_pure[~np.in1d(connect_mat_1_pure,pure_pix)]=0;
    connect_mat_1_pure = connect_mat_1_pure.reshape(dims,order=order);

    ax1.imshow(connect_mat_1_pure,cmap="nipy_spectral_r");

    if text:
        for ii in range(len(pure_pix)):
            pos = np.where(connect_mat_1_pure[:,:] == pure_pix[ii]);
            pos0 = pos[0];
            pos1 = pos[1];
            ax1.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{brightness_rank[ii]+1}",
                verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)#, fontweight="bold")
    ax1.set(title="Pure superpixels")
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold");

    ax2 = plt.subplot(3,1,3);
    show_img(ax2, Cnt);
    ax2.set(title="Local mean correlation")
    ax2.title.set_fontsize(15)
    ax2.title.set_fontweight("bold")
    plt.tight_layout()
    plt.show();
    return fig, connect_mat_1_pure


def show_img(ax, img,vmin=None,vmax=None):
    # Visualize local correlation, adapt from kelly's code
    im = ax.imshow(img,cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    if np.abs(img.min())< 1:
        format_tile ='%.2f'
    else:
        format_tile ='%5d'
    plt.colorbar(im, cax=cax,orientation='vertical',spacing='uniform')



def local_correlation_mat(U, R, V, dims, pseudo, a=None, c=None, order="C", batch_size=200):
    '''
    Local correlation matrix for U and V computations
    
    Inputs: 
        U: torch_sparse.Tensor. Shape (d1*d2, R)
        R: torch.Tensor. Shape (R, R)
        V: torch.Tensor. Shape (R, T)
        pseudo: value typically near 0.1. Noise variance parameter used for correlation image calculation
        a: torch_sparse.Tensor. Shape (d1*d2, K), where K is # of neural signals
        c: torch.Tensor. Shape (T, K). 
        order: either "F" or "C" indicating how to reshape data back into (d1, d2, whatever) format from (d1*d2, whatever)
    '''
    if a is not None and c is not None:
        X = torch.matmul(V,c).t() #Equivalently a linear subspace projection of c onto V...
    else:
        X = None
    
    m, s = get_mean(U,R,V, a=a, X=X)
    norm = get_pixel_normalizer(U, R, V, m, s, pseudo, a=a, X=X, batch_size = batch_size)
    
    (r,c,v) = construct_index_mat(dims[0], dims[1], order=order, device=R.device)
    
    I = torch_sparse.tensor.SparseTensor(row = r, col = c, value=v, sparse_sizes=(dims[0]*dims[1], dims[0]*dims[1]))
    
    return compute_correlation(I, U, R, m, s, norm, a=a, X=X, batch_size=batch_size).cpu().numpy().reshape((dims[0], dims[1], -1), order=order)


def single_pixel_correlation_image(row_index, U_sparse, R, V, a=None, c=None, batch_size=100):
    device = R.device
    if a is not None and c is not None:
        X = torch.matmul(V,c).t() #Equivalently a linear subspace projection of c onto V...
    else:
        row = torch.Tensor([]).to(device).long()
        col = torch.Tensor([]).to(device).long()
        value = torch.Tensor([]).to(device).bool()
        a = torch_sparse.tensor.SparseTensor(row = row, col = col, value=value, sparse_sizes=(U_sparse.sparse_sizes()[0], 1))
        X = torch.zeros(1, V.shape[0], device=device)
        
    m, s = get_mean(U_sparse, R, V, a=a, X=X)
    norm = get_pixel_normalizer(U_sparse, R, V, m, s, 0, a=a, X=X, batch_size=batch_size)
    
    #Finally, get the centered row of interest: 
    row_index_tensor = torch.Tensor([row_index]).to(device).long()
    U_sparse_row = torch_sparse.index_select(U_sparse, 0, row_index_tensor )
    a_sparse_row = torch_sparse.index_select(a, 0, row_index_tensor)
    
    centered_pixel = torch_sparse.matmul(U_sparse_row, R) - torch_sparse.matmul(a_sparse_row, X) - torch.matmul(m[[row_index], :], s)
    centered_pixel = centered_pixel.t()
    
    Rprod = torch.matmul(R, centered_pixel)
    URprod = torch_sparse.matmul(U_sparse, Rprod)
    
    Xprod = torch.matmul(X, centered_pixel)
    aXprod = torch_sparse.matmul(a, Xprod) 
    
    sprod = torch.matmul(s, centered_pixel) 
    msprod = torch.matmul(m, sprod) 
    
    final_result = URprod - aXprod - msprod
    
    final_normalizer = norm * norm[row_index]
    final_normalizer[ final_normalizer == 0] = 1
    return final_result / final_normalizer
    
def prepare_iteration_UV(dims, connect_mat_1, permute_col, pure_pix, U_mat, V_mat, more=False):
    """
    Get some needed variables for the successive nmf iterations.
    Parameters:
    ----------------
    U: 2d np.ndarray, dimension (d1*d2) x R
        thresholded data
    V: 2d np.ndarray, dimension R x T
    connect_mat_1: 2d np.darray, d1 x d2
        illustrate position of each superpixel, same value means same superpixel
    permute_col: list, length = number of superpixels
        random number used to idicate superpixels in connect_mat_1
    pure_pix: 1d np.darray, dimension d x 1. (d is number of pure superpixels)
        pure superpixels for these superpixels, actually column indices of M.
    V_mat: 2d np.darray, dimension T x number of superpixel
        temporal initilization
    U_mat: 2d np.darray, dimension (d1*d2) x number of superpixel
        spatial initilization
    Return:
    ----------------
    U_mat: 2d np.darray, number pixels x number of pure superpixels
        initialization of spatial components
    V_mat: 2d np.darray, T x number of pure superpixels
        initialization of temporal components
    brightness_rank: 2d np.darray, dimension d x 1
        brightness rank for pure superpixels in this patch. Rank 1 means the brightest.
    B_mat: 2d np.darray
        initialization of constant background
    normalize_factor: std of Y
    """

    
    T = dims[2];

    ####################### pull out all the pure superpixels ################################
    permute_col = list(permute_col);
    pos = [permute_col.index(x) for x in pure_pix];
    U_mat = U_mat[:,pos];
    V_mat = V_mat[:,pos];
    ####################### order pure superpixel according to brightness ############################
    brightness = np.zeros(len(pure_pix));

    u_max = U_mat.max(axis=0);
    v_max = V_mat.max(axis=0);
    brightness = u_max * v_max;
    brightness_arg = np.argsort(-brightness); #
    brightness_rank = U_mat.shape[1] - ss.rankdata(brightness,method="ordinal");
    U_mat = U_mat[:,brightness_arg];
    V_mat = V_mat[:,brightness_arg];

    temp = np.sqrt((U_mat**2).sum(axis=0,keepdims=True));
    V_mat = V_mat*temp
    U_mat = U_mat/temp;
    return U_mat, V_mat, brightness_rank


def fit_large_spatial_support(comp, c_init, U_sparse_torch, V_torch, th, a_sparse=None, c=None, batch_size = 500):
    '''
    Routine for estimating 
    '''
    print("Fitting larger spatial support")
    comp = list(comp)
    num_iters = math.ceil(len(comp)/batch_size)
    final_values = torch.zeros(0, device=V_torch.device)
    
    for k in range(num_iters):
        start_pt = batch_size * k
        end_pt = min(len(comp), batch_size*(k+1))
        components = comp[start_pt:end_pt]
        comp_tensor = torch.LongTensor(components).to(V_torch.device)
        U_subset = torch_sparse.index_select(U_sparse_torch, 0, comp_tensor)
        y_temp = torch_sparse.matmul(U_subset, V_torch)
        

        if a_sparse is not None and c is not None:
            a_subset = torch_sparse.index_select(a_sparse, 0, comp_tensor)
            ac_prod = torch_sparse.matmul(a_subset, c)
            y_temp = torch.sub(y_temp, ac_prod)
            
        y_temp = threshold_data_inplace(y_temp, th, axisVal=1)
        
        normalizer = torch.sum(c_init * c_init)
        elt_product = torch.sum(c_init[None, :] * y_temp, axis = 1)
        
        curr_values = elt_product / normalizer
        threshold_function = torch.nn.ReLU()
        curr_values_thr = threshold_function(curr_values)
    
        final_values = torch.cat((final_values, curr_values_thr.type(final_values.dtype)), dim=0)
        
    return final_values


def superpixel_init(U_sparse, R, s, V, patch_size, num_plane, data_order, dims, cut_off_point, residual_cut, length_cut, th, batch_size, pseudo, device, dim1_coordinates, dim2_coordinates, correlations, text =True, plot_en = False, a = None, c = None):
    '''
    API: 
        Inputs (define variables here)
            - U_sparse: torch_sparse.Tensor, dims (d1*d2, R)
            - R: torch.Tensor, dims (R1, R2) where R = R1 and R2 is either R1 + 1 or R1
            - V: torch.Tensor, dims (R2, T)
            - V_PMD: torch.Tensor, dims (R2, T)
            - patch_size: tuple (p1, p2) of integers. describes patches into which FOV is subdivided when finding pure superpixels (via SPA) 
            - num_plane: number of planes to demix. Only valid value is 1 (for now) 
            - data_order: "F" or "C" depending on how the field of view "collapsed" into 1D vectors
            - dims: tuple containing (d1, d2, T), the dimensions of the data
            - cut_off_point: float between 0 and 1. Correlation thresholds used 
        in the superpixelization process
            - residual_cut: list of values between 0 and 1. Length of list = pass_num
                sqrt(1 - r_sqare of SPA)
                Standard value = 0.6
            - length_cut: integer. Minimum allowed sizes of superpixels
            - th: integer. MAD threshold factor
            - batch_size: integer. Batch size for various memory-constrained GPU calculations
            - pseudo: the robust correlation threshold
            - device: string, either 'cpu' or 'cuda'

            Optional parameters: 
            - text: Flag (boolean). Indicates whether some text is displayed in the plotting function
            - plot_en: Flag (boolean). Used to determine whether or not we plot some results here 
            - a: np.ndarray, shape (d1*d2, K)
            - c: np.ndarray, shape (T, K)

        Outputs
            - a. torch_sparse.Tensor, shape (d1*d2, K) where d1, d2 are the FOV dimensions and K is the number of signals identified
            - mask_ab. either None or torch_sparse.Tensor of shape same as "a"
            - c. torch_sparse.Tensor, shape (T,  K) where
            - b: torch.Tensor, shape(d1*d2) 
            - superpixel_img, shape (d1, d2)
    '''
    assert num_plane == 1, 'number of planes to demix must be 1' 
    
    if a is None and c is None: 
        first_init_flag = True
    elif a is not None and c is not None: 
        first_init_flag = False
    else:
        raise ValueError("Invalid configuration of c and a values were provided") 
    
    assert num_plane == 1, "Only single-plane data is currently supported"
    
    print("find superpixels!")
    connect_mat_1, idx, comps, permute_col= find_superpixel_UV(dims, cut_off_point, length_cut, dim1_coordinates, dim2_coordinates, correlations, data_order)
    
    c_ini, a_ini = spatial_temporal_ini_UV(U_sparse,R, s, V, dims, th, comps, idx, length_cut, a = a, c = c)
    start = time.time();
    
    print("find pure superpixels!")
    ## cut image into small parts to find pure superpixels ##
    patch_height = patch_size[0];
    patch_width = patch_size[1];
    height_num = int(np.ceil(dims[0]/patch_height));  ########### if need less data to find pure superpixel, change dims[0] here #################
    width_num = int(np.ceil(dims[1]/(patch_width*num_plane)));
    num_patch = height_num*width_num;
    patch_ref_mat = np.array(range(num_patch)).reshape(height_num, width_num, order=data_order);

    unique_pix = np.asarray(np.sort(np.unique(connect_mat_1)),dtype="int");
    unique_pix = unique_pix[np.nonzero(unique_pix)];
    brightness_rank_sup = order_superpixels(permute_col, unique_pix, a_ini, c_ini);
    pure_pix = [];
    
    for kk in range(num_patch):
        pos = np.where(patch_ref_mat==kk);
        up=pos[0][0]*patch_height;
        down=min(up+patch_height, dims[0]);
        left=pos[1][0]*patch_width;
        right=min(left+patch_width, dims[1]);
        unique_pix_temp, M = search_superpixel_in_range((connect_mat_1.reshape(dims[0],int(dims[1]/num_plane),num_plane,order=data_order))[up:down,left:right], permute_col, c_ini);
        pure_pix_temp = fast_sep_nmf(M, M.shape[1], residual_cut);
        if len(pure_pix_temp)>0:
            pure_pix = np.hstack((pure_pix, unique_pix_temp[pure_pix_temp]));
    pure_pix = np.unique(pure_pix);


    start = time.time();
    print("prepare iteration!")
    mask_a=None ## Disable the mask after first pass over data
    if not first_init_flag:
        a_ini, c_ini, brightness_rank = prepare_iteration_UV((dims[0], dims[1], dims[2]), connect_mat_1, permute_col, pure_pix, a_ini, c_ini);
        a = np.hstack((a, a_ini));
        c = np.hstack((c, c_ini));
        a = torch_sparse.tensor.from_scipy(scipy.sparse.coo_matrix(a)).float().to(device)
        c = torch.from_numpy(c).float().to(device)
        uv_mean = get_mean_data(U_sparse, R, s, V)
        b = regression_update.baseline_update(uv_mean, a, c)
    else:
        a, c, brightness_rank = prepare_iteration_UV((dims[0], dims[1], dims[2]), connect_mat_1, permute_col, pure_pix, a_ini, c_ini, more=True)
        a = torch_sparse.tensor.from_scipy(scipy.sparse.coo_matrix(a)).float().to(device)
        c = torch.from_numpy(c).float().to(device)
        uv_mean = get_mean_data(U_sparse, R, s, V)
        b = regression_update.baseline_update(uv_mean, a, c)

    assert a.sparse_sizes()[1] > 0, 'Superpixels did not identify any components, re-run with different parameters before proceeding'

    #Plot superpixel correlation image
    if plot_en:
        Cnt = local_correlation_mat(U_sparse, R*s[None, :], V, dims, pseudo, a=a, c=c, order=data_order)
        _, superpixel_img = pure_superpixel_corr_compare_plot(connect_mat_1, unique_pix, pure_pix, brightness_rank_sup, brightness_rank, Cnt, text, order=data_order);
    else:
        superpixel_img = None
        
    superpixel_dict = {'connect_mat_1':connect_mat_1, 'pure_pix':pure_pix,\
                       'unique_pix':unique_pix, 'brightness_rank':brightness_rank, 'brightness_rank_sup':brightness_rank_sup}

    return a,mask_a,c,b,superpixel_dict, superpixel_img

def merge_components(a,c,corr_img_all_r,num_list,patch_size,merge_corr_thr=0.6,merge_overlap_thr=0.6,plot_en=False, data_order="C"):
    """ want to merge components whose correlation images are highly overlapped,
    and update a and c after merge with region constrain
    Parameters:
    -----------
    a: torch_sparse.tensor
         sparse matrix describing the spatial supports of all signals. Shape (d, K) where d is the number of pixels in the movie and K is the number of neural signals
    c: torch.Tensor
         torch Tensor describing the temporal profiles of all signals. Shape (T, K), where T is the number of frames in the movie
    corr_img_all_r: np.ndarray (TODO: for now...) 
         corr image
    num_list: list. numbered list assigning an index value to each component (for tracking purposes)
    patch_size: (list-like) dimensions for data
    merge_corr_thr: scalar between 0 and 1
        temporal correlation threshold for truncating corr image (corr(Y,c)) (default 0.6)
    merge_overlap_thr: scalar between 0 and 1
        overlap ratio threshold for two corr images (default 0.6)
    plot_en: Boolean. Whether or not to plot the results. This is useful for development, not production (TODO: Check what things need to be moved to CPU for this)
    data_order: string. Either "C" or "F". 
    Returns:
    --------
    a_pri: torch_sparse.tensor. 
        sparse matrix describing the spatial supports of all signals. Shape (d, K') where d is the number of pixels in the movie and K' 
            is the number of neural signals after this merging procedure (entirely possible no merge happens and K' = K)
    c_pri: torch.Tensor.
        torch Tensor of merged temporal components, shape (T,K')
    num_list: np.ndarray. numbered list assigning an index value to each component (for tracking purposes) 
    flag: int. if merging occurred, return 1. Otherwise, returns 0
    """
    device = c.device
    f = torch.ones([c.shape[0], 1], device=device)
    corr_img_all_r = torch.from_numpy(corr_img_all_r).to(device) #TODO: Enforce this in the function definition
    ############ calculate overlap area ###########
   
    a_corr = torch_sparse.matmul(a.t(), a).to_dense()
    a_corr = torch.triu(a_corr, diagonal = 1)
    cor = ((corr_img_all_r > merge_corr_thr) * 1).float()
    temp = torch.sum(cor, dim=0)
    temp[temp == 0] = 1
    cor_corr = torch.matmul(cor.t(), cor)
    cor_corr = torch.triu(cor_corr, diagonal=1)
    cri = ((cor_corr / temp.t()) > merge_overlap_thr) * ((cor_corr/temp) > merge_overlap_thr) * (a_corr > 0)



    connect_comps = torch.argwhere(cri)
    
    if torch.numel(connect_comps) > 0:
        flag = 1
        G = nx.Graph();
        G.add_edges_from(list(zip(connect_comps[:, 0].cpu().numpy(), connect_comps[:, 1].cpu().numpy())))
        comps = list(nx.connected_components(G))
        remove_indices = torch.unique(torch.flatten(connect_comps))
        all_indices = torch.ones([c.shape[1]], device=device)
        all_indices[remove_indices] = 0
        all_indices = all_indices.bool()
        
        indices_arange = torch.arange(a.sparse_sizes()[1], device=device)[all_indices]
        
        a_preserved = torch_sparse.index_select(a, 1, indices_arange)
        c_preserved = torch.index_select(c, 1, indices_arange)
        num_preserved = num_list[indices_arange.cpu().numpy()]
        
        c_append_list = [c_preserved]
        num_append_list = [num_preserved]

        row_indices = [a_preserved.storage.row()] 
        col_indices = [a_preserved.storage.col()]
        values_indices = [a_preserved.storage.value()]
        num_preserved_comps = a_preserved.sparse_sizes()[1]
        added_counter = 0
        for comp in comps:
            comp = list(comp)
            print("merge" + str(num_list[comp]+1));
            good_comps = torch.Tensor(comp).to(device).long()
            
            a_zero = torch.zeros([a.sparse_sizes()[0], 1], device=device)
            a_merge = torch_sparse.index_select(a, 1, good_comps)
            c_merge = torch.index_select(c, 1, good_comps)
            
            a_rank1, c_rank1 = rank_1_NMF_fit(a_merge, c_merge)
        
            if plot_en:
                spatial_comp_plot(a_merge.cpu().to_dense().numpy(),\
                                  corr_img_all_r[:,comp].cpu().numpy().reshape(patch_size[0],patch_size[1],-1,order=data_order),num_list[comp],ini=False, order=data_order);
            
            nonzero_indices = torch.nonzero(a_rank1)
            row_temp = nonzero_indices[:, 0]
            col_temp = nonzero_indices[:, 1]
            
            nonzero_values = a_rank1[row_temp, col_temp]
            
            row_indices.append(row_temp)
            col_indices.append(col_temp + num_preserved_comps + added_counter)
            added_counter += 1
            values_indices.append(nonzero_values)

            c_append_list.append(c_rank1)
            num_append_list.append(num_list[comp[0]])
            
         
        row_indices_net = torch.cat(row_indices, dim=0)
        col_indices_net = torch.cat(col_indices, dim=0)
        value_indices_net = torch.cat(values_indices, dim=0)     
        c = torch.cat(c_append_list, dim=1)
        num_list = np.hstack(num_append_list)
        a = torch_sparse.tensor.SparseTensor(row=row_indices_net, col=col_indices_net, value=value_indices_net, \
                                                    sparse_sizes = (a.storage.sparse_sizes()[0], c.shape[1])).coalesce()   
    else:
        flag = 0
    
    return a, c, num_list, flag


def rank_1_NMF_fit(a_merge, c_merge):
    '''
    Fast HALS_based routine to perform a rank-1 NMF fit, constrained by the support of a_merge
    Inputs:
        a_merge: torch_sparse.Tensor. Shape (d, K), where d is the number of pixels and K is the number of neural signals to be merged
        c_merge: torch.Tensor. Shape (T, K), where T is the number of frames in the movie
        
    Returns: 
        spatial_component: torch.Tensor. Shape (d, 1)
        temporal_component: Torch.Tensor. Shape (T, 1). These two tensors, when multiplied like so: 
                    torch.matmul(spatial_component, temporal_component.t()), give the rank-1 constrained NMF approximation to the movie given by torch.matmul(a_merge, c_merge.t())
    '''
    device = c_merge.device
    
    #Step 1: Figure out how to initialize the first and second components of the rank-1 factorization. We init the first component to the mean:
    summand = torch.ones([a_merge.sparse_sizes()[1], 1], device=device)
    summand /= a_merge.sparse_sizes()[1]
    spatial_component = torch_sparse.matmul(a_merge, summand)
    mask = (spatial_component > 0)
    
    temporal_component = torch.zeros([c_merge.shape[0], 1], device=device)
    
    my_relu_obj = torch.nn.ReLU()
    
    num_iters = 5
    
    for k in range(num_iters):
        temporal_component = my_relu_obj(_temporal_fit_routine(a_merge, c_merge, spatial_component))
        spatial_component = my_relu_obj(_spatial_fit_routine(a_merge, c_merge, temporal_component, mask))
        
    
    return spatial_component, temporal_component

def _spatial_fit_routine(a_merge, c_merge, temporal_component, mask):
    '''
    Fits a spatial component in the rank-1 nonnegative merging fit via standard least squares
    Inputs: 
        a_merge: torch_sparse.tensor of shape (d, K) where K is the number of signals slated to be merged
        c_merge: torch.Tensor of shape (T, K) where T is the number of frames
        temporal_component: torch.Tensor of shape (T, 1)
        mask: torch.Tensor of shape (d, 1). Has values 1 where the support is defined and 0 elsewhere
    Output:
        spatial_component: torch.Tensor of shape (d, 1)
    '''
    
    temporal_dot_product = torch.matmul(temporal_component.t(), temporal_component)
    temporal_dot_product[temporal_dot_product == 0] = 1 #Avoid division by zero issues
    
    merge_dots = torch.matmul(c_merge.t(), temporal_component) 
    row_dots = torch_sparse.matmul(a_merge, merge_dots)
    
    least_squares_fits = row_dots / temporal_dot_product
    return least_squares_fits * mask #Set other elts outside of support to 0 

def _temporal_fit_routine(a_merge, c_merge, spatial_component):
    '''
    Fits a nonnegative temporal component in the rank-1 nonnegative merging fit
    Inputs: 
        a_merge: torch_sparse.tensor of shape (d, K) where K is the number of signals slated to be merged
        c_merge: torch.Tensor of shape (T, K) where T is the number of frames
        spatial_component: torch.Tensor of shape (d, 1)
    Output:
        temporal_component: torch.Tensor of shape (T, 1)
    '''
    
    aA = torch_sparse.matmul(a_merge.t(),spatial_component).t()
    aAC = torch.matmul(aA, c_merge.t())
    
    spatial_norm = torch.matmul(spatial_component.t(), spatial_component)
    spatial_norm[spatial_norm == 0] = 1
    
    least_squares_fits = aAC / spatial_norm
    
    return least_squares_fits.T  

def spatial_comp_plot(a, corr_img_all_r, num_list=None, ini=False, order="C"):
    print("DISPLAYING SOME OF THE COMPONENTS")
    num = min(3, a.shape[1]);
    patch_size = corr_img_all_r.shape[:2];
    scale = np.maximum(1, (corr_img_all_r.shape[1]/corr_img_all_r.shape[0]));
    fig = plt.figure(figsize=(8*scale,4*num));
    if num_list is None:
        num_list = np.arange(num);
    for ii in range(num):
        plt.subplot(num,2,2*ii+1);
        plt.imshow(a[:,ii].reshape(patch_size,order=order),cmap='nipy_spectral_r');
        plt.ylabel(str(num_list[ii]+1),fontsize=15,fontweight="bold");
        if ii==0:
            if ini:
                plt.title("Spatial components ini",fontweight="bold",fontsize=15);
            else:
                plt.title("Spatial components",fontweight="bold",fontsize=15);
        ax1 = plt.subplot(num,2,2*(ii+1));
        show_img(ax1, corr_img_all_r[:,:,ii]);
        if ii==0:
            ax1.set(title="corr image")
            ax1.title.set_fontsize(15)
            ax1.title.set_fontweight("bold")
    plt.tight_layout()
    plt.show()
    return fig


##TODO: Finalize the API and make this class inherit "FunctionalVideo"
class PMDVideo():
    
    
    def __init__(self, U_sparse, R, s, V, dimensions, data_order="F", device='cpu'):
        '''
        Things to manage: 
            - 'a', 'c' and the like should be optional. If they are not None, it's implied custom init, and we should do the standard custom init pipeline
        
        '''
        self.device = device
        self.R = torch.Tensor(R).float().to(self.device)
        self.s = torch.from_numpy(s).float().to(self.device)
        self.U_sparse = torch_sparse.tensor.from_scipy(U_sparse).float().to(self.device)
        self.V = torch.Tensor(V).float().to(self.device)
        self.shape = dimensions
        self.d1 = dimensions[0]
        self.d2 = dimensions[1]
        self.T = dimensions[2]
        self.data_order = data_order
        self.active_weights = None
       
        self.demixing_state = False
        self.precomputed = False 
        
        self.U_sparse, self.R, self.s, self.V = PMD_setup_routine(self.U_sparse, self.R, self.s, self.V) 
        self.batch_size = 1000 #Change this long term
        
        
        
        ##These are the "signal" components
        self.a = None
        self.c = None
        self.b = None
        self.a_init = None
        self.mask_a_init = None
        self.c_init = None
        self.b_init = None
        self.num_passes_run = 0
        
        self.superpixel_rlt_recent = None
        self.superpixel_rlt = []
        self.superpixel_image_recent = None
        self.superpixel_image_list=[]
        
        #Reset the local pixelwise correlation connectivity data structure
        self.dim1_coordinates = None
        self.dim2_coordinates = None
        self.correlations = None
        
        
        ## Initialize ring model object for neuropil estimation
        ring_placeholder = 5
        self.W = ring_model(self.d1, self.d2, ring_placeholder, device=self.device, order=self.data_order, empty=True)
        
     
    def finalize_initialization(self):
        self.demixing_state = True
        self.a = self.a_init
        self.b = self.b_init
        self.c = self.c_init
        self.mask_a = self.mask_a_init
        
        #Reset the local pixelwise correlation connectivity data structure
        self.dim1_coordinates = None
        self.dim2_coordinates = None
        self.correlations = None
        
        
        if self.superpixel_rlt_recent is not None: 
            self.superpixel_rlt.append(self.superpixel_rlt_recent)
            self.superpixel_image_list.append(self.superpixel_image_recent)
            
            self.superpixel_rlt_recent = None
            self.superpixel_image_recent = None
        
        
    def initialize_signals_superpixels(self, num_plane, cut_off_point, residual_cut, length_cut, th, pseudo_2, \
                                       text =True, plot_en = False):
        '''
        See superpixel_init function above for a clear explanation of what each of these parameters should be
        '''

        import time
        if not self.demixing_state:
            patch_size = [100, 100]
            if self.dim1_coordinates is None:
                #This indicates that it is the first time we are running the superpixel init with this set of pre-existing self.a and self.c values, so we need to compute the local correlation data
                start_time = time.time()
                self.dim1_coordinates, self.dim2_coordinates, self.correlations = get_local_correlation_structure(self.U_sparse, torch.matmul(self.R*self.s[None, :], self.V), self.shape, th, order = self.data_order,batch_size = self.batch_size, pseudo=pseudo_2, a = self.a, c = self.c)
                print("the time to run the one time only local corr data structure computation is {}".format(time.time() - start_time))

            start_time = time.time()
            self.a_init, self.mask_a_init, self.c_init, self.b_init, output_dictionary, superpixel_image = superpixel_init(self.U_sparse,self.R,self.s, self.V, patch_size, num_plane, self.data_order, self.shape, cut_off_point, residual_cut, length_cut, th, self.batch_size, pseudo_2, self.device, self.dim1_coordinates, self.dim2_coordinates, self.correlations, text = text, plot_en = plot_en, a = self.a, c = self.c)
            print("the time to run pure superpixelization is {}".format(time.time() - start_time))

            self.superpixel_rlt_recent = output_dictionary
            self.superpixel_image_recent = superpixel_image

        else:
            print("Cannot run initialization until current round of demixing is done")

    def initialize_signals_custom(self, custom_init):
        
        assert custom_init['a'].shape[2] > 0, 'Must provide at least 1 spatial footprint'
        assert self.a is None and self.c is None, 'Custom init is only supported for the first initialization, not for subsequent initializations'
        self.a_init, self.mask_a_init, self.c_init, self.b_init = process_custom_signals(custom_init['a'].copy(), self.U_sparse, self.R, self.s, self.V, order=self.data_order)
        
    def precompute_quantities(self, maxiter, ring_radius=15):
        '''
        Args: 
            maxiter: int. Number of iterations to be run (long term eliminate this)
            ring_radius. int, default of 15 
        '''
        self.finalize_initialization()
        self.r = ring_radius
        self.K = self.c.shape[1]
        self.res = np.zeros(maxiter)
        self.uv_mean = get_mean_data(self.U_sparse, self.R, self.s, self.V)
        self.num_list = np.arange(self.K)

        if self.mask_a is None:
            self.mask_a = self.a.bool()
        else:
            print("MASK IS NOT NONE")
        self.mask_ab = self.mask_a

        self.W = ring_model(self.d1, self.d2, self.r, device=self.device, order = self.data_order, empty=True)
        self.a_summand = torch.ones((self.d1*self.d2, 1)).to(self.device)
        
        #Initialize correlation image fields here
        self.standard_correlation_image = None
        self.residual_correlation_image = None
        self.precomputed=True
        
    def _assert_initialization(self):
        assert self.a is not None and self.c is not None, "Initialization was not run"
        
    def _assert_ready_to_demix(self):
        assert self.precomputed, "The values were precomputed"
        assert self.demixing_state, "Not ready to demix"
    
    def delete_precomputed(self):
        '''
        For now, this is a memory-saving technique: we delete the precomputed quantities which actually can occupy significant space on GPU 
        '''
        self.precomputed=False
        
    def halt_demixing(self):
        self._delete_precomputed()
        self.demixing_state = False
        
    def static_baseline_update(self):
        self._assert_initialization()
        self._assert_ready_to_demix()
        self.b = regression_update.baseline_update(self.uv_mean, self.a, self.c)
        
    def fluctuating_baseline_update(self):
        self._assert_initialization()
        self._assert_ready_to_demix()
        
        if self.W.empty:
            #This means we need to create the actual W matrix (i.e. it can't just be empty)
            self.W = ring_model(self.d1, self.d2, self.r, empty=False, device=self.device, order=self.data_order)
        ring_model_update(self.U_sparse, self.R*self.s[None, :], self.V, self.W, self.c, self.b, self.a, self.d1, self.d2, device=self.device)
        
    def temporal_update(self, denoise=False, plot_en=False):
        self._assert_initialization()
        self._assert_ready_to_demix()
        self.c = regression_update.temporal_update_HALS(self.U_sparse, self.R, self.s, self.V, self.W, self.a, self.c, self.b)
        
        #Denoise 'c' components if desired
        if denoise:
            c = self.c.cpu().numpy()
            c = ca_utils.denoise(c) #We now use OASIS denoising to improve improve signals
            c = np.nan_to_num(c, posinf = 0, neginf = 0, nan = 0) #Gracefully handle invalid values
            self.c = torch.from_numpy(c).float().to(self.device)
        
        #Delete bad components
        temp = (torch.sum(self.c, dim=0) == 0);
        if torch.sum(temp):
            self.a, self.c, self.standard_correlation_image, self.residual_correlation_image, self.mask_ab, self.num_list = delete_comp(self.a, self.c, self.standard_correlation_image, self.residual_correlation_image, self.mask_ab, self.num_list, temp, "zero c!", plot_en, (self.d1, self.d2), order=self.data_order)
            
    def spatial_update(self, plot_en = False):
        self.a = regression_update.spatial_update_HALS(self.U_sparse,self.R, self.s, self.V, self.W, self.a, self.c, self.b, mask_ab=self.mask_ab)
        
        ## Delete Bad Components
        temp = torch_sparse.matmul(self.a.t(), self.a_summand).t() == 0 #Identify which columns of 'a' are all zeros
        if torch.sum(temp):
            self.a, self.c, self.standard_correlation_image, self.residual_correlation_image, self.mask_ab, self.num_list = delete_comp(self.a, self.c, self.standard_correlation_image, self.residual_correlation_image, self.mask_ab, self.num_list, temp, "zero a!", plot_en, (self.d1, self.d2), order=self.data_order);
            
    def compute_local_correlation_image(self, pseudo):
        ##Long term: the currently initialized "a" and "c" should just be on the same device as U_sparse,R,V
        a_device = torch_sparse.tensor.from_scipy(scipy.sparse.coo_matrix(self.a)).float().to(self.device)
        c_device = torch.from_numpy(self.c).float().to(self.device)
        return local_correlation_mat(self.U_sparse, self.R*self.s[None, :], self.V, self.shape, pseudo, a=a_device, c=c_device, order=self.data_order, batch_size=self.batch_size)
    
    def compute_single_pixel_correlation_image(self, row_index, residual=True):
        '''
        Calculates the residual correlation image
        '''
        if self.a is not None and self.c is not None and residual: 
            a_device = torch_sparse.tensor.from_scipy(scipy.sparse.coo_matrix(self.a)).float().to(self.device)
            c_device = torch.from_numpy(self.c).float().to(self.device)
        else:
            a_device = None
            c_device = None
        return single_pixel_correlation_image(row_index, self.U_sparse, self.R*self.s[None, :], self.V, a=a_device, c=c_device, batch_size=self.batch_size)
            
    def compute_standard_correlation_image(self):
        self.standard_correlation_image = vcorrcoef_UV_noise(self.U_sparse, self.R*self.s[None, :], self.V, self.c, batch_size = self.batch_size,  device=self.device)
        
    #TODO: Profile this to see if you can precompute some quantities ahead of time
    def compute_residual_correlation_image(self):
        self.residual_correlation_image = vcorrcoef_resid(self.U_sparse, self.R*self.s[None, :], self.V, self.a, self.c, batch_size = self.batch_size)
        
    def merge_signals(self, merge_corr_thr, merge_overlap_thr, plot_en, data_order):
        rlt = merge_components(self.a,self.c,self.standard_correlation_image,self.num_list,\
                               self.shape,merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr,plot_en=plot_en, data_order=self.data_order);

        self.a = rlt[0]
        self.c = rlt[1]
        self.num_list = rlt[2]

    def support_update_prune_elements_apply_mask(self, corr_th_fix, corr_th_del, plot_en):
        
        #Currently using rigid mask
        print("making dynamic support updates")
        self.mask_ab = self.a.bool()
        corr_img_all_r = self.residual_correlation_image.reshape(self.d1, self.d2, -1, order=self.data_order)
        mask_a_rigid = make_mask_dynamic(corr_img_all_r, corr_th_fix, self.mask_ab.cpu().to_dense().numpy().astype('int'), data_order=self.data_order)
        mask_a_rigid_scipy = scipy.sparse.csr_matrix(mask_a_rigid)
        self.mask_ab = torch_sparse.tensor.from_scipy(mask_a_rigid_scipy).float().to(self.device)

        ## Now we delete components based on whether they have a 0 residual corr img with their supports or not...

        mask_ab_corr = mask_a_rigid_scipy.multiply(self.residual_correlation_image)
        mask_ab_corr = np.array((mask_ab_corr > corr_th_del).sum(axis=0))
        mask_ab_corr = torch.from_numpy(mask_ab_corr).float().squeeze().to(self.device)
        temp = (mask_ab_corr == 0)
        if torch.sum(temp):
            print("we are at the mask update delete step... corr img is {}".format(corr_th_del))
            self.a, self.c, self.standard_correlation_image, self.residual_correlation_image, self.mask_ab, self.num_list = delete_comp(self.a, self.c, self.standard_correlation_image, self.residual_correlation_image, self.mask_ab, self.num_list, temp, "zero mask!", plot_en, (self.d1,self.d2), order=self.data_order)
            
        ##Apply mask to existing 'a'
        a_scipy = self.a.to_scipy(layout='csr')
        mask_ab_scipy = self.mask_ab.to_scipy(layout='csr')
        a_scipy = a_scipy.multiply(mask_ab_scipy)
        self.a = torch_sparse.tensor.from_scipy(a_scipy).float().to(self.device)        
        
    def get_PMD_row(self, row_index):
        '''
        Utility function for accessing a single row of the PMD denoised data
        '''
        assert 0 <= row_index and row_index < self.U_sparse.sparse_sizes()[0], "Row Index is out of range"
        
        row_index_tensor = torch.arange(row_index, row_index+1, device=self.device)
        my_row = torch_sparse.index_select(self.U_sparse, 0, row_index_tensor)
        my_row_R = torch_sparse.matmul(my_row, self.R)
        my_row_R = my_row_R * self.s[None,:]
        my_row_RV = torch.matmul(my_row_R, self.V)
        
        return my_row_RV
        
    def get_PMD_frame(self, frame):
        '''
        Returns the index of a frame of the movie (this is pythonic indexing, so to generate the first frame, you should use frame = 0)
        '''
        assert 0 <= frame and frame < self.V.shape[1], "Requested frame is out of bounds"
        
        curr_index = torch.arange(frame, frame+1, device=self.device)
        curr_sV = self.s[:, None] *torch.index_select(self.V, 1, curr_index)
        RsV = torch.matmul(self.R, curr_sV)
        URsV = torch.squeeze(torch_sparse.matmul(self.U_sparse, RsV))
        
        if self.data_order == "C":
            return reshape_c(URsV, self.shape[:2])
        elif self.data_order == "F":
            return reshape_fortran(URsV, self.shape[:2])
        else:
            raise ValueError("Invalid order")
    
    def get_background_row(self, row_index):
        #Add assert statement to make sure there even is "a"
        assert self.a is not None, "There are no signals, so the background estimate will not be valid"
        assert self.demixing_state is False, "This function is currently meant to be run after demixing, not during it" 
        return self.W.compute_fluctuating_background_row(self.U_sparse, self.R, self.s, self.V, self.active_weights, self.b, row_index)
    
    
    def reset(self):
        '''
        Generic reset to "initial" state of PMD demixing object
        '''
        self.a = None
        self.c = None
        self.mask_a = None
        self.b = None
        self.a_init = None
        self.c_init = None
        self.mask_a_init = None
        self.b_init = None
        self.active_weights = None
        
        self.superpixel_rlt_recent = None
        self.superpixel_rlt = []
        self.superpixel_image_recent = None
        self.superpixel_image_list=[]
            

    def brightness_order_and_return_state(self):
        '''
        This is a compatibility function. Long term the api for this should change. 
        Assumption here is that before running this function, the data is on the "device" in "demixing" state. After, it will not be.
        '''
        a_sum_vec = torch.ones((self.a.sparse_sizes()[1], 1), device=self.device)
        self.active_weights = (torch_sparse.matmul(self.a, a_sum_vec) == 0).float().cpu().numpy()

        self.a = self.a.cpu().to_dense().numpy()
        self.c = self.c.cpu().numpy()
        self.b = self.b.cpu().numpy()

        temp = np.sqrt((self.a**2).sum(axis=0,keepdims=True));
        c = self.c*temp;
        a = self.a/temp;
        brightness = np.zeros(self.a.shape[1]);
        a_max = self.a.max(axis=0);
        c_max = self.c.max(axis=0);
        brightness = a_max * c_max;
        brightness_rank = np.argsort(-brightness);
        self.a = self.a[:,brightness_rank];
        self.c = self.c[:,brightness_rank];
        corr_img_all_r = self.residual_correlation_image.reshape((self.d1, self.d2, -1), order=self.data_order)[:,:,brightness_rank];
        self.num_list = self.num_list[brightness_rank];
        
        self.demixing_state = False
        self.precomputed=False

        #Reset the local pixelwise correlation connectivity data structure
        self.dim1_coordinates = None
        self.dim2_coordinates = None
        self.correlations = None
        
        return self.a, self.c, self.b, self.W, self.res, corr_img_all_r, self.num_list