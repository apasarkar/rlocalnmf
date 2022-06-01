# from typing import list
import copy
import math
import time
import torch

import numpy as np
from scipy import ndimage
import scipy.sparse
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import coo_matrix
from skimage.morphology import disk as sk_disk
import cv2 as cv
from scipy.optimize import lsq_linear
from numpy import linalg as LA
from sklearn.decomposition import non_negative_factorization
import scipy.optimize as optimization
import os

print(os.getcwd())
print("that was in cnmf_e.py")
from . import nnls



######
###
######

def init_w_old(d1, d2, r):
    """Compute the initial W weighting matrix
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param r: ring radius of ring model
    :return: W weighting matrix of shape [d1, d2]
    """

    # Compute XY distance tile
    x_tile = np.tile(range(-(r + 1), (r + 2)), [(2 * r + 3), 1])
    y_tile = np.transpose(np.tile(range(-(r + 1), (r + 2)), [(2 * r + 3), 1]))
    xy_tile = np.sqrt(np.multiply(x_tile, x_tile) + np.multiply(y_tile, y_tile))

    # Find ring index
    r_tile = np.ones((2 * r + 3, 2 * r + 3)) * r
    r1_tile = r_tile + 1
    ring = np.logical_and(np.greater_equal(xy_tile, r_tile),
                          np.less(xy_tile, r1_tile))
    ring_idx = np.argwhere(ring)
    ring_idx_T = np.transpose(ring_idx)
    ring_idx_T = ring_idx_T - (r + 1)  # shift index so that center has zero index

    # Create a weighting matrix to store initial value, the matrix size is padded
    # r cells along 2nd and 3rd dimensions to avoid out of index

    d = d1 * d2
    W = lil_matrix((d, d), dtype=np.float)
    for i in range(d1):
        for j in range(d2):
            ij = i * d2 + j
            x_base, y_base = i + r, j + r
            ring_idx_T2 = copy.deepcopy(ring_idx_T)
            ring_idx_T2[0, :] += x_base
            ring_idx_T2[1, :] += y_base
            selection_0 = np.logical_and(ring_idx_T2[0, :] >= r, ring_idx_T2[0, :] < r + d1)
            selection_1 = np.logical_and(ring_idx_T2[1, :] >= r, ring_idx_T2[1, :] < r + d2)
            selection = np.logical_and(selection_0, selection_1)
            selection_idx = np.argwhere(selection)
            ring_idx_T3 = ring_idx_T2[:, selection_idx[:, 0]]
            ring_idx = (ring_idx_T3[0, :] - r) * d2 + ring_idx_T3[1, :] - r
            W[ij, ring_idx] = 1.0
    return W


def update_ring_model_w_const_old(U, V, A, X, b, W, d1, d2, T, r, mask_a=None, device = 'cpu', batch_size = 10000):
    """Update W matrix using ring model
    :param U: spatial component matrix from denoiser, R(d=d1xd2, N)
    :param V: temporal component matrix from denoiser, R(N, T). 
        MUST BE ORTHOGONAL: V(V^t) = I
    :param A: spatial component matrix, R(d, K)
    :param X: temporal component matrix (the actual temporal matrix C = XV),  R(K, N)
    :param b: constant baseline, dimension d x 1
    :param W: weighting matrix, R(d, d)
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param T: number to time steps along time axis
    :param r: ring radius of ring model
    :return:
        W: updated weighting matrix
        b0: constant baseline of background image
    """

    if W is None:
        W = init_w_old(d1, d2, r)
    bg_basis = b.dot((V.T).sum(axis = 0, keepdims = True))
    Y = U - np.matmul(A, X) - bg_basis
    print(np.min(Y))
    print(np.max(Y))
    
    print('asum start')
    if mask_a is None:
        print("in 1")
        A_sum = np.sum(A, axis=1)
    else:
        print("in 2")
        print(mask_a.shape)
        A_sum = np.sum(mask_a, axis=1)
    print('asum done')
    
    print('call subfunc')
    W = update_w_1p_const_old(Y, W, A_sum, d1, d2, device = device, batch_size = batch_size) #For now leave out batch size..
    return W



def update_w_1p_const_old(Y, W, A_sum, d1, d2, device = 'cpu', batch_size = 10000):
    """Constant Ring Model codebase 
    params:
        Y: 2D matrix that is equal to U - AX
        W: weighting matrix. Type: lil_matrix. Note: assumes W has been initialized already
        d1: x axis frame size
        A_sum: the "summed" A onto 1 plane
        d2: y axis frame size
        batch_size: number of ring indices to simultaneously update
    return:
        W: Sparse matrix (d x d) representing ring weights
    
    This is the CONST pipeline
    """
#     Y = Y/np.amax(Y)   #Note... why are we doing this?
    d = d1 * d2
    iters = math.ceil(d / batch_size)
    weights = np.zeros((d, 1))
    
    if device == 'cpu':
        batch_size = d
    
    start_time = time.time()
    print("preprocessing W")
    #Preprocess W
    W[W > 0] = 1 #Reset all values of support of W back to 1
    print("set W to 1 at {}".format(time.time() - start_time))
    W[:, (A_sum > 0)] = 0
    print("turn off support at {}".format(time.time() - start_time))
    W = W.tocsr() #Convert to csr for quick multiply
    print("convert to CSR at {}".format(time.time() - start_time))
    
    test_time = time.time()
    print("WdotY...")
    C_total = (W.dot(Y)) 
    print("the time taken for C_total is {}".format(time.time() - test_time))
    
    for batch in range(iters):
        print("batch {} of {}".format(batch, iters))
        start_pt = batch_size * (batch)
        Y_crop = Y[start_pt:(start_pt + batch_size), :] 
        C_mat = C_total[start_pt:(start_pt + batch_size), :]
        
        
        
        #Now move values to device
        C_tensor = torch.Tensor(C_mat).to(device)
        Y_tensor = torch.Tensor(Y_crop).to(device)
        
        #Find proj and norm
        proj = (C_tensor * Y_tensor).sum(dim=1)
        norm = (C_tensor * C_tensor).sum(dim=1)
        
        #Final parameter estimates
        values = torch.div(proj, norm)

        values = values.to('cpu').detach().numpy().astype("double")
            
            
        #Empty Cuda   
        torch.cuda.empty_cache()
        
        #Filter values
        values[values<0] = 0 
        values = np.nan_to_num(values, nan=0, posinf=0)
        
        #Add weights
        weights[start_pt:(start_pt + batch_size), 0] = values
        
    
    #Create new CSR matrix   
    pos = W.nonzero()
    rows = pos[0]
    values = weights[rows]
    W = csr_matrix((values.squeeze(), (pos[0], pos[1])), shape = (d, d))
    
    #Convert to lil (temp)
    W = W.tolil()
    return W








######
###
######


def init_w(d1, d2, r):
    """Compute the initial W weighting matrix
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param r: ring radius of ring model
    :return: W weighting matrix of shape [d1, d2]
    """

    # Compute XY distance tile
    x_tile = np.tile(range(-(r + 1), (r + 2)), [(2 * r + 3), 1])
    y_tile = np.transpose(np.tile(range(-(r + 1), (r + 2)), [(2 * r + 3), 1]))
    xy_tile = np.sqrt(np.multiply(x_tile, x_tile) + np.multiply(y_tile, y_tile))

    # Find ring index
    r_tile = np.ones((2 * r + 3, 2 * r + 3)) * r
    r1_tile = r_tile + 1
    ring = np.logical_and(np.greater_equal(xy_tile, r_tile),
                          np.less(xy_tile, r1_tile))
    ring_idx = np.argwhere(ring)
    ring_idx_T = np.transpose(ring_idx)
    ring_idx_T = ring_idx_T - (r + 1)  # shift index so that center has zero index

    # Create a weighting matrix to store initial value, the matrix size is padded
    # r cells along 2nd and 3rd dimensions to avoid out of index

    d = d1 * d2
    rows = []
    cols = []
    vals = []
    for i in range(d1):
        for j in range(d2):
            ij = i * d2 + j
            x_base, y_base = i + r, j + r
            ring_idx_T2 = copy.deepcopy(ring_idx_T)
            ring_idx_T2[0, :] += x_base
            ring_idx_T2[1, :] += y_base
            selection_0 = np.logical_and(ring_idx_T2[0, :] >= r, ring_idx_T2[0, :] < r + d1)
            selection_1 = np.logical_and(ring_idx_T2[1, :] >= r, ring_idx_T2[1, :] < r + d2)
            selection = np.logical_and(selection_0, selection_1)
            selection_idx = np.argwhere(selection)
            ring_idx_T3 = ring_idx_T2[:, selection_idx[:, 0]]
            ring_idx = (ring_idx_T3[0, :] - r) * d2 + ring_idx_T3[1, :] - r
#             W[ij, ring_idx] = 1.0
            for k in range(len(ring_idx)):
                rows.append(ij)
                cols.append(ring_idx[k])
                vals.append(1)
    W = coo_matrix((vals, (rows, cols)), shape=(d,d))
    return W

def update_ring_model_w_full(U, V, A, X, W, d1, d2, T, r, mask_a=None):
    """Update W matrix using ring model
    :param U: spatial component matrix from denoiser, R(d=d1xd2, N)
    :param V: temporal component matrix from denoiser, R(N, T)
    :param A: spatial component matrix, R(d, K)
    :param X: temporal component matrix (the actual temporal matrix C = XV),  R(K, N)
    :param W: weighting matrix, R(d, d)
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param T: number to time steps along time axis
    :param r: ring radius of ring model
    :return:
        W: updated weighting matrix
        b0: constant baseline of background image
    """

    if W is None:
        W = init_w(d1, d2, r)
    W = init_w(d1, d2, r)
    Y = U - np.matmul(A, X)
    print(np.min(Y))
    print(np.max(Y))
    b0 = np.matmul(Y, np.mean(V, axis=1) / T)
    if mask_a is None:
        A_sum = np.sum(A, axis=1)
    else:
        A_sum = np.sum(mask_a, axis=1)
    update_w_1p_full(Y, W,A_sum, d1, d2)
    return b0, W
    
    

def update_w_1p_full(Y, W, A_sum, d1, d2):
    """Update weighting matrix W in place 
    :param Y: 2D matrix that is equal to U - AX
    :param W: weighting matrix
    :param d1: x axis frame size
    :param A_sum: the "summed" A onto 1 plane
    :param d2: y axis frame size
    :return: None, update W in place
    This is the FULL pipeline
    """
    Y = Y/np.amax(Y)
    d = d1 * d2
#     print('modified_constraints')
    # Solving d1*d2 subproblems min_0<=w<=upper_bound ||y - wX||^2,
    # where y here is each u_i, a row vector of dimension 1*r
    # w is a row vector weights, representing the weights on the ring of dimension 1*omega.shape[0] (size of the ring)
    # X are those u_i on the ring of each center pixel ( of dimension omega.shape[0]*r)
    d = d1*d2
    for i in range(d):
        if i%5000==0:
            print(i)
        omega = np.argwhere(W[i, :] > 0)[:, 1]  # use greater than 0, instead of not-equal to 0
        W[i, omega] = 0
        
        values = A_sum[omega]
        omega = omega[values == 0]

        if omega.shape[0] == 0:
            W[i, omega] = 1
            divisor = 1
        else:
            W[i,omega] = 1/omega.shape[0]
            divisor = omega.shape[0]
            
        y = Y[i, :]
        y = y.reshape([1, -1])
        W_re = W[i,omega].toarray()
        W_re = W_re.reshape([1, -1])
        XX = Y[omega, :]
        W[i, omega], _, _ = nnls.non_negative_factorization(y, 5/divisor, W_re, XX, n_components=omega.shape[0],init='custom', update_H=False, l2_reg_W=0, solver='cd', verbose=0)


        
def update_ring_model_w_const(U_sparse, R, V, A, X, b, W, d1, d2, T, r, mask_a=None, device = 'cpu', batch_size = 10000):
    """Update W matrix using ring model
    :param U: spatial component matrix from denoiser, R(d=d1xd2, N)
    :param V: temporal component matrix from denoiser, R(N, T). 
        MUST BE ORTHOGONAL: V(V^t) = I
    :param A: spatial component matrix, R(d, K)
    :param X: temporal component matrix (the actual temporal matrix C = XV),  R(K, N)
    :param b: constant baseline, dimension d x 1
    :param W: weighting matrix, scipy.sparse.csr_matrix. Dimensions (d, d)
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param T: number to time steps along time axis
    :param r: ring radius of ring model
    :return:
        W: updated weighting matrix
        b0: constant baseline of background image
    """
    
    start_time = time.time()
    if W is None:
        W = init_w(d1, d2, r)
    A_sparse = scipy.sparse.csr_matrix(A)
    
    if mask_a is None:
        A_sum = np.sum(A, axis=1)
    else:
        A_sum = np.sum(mask_a, axis=1)
    
    print("A_sum at {}".format(time.time() - start_time))
    
    W = update_w_1p_const(U_sparse, R, V, W.tocsr(), X, b, A_sparse, A_sum, d1, d2, batch_size = batch_size) #For now leave out batch size..
    return W
     

def update_w_1p_const(U_sparse, R, V, W, X, b, A_sparse, A_sum, d1, d2, batch_size = 10000):
    """Constant Ring Model codebase 
    params:
        U_sparse: scipy.sparse.csr_matrix. Dimensions d x r
        R: np.ndarray. Dimensions r x r
        V: np.ndarray. Dimensions r x T
        W: scipy.sparse.csr_matrix. Dimensions d x d
        A_sparse: scipy.sparse.csr_matrix. Dimensions d x k (k neurons)
        A_sum: the "summed" A onto 1 plane
        d1: x axis frame size
        d2: y axis frame size
        batch_size: number of ring indices to simultaneously update
    return:
        W: Sparse matrix (d x d) representing ring weights
    
    This is the CONST pipeline
    """
    first_time = time.time()
    d = d1 * d2
    weights = np.zeros((d, 1))

    iters = math.ceil(d / batch_size)
    
    start_time = time.time()
    #Preprocess W
    print("WE ARE IN THE SPATIAL W")
    W = W.tocoo()
    rows, cols, values = (W.row, W.col, W.data)
    indices = (values > 0)
    rows = rows[indices]
    cols = cols[indices]
    values[indices] = 1
    values = values[indices]
    
    #Turn off pixels in columns in which neurons are active
    support = (A_sum > 0)
    intersection = support[cols]
    inter_keep = (intersection == 0)
    
    rows = rows[inter_keep]
    cols = cols[inter_keep]
    values = values[inter_keep]
    
    W = coo_matrix((values, (rows, cols)), shape = (d,d))
    W = W.tocsr() #Convert to csr for quick multiply
    W.eliminate_zeros()
    
    #Concisely represent 'b' using the V basis. We want to say static_bg_movie = bsV for some row vector s. 
    # s should satisfy sV = 1 (the row vector of 1's)
    s = np.ones((1, V.shape[1])).dot(V.T)

    
    ##We want to calculate two diagonals: 
    ## (1) diag(W*(UR - AX - bs)*(UR - AX - bs)^t)
    ## (2) diag(W*(UR - AX - bs)*(UR - AX - bs)^t*W^t)
    
    #Note that WA is 0 (because we exclude ring weights over pixels of support of A). So we really want to find: 
    
    ## (1) T = diag(W*(UR - bs)*(UR - AX - bs)^t)
    ## (2) P = diag(W*(UR - bs)*(UR - bs)^t*W^t)
    
    ####NOTATION: rowsum(A, B) refers to the rowsum of the elementwise product of A and B
    
    ##We start with objective 1, and abbreviate diag by d() 
    ## diag(W*(UR - bs)*(UR - AX - bs)^t)
    ## = d(W(UR)(UR)^t) - d(W(UR)(AX)^t) - d(W(UR)(bs)^t) - d(W(bs)(UR)^t) + d(W(bs)(AX)^t) - d(W(bs)(bs)^t)
    
    T = np.zeros((d, 1))
    #We calculate the first three terms: d(W(UR)(UR)^t) - d(W(UR)(AX)^t) - d(W(UR)(bs)^t)
    
    #Precompute WU, which is sparse assuming W has ring structure, U is overlapping block-sparse
    WU = W.dot(U_sparse)

    
    ##Step 1.1: Find
    #d(W(UR)(UR)^t) = rowsum(WU, U*R*R^t)
    #d(W(UR)(AX)^t) = rowsum(WU, AXR^t)
    
    RRt = R.dot(R.T)
    XRt = X.dot(R.T)
    diag_WURRU = np.zeros((d,1))
    diag_WURAX = np.zeros((d,1))
    for k in range(iters):
        iter_time = time.time()
        start = batch_size*(k)
        end = start + batch_size
        WU_crop = WU[start:end, :]
        
                
        U_sparse_crop = U_sparse[start:end, :]
        URRtt_crop = U_sparse_crop.dot(RRt)

        WU_URRt = (WU_crop.multiply(URRtt_crop)).tocsr()


        diag_WURRU[start:end, :] = WU_URRt.sum(1)  

        
        AXRt_crop = A_sparse[start:end, :].dot(XRt)
        WU_AXRt = (WU_crop.multiply(AXRt_crop)).tocsr()
        diag_WURAX[start:end, :] = WU_AXRt.sum(1)
    
    T += diag_WURRU
    T -= diag_WURAX
    

    
    #Step 1.2: Find d(W(UR)(bs)^t)
    Rst = R.dot(s.T) #Dims r x 1
    URst = U_sparse.dot(Rst) #Dims d x 1
    WURst = W.dot(URst)
    diag_WURbs = WURst * b
    T -= diag_WURbs
    

    
    #Step 1.3: Get d(W(bs)(UR)^t) and d(W(bs)(AX)^t)
    ## d(W(bs)(UR)^t) = rowsum (Wb, URs^t)
    ## d(W(bs)(AX)^t) = rowsum(Wb, AXs^t)
    
    #Precompute Wb: 
    Wb = W.dot(b) 
    diag_WbsUR = Wb * URst
    
    T -= diag_WbsUR
    
    Xst = X.dot(s.T)
    AXst = A_sparse.dot(Xst)
    
    diag_WbsAX = Wb * AXst
    
    T += diag_WbsAX
    

    
    #Step 1.4: Add diag((bs)(bs)^t)
    T += Wb * ((s.dot(s.T))*b)
    

    

    
    ###Second objective: compute P = diag(W*(UR - bs)*(UR - bs)^t*W^t)
    P = np.zeros((d,1))
    ##Four terms to calculate..
    
    ##Step 2.1: Find diag(W(UR)(UR)^tW^t) = rowsum(WU, WURR^t)
    diag_WURURW = np.zeros((d,1))
    for k in range(iters):
        start = batch_size*(k)
        end = start + batch_size
        
        WU_crop = WU[start:end, :]
        WURRt_crop = WU_crop.dot(RRt)
        WURUR_prod = WU_crop.multiply(WURRt_crop)
        diag_WURURW[start:end, :] = WURUR_prod.sum(1)
    
    P += diag_WURURW
    

    
    ##Step 2.2: Find diag(W(UR)(bs)^tW^t) = diag(W(bs)(UR)^tW^t) and subtract these from P
    ## diag(W(UR)(bs)^tW^t) = diag(WURs^tb^tW^t) = rowsum(Wb, WURs^t)
    diag_WbsURW = Wb*WURst
    P -= 2*diag_WbsURW
    

    
    ## Step 2.3: Find diag(W(bs)(bs)^tW^t)
    ## = (ss^t)*rowsum(Wb, Wb)
    diag_WbsbsW = s.dot(s.T) * Wb * Wb
    P += diag_WbsbsW
    

    

    
    
    values = T / P
    values[values < 0] = 0
    values = np.nan_to_num(values, nan = 0, posinf = 0)
        
        
           
    #Add weights
    weights = values
    
    #Create new CSR matrix   
    pos = W.nonzero()
    rows = pos[0]
    values = weights[rows]
    W = csr_matrix((values.squeeze(), (pos[0], pos[1])), shape = (d, d))
    
    print("we are done. took {}".format(time.time() - first_time))
    return W.tocoo()