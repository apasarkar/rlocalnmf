#This file contains code for ring model localNMF
import cv2
import time


#Experimental import: 
import gc
# from memory_profiler import profile

import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as ss
import scipy.ndimage
import scipy.signal
import scipy.sparse
import scipy
import cvxpy as cvx

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import NMF
from sklearn import linear_model
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import distance_transform_bf
from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD
from matplotlib import ticker
from scipy.sparse import lil_matrix
from scipy import ndimage as ndimage
import math


import os
import sys
print(os.getcwd())


#Repo-specific imports:
from localnmf import ca_utils
from localnmf.constrained_ring.cnmf_e import update_ring_model_w_full, update_ring_model_w_const, init_w
from localnmf import regression_update


# import constrained_ring
# from constrained_ring.cnmf_e import update_ring_model_w_full, update_ring_model_w_const, init_w
# import ca_utils
# import regression_update


import multiprocessing

#Change the CPU affinity to allow multiprocessing


# To do
# split and merge functions

# ----- utility functions (to decimate data and estimate noise level) -----
#########################################################################################################


def resize(Y, size, interpolation=cv2.INTER_AREA):
    """
    :param Y:
    :param size:
    :param interpolation:
    :return:
    faster and 3D compatible version of skimage.transform.resize
    """
    if Y.ndim == 2:
        return cv2.resize(Y, tuple(size[::-1]), interpolation=interpolation)

    elif Y.ndim == 3:
        if np.isfortran(Y):
            return (cv2.resize(np.array(
                [cv2.resize(y, size[:2], interpolation=interpolation) for y in Y.T]).T
                .reshape((-1, Y.shape[-1]), order='F'),
                (size[-1], np.prod(size[:2])), interpolation=interpolation).reshape(size, order='F'))
        else:
            return np.array([cv2.resize(y, size[:0:-1], interpolation=interpolation) for y in
                    cv2.resize(Y.reshape((len(Y), -1), order='F'),
                        (np.prod(Y.shape[1:]), size[0]), interpolation=interpolation)
                    .reshape((size[0],) + Y.shape[1:], order='F')])
    else:  # TODO deal with ndim=4
        raise NotImplementedError
    return


def local_correlations_fft(Y, eight_neighbours=True, swap_dim=True, opencv=True):
    """
    Computes the correlation image for the input dataset Y using a faster FFT based method, adapt from caiman
    Parameters:
    -----------
    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format
    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively
    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front
    opencv: Boolean
        If True process using open cv method
    Returns:
    --------
    Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    Y -= np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Ystd[Ystd == 0] = np.inf
    Y /= Ystd

    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype='float32')
            sz[1, 1, 1] = 0
        else:
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype='float32')
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='float32')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    if opencv and Y.ndim == 3:
        Yconv = Y.copy()
        for idx, img in enumerate(Yconv):
            Yconv[idx] = cv2.filter2D(img, -1, sz, borderType=0)
        MASK = cv2.filter2D(
            np.ones(Y.shape[1:], dtype='float32'), -1, sz, borderType=0)
    else:
        Yconv = convolve(Y, sz[np.newaxis, :], mode='constant')
        MASK = convolve(
            np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')
    Cn = np.mean(Yconv * Y, axis=0) / MASK
    return Cn


def local_correlations_fft_UV(U, V, dims, eight_neighbours=True, a = None, c = None):
    """
    Computes the correlation image for the input dataset Y using a faster FFT based method, adapt from caiman
    Parameters:
    -----------
    U:  np.ndarray (d x R). Compressed spatial representation
    V:  np.ndarray (R x T). Compressed temporal representation
    
    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively
    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front
    opencv: Boolean
        If True process using open cv method
    Returns:
    --------
    Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    T = V.shape[1]
    
    if a is not None and c is not None:
        X = (c.T).dot(V.T)
        AX = a.dot(X)
        U = U - AX 
    
    UV_mean = (U.dot(V.sum(axis = 1, keepdims = True)))/T
    UV_mean_basis = UV_mean.dot((V.T).sum(axis = 0, keepdims = True))
    U_curr = U - UV_mean_basis
    norm = np.sqrt(np.sum(U_curr * U_curr, axis = 1, keepdims = True))
    U_norm = U_curr/norm
    
    U_norm = np.nan_to_num(U_norm, nan = 0, posinf = 0)
    
    U_norm = U_norm.reshape((dims[0], dims[1], U_norm.shape[1]), order = "F")
    
    
    if eight_neighbours:
        sz = np.ones((3, 3), dtype='float32')
        sz[1, 1] = 0
    else:
        sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    sz = sz.reshape((sz.shape[0], sz.shape[1], 1))
    Uconv = convolve(U_norm, sz, mode='constant')
    MASK = convolve(
            np.ones(U_norm.shape[:2], dtype='float32'), sz.squeeze(), mode='constant')
    Cn = np.sum(Uconv * U_norm, axis=2)/ MASK
    
    return Cn


def mean_psd(y, method ='logmexp'):
    """
    Averaging the PSD, adapt from caiman
    Parameters:
    ----------
        y: np.ndarray
             PSD values
        method: string
            method of averaging the noise.
            Choices:
             'mean': Mean
             'median': Median
             'logmexp': Exponential of the mean of the logarithm of PSD (default)
    Returns:
    -------
        mp: array
            mean psd
    """
    if method == 'mean':
        mp = np.sqrt(np.mean(np.divide(y, 2), axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(np.divide(y, 2), axis=-1))
    else:
        mp = np.log(np.divide((y + 1e-10), 2))
        mp = np.mean(mp, axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)

    return mp


def noise_estimator(Y, noise_range=[0.25, 0.5], noise_method='logmexp', max_num_samples_fft=4000,
                    opencv=True):
    """Estimate the noise level for each pixel by averaging the power spectral density.
    Inputs:
    -------
    Y: np.ndarray
    Input movie data with time in the last axis
    noise_range: np.ndarray [2 x 1] between 0 and 0.5
        Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
        default: [0.25,0.5]
    noise method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)
    Output:
    ------
    sn: np.ndarray
        Noise level for each pixel
    """
    T = Y.shape[-1]
    # Y=np.array(Y,dtype=np.float64)

    if T > max_num_samples_fft:
        Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1],
                            Y[..., np.int(T // 2 - max_num_samples_fft / 3 / 2):np.int(T // 2 + max_num_samples_fft / 3 / 2)],
                            Y[..., -max_num_samples_fft // 3:]), axis=-1)
        T = np.shape(Y)[-1]

    # we create a map of what is the noise on the FFT space
    ff = np.arange(0, 0.5 + 1. / T, 1. / T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1, ind2)
    # we compute the mean of the noise spectral density s
    if Y.ndim > 1:
        if opencv:
            import cv2
            psdx = []
            for y in Y.reshape(-1, T):
                dft = cv2.dft(y, flags=cv2.DFT_COMPLEX_OUTPUT).squeeze()[
                    :len(ind)][ind]
                psdx.append(np.sum(1. / T * dft * dft, 1))
            psdx = np.reshape(psdx, Y.shape[:-1] + (-1,))
        else:
            xdft = np.fft.rfft(Y, axis=-1)
            xdft = xdft[..., ind[:xdft.shape[-1]]]
            psdx = 1. / T * abs(xdft)**2
        psdx *= 2
        sn = mean_psd(psdx, method=noise_method)

    else:
        xdft = np.fliplr(np.fft.rfft(Y))
        psdx = 1. / T * (xdft**2)
        psdx[1:] *= 2
        sn = mean_psd(psdx[ind[:psdx.shape[0]]], method=noise_method)

    return sn

################################################# begin functions for superpixel analysis ##################################################
############################################################################################################################################

def threshold_data(Yd, th=2):
    """
    Threshold data: in each pixel, compute the median and median absolute deviation (MAD),
    then zero all bins (x,t) such that Yd(x,t) < med(x) + th * MAD(x).  Default value of th is 2.
 
    Parameters:
    ----------------
    Yd: 3d np.darray: dimension d1 x d2 x T
        denoised data
    Return:
    ----------------
    Yt: 3d np.darray: dimension d1 x d2 x T
        cleaned, thresholded data
    """
    dims = Yd.shape;
    Yt = np.zeros(dims);
    ii=0;
    for array in [Yd]:
        Yd_median = np.median(array, axis=2, keepdims=True)
        Yd_mad = np.median(abs(array - Yd_median), axis=2, keepdims=True)
        for i in range(dims[2]):
            Yt[:,:,i] = np.clip(array[:,:,i], a_min = (Yd_median + th*Yd_mad)[:,:,0], a_max = None) - (Yd_median + th*Yd_mad)[:,:,0]
    return Yt

def find_superpixel(Yt, cut_off_point, length_cut, eight_neighbours=True):
    """
    Find superpixels in Yt.  For each pixel, calculate its correlation with neighborhood pixels.
    If it's larger than threshold, we connect them together.  In this way, we form a lot of connected components.
    If its length is larger than threshold, we keep it as a superpixel.
    Parameters:
    ----------------
    Yt: 3d np.darray, dimension d1 x d2 x T
        thresholded data
    cut_off_point: double scalar
        correlation threshold
    length_cut: double scalar
        length threshold
    eight_neighbours: Boolean
        Use 8 neighbors if true.  Defalut value is True.
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

    dims = Yt.shape;
    ref_mat = np.arange(np.prod(dims[:-1])).reshape(dims[:-1],order='F')
    ######### calculate correlation ############
    w_mov = (Yt.transpose(2,0,1) - np.mean(Yt, axis=2)) / np.std(Yt, axis=2);
    w_mov[np.isnan(w_mov)] = 0;

    rho_v = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_h = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    if eight_neighbours:
        rho_l = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:]), axis=0)
        rho_r = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:]), axis=0)

    rho_v = np.concatenate([rho_v, np.zeros([1, rho_v.shape[1]])], axis=0)
    rho_h = np.concatenate([rho_h, np.zeros([rho_h.shape[0],1])], axis=1)
    if eight_neighbours:
        rho_r = np.concatenate([rho_r, np.zeros([rho_r.shape[0],1])], axis=1)
        rho_r = np.concatenate([rho_r, np.zeros([1, rho_r.shape[1]])], axis=0)
        rho_l = np.concatenate([np.zeros([rho_l.shape[0],1]), rho_l], axis=1)
        rho_l = np.concatenate([rho_l, np.zeros([1, rho_l.shape[1]])], axis=0)

    ################## find pairs where correlation above threshold
    temp_v = np.where(rho_v > cut_off_point);
    A_v = ref_mat[temp_v];
    B_v = ref_mat[(temp_v[0] + 1, temp_v[1])]

    temp_h = np.where(rho_h > cut_off_point);
    A_h = ref_mat[temp_h];
    B_h = ref_mat[(temp_h[0], temp_h[1] + 1)]

    if eight_neighbours:
        temp_l = np.where(rho_l > cut_off_point);
        A_l = ref_mat[temp_l];
        B_l = ref_mat[(temp_l[0] + 1, temp_l[1] - 1)]

        temp_r = np.where(rho_r > cut_off_point);
        A_r = ref_mat[temp_r];
        B_r = ref_mat[(temp_r[0] + 1, temp_r[1] + 1)]

        A = np.concatenate([A_v,A_h,A_l,A_r])
        B = np.concatenate([B_v,B_h,B_l,B_r])
    else:
        A = np.concatenate([A_v,A_h])
        B = np.concatenate([B_v,B_h])

    ########### form connected componnents #########
    G = nx.Graph();
    G.add_edges_from(list(zip(A, B)))
    comps=list(nx.connected_components(G))

    connect_mat=np.zeros(np.prod(dims[:2]));
    idx=0;
    for comp in comps:
        if(len(comp) > length_cut):
            idx = idx+1;

    permute_col = np.random.permutation(idx)+1;

    ii=0;
    for comp in comps:
        if(len(comp) > length_cut):
            connect_mat[list(comp)] = permute_col[ii];
            ii = ii+1;
    connect_mat_1 = connect_mat.reshape(dims[0],dims[1],order='F');
    return connect_mat_1, idx, comps, permute_col



def vcorrcoef(U, V, c):
    """
    fast way to calculate correlation between c and Y(UV).
    """
    temp = (c - c.mean(axis=0,keepdims=True));
    return np.matmul(U, np.matmul(V - V.mean(axis=1,keepdims=True), temp/np.std(temp, axis=0, keepdims=True)));


def vcorrcoef_robust(U, V, c, pseudo=0):
    """
    fast way to calculate correlation between c and Y(UV).
    """
    temp = (c - c.mean(axis=0,keepdims=True))
    return np.matmul(U, np.matmul(V - V.mean(axis=1,keepdims=True),temp/np.sqrt(np.std(temp, axis=0, keepdims=True)**2 + pseudo**2)));


def vcorrcoef2(X,y):
    """
    calculate correlation between vector y and matrix X.
    """
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r


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
    #unique_pix = list(unique_pix);

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
    #coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(M[:,pure_pixels].T, M[:,pure_pixels])), M[:,pure_pixels].T), M);
    pure_pixels = np.array(pure_pixels);
    #coef_rank = coef.copy(); ##### from large to small
    #for ii in range(len(pure_pixels)):
    #	coef_rank[:,ii] = [x for _,x in sorted(zip(len(pure_pixels) - ss.rankdata(coef[:,ii]), pure_pixels))];
    return pure_pixels #, coef, coef_rank


    
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
#     Yd = Yd.reshape(np.prod(dims[:-1]),-1, order="F");

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


def make_mask(corr_img_all_r, corr, mask_a, num_plane=1,times=10,max_allow_neuron_size=0.2):
    """
    update the spatial support: connected region in corr_img(corr(Y,c)) which is connected with previous spatial support
    """
    s = np.ones([3,3]);
    unit_length = int(mask_a.shape[0]/num_plane);
    dims = corr_img_all_r.shape;
    corr_img_all_r = corr_img_all_r.reshape(dims[0],int(dims[1]/num_plane),num_plane,-1,order="F");
    mask_a = mask_a.reshape(corr_img_all_r.shape,order="F");
    corr_ini = corr;
    for ii in range(mask_a.shape[-1]):
        for kk in range(num_plane):
            jj=0;
            corr = corr_ini;
            if mask_a[:,:,kk,ii].sum()>0:
                while jj<=times:
                    labeled_array, num_features = scipy.ndimage.measurements.label(corr_img_all_r[:,:,kk,ii] > corr,structure=s);
                    u, indices, counts = np.unique(labeled_array*mask_a[:,:,kk,ii], return_inverse=True, return_counts=True);
                    #print(u);
                    if len(u)==1:
                        labeled_array = np.zeros(labeled_array.shape);
                        if corr == 0 or corr == 1:
                            break;
                        else:
                            print("corr too high!")
                            corr = np.maximum(0, corr - 0.1);
                            jj = jj+1;
                    else:
                        if num_features>1:
                            c = u[1:][np.argmax(counts[1:])];
                            #print(c);
                            labeled_array = (labeled_array==c);
                            del(c);

                        if labeled_array.sum()/unit_length < max_allow_neuron_size or corr==1 or corr==0:
                            break;
                        else:
                            print("corr too low!")
                            corr = np.minimum(1, corr + 0.1);
                            jj = jj+1;
                mask_a[:,:,kk,ii] = labeled_array;
    mask_a = (mask_a*1).reshape(unit_length*num_plane,-1,order="F");
    return mask_a


def make_mask_rigid(corr_img_all_r, corr, mask_a):
    """
    update the spatial support: connected region in corr_img(corr(Y,c)) which is connected with previous spatial support
    """
    s = np.ones([3,3]);
    mask_a = (mask_a.reshape(corr_img_all_r.shape,order="F")).copy()
    for ii in range(mask_a.shape[2]):
        labeled_array, num_features = scipy.ndimage.measurements.label(corr_img_all_r[:,:,ii] > corr,structure=s);
        u, indices, counts = np.unique(labeled_array*mask_a[:,:,ii], return_inverse=True, return_counts=True);
        
        if len(u)==1:
            mask_a[:, :, ii] *= 0
        else:
            c = u[1:][np.argmax(counts[1:])];
            #print(c);
            labeled_array = (labeled_array==c);
            mask_a[:,:,ii] = labeled_array;

    return mask_a.reshape((-1, mask_a.shape[2]), order = "F")
    
    

def merge_components(a,c,corr_img_all_r,num_list,patch_size,merge_corr_thr=0.6,merge_overlap_thr=0.6,plot_en=False):
    """ want to merge components whose correlation images are highly overlapped,
    and update a and c after merge with region constrain
    Parameters:
    -----------
    a: np.ndarray
         matrix of spatial components (d x K)
    c: np.ndarray
         matrix of temporal components (T x K)
    corr_img_all_r: np.ndarray
         corr image
    U, V: low rank decomposition of Y
    normalize_factor: std of Y
    num_list: indices of components
    patch_size: dimensions for data
    merge_corr_thr:   scalar between 0 and 1
        temporal correlation threshold for truncating corr image (corr(Y,c)) (default 0.6)
    merge_overlap_thr: scalar between 0 and 1
        overlap ratio threshold for two corr images (default 0.6)
    Returns:
    --------
    a_pri:     np.ndarray
            matrix of merged spatial components (d x K')
    c_pri:     np.ndarray
            matrix of merged temporal components (T x K')
    corr_pri:   np.ndarray
            matrix of correlation images for the merged components (d x K')
    flag: merge or not
    """

    f = np.ones([c.shape[0],1]);
    ############ calculate overlap area ###########
    a = csc_matrix(a);
    a_corr = scipy.sparse.triu(a.T.dot(a),k=1);
    cor = csc_matrix((corr_img_all_r>merge_corr_thr)*1);
    temp = cor.sum(axis=0);
    cor_corr = scipy.sparse.triu(cor.T.dot(cor),k=1);
    cri = np.asarray((cor_corr/(temp.T)) > merge_overlap_thr)*np.asarray((cor_corr/temp) > merge_overlap_thr)*((a_corr>0).toarray());
    a = a.toarray();

    connect_comps = np.where(cri > 0);
    if len(connect_comps[0]) > 0:
        flag = 1;
        a_pri = a.copy();
        c_pri = c.copy();
        G = nx.Graph();
        G.add_edges_from(list(zip(connect_comps[0], connect_comps[1])))
        comps=list(nx.connected_components(G))
        merge_idx = np.unique(np.concatenate([connect_comps[0], connect_comps[1]],axis=0));
        a_pri = np.delete(a_pri, merge_idx, axis=1);
        c_pri = np.delete(c_pri, merge_idx, axis=1);
#         corr_pri = np.delete(corr_img_all_r, merge_idx, axis=1);
        num_pri = np.delete(num_list,merge_idx);
        for comp in comps:
            comp=list(comp);
            print("merge" + str(num_list[comp]+1));
            a_zero = np.zeros([a.shape[0],1]);
            a_temp = a[:,comp];
            if plot_en:
                spatial_comp_plot(a_temp, corr_img_all_r[:,comp].reshape(patch_size[0],patch_size[1],-1,order="F"),num_list[comp],ini=False);
            mask_temp = np.where(a_temp.sum(axis=1,keepdims=True) > 0)[0];
            
            a_temp = a_temp[mask_temp,:];
            y_temp = np.matmul(a_temp, c[:,comp].T);
            a_temp = a_temp.mean(axis=1,keepdims=True);
            c_temp = c[:,comp].mean(axis=1,keepdims=True);
            model = NMF(n_components=1, init='custom')
            a_temp = model.fit_transform(y_temp, W=a_temp, H = (c_temp.T));
            a_zero[mask_temp] = a_temp;
            c_temp = model.components_.T;
            
            a_pri = np.hstack((a_pri,a_zero));
            c_pri = np.hstack((c_pri,c_temp));
            num_pri = np.hstack((num_pri,num_list[comp[0]]));
        return flag, a_pri, c_pri, num_pri
    else:
        flag = 0;
        return flag


def delete_comp(a, c, corr_img_all_reg, corr_img_all, mask_a, num_list, temp, word, plot_en, fov_dims):
    """
    delete those zero components
    """
    print(word);
    pos = np.where(temp)[0];
    print("delete components" + str(num_list[pos]+1));
    corr_img_all_reg_r = corr_img_all_reg.reshape((fov_dims[0], fov_dims[1], -1), order = "F")
    if plot_en:
        spatial_comp_plot(a[:,pos], corr_img_all_reg_r[:,:,pos], num_list=num_list[pos], ini=False);
    corr_img_all_reg = np.delete(corr_img_all_reg, pos, axis=1);
    corr_img_all = np.delete(corr_img_all, pos, axis = 1);
    mask_a = np.delete(mask_a, pos, axis=1);
    a = np.delete(a, pos, axis=1);
    c = np.delete(c, pos, axis=1);
    num_list = np.delete(num_list, pos);
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


def l1_tf(y, sigma):
    """
    L1_trend filter to denoise the final temporal traces
    """
    if np.abs(sigma/y.max())<=1e-3:
        print('Do not denoise (high SNR: noise_level=%.3e)'%sigma);
        return y
#
    n = y.size
    # Form second difference matrix.
    D = (np.diag(2*np.ones(n),0)+np.diag(-1*np.ones(n-1),1)+np.diag(-1*np.ones(n-1),-1))[1:n-1];
    x = cvx.Variable(n)
    obj = cvx.Minimize(cvx.norm(D*x, 1));
    constraints = [cvx.norm(y-x,2)<=sigma*np.sqrt(n)]
    prob = cvx.Problem(obj, constraints)
#
    prob.solve(solver=cvx.ECOS,verbose=False)

    # Check for error.
    if prob.status != cvx.OPTIMAL:
        raise Exception("Solver did not converge!")
        return y
    return np.asarray(x.value).flatten()

##################################################### plot functions ############################################################################
#################################################################################################################################################

def pure_superpixel_corr_compare_plot(connect_mat_1, unique_pix, pure_pix, brightness_rank_sup, brightness_rank, Cnt, text=False):
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
    connect_mat_1_pure = connect_mat_1_pure.reshape(np.prod(dims),order="F");
    connect_mat_1_pure[~np.in1d(connect_mat_1_pure,pure_pix)]=0;
    connect_mat_1_pure = connect_mat_1_pure.reshape(dims,order="F");

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
    return fig


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


def temporal_comp_plot(c, num_list=None, ini = False):
    num = c.shape[1];
    fig = plt.figure(figsize=(20,1.5*num))
    if num_list is None:
        num_list = np.arange(num);
    for ii in range(num):
        plt.subplot(num,1, ii+1);
        plt.plot(c[:,ii]);
        if ii == 0:
            if ini:
                plt.title("Temporal components initialization for pure superpixels",fontweight="bold",fontsize=15);
            else:
                plt.title("Temporal components",fontweight="bold",fontsize=15);
        plt.ylabel(f"{num_list[ii]+1}",fontweight="bold",fontsize=15)
        if (ii > 0 and ii < num-1):
            plt.tick_params(axis='x',which='both',labelbottom='off')
        else:
            plt.xlabel("frames");
    plt.tight_layout()
    plt.show()
    return fig


def spatial_comp_plot(a, corr_img_all_r, num_list=None, ini=False):
    num = a.shape[1];
    patch_size = corr_img_all_r.shape[:2];
    scale = np.maximum(1, (corr_img_all_r.shape[1]/corr_img_all_r.shape[0]));
    fig = plt.figure(figsize=(8*scale,4*num));
    if num_list is None:
        num_list = np.arange(num);
    for ii in range(num):
        plt.subplot(num,2,2*ii+1);
        plt.imshow(a[:,ii].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
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


def spatial_sum_plot(a, a_fin, patch_size, num_list_fin=None, text=False):
    scale = np.maximum(1, (patch_size[1]/patch_size[0]));
    fig = plt.figure(figsize=(16*scale,8));
    ax = plt.subplot(1,2,1);
    ax.imshow(a_fin.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");

    if num_list_fin is None:
        num_list_fin = np.arange(a_fin.shape[1]);
    if text:
        for ii in range(a_fin.shape[1]):
            temp = a_fin[:,ii].reshape(patch_size,order="F");
            pos0 = np.where(temp == temp.max())[0][0];
            pos1 = np.where(temp == temp.max())[1][0];
            ax.text(pos1, pos0, f"{num_list_fin[ii]+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")

    ax.set(title="more passes spatial components")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")

    ax1 = plt.subplot(1,2,2);
    ax1.imshow(a.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");

    if text:
        for ii in range(a.shape[1]):
            temp = a[:,ii].reshape(patch_size,order="F");
            pos0 = np.where(temp == temp.max())[0][0];
            pos1 = np.where(temp == temp.max())[1][0];
            ax1.text(pos1, pos0, f"{ii+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")

    ax1.set(title="1 pass spatial components")
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold")
    plt.tight_layout();
    plt.show()
    return fig


def spatial_sum_plot_single(a_fin, patch_size, num_list_fin=None, text=False):
    scale = np.maximum(1, (patch_size[1]/patch_size[0]));
    fig = plt.figure(figsize=(4*scale,4));
    ax = plt.subplot(1,1,1);
    ax.imshow(a_fin.sum(axis=1).reshape(patch_size,order="F"),cmap="nipy_spectral_r");

    if num_list_fin is None:
        num_list_fin = np.arange(a_fin.shape[1]);
    if text:
        for ii in range(a_fin.shape[1]):
            temp = a_fin[:,ii].reshape(patch_size,order="F");
            pos0 = np.where(temp == temp.max())[0][0];
            pos1 = np.where(temp == temp.max())[1][0];
            ax.text(pos1, pos0, f"{num_list_fin[ii]+1}", verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)

    ax.set(title="Cumulative spatial components")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")

    plt.tight_layout();
    plt.show()
    return fig


def spatial_match_projection_plot(order, number, rlt_xya, rlt_yza, dims1, dims2):
    number = (order>=0).sum();
    scale = (dims1[1]+dims2[1])/max(dims1[0],dims2[0]);
    fig = plt.figure(figsize=(scale*2, 2*number));
    temp0 = np.where(order>=0)[0];
    temp1 = order[temp0];
    for ii in range(number):
        plt.subplot(number,2,2*ii+1);
        plt.imshow(rlt_xya[:,temp0[ii]].reshape(dims1[:2],order="F"),cmap="jet",aspect="auto");
        if ii == 0:
            plt.title("xy",fontsize=15,fontweight="bold");
            plt.ylabel("x",fontsize=15,fontweight="bold");
            plt.xlabel("y",fontsize=15,fontweight="bold");

        plt.subplot(number,2,2*ii+2);
        plt.imshow(rlt_yza[:,temp1[ii]].reshape(dims2[:2],order="F"),cmap="jet",aspect="auto");
        if ii == 0:
            plt.title("zy",fontsize=15,fontweight="bold");
            plt.ylabel("z",fontsize=15,fontweight="bold");
            plt.xlabel("y",fontsize=15,fontweight="bold");
    plt.tight_layout()
    return fig


def spatial_compare_single_plot(a, patch_size):
    scale = (patch_size[1]/patch_size[0]);
    fig = plt.figure(figsize=(4*scale,4));
    ax1 = plt.subplot(1,1,1);
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    img1 = ax1.imshow(a.reshape(patch_size,order="F"),cmap='nipy_spectral_r');
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(img1, cax=cax,orientation='vertical',spacing='uniform')
    plt.tight_layout();
    plt.show();
    return fig


def spatial_compare_nmf_plot(a, a_lasso_den, a_lasso_raw, order_Yd, order_Yraw, patch_size):
    num = a.shape[1];
    scale = (patch_size[1]/patch_size[0]);
    fig = plt.figure(figsize=(12*scale,4*num));

    for ii in range(num):
        ax0=plt.subplot(num,3,3*ii+1);
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        img0=plt.imshow(a[:,ii].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        if ii==0:
            plt.title("Our method",fontweight="bold",fontsize=15);

        ax1=plt.subplot(num,3,3*ii+2);
        if ii==0:
            plt.title("Sparse nmf on denoised data",fontweight="bold",fontsize=15);
        if order_Yd[ii]>=0:
            img1=plt.imshow(a_lasso_den[:,order_Yd[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

        ax2=plt.subplot(num,3,3*ii+3);
        if ii==0:
            plt.title("Sparse nmf on raw data",fontweight="bold",fontsize=15);
        if order_Yraw[ii]>=0:
            img2=plt.imshow(a_lasso_raw[:,order_Yraw[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    plt.tight_layout()
    plt.show()
    return fig


def spatial_compare_nmf_gt_plot(a_gt, a, a_lasso_den, a_lasso_raw, order_Ys, order_Yd, order_Yraw, patch_size):
    num = a_gt.shape[1];
    scale = np.maximum(1, (patch_size[1]/patch_size[0]));
    fig = plt.figure(figsize=(16*scale,4*num));

    for ii in range(num):
        ax00=plt.subplot(num,4,4*ii+1);
        img00=plt.imshow(a_gt[:,ii].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        if ii==0:
            plt.title("Ground truth",fontweight="bold",fontsize=15);

        ax0=plt.subplot(num,4,4*ii+2);
        if order_Ys[ii]>=0:
            img0=plt.imshow(a[:,order_Ys[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        if ii==0:
            plt.title("Our method",fontweight="bold",fontsize=15);

        ax1=plt.subplot(num,4,4*ii+3);
        if ii==0:
            plt.title("Sparse nmf on denoised data",fontweight="bold",fontsize=15);
        if order_Yd[ii]>=0:
            img1=plt.imshow(a_lasso_den[:,order_Yd[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

        ax2=plt.subplot(num,4,4*ii+4);
        if ii==0:
            plt.title("Sparse nmf on raw data",fontweight="bold",fontsize=15);
        if order_Yraw[ii]>=0:
            img2=plt.imshow(a_lasso_raw[:,order_Yraw[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    plt.tight_layout()
    plt.show()
    return fig


def temporal_compare_nmf_plot(c, c_lasso_den, c_lasso_raw, order_Yd, order_Yraw):
    num = c.shape[1];
    fig = plt.figure(figsize=(20,1.5*num))
    for ii in range(num):
        plt.subplot(num,1, ii+1);
        plt.plot(c[:,ii],label="our method");
        if order_Yd[ii]>=0:
            plt.plot(c_lasso_den[:,order_Yd[ii]],label="sparse nmf on denoised data");
        if order_Yraw[ii]>=0:
            plt.plot(c_lasso_raw[:,order_Yraw[ii]],label="sparse nmf on raw data");
        plt.legend();
        if ii == 0:
            plt.title("Temporal components",fontweight="bold",fontsize=15);
        plt.ylabel(f"{ii+1}",fontweight="bold",fontsize=15)
        if (ii > 0 and ii < num-1):
            plt.tick_params(axis='x',which='both',labelbottom='off')
        else:
            plt.xlabel("frames");
    plt.tight_layout()
    plt.show()
    return fig


def temporal_compare_plot(c, c_tf, ini = False):
    num = c.shape[1];
    fig = plt.figure(figsize=(20,1.5*num))
    for ii in range(num):
        plt.subplot(num,1, ii+1);
        plt.plot(c[:,ii],label="c");
        plt.plot(c_tf[:,ii],label="c_tf");
        plt.legend();
        if ii == 0:
            if ini:
                plt.title("Temporal components initialization for pure superpixels",fontweight="bold",fontsize=15);
            else:
                plt.title("Temporal components",fontweight="bold",fontsize=15);
        plt.ylabel(f"{ii+1}",fontweight="bold",fontsize=15)
        if (ii > 0 and ii < num-1):
            plt.tick_params(axis='x',which='both',labelbottom='off')
        else:
            plt.xlabel("frames");
    plt.tight_layout()
    plt.show()
    return fig

###########################    
######## Updated code/functions with 
######## robust correlation measures
###########################
    

def find_superpixel_robust(Yt, cut_off_point, length_cut, eight_neighbours=True, pseudo=0):
    """
    Find superpixels in Yt.  For each pixel, calculate its correlation with neighborhood pixels.
    If it's larger than threshold, we connect them together.  In this way, we form a lot of connected components.
    If its length is larger than threshold, we keep it as a superpixel.
    Parameters:
    ----------------
    Yt: 3d np.darray, dimension d1 x d2 x T
        thresholded data
    cut_off_point: double scalar
        correlation threshold
    length_cut: double scalar
        length threshold
    eight_neighbours: Boolean
        Use 8 neighbors if true.  Defalut value is True.
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

    dims = Yt.shape;
    ref_mat = np.arange(np.prod(dims[:-1])).reshape(dims[:-1],order='F')
    ######### calculate correlation ############
#     w_mov = (Yt.transpose(2,0,1) - np.mean(Yt, axis=2)) / (np.sqrt((np.std(Yt, axis=2)**2)+pseudo**2));
    w_mov = (Yt.transpose(2,0,1) - np.mean(Yt, axis=2)) / (np.sqrt((np.std(Yt, axis=2)**2)+(pseudo**2)));
    w_mov[np.isnan(w_mov)] = 0;

    rho_v = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_h = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    if eight_neighbours:
        rho_l = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:]), axis=0)
        rho_r = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:]), axis=0)

    rho_v = np.concatenate([rho_v, np.zeros([1, rho_v.shape[1]])], axis=0)
    rho_h = np.concatenate([rho_h, np.zeros([rho_h.shape[0],1])], axis=1)
    if eight_neighbours:
        rho_r = np.concatenate([rho_r, np.zeros([rho_r.shape[0],1])], axis=1)
        rho_r = np.concatenate([rho_r, np.zeros([1, rho_r.shape[1]])], axis=0)
        rho_l = np.concatenate([np.zeros([rho_l.shape[0],1]), rho_l], axis=1)
        rho_l = np.concatenate([rho_l, np.zeros([1, rho_l.shape[1]])], axis=0)

    ################## find pairs where correlation above threshold
    temp_v = np.where(rho_v > cut_off_point);
    A_v = ref_mat[temp_v];
    B_v = ref_mat[(temp_v[0] + 1, temp_v[1])]

    temp_h = np.where(rho_h > cut_off_point);
    A_h = ref_mat[temp_h];
    B_h = ref_mat[(temp_h[0], temp_h[1] + 1)]

    if eight_neighbours:
        temp_l = np.where(rho_l > cut_off_point);
        A_l = ref_mat[temp_l];
        B_l = ref_mat[(temp_l[0] + 1, temp_l[1] - 1)]

        temp_r = np.where(rho_r > cut_off_point);
        A_r = ref_mat[temp_r];
        B_r = ref_mat[(temp_r[0] + 1, temp_r[1] + 1)]

        A = np.concatenate([A_v,A_h,A_l,A_r])
        B = np.concatenate([B_v,B_h,B_l,B_r])
    else:
        A = np.concatenate([A_v,A_h])
        B = np.concatenate([B_v,B_h])

    ########### form connected componnents #########
    G = nx.Graph();
    G.add_edges_from(list(zip(A, B)))
    comps=list(nx.connected_components(G))

    connect_mat=np.zeros(np.prod(dims[:2]));
    idx=0;
    for comp in comps:
        if(len(comp) > length_cut):
            idx = idx+1;

    np.random.seed(2)
    permute_col = np.random.permutation(idx)+1;

    ii=0;
    for comp in comps:
        if(len(comp) > length_cut):
            connect_mat[list(comp)] = permute_col[ii];
            ii = ii+1;
    connect_mat_1 = connect_mat.reshape(dims[0],dims[1],order='F');
    return connect_mat_1, idx, comps, permute_col


def threshold_data_robust(Yd, th=2, c=1/4):
    """
    Threshold data: in each pixel, compute the median and median absolute deviation (MAD),
    then zero all bins (x,t) such that Yd(x,t) < med(x) + th * MAD(x).  Default value of th is 2.
 
    Parameters:
    ----------------
    Yd: 3d np.darray: dimension d1 x d2 x T
        denoised data
    Return:
    ----------------
    Yt: 3d np.darray: dimension d1 x d2 x T
        cleaned, thresholded data
    """
    dims = Yd.shape;
    Yt = np.zeros(dims);
    ii=0;
    for array in [Yd]:
        Yd_median = np.median(array, axis=2, keepdims=True)
        Yd_mad = np.median(abs(array - Yd_median), axis=2, keepdims=True)
        for i in range(dims[2]):
            Yt[:,:,i] = np.clip(array[:,:,i], a_min = (Yd_median + th*Yd_mad)[:,:,0], a_max = None) - (Yd_median + th*Yd_mad)[:,:,0]
            
    ##Finally, threshold on c
    Yt[Yt < c] = 0
    return Yt
    


def delete_small_comps_mnmf(a, c, mask_a, U,V, corr_th_fix, cut_off, dims, normalize_factor):
    """
    Delete small components and return 
    a_new: updated a matrix
    c_new: updated c matrix
    mask_new: updated mask matrix
    """
    corr_img_all = vcorrcoef(U/normalize_factor, V, c);
    corr_img_all_r = corr_img_all.reshape(dims[0],dims[1],-1,order="F");
    keeps = []
    for k in range(a.shape[1]):
        val = np.count_nonzero(corr_img_all_r[:,:, k]*(corr_img_all_r[:, :, k]>corr_th_fix))
        print(val)
        if val >= cut_off:
            keeps.append(k)
    a_new = a[:, keeps]
    c_new = c[:, keeps]
    mask_a_new = mask_a[:, keeps]
    if len(keeps) < a.shape[1]:
        print("we deleted components!")
    
    return a_new, c_new, mask_a_new


def print_signals(ai, ci, dims):
    for k in range(ai.shape[1]):
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(ai[:, k].reshape(dims[0], dims[1], order="F"))
        ax[1].plot(ci[:, k])
        plt.show()

def print_signals_corr(ai, ci, U, V, corr_th_fix, normalize_factor, dims, figsize = (12,6)):
    corr_img_all = vcorrcoef_robust(U/normalize_factor, V, ci, 0);
    corr_img_all_r = corr_img_all.reshape(dims[0],dims[1],-1,order="F");
    bin_ai = np.zeros_like(ai)
    bin_ai[ai > 0] = 1
    for k in range(ai.shape[1]):
        fig, ax = plt.subplots(1,4, figsize=figsize)
        #Calculate the correlation image
        
        ax[0].imshow(ai[:, k].reshape(dims[0], dims[1], order="F"))
        ax[0].set_title("A_i")
        ax[1].imshow(corr_img_all_r[:, :, k]*(corr_img_all_r[:, :, k]>0))
        ax[1].set_title("Corr Img Th: {}".format(corr_th_fix))
        ax[2].imshow(bin_ai[:, k].reshape(dims[0], dims[1], order="F"))
        ax[2].set_title("A_i Support")
        ax[3].plot(ci[:, k])
        ax[3].set_title("C_i trace")
        plt.show()
        
        
####
## Implementation of localNMF with bessel initialization options
## Main changes: 
## (1) Add explicit step to specify local tiles
## (2) Add different correlation thresholds for the resid and regular corr images
####


def avg_interpolation(W, d1, d2, order = "F"):
    '''
    args:
        W: (d1 x d2 x (d1 * d2)): the weights matix
    returns: 
        W_new (d1 x d2 x (d1*d2)): the new matrix
    '''
    print(d1)
    print(d2)
    print(W.shape)
    print(type(W))
    W_r = (W.reshape((d1, d2, -1), order=order)).copy()
    sum_W = np.sum(W, axis = 1, keepdims = True)
    zeros = (sum_W == 0)
    if np.count_nonzero(zeros == 0) == 0:
        return W #Everything is zero..
    zeros_r = zeros.reshape((d1,d2), order=order)
    
    output = ndimage.distance_transform_bf(zeros_r, metric="taxicab")
    max_val = np.amax(output)
    if max_val == 0:
        return W #No modification needed

    #Skip component
    for k in range(max_val + 2):
        if k == 0:
            continue
        else:
            maxIndices = np.where(output == k)
            maxIndicesX, maxIndicesY = maxIndices[:]
            
            for index in range(len(maxIndicesX)):
                curr_x = maxIndicesX[index]
                curr_y = maxIndicesY[index]
                
#                 print(" {} {}".format(curr_x, curr_y))
                
                #For each coordinate, we find how many components in its neighborhood are nonneg. We average their 
                #ring values...
                count = 0
                #Left
                if curr_y - 1 >= 0:
                    if zeros_r[curr_x, curr_y - 1] == 0:
                        count += 1
                        W_r[curr_x, curr_y, :] += W_r[curr_x, curr_y - 1, :]
                        zeros_r[curr_x, curr_y] = 0
                #Right
                if curr_y + 1 < d2:
                    if zeros_r[curr_x, curr_y + 1] == 0: 
                        count += 1
                        W_r[curr_x, curr_y, :] += W_r[curr_x, curr_y + 1, :]
                        zeros_r[curr_x, curr_y] = 0
                
                if curr_x - 1 >= 0:
                    if zeros_r[curr_x - 1, curr_y] == 0:
                        count += 1
                        W_r[curr_x , curr_y, :] += W_r[curr_x - 1, curr_y, :]
                        zeros_r[curr_x, curr_y] = 0
                
                if curr_x + 1 < d1:
                    if zeros_r[curr_x + 1, curr_y] == 0:
                        count += 1
                        W_r[curr_x, curr_y, :] += W_r[curr_x + 1, curr_y]
                        zeros_r[curr_x, curr_y] = 0
                
                if count > 0:
                    W_r[curr_x, curr_y, :] /= count
                else:
                    print(k)
                
                    
    return W_r.reshape((d1*d2, d1*d2), order = "F")


def print_signals_corrimg(ai, ci, corr_th_fix, corr_img_r, dims, figsize = (12,6)):
    bin_ai = np.zeros_like(ai)
    bin_ai[ai > 0] = 1
    for k in range(ai.shape[1]):
        fig, ax = plt.subplots(1,4, figsize=figsize)
        #Calculate the correlation image
        
        ax[0].imshow(ai[:, k].reshape(dims[0], dims[1], order="F"))
        ax[0].set_title("A_i")
        ax[1].imshow(corr_img_r[:, :, k]*(corr_img_r[:, :, k]>corr_th_fix))
        ax[1].set_title("Corr Img Th: {}".format(corr_th_fix))
        ax[2].imshow(bin_ai[:, k].reshape(dims[0], dims[1], order="F"))
        ax[2].set_title("A_i Support")
        ax[3].plot(ci[:, k])
        ax[3].set_title("C_i trace")
        plt.show()

        
 
def vcorrcoef_resid(U_sparse, R, V, a, c, batch_size = 10000, tol = 0.000001):
    '''
    Residual correlation image calculation 
    Params:
        U_sparse: scipy.sparse.coo_matrix. Dimensions (d x d)
        R: numpy.ndarray. Dimensions (r x r)
        V: numpy.ndarray. Dimensions (r x T). V is orthogonal; i.e. V.dot(V.T) is identity
            The row of 1's must belong in the row space of V
        a: numpy.ndarray. Dimensions (d x k)
        c: numpy.ndaray. Dimensions (T x k)
        batch_size: number of pixels to process at once. Limits matrix sizes to O((batch_size+T)*r)
    '''
    T = V.shape[1]
    X = (c.T).dot(V.T) 
    d = U_sparse.shape[0]
    
    
    corr_img = np.zeros((d, a.shape[1]))
    
    c_mean = np.mean(c, axis = 0, keepdims = True)
    c_norm = np.sqrt(np.sum((c-c_mean) * (c-c_mean), axis = 0, keepdims = True))
    c_standard = (c - c_mean) / c_norm
    
    a_sparse = scipy.sparse.coo_matrix(a).tocsr()
    
        
    #Define number of iterations so we only process batch_size pixels at a time
    batch_iters = math.ceil(d / batch_size)
    
    
    #Precompute V_mean quantity for faster calculations
    V_mean = np.mean(V, axis = 1, keepdims = True)
    

    ##Step 2: For each iteration below, we will express the 'consant'-mean movie in terms of V basis: Mean_Movie = m*s*V for some (1 x r) vector s. We know sV should be a row vector of 1's. So we solve sV = 1; since V is orthogonal:
    s = np.ones((1, V.shape[1])).dot(V.T)

    
    #Precompute diag((UR)(UR)^t)
    diag_URRtUt = np.zeros((d, 1))
    RRt = R.dot(R.T)
    for k in range(batch_iters):
        start = batch_size * k
        end = batch_size * (k+1)
        U_crop = U_sparse[start:end, :]
        UR_crop = U_sparse[start:end, :].dot(RRt)
        UmulUR = (U_crop.multiply(UR_crop)).tocsr()
        
        diag_URRtUt[start:end, :] = UmulUR.sum(1)

    #Precompute diag((AX)(AX)^t)
    diag_AXXtAt = np.zeros((d, 1))
    XXt = X.dot(X.T)
    for k in range(batch_iters):
        start = batch_size * k
        end = batch_size * (k+1)
        A_crop = a_sparse[start:end, :]
        AX_crop = A_crop.dot(XXt)
        AmulAXXt = (A_crop.multiply(AX_crop)).tocsr()
        
        diag_AXXtAt[start:end, :] = AmulAXXt.sum(1)
        
    #Precompute diag((AX)(UR)^t) and diag(UR(AX)^t) (they are the same thing)
    diag_AXUR = np.zeros((d,1))
    XRt = X.dot(R.T)
    for k in range(batch_iters):
        start = batch_size * k
        end = batch_size * (k+1)
        A_crop = a_sparse[start:end, :]
        U_crop = U_sparse[start:end, :]
        AXRt_crop = A_crop.dot(XRt)
        U_mulAXRt = (U_crop.multiply(AXRt_crop)).tocsr()
        
        diag_AXUR[start:end, :] = U_mulAXRt.sum(1)
    

    for k in range(c.shape[1]):
        c_curr = c_standard[:, [k]]
                
            
        #Define X_i and A_i
        keeps = [i for i in range(a.shape[1])]
        del keeps[k]
        A_k = a_sparse[:, keeps].tocsr()
        X_k = X[keeps, :]
        
        ##Step 1: Get mean of the movie: (UR - A_k * X_k)V
        RV_mean = R.dot(V_mean)
        m_UR = U_sparse.dot(RV_mean) #Dims: d x 1 
        
        XkV_mean = X_k.dot(V_mean) #Dims: k x 1
        m_AX = A_k.dot(XkV_mean) #Dims: d x 1
        m = m_UR - m_AX #Final mean
                
        
        ##Step 2: Get square of norm of mean-subtracted movie. Let A_k denote k-th neuron of A and X_k denote k-th row of X
        ## We have: Y_res = (UR - AX - A_k * X_k - ms)V. 
        ## Square of norm is equivalent to diag(Y_res * Y_res^t). Abbreviate diag() by d()
        ## = d((UR)(UR)^t) - d((AX)(UR)^t) + d((A_k*X_k)(UR)^t) - d(ms(UR)^t)
        ##     - d(UR(AX)^t) + d(AX(AX)^t) - d((A_kX_k)(A_kX_k)^t) + d((ms)(AX)^t)
        ##     + d((UR)(A_kX_k)^t) - d((AX)((A_kX_k)^t)) + d(((A_kX_k)(A_kX_k)^t) - d((ms)(A_kX_k)^t)
        ##     - d(UR(ms)^t) + d((AX)(ms)^t) - d((A_kX_k)(ms)^t) + d((ms)(ms)^t)
        
        ##We find the diagonal of each term individually, and add the results.
        
        final_diagonal = np.zeros((d, 1))
        
        ## Step 2.a Add/subtract precomputed values, d((UR)(UR)^t) and  d(AX(AX)^t) to final_diagonal
        final_diagonal += diag_URRtUt
        final_diagonal += diag_AXXtAt
        final_diagonal -= 2*diag_AXUR
        
        ## Step 2.b. Add diag(A_k * X_k (UR)^t) + diag(UR(A_k * X_k)^t) = 2*diag(URX_k^t * A_k^t) to final_diagonal
        RX_k = R.dot(X[[k], :].T)
        URX_k = U_sparse.dot(RX_k) #Dims: d x 1
        diag_URAkXk  = a[:,[k]] * URX_k
        final_diagonal += 2*diag_URAkXk
        
        ## Step 2.c. Subtract diag((AX)(A_k*X_k)^t) + diag((A_k*X_k)*(AX)^t) = 2*diag((A_k*X_k)*(AX)^t) from final diagonal
        XX_k = X.dot(X[[k], :].T) #Dims k x 1
        AXX_k = a_sparse.dot(XX_k) #Dims d x 1
        diag_AXAkXk = a[:, [k]] * AXX_k
        
        final_diagonal -= 2*diag_AXAkXk
        
        ## 2.d. Subtract d(UR(ms)^t) + d(ms(UR)^t) = 2*d(URs^tm^t)
        Rst = R.dot(s.T)
        URst = U_sparse.dot(Rst)
        diag_URstm = URst*m
        
        final_diagonal -= 2*diag_URstm
        
        ## 2.e. Add d(AX(ms)^t) + d(ms(AX)^t) = 2*d(AXs^tm^t) to final_diagonal
        Xst = X.dot(s.T)
        AXst = a_sparse.dot(Xst) #d x 1
        diag_AXstm = AXst*m
        
        final_diagonal += 2*diag_AXstm
        
        ## 2.f. Subtract d((A_kX_k)(ms)^t) + d((ms)(A_kX_k)^t) = 2*d(A_kX_ks^tm^t)
        Xkst = X[[k], :].dot(s.T) # 1 x 1
        diag_AkXkms = Xkst * (a[:, [k]]*m)
        final_diagonal -= 2* diag_AkXkms
        
        ## 2.g. Add d((ms)(ms)^2)
        sst = s.dot(s.T)
        diag_msms = m*m*sst
        final_diagonal += diag_msms
        
        ## 2.g. Add d(((A_kX_k)(A_kX_k)^t)
        XkXk = X[[k], :].dot(X[[k], :].T) #Single value
        a_norm = a[:, [k]] * a[:, [k]] 
        diag_axxa = (a_norm) * XkXk
        
        final_diagonal += diag_axxa
    
        norm = np.sqrt(final_diagonal)
        norm[norm < tol] = 0 #Set arbitrarily small values to 0 
        
        #Find the unnormalized pixel-wise product, and normalize after..
        Vc = V.dot(c_curr)
        
        corr_fin = np.zeros((d,1))
        
        sVc = s.dot(Vc)
        msVc = m.dot(sVc)
        corr_fin -= msVc
        
        XVc = X_k.dot(Vc)
        AXVc = A_k.dot(XVc)
        corr_fin -= AXVc
        
        RVc = R.dot(Vc)
        URVc = U_sparse.dot(RVc)
        corr_fin += URVc
        
        ##Finally, divide by norm of the residual movie 
        corr_fin /= norm
        corr_fin = np.nan_to_num(corr_fin, nan = 0, posinf = 0, neginf = 0)
        
        
        corr_img[:, [k]] = corr_fin
        
        
    return corr_img


        
        


def vcorrcoef_UV_noise(U_sparse, R, V, c, pseudo = 0, batch_size = 10000, tol = 0.000001):
    '''
    New standard correlation calculation. Finds the correlation image of each neuron in 'c' 
    with the denoised movie URV
    
    Params:
        U_sparse: scipy.sparse.coo_matrix. dims (d x r), where the FOV has d pixels
        R: np.ndarray. dims (r x r), where r is the rank of the PMD decomposition
        V: np.ndarray. dims (r x T), where T is the number of frames in the movie
        c: np.ndarray. dims (T x k), where k is the number of neurons
        pseudo: nonnegative integer
        batch_size: maximum number of pixels to process at a time. (batch_size x R)-sized matrices will be constructed
    '''
    T = V.shape[1]
    d = U_sparse.shape[0]
    
    U_sparse = U_sparse.tocsr()
    U_sparse_t = U_sparse.transpose().tocsc() ##Is this needed? 
    
    ##Step 1: Standardize c
    c_mean = np.mean(c, axis = 0, keepdims = True)
    c_norm = np.sqrt(np.sum((c-c_mean) * (c-c_mean), axis = 0, keepdims = True))
    c_standard = (c - c_mean) / c_norm
    
    ##Step 2: Calculate the mean of Yd = URV:
    V_mean = np.mean(V, axis = 1, keepdims = True) #V_mean has dims r x 1
    RV_mean = R.dot(V_mean)
    m = U_sparse.dot(RV_mean) #Dims: d x 1 
    
    ## Step 3: Express the movie mean in the V basis as: m*s*V, where s is a 1 x r vector. s*V should be a row of 1's, length T
    ## Since V is orthogonal, can find s easily: 
    s = np.ones((1, V.shape[1])).dot(V.T)
    
    ##Note: Now the mean subtracted movie is: (U*R - m*s)*V
    
    ##Step 4: Find the pixelwise norm: sqrt(diag((U*R - m*s)*V*V^t*(U*R - m*s)^t)) 
    ## diag((U*R - mov_mean*S)*V*V^t*(U*R - mov_mean*S)^t) = diag((U*R - m*s)*(U*R - m*s)^t) since V orthogonal
    ## diag((U*R - m*s)*(U*R - m*s)^t) = diag(U*R*R^t*U^t - U*R*s^t*m^t - m*s*R^t*U^t + m*s*s^t*m^t)
    ## diag(U*R*R^t*U^t - U*R*s^t*m^t - m*s*R^t*U^t + m*s*s^t*m^t) = diag(U*R*R^t*U^t) - diag(U*R*s^t*m^t) - diag(m*s*R^t*U^t) + diag(m*s*s^t*m^t)
    
    ##Step 4a: Get diag(U*R*s^t*m^t) and diag(m*s*R^t*U^t)
    #These are easy because U*R*s^t and s*R^t*U^t are 1-dimensional and transposes of each other: 
    
    Rst = R.dot(s.T)
    URst = U_sparse.dot(Rst) #This is d x 1
    
    #Now diag(U*R*s^t*m^t) is easy:
    diag_URstmt = URst*m #Element-wise product
    
    #Now diag(m*s*R^t*U^t) is easy: 
    diag_msRtUt = m*URst
    
    ##Step 4b: Get diag(m*s*s^t*m^t)
    #Note that s*s^t just a dot product
    s_dot = s.dot(s.T)
    diag_msstmt = s_dot * (m*m)
    
    ## Step 4c: Get diag(U*R*R^t*U^t)
    diag_URRtUt = np.zeros((U_sparse.shape[0], 1))
    RRt = R.dot(R.T) #r x r matrix
    
    batch_iters = math.ceil(d / batch_size)
    for k in range(batch_iters):
        start = batch_size * k
        end = batch_size * (k+1)
        U_crop = U_sparse[start:end, :]
        UR_crop = U_sparse[start:end, :].dot(RRt)
        UmulUR = (U_crop.multiply(UR_crop)).tocsr()
        
        diag_URRtUt[start:end, :] = UmulUR.sum(1)
    
        
    norm_sqrd = diag_URRtUt - diag_msRtUt - diag_URstmt + diag_msstmt
    norm = np.sqrt(norm_sqrd)
    norm[norm < tol] = 0
    
    ## Step 5: Get the dot product between normalized c and each unnormalized pixel
    
    #First precompute Vc: 
    Vc = V.dot(c_standard) #r x k matrix
    
    #Find (UR - ms)V*c
    RVc = R.dot(Vc)
    URVc = U_sparse.dot(RVc)
    
    sVc = s.dot(Vc)
    msVc = m.dot(sVc)
    
    fin_corr = URVc - msVc
    
    #Step 6: Divide by pixelwise norm from step 4
    fin_corr /= norm

    
    fin_corr = np.nan_to_num(fin_corr, nan = 0, posinf = 0, neginf = 0)
    return fin_corr    


def merge_components_priors(a,c,corr_img_all_r,num_list,patch_size,merge_corr_thr=0.6,merge_overlap_thr=0.6,plot_en=False, dims = (64,64,3600), \
                           frame_len = 25, confidence = 0.99, spatial_thres = [0.8,0.8], allowed_overlap = 30, model = None, plot_mnmf = True):
    """ want to merge components whose correlation images are highly overlapped,
    and update a and c after merge with region constrain
    Parameters:
    -----------
    a: np.ndarray
         matrix of spatial components (d x K)
    c: np.ndarray
         matrix of temporal components (T x K)
    corr_img_all_r: np.ndarray
         corr image
    U, V: low rank decomposition of Y
    normalize_factor: std of Y
    num_list: indices of components
    patch_size: dimensions for data
    merge_corr_thr:   scalar between 0 and 1
        temporal correlation threshold for truncating corr image (corr(Y,c)) (default 0.6)
    merge_overlap_thr: scalar between 0 and 1
        overlap ratio threshold for two corr images (default 0.6)
    Returns:
    --------
    a_pri:     np.ndarray
            matrix of merged spatial components (d x K')
    c_pri:     np.ndarray
            matrix of merged temporal components (T x K')
    corr_pri:   np.ndarray
            matrix of correlation images for the merged components (d x K')
    flag: merge or not
    Model: Neural network we use to analyze the summary image..
    """

    f = np.ones([c.shape[0],1]);
    ############ calculate overlap area ###########
    a = csc_matrix(a);
    a_corr = scipy.sparse.triu(a.T.dot(a),k=1);
    cor = csc_matrix((corr_img_all_r>merge_corr_thr)*1);
    temp = cor.sum(axis=0);
    cor_corr = scipy.sparse.triu(cor.T.dot(cor),k=1);
    cri = np.asarray((cor_corr/(temp.T)) > merge_overlap_thr)*np.asarray((cor_corr/temp) > merge_overlap_thr)*((a_corr>0).toarray());
    a = a.toarray();

    connect_comps = np.where(cri > 0);
    if len(connect_comps[0]) > 0:
        flag = 1;
        a_pri = a.copy();
        c_pri = c.copy();
        G = nx.Graph();
        G.add_edges_from(list(zip(connect_comps[0], connect_comps[1])))
        comps=list(nx.connected_components(G))
        merge_idx = np.unique(np.concatenate([connect_comps[0], connect_comps[1]],axis=0));
        a_pri = np.delete(a_pri, merge_idx, axis=1);
        c_pri = np.delete(c_pri, merge_idx, axis=1);
        num_pri = np.delete(num_list,merge_idx);
        print("num_list is {}".format(num_list))
        
        ##Load MASK_RCNN network only if we need to do merges
        model_nn = mnmf.load_MASKRCNN(model)
        
        for comp in comps: 
            print(comp)
        for comp in comps:
            comp=list(comp);
            print("merge" + str(num_list[comp]+1));
            a_zero = np.zeros([a.shape[0],1]);
            a_temp = a[:,comp];
            if plot_en or True:
                spatial_comp_plot(a_temp, corr_img_all_r[:,comp].reshape(patch_size[0],patch_size[1],-1,order="F"),num_list[comp],ini=False);
            mask_temp = np.where(a_temp.sum(axis=1,keepdims=True) > 0)[0];
            a_temp = a_temp[mask_temp,:];
            y_temp = np.matmul(a_temp, c[:,comp].T);
            
            #Here we do the visualization we need to diagnose things:
            debug = False
            if debug: 
                y_temp_avg = np.mean(y_temp, axis = 1)
                y_temp_avg_img = np.zeros((dims[0]*dims[1],))
                y_temp_avg_img[mask_temp] = y_temp_avg
                y_temp_avg_img = y_temp_avg_img.reshape((dims[0],dims[1]), order="F")
                plt.figure()
                plt.imshow(y_temp_avg_img)
                plt.show()

                for k in range(len(comp)):
                    plt.figure()
                    plt.plot(c[:, comp[k]])
                    plt.show()
                print("THAT WAS THE MEAN MERGE IMAGE! and the traces") 
            
            
                a_debug = a[:, comp]
                c_debug = c[:, comp]
                print("HERE ARE THE NEURONS IN QUESTION BEFORE MRCNN")
                for index in range(len(comp)):
                    fig, ax = plt.subplots(1,2)
                    ax[0].imshow(a_debug[:, index].reshape((dims[0],dims[1]), order="F"))
                    ax[1].plot(c_debug[:, index])
                    plt.show()
            
            
            a_temp, c_temp = mnmf.maskNMF_merge(a[:, comp], c[:, comp], frame_len,confidence, spatial_thres, model_nn, plot_mnmf, allowed_overlap = allowed_overlap, max_comps = len(comp),\
                                                 dims = dims)

            #In this case all components were discarded
            if a_temp is None or c_temp is None:
                continue
                
                
            print("THE NUMBER OF COMPONENTS DEMIXED HERE IS {}".format(a_temp.shape[1]))
            print("THE SIZE OF COMPS WAS {}".format(len(comps)))
            
            print("HERE IS HOW THINGS LOOK AFTER THE BESSEL INIT")
            for index in range(a_temp.shape[1]):
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(a_temp[:, index].reshape((dims[0],dims[1]), order="F"))
                ax[1].plot(c_temp[:, index])
                plt.show()

#             corr_temp = vcorrcoef(U/normalize_factor, V.T, c_temp);

            a_pri = np.hstack((a_pri,a_temp));
            c_pri = np.hstack((c_pri,c_temp));
            for num_add in range(a_temp.shape[1]):
                num_pri = np.hstack((num_pri,num_list[comp[num_add]]));
        del model_nn
        torch.cuda.empty_cache()
#         num_pri = c_pri.shape[1] #Re-order relevant components
        return flag, a_pri, c_pri, num_pri
    else:
        flag = 0;
        return flag

    
def dilate_mask(mask_a, dilation_size):
    """
    Function to dilate a given mask:
    params:
        mask_a: ndarray, d1 x d2 x K. d1 and d2 are the x and y dimensions of the field of view. K is the number of distinct objects
            Expected that mask_a is a binary matrix
    """
    new_mask = np.zeros_like(mask_a)
    for k in range(mask_a.shape[2]):
        curr_frame = mask_a[:, :, k]
        support = (curr_frame == 0)
        distances = distance_transform_bf(support, metric = 'chessboard')
#         print("SHOWINIG THE DISTANCES AND SUPPORT")
#         fig, ax = plt.subplots(1,2)
#         ax[0].imshow(support)
#         ax[1].imshow(distances)
#         plt.show()
        
        values = (distances < dilation_size + 1)
        new_mask[:, :, k] = values
    
#     print("NOW SHOWING THE FINAL UPDATES")
#     for k in range(mask_a.shape[2]):
#         fig, ax = plt.subplots(1,2)
#         ax[0].imshow(mask_a[:, :, k])
#         ax[1].imshow(new_mask[:, :, k])
#         plt.show()
    
    return new_mask
        

def process_custom_signals(a_init, U_sparse, V):
    '''
    Custom initialization: Given a set of neuron spatial footprints ('a'), this provides initial estimates to the other component (temporal traces, baseline, fluctuating background)
    Terms:
        d1, d2: the dimensions of the FOV
        K: number of neurons identified
        R: rank of the PMD decomposition
    
    Params:
        a: np.ndarray, dimensions (d1, d2, K) where K is the number of neurons, 
    
    
    TODO: Eliminate awkward naming issues in 'process custom signals'
    '''
    dims = (a_init.shape[0], a_init.shape[1], V.shape[1])
    
    A = a_init
    A_mask = (a_init>0)
    
    A_re = A.reshape(dims[0]*dims[1],-1, order="F")
    A_mask_re = A_mask.reshape(dims[0]*dims[1],-1,order="F")

    c = np.zeros((dims[2], A_re.shape[1]))
    
    a = A_re
    a_mask=A_mask_re
    
    
    W = scipy.sparse.csr_matrix((U_sparse.shape[0],U_sparse.shape[0]))
    X = np.zeros((A_re.shape[1], V.shape[0]))
    b = np.zeros((U_sparse.shape[0], 1))
    
    C_new = regression_update.temporal_update_HALS(U_sparse, V, W, X, A_re, c, b)
    c = C_new
    
      
    ####Delete any zero components
    ####TODO: completely eliminate 
    deletions = []
    for k in range(a.shape[1]):
        curr_trace = c[:, k]
        if np.count_nonzero(curr_trace) == 0:
            deletions.append(k)
            print("delete {}".format(k))
    keeps = []
    for m in range(a.shape[1]):
        if m in deletions:
            continue
        else:
            keeps.append(m)
    a = a[:, keeps]
    a_mask=a_mask[:,keeps]
    c = c[:, keeps]
    print("THIS IS THE SHAPE AFTER DELETIONS")
    print(a.shape)
    print(c.shape)
 
    ###
    # Estimate background (stationary and fluctuating) for localNMF 
    ###
    ##We estimate stationary background
    uv_mean = U_sparse.dot((V.mean(axis = 1, keepdims = True))) #Pixel-wise mean
    b = regression_update.baseline_update(uv_mean, a, c)
    
    
    a_used = a.astype("double")
    b_used = b.astype("double")
    c_used = c.astype("double")
    
    return a_used, a_mask, b_used, c_used

    
# def process_custom_signals(a, c, b, U, V, dims, dilation = 2,  scale_divisor = 10):
#     """
#     Function to initialize "a" and "c" matrices. "a" and "c" are estimated outside of the standard demixing pipeline
#     params: 
#         a: ndarray, d x K. K = number of neurons, d = number of pixels in spatial support
#         c: ndarray, T x K. T = number of frames in imaging movie, K = number of neurons
#         b: ndarray, d x 1. d = number of pixels in spatial support
#         U: ndarray, d x r. d = number of pixels in spatial support, r = rank of compressed movie. U is a spatial representation for the movie
#         V: ndarray, r x T. r = rank of compressed movie. T = number of frames in imaging movie. V is a temporal representation for the movie  
#         dims: tuple (x, y). x * y = d (x and y are the dimensions of the field of view)
#     """
    
#     #First scale a*c to be compatible with the U*V movie. Using a constant scale factor, this becomes a 1d regression problem
# #     print("WE ARE NOW PRINTINIG A")
# #     for k in range(a.shape[1]):
# #         plt.figure()
# #         plt.imshow(a[:, k].reshape((dims[0], dims[1]), order = 'F'))
# #         plt.show()
#     a_sum = np.sum(a, axis = 1) > 0
#     U_crop = U[a_sum, :].dot(V.sum(axis = 1, keepdims = True))
#     Yd_sum = np.sum(U_crop.flatten())
    
    
#     ac_interm = a.dot(c.T.sum(axis = 1, keepdims = True))
#     ac_sum = np.sum(ac_interm.flatten())
    
#     scale = Yd_sum / ac_sum
#     scale /= scale_divisor
    
#     uv_mean = (U*((V.T).mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)
    
#     #Rescale ac based on above estimate
#     c = c * scale
    
#     mask_a = (a > 0)*1
#     b = b * scale
# #     c = ls_solve_acc_ring(a, U - bg_basis, V.T, mask=None, beta_LS=c).T;
    
# #     mask_a = (a>0)*1
# #     for k in range(0):
# #         bg_basis = b.dot((V.T).sum(axis = 0, keepdims = True))
# #         c = ls_solve_acc_ring(a, U - bg_basis, V.T, mask=None, beta_LS=c).T;
        
# #         b = np.maximum(0, uv_mean-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));
# #         bg_basis = b.dot((V.T).sum(axis = 0, keepdims = True))
        
# #         a = ls_solve_ac(c, V.T, U - bg_basis, mask=mask_a.T, beta_LS=a).T;
# #         b = np.maximum(0, uv_mean-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));
        
    
#     #Now perform a baseline update
# #     uv_mean = (U*((V.T).mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)
    
#     new_mask = dilate_mask(mask_a.reshape((dims[0], dims[1], -1), order = "F"), dilation_size = dilation)
    
# #     for k in range(new_mask.shape[2]):
# #         plt.figure()
# #         plt.imshow(new_mask[:, :, k])
# #         plt.show()
#     new_mask = new_mask.reshape((dims[0]*dims[1], -1), order = "F")
    
#     return a, b, c, new_mask
  


def orthogonalize_UV(U, V):
    """
    U: scipy.sparse.csr matrix, dimensions (d x R)
    V: np.ndarray, dimensions (R x T)
    
    """
    print("Orthogonalizing UV")
    start_time = time.time()
    Q, R = np.linalg.qr(V.T) #Now UV = U(R^t)(Q^t)
    U_orth = U.dot(R.T)
    V = Q.T 
    print("Orthogonalization took {}".format(time.time() - start_time))
    return U_orth, V, R.T




#Define min_subtraction function
def get_min_vals_torch(U, V, device = 'cpu', batch_size = 10000):
    '''
    Function which min_subtracts the product UV
    params: 
        U: np.ndarray. d x R matrix
        V: np.ndarray. R x T matrix
        device: the device on which these computations will be executed (as defined in PyTorch)
        batch_size: 
    '''
    
    batch_values = math.ceil(U.shape[0]/batch_size)
    V_t = torch.from_numpy(V).float().to(device)
    prod_vals = np.zeros((U.shape[0],1))
    for j in range(batch_values):
        start = j*batch_size
        end = start + batch_size
        U_t = torch.from_numpy(U[start:end, :]).float().to(device)
        prod_orig = torch.matmul(U_t, V_t)
        prod = torch.min(prod_orig, dim = 1)[0]
        prod_store = prod.to('cpu').detach().numpy().astype("double")
        prod_vals[start:end,0] = prod_store
    torch.cuda.empty_cache()

    return prod_vals
        
    
def get_min_vals(U_sparse, V, batch_size = 10000):
    '''
    Function which calculates the pixel-wise minimum of every pixel in the movie, U_sparse * V
    params: 
        U_sparse: scipy.sparse.csr_matrix. d x r matrix
        V: np.ndarray. r x d matrix
        batch_size: number of pixels to use at a time
    '''
    start_time = time.time()
    d = U_sparse.shape[0]
    min_vals = np.zeros((d, 1))
    batches = math.ceil(d/batch_size)
    for k in range(batches):
        start = k*batch_size
        end = start+batch_size
        U_crop = U_sparse[start:end, :]
        UV_crop = U_crop.dot(V)
        UV_crop_min = np.amin(UV_crop, axis = 1, keepdims = True)
        
        min_vals[start:end, :] = UV_crop_min
        
    print("took {} seconds to find min_vals".format(time.time() - start_time))
    return min_vals

def get_median(tensor, axis):
    max_val = torch.max(tensor, dim=axis, keepdim=True)[0]
    tensor_med_1 = torch.median(torch.cat((tensor, max_val), dim = axis), dim = axis, keepdim=True)[0]
    tensor_med_2 = torch.median(tensor, dim = 2, keepdim = True)[0]
    
    tensor_med = torch.mul(tensor_med_1 + tensor_med_2, 0.5)
    return tensor_med

def threshold_data_inplace(Yd, th = 2):
    '''
    Threshold data: in each pixel, compute the median and median absolute deviation (MAD),
    then zero all bins (x,t) such that Yd(x,t) < med(x) + th * MAD(x).  Default value of th is 2.
    Inputs: 
        Yd: torch.Tensor, shape (d1, d2, T)
    Outputs:
        Yd: This is an in-place operation 
    '''
    
    #Get per-pixel medians
    Yd_med = get_median(Yd, axis = 2)
    diff = torch.sub(Yd, Yd_med)
    
    #Calculate MAD values
    torch.abs(diff, out=diff)
    MAD = get_median(diff, axis = 2)
    
    #Calculate actual threshold
    torch.mul(MAD, th, out=MAD)
    th_val = Yd_med.add(MAD)  
    
    #Subtract threshold values
    torch.sub(Yd,th_val, out = Yd)
    torch.clamp(Yd, min = 0, out = Yd)
    return Yd




def find_superpixel_UV(U_sparse, V, dims, cut_off_point, length_cut, th, eight_neighbours=True, \
                        device = 'cuda', a = None, c = None, batch_size = 10000, pseudo = 0, tol = 0.000001):
    """
    Find superpixels in Yt.  For each pixel, calculate its correlation with neighborhood pixels.
    If it's larger than threshold, we connect them together.  In this way, we form a lot of connected components.
    If its length is larger than threshold, we keep it as a superpixel.
    Parameters:
    ----------------
    Yt: 3d np.darray, dimension d1 x d2 x T
        thresholded data
    cut_off_point: double scalar
        correlation threshold
    length_cut: double scalar
        length threshold
    eight_neighbours: Boolean
        Use 8 neighbors if true.  Defalut value is True.
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

    dims = (dims[0], dims[1], V.shape[1])
    T = V.shape[1]
    ref_mat = np.arange(np.prod(dims[:-1])).reshape(dims[:-1],order='F')
    
    if a is not None and c is not None: 
        resid_flag = True
    else:
        resid_flag = False
        
    
    tiles = math.floor(math.sqrt(batch_size))
    
    iters_x = math.ceil((dims[0]/(tiles-1)))
    iters_y = math.ceil((dims[1]/(tiles-1)))
    
    
    if resid_flag: 
        a_sparse = scipy.sparse.csr_matrix(a)
        
    
    
    #Pixel-to-pixel coordinates for highly-correlated neighbors
    A_indices = []
    B_indices = []
    
    for tile_x in range(iters_x):
        for tile_y in range(iters_y):
            x_pt = (tiles-1)*tile_x
            x_end = x_pt + tiles
            y_pt = (tiles - 1)*tile_y
            y_end = y_pt + tiles
            
            indices_curr_2d = ref_mat[x_pt:x_end, y_pt:y_end]
            x_interval = indices_curr_2d.shape[0]
            y_interval = indices_curr_2d.shape[1]
            indices_curr = indices_curr_2d.reshape((x_interval*y_interval,), order = "F")
            
            Yd = U_sparse[indices_curr, :].dot(V).reshape((x_interval,y_interval, -1), order = "F")
            if resid_flag:
                ac_mov = a_sparse[indices_curr, :].dot(c.T)
                ac_mov = ac_mov.reshape((x_interval, y_interval, -1), order = "F")
                Yd -= ac_mov
            
            print("{}{}".format(tile_x, tile_y))
                
            Yd = torch.from_numpy(Yd).to(device)   
            
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
            print("vertical. after the elt wise mult, rho shape is {}".format(rho.shape))
            rho = torch.cat( (rho,torch.zeros([1, rho.shape[1]]).double().to(device)), dim = 0)
            temp_rho = rho.cpu().numpy()
            temp_indices = np.where(temp_rho > cut_off_point)
            A_curr = ref_mat[(temp_indices[0] + x_pt, temp_indices[1] + y_pt)]
            B_curr = ref_mat[(temp_indices[0] + x_pt + 1, temp_indices[1] + y_pt)]
            A_indices.append(A_curr)
            B_indices.append(B_curr)
            
            
            #Horizonal pixel correlations
            rho = torch.mean(Yd[:, :, :-1] * Yd[:, :, 1:], dim = 0)
            print("horizontal. after the element-wise multiply, rrho shape is {}".format(rho.shape))
            rho = torch.cat( (rho,torch.zeros([rho.shape[0], 1]).double().to(device)), dim = 1)
            temp_rho = rho.cpu().numpy()
            print("the shape of rho in horizonttal is {}".format(temp_rho.shape))
            temp_indices = np.where(temp_rho > cut_off_point)
            A_curr = ref_mat[(temp_indices[0] + x_pt, temp_indices[1] + y_pt)]
            B_curr = ref_mat[(temp_indices[0] + x_pt, temp_indices[1] + y_pt + 1)]
            A_indices.append(A_curr)
            B_indices.append(B_curr)
            
            if eight_neighbours:
                #Right-sided pixel correlations
                rho = torch.mean(Yd[:, :-1, :-1]*Yd[:, 1:, 1:], dim=0)
                rho = torch.cat((rho, torch.zeros([rho.shape[0],1]).double().to(device)), dim=1)
                rho = torch.cat((rho, torch.zeros([1, rho.shape[1]]).double().to(device)), dim=0)
                temp_rho = rho.cpu().numpy()
                temp_indices = np.where(temp_rho > cut_off_point)
                A_curr = ref_mat[(temp_indices[0] + x_pt, temp_indices[1] + y_pt)]
                B_curr = ref_mat[(temp_indices[0] + x_pt + 1, temp_indices[1] + y_pt + 1)]
                A_indices.append(A_curr)
                B_indices.append(B_curr)
                
                #Left-sided pixel correlations
                rho = torch.mean(Yd[:, 1:, :-1]*Yd[:, :-1, 1:], dim=0)
                rho = torch.cat( (torch.zeros([rho.shape[0],1]).double().to(device), rho), dim=1)
                rho = torch.cat((rho, torch.zeros([1, rho.shape[1]]).double().to(device)), dim=0)
                temp_rho = rho.cpu().numpy()
                temp_indices = np.where(temp_rho > cut_off_point)
                A_curr = ref_mat[(temp_indices[0] + x_pt, temp_indices[1] + y_pt)]
                B_curr = ref_mat[(temp_indices[0] + x_pt + 1, temp_indices[1] + y_pt - 1)]
                A_indices.append(A_curr)
                B_indices.append(B_curr)
                
            del rho
            del Yd
            torch.cuda.empty_cache()
            
     
    torch.cuda.empty_cache()
    A = np.concatenate(A_indices)
    B = np.concatenate(B_indices)

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
    connect_mat_1 = connect_mat.reshape(dims[0],dims[1],order='F');
    return connect_mat_1, idx, comps, permute_col




def spatial_temporal_ini_UV(U,V, dims, th, comps, idx, length_cut, a = None, c = None, device = 'cpu'):
    """
    Apply rank 1 NMF to find spatial and temporal initialization for each superpixel in Yt.
    """

    dims = (dims[0], dims[1], V.shape[1])
    T = V.shape[1]
    ii = 0;
    U_mat = np.zeros([np.prod(dims[:2]),idx]);
    V_mat = np.zeros([T,idx]);

    for comp in comps:
        if(len(comp) > length_cut):
            if ii % 100 == 0:
                print("we are initializing component {} out of {}".format(ii, idx))
            y_temp = U[list(comp), :].dot(V)
            
            if a is not None and c is not None:
                y_temp = y_temp - a[list(comp), :].dot(c.T)
            y_temp = threshold_data(y_temp[:, None, :], th)
#             y_temp[y_temp<0] = 0
            y_temp = y_temp.squeeze()
            model = NMF(n_components=1, init='custom');
            U_mat[list(comp),ii] = model.fit_transform(y_temp, W=y_temp.mean(axis=1,keepdims=True),
                                        H = y_temp.mean(axis=0,keepdims=True))[:,0];
            V_mat[:,ii] = model.components_;
            ii = ii+1;
            
    return V_mat, U_mat


def prune_zero_columns_UV(U, V):
    '''
    Prunes unused columns of U (and corresponding rows of V)
    
    '''
    print('the original shape of U and V is {} and {}'.format(U.shape, V.shape))
    col_count = np.sum((U != 0), axis = 0)
    print("the shape of col_count is {}".format(col_count.shape))
    keeps = np.argwhere(col_count != 0)
    U_new = U[:, keeps].squeeze()
    V_new = V[keeps, :].squeeze()
    
    print("the new shapes of U and V are {} and {}".format(U_new.shape, V_new.shape))
    
    return U_new, V_new
    
def demix_whole_data_robust_ring_lowrank(U,V_PMD,r=10, cut_off_point=[0.95,0.9], length_cut=[15,10], th=[2,1], pass_num=1, residual_cut = [0.6,0.6],
                    corr_th_fix=0.31, corr_th_fix_sec = 0.4, corr_th_del = 0.2, switch_point=10, max_allow_neuron_size=0.3, merge_corr_thr=0.6, merge_overlap_thr=0.6, num_plane=1, patch_size=[100,100],
                    plot_en=False, TF=False, fudge_factor=1, text=True, max_iter=35, max_iter_fin=50,
                    update_after=4, pseudo_1=[1/4, 1/4], pseudo_2=[5, 5], skips=2, update_type="Constant", init = ['mnmf', 'mnmf', 'lnmf'], custom_init = {}, block_dims=(16,16),frame_len=[100,50,50],confidence=0.99, spatial_thres=(0.75,0.50), model=None, allowed_overlap=5, \
                                              plot_mnmf = True, sb=True, pseudo_corr = [0,0], device = 'cpu', batch_size = 10000, plot_debug = False, denoise = False):
    '''
    This function is a low-rank pipeline with robust correlation measures and a ring background model. The low-rank implementation is in the HALS updates.
    Args:
        Yd_raw: ndarray, (d1 x d2 x T). 3D matrix describing raw video
        Yd: ndarray, (d1 x d2 x T). 3D matrix which is suitably preprocessed. Might include denoising, motion correction
            standardization, etc.
        U: ndarray, (d1 x d2 x K). A PMD-denoised and compressed spatial representation of Yd_raw. K is the rank of the th
            PMD decomposition.
        V: ndarray, (K x T). A PMD-denoised and compressed temporal representation of Yd_raw. K is the rank of the PMD 
            decomposition
        r: integer. The radius used in the ring model
        cut_off_point: list of values between 0 and 1, length = pass_num. Correlation thresholds used 
            in the superpixelization process
        length_cut: list of integers, length = pass_num. Minimum allowed sizes of superpixels in
            different passes. If length_cut = [2,3], it means in first pass, 
            minimum size is 2. In second pass, minimum size is 3
        th: list of integers, length = pass_num. Describes, for each pass, what median absolute deviation (MAD) 
            threshold is applied to the data during superpixelization process
        pass_num: number of passes localNMF algorithm takes over the dataset to identify neurons
        residual_cut: list of values between 0 and 1. Length of list = pass_num
            sqrt(1 - r_sqare of SPA)
            Standard value = 0.6
        corr_th_fix: correlation value (between 0 and 1). Correlation threshold used to update 
            support of neurons during localNMF updates
        corr_th_fix_sec: correlation value (between 0 and 1). Correlation threshold
            after 'switch_point' number of HALS updates 
        max_allow_neuron_size: value betwee 0 and 1. Max allowed max_i supp(ai) / (d1 x d2).
            If neuron i exceed this range, then when updating spatial support of ai, corr_th_fix will automatically increase 0.1; and will print("corr too low!") on screen.
            If there're too many corr too low on screen, you should consider increasing corr_th_fix.
        merge_corr_thr: float, correlation threshold for truncating corr(Yd, ci) when merging
        merge_overlap_thr: float, overlapped threshold for truncated correlation images (corr(Yd, ci)) when merging
        num_plane: integer. Currently, only num_plane = 1 is fully supporte
        patch_size: list, length = 2. Patch size used to find pure superpixels. Typical values = [100,100]. This parameter automatically adjusted if the field of view has any dimension with length less than 100.
        plot_en: boolean. If true, pipeline plots initialization and neuron estimates intermittently.
        TF: boolean. If True, then run l1_TF on temporal components after local NMF
        fudge_factor: float, usually set to 1
            do l1_TF up to fudge_factor*noise level i.e.
            min_ci' |ci'|_1 s.t. |ci' - ci|_F <= fudge_factor*sigma_i\sqrt(T)
        text: boolean. If true, prints numbers for each superpixel in the superpixel plot
        max_iter_fin: integer, number of HALS iterations in final pass
        max_iter: integer. Number of HALS updates for all pre-final passes (if pass_num > 0)
        update_after: integer. Merge and update spatial support every 'update_after' iterations
        pseudo_1: float (nonnegative). Robust parameter for MAD thresholding step
        pseudo_2: float nonnegsative). Robust parameter for correlation measures used in superpixel step
        skips: integer (nonnegative). For each pass of localNMF, the first 'skips' HALS updates 
            do not estimate the fluctuating background. NOTE: if you do not want fluctuating background at all, set skips
            to be any value greater than both max_iter and max_iter_fin.
        update_type: string, either "Constant" or "Full". Describes the type of ring update being performed
        init: list of strings, length = pass_num. Options:
                (a) 'mnmf' (neural network-based initialization)
                (b) 'lnmf' (superpixel based initialization)
                (c) 'custom' (custom init values provided). NOTE: only be used for pass #1
            For example, init = ['mnmf', 'lnmf'] means the first pass is initialized with 
            neural network, the second with superpixels.
        custom_init: dict. keys: 'a','b', c'. A dictionary describing the custom values (a,b,c) provided for demixing.
            'a' is the spatial footprint, 'b' is the baseline, 'c' is the temporal trace
        block_dims: tuple of integers, (a,b) (length 2). In 'mnmf' init, the algorithm breaks the field of view into tiles
            of size (a x b). For each tile, we find the 'frame_len' brightest frames over that tile and run the network on 
            those frames.
        frame_len: list of integers (length = pass_num). For each pass, if the initialization method is 'mnmf', this describes
            how many frames the algorithm looks at.
        confidence: float betwee 0 and 1. Accepted confidence level at which we accept neural network's output. 
            Generally keep at 0.99
        spatial_thres: tuple (length 2), both values are float between 0 and 1. Maximum allowed cosine similarity between
            neuron footprints identified in the 'mnmf' initialization. One threshold is between the footprints themselves, 
            the other is between their masks
        model: dict. data used to construct neural network Mask-RCNN data. Keys: 
            (1) 'path': path to Mask-RCNN saved weights (.h5 file)
            (2) 'inference_config' object, specifying details of Mask-RCNN execution
        allowed_overlap: amount of acceptable overlap (in pixels) between two masks identified by Mask-RCNN from single frame
        plot_mnmf: boolean. If true, we plot and display Mask-RCNN output as it is run at various point of pipeline
        sb: boolean. Stands for 'static background'. If false, we turn off static background estimates throughout pipeline
            Usually kept as true. 
        pseudo_corrr: list of nonnegative floats, length = pass_num. This is a robust correlation measure used when 
            calculating correlation images of neurons in the HALS updates. For data in which neurons don't overlap much, this
            should be left at 0.
        device: string identifying whether certain operations (matrix updates, etc.) should be moved to gpu and 
            accelerated. Standard options: 'cpu' for CPU-only. 'cuda' for GPU computations.
        batch_size: int. For GPU computing, identifies how many pixels to process at a time (to make the pipeline compatible 
            with GPUs of various memory constraints)
        plot_debug: boolean. Indicates whether intermediate-step visualizations should be generated during demixing. Used for purposes
            of visualization.
        
    '''
    ## Define Ring Model values..
    
    #First prune empty spatial basis vectors
    
    d1,d2 = U.shape[:2]
    dims = U.shape[:2];
    T = V_PMD.shape[1];
    
    U_used = U.reshape((-1, U.shape[2]), order="F")
    
    #Prune any columns which are all 0
    U_used, V_PMD = prune_zero_columns_UV(U_used, V_PMD)
    
    ##Subtraction Step
    

    print("BEFORE MIN SUB we have U is {} and V is {}".format((d1, d2, U_used.shape[1]), \
                                                              V_PMD.shape))
    ## if data has negative values then do pixel-wise minimum subtraction ##
    
    
    U_sparse = ca_utils.construct_sparse(U_used).tocsr()
    min_vals = get_min_vals(U_sparse, V_PMD, batch_size = batch_size)
    U_sparse = scipy.sparse.hstack([U_sparse.tocsc(), min_vals], format = 'csc').tocsr()
    V_PMD = np.hstack((V_PMD.T, -1*np.ones((V_PMD.shape[1], 1)))).T
    
    
    U_used, V, R = orthogonalize_UV(U_sparse, V_PMD)   
    
    if not plot_en:
        del U_used
    
    superpixel_rlt = [];
    
    ii = 0;
    while ii < pass_num:
        print("start " + str(ii+1) + " pass!");
                
        #######
        ### Initialize components
        #######
                
        start = time.time();
        if init[ii] == 'lnmf':   
            ## cut image into small parts to find pure superpixels ##

            patch_height = patch_size[0];
            patch_width = patch_size[1];
            height_num = int(np.ceil(dims[0]/patch_height));  ########### if need less data to find pure superpixel, change dims[0] here #################
            width_num = int(np.ceil(dims[1]/(patch_width*num_plane)));
            num_patch = height_num*width_num;
            patch_ref_mat = np.array(range(num_patch)).reshape(height_num, width_num, order="F");
    
            if num_plane > 1:
                raise ValueError('num_plane > 2 (higher dimensional data) not supported!')
            else:
                print("find superpixels!")
                
                if ii == 0:
                    connect_mat_1, idx, comps, permute_col = find_superpixel_UV(U_sparse, V_PMD, dims, cut_off_point[ii],length_cut[ii], th[ii], eight_neighbours=True, device = device, batch_size = batch_size, pseudo=pseudo_2[ii]); 
                else:
                    connect_mat_1, idx, comps, permute_col = find_superpixel_UV(U_sparse, V_PMD, dims, cut_off_point[ii],length_cut[ii], th[ii], eight_neighbours=True, device = device, a=a, c=c, batch_size = batch_size, pseudo=pseudo_2[ii]);
            print("time: " + str(time.time()-start));

            start = time.time();
            print("rank 1 svd!")
            if ii == 0:
                c_ini, a_ini = spatial_temporal_ini_UV(U_sparse.tocsr(), V_PMD, dims, th[ii], comps, idx, length_cut[ii])
            else:
                c_ini, a_ini = spatial_temporal_ini_UV(U_sparse.tocsr(), V_PMD, dims, th[ii], comps, idx, length_cut[ii], a = a, c = c)
                
            
            mask_a=None ## Disable the mask after first pass over data
            unique_pix = np.asarray(np.sort(np.unique(connect_mat_1)),dtype="int");
            unique_pix = unique_pix[np.nonzero(unique_pix)];
            brightness_rank_sup = order_superpixels(permute_col, unique_pix, a_ini, c_ini);
            pure_pix = [];

            start = time.time();
            print("find pure superpixels!")
            for kk in range(num_patch):
                pos = np.where(patch_ref_mat==kk);
                up=pos[0][0]*patch_height;
                down=min(up+patch_height, dims[0]);
                left=pos[1][0]*patch_width;
                right=min(left+patch_width, dims[1]);
                unique_pix_temp, M = search_superpixel_in_range((connect_mat_1.reshape(dims[0],int(dims[1]/num_plane),num_plane,order="F"))[up:down,left:right], permute_col, c_ini);
                pure_pix_temp = fast_sep_nmf(M, M.shape[1], residual_cut[ii]);
                if len(pure_pix_temp)>0:
                    pure_pix = np.hstack((pure_pix, unique_pix_temp[pure_pix_temp]));
            pure_pix = np.unique(pure_pix);
            
            
            start = time.time();
            print("prepare iteration!")
            if ii > 0:
                a_ini, c_ini, brightness_rank = prepare_iteration_UV((dims[0], dims[1], T), connect_mat_1, permute_col, pure_pix, a_ini, c_ini);
                a = np.hstack((a, a_ini));
                c = np.hstack((c, c_ini));
            elif ii == 0:
                a, c, brightness_rank = prepare_iteration_UV((dims[0], dims[1], T), connect_mat_1, permute_col, pure_pix, a_ini, c_ini, more=True);
#                 uv_mean = (U_used*((V.T).mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True);
                uv_mean = U_sparse.dot(R.dot(V.mean(axis = 1, keepdims = True))) #Pixel-wise mean
                b = regression_update.baseline_update(uv_mean, a, c)
            
            #Plot superpixel correlation image
            if plot_en:
                Cnt = local_correlations_fft_UV(U_used, V, dims);
                pure_superpixel_corr_compare_plot(connect_mat_1, unique_pix, pure_pix, brightness_rank_sup, brightness_rank, Cnt, text);
            
            print("time: " + str(time.time()-start));


        elif init[ii] == 'mnmf':  
            ##Here we use the bessel init:
            if ii == 0:
                a_ini, mask_a, b, c_ini = mnmf.bessel_init_local_UV(U_sparse, V_PMD, (d1,d2,T), block_dims, frame_len[ii], confidence, spatial_thres, model, \
                        plot_mnmf = plot_mnmf, allowed_overlap = allowed_overlap, device = device, \
                                                                   batch_size = batch_size)
                a = a_ini
                c = c_ini
            else:
                raise ValueError('maskNMF can only be run on the first pass')            
            
            
            
            start = time.time()
            print("prepare iteration!")
            
            print("time: " + str(time.time()-start));
                
        elif init[ii]=='custom' and ii == 0:
            a_ini, mask_a, b, c_ini = process_custom_signals(custom_init['a'].copy(), U_sparse, V_PMD)
            a = a_ini
            c = c_ini
       
        

        
        print("start " + str(ii+1) + " pass iteration!")
        if ii == pass_num - 1:
            maxiter = max_iter_fin;
        else:
            maxiter=max_iter;
        start = time.time();
        
        
        print("pre update AC")

        #######
        ## Run demixing pipeline
        #######
        print("BEFORE ENTERINIG DEMIX THE PATCH SIZE IS {}".format(patch_size))

      
        a, c, b, X, W, res, corr_img_all_r, num_list = update_AC_bg_l2_Y_ring_lowrank(U_sparse, R, V, V_PMD, r,dims,\
                                                                                   a, c, b, dims,
                                        corr_th_fix, corr_th_fix_sec, corr_th_del, switch_point, maxiter=maxiter, tol=1e-8, update_after=update_after,
                                        merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane, plot_en=plot_en, max_allow_neuron_size=max_allow_neuron_size, skips=skips, update_type=update_type, mask_a=mask_a,sb=sb, pseudo_corr = pseudo_corr[ii], model = model, plot_mnmf = plot_mnmf, device = device, batch_size = batch_size, plot_debug = plot_debug, denoise = denoise);
        print("time: " + str(time.time()-start));
        
        
        
        print("POST update AC")
        
        
        if init[ii] == 'lnmf':
            superpixel_rlt.append({'connect_mat_1':connect_mat_1, 'pure_pix':pure_pix, 'unique_pix':unique_pix, 'brightness_rank':brightness_rank, 'brightness_rank_sup':brightness_rank_sup});
        
        #If multi-pass, save results from first pass
        if pass_num > 1 and ii == 0:
            rlt = {'a':a, 'c':c, 'b':b, "X":X, "W":W, 'res':res, 'corr_img_all_r':corr_img_all_r, 'num_list':num_list};
            a0 = a.copy();
        ii = ii+1;

    c_tf = [];
    start = time.time();
    if TF:
        sigma = noise_estimator(c.T);
        sigma *= fudge_factor
        for ii in range(c.shape[1]):
            c_tf = np.hstack((c_tf, l1_tf(c[:,ii], sigma[ii])));
        c_tf = c_tf.reshape(T,int(c_tf.shape[0]/T),order="F");
    print("time: " + str(time.time()-start));
    if plot_en:
        if pass_num > 1:
            spatial_sum_plot(a0, a, dims, num_list, text);
        Cnt = local_correlations_fft_UV(U_used, V, dims, a=a, c=c);
        scale = np.maximum(1, int(Cnt.shape[1]/Cnt.shape[0]));
        plt.figure(figsize=(8*scale,8))
        ax1 = plt.subplot(1,1,1);
        show_img(ax1, Cnt);
        ax1.set(title="Local mean correlation for residual")
        ax1.title.set_fontsize(15)
        ax1.title.set_fontweight("bold")
        plt.show();
    
    fin_rlt = {'a':a, 'c':c, 'c_tf':c_tf, 'b':b, "X":X, "W":W, 'res':res, 'corr_img_all_r':corr_img_all_r, 'num_list':num_list};
    
    
    if pass_num > 1:
        return {'rlt':rlt, 'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}
    else:
        return {'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}


def update_AC_bg_l2_Y_ring_lowrank(U_sparse, R, V, V_orig,r,dims, a, c, b, patch_size, corr_th_fix, corr_th_fix_sec = 0.4, corr_th_del = 0.2, switch_point = 10,
            maxiter=50, tol=1e-8, update_after=None,merge_corr_thr=0.5,
            merge_overlap_thr=0.7, num_plane=1, plot_en=False,
            max_allow_neuron_size=0.2, skips=2, update_type="Constant",\
                                mask_a=None, sb=True, pseudo_corr = 0, model = None, plot_mnmf = False, device = 'cpu', batch_size = 10000, plot_debug = False, denoise = False):
    
    '''
    Function for computing background, spatial and temporal components of neurons. Uses HALS updates to iteratively
    refine spatial and temporal estimates. 
    
    See 'demix_whole_data_robust_ring_lowrank' for all documentation (parameters are identical)
    '''
    K = c.shape[1];
    res = np.zeros(maxiter);
    uv_mean = U_sparse.dot(R.dot(V.mean(axis = 1, keepdims = True))) #Pixel-wise mean
    num_list = np.arange(K);
    d1, d2 = dims[:2]
    T = V.shape[1]
    
    print("the patch size is {}".format(patch_size))

    ## initialize spatial support ##
    if mask_a is None: 
        mask_a = (a>0)*1;
    else:
        print("MASK IS NOT NONE")
    mask_ab = mask_a;
    
    
    #Precompute VV^t for X updates
    VVt = V.dot(V.T) #This is for the orthogonal V
    VVt_orig = V_orig.dot(V_orig.T) #This is for the original V

        
    #Get residual correlation image
    corr_time = time.time()
    corr_img_all = vcorrcoef_resid(U_sparse, R, V, a, c, batch_size = batch_size)
    corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");
    print("Resid Corr Image Took {}".format(time.time() - corr_time))
    #Get regular correlation image
    
        
    corr_time = time.time()
    corr_img_all_reg = vcorrcoef_UV_noise(U_sparse, R, V, c, batch_size = batch_size)
    corr_img_all_reg_r = corr_img_all_reg.reshape(patch_size[0],patch_size[1],-1,order="F");
    print("Standard Corr Image Took {}".format(time.time() - corr_time))
   
    print("shape of corr_img_all_r is {}".format(corr_img_all_r.shape)) 
    if plot_debug:
        for k in range(corr_img_all_r.shape[2]):
            fig, ax = plt.subplots(1,4, figsize = (12, 5)) 


            im1 = ax[0].imshow(corr_img_all_r[:, :, k])
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax = cax)
            ax[0].set_title("Robust Resid Corr")

            im2 = ax[1].imshow(corr_img_all_r[:, :, k] * (corr_img_all_r[:, :, k]>0))
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax = cax)
            ax[1].set_title("Robust Resid Corr > 0")


            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im3 = ax[2].imshow(corr_img_all_reg_r[:, :, k])
            fig.colorbar(im3, cax = cax)
            ax[2].set_title("Orig Corr Img")

            divider = make_axes_locatable(ax[3])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im4 = ax[3].imshow(corr_img_all_reg_r[:, :, k] * (corr_img_all_reg_r[:, :, k] > 0))
            fig.colorbar(im4, cax = cax)
            ax[3].set_title("Orig Corr > 0")
            plt.show()

     
    
   
           
    for iters in range(maxiter):
        print(iters)
        
                
        #Change correlation for last few iterations to pick dendrites
        if iters >= maxiter - switch_point:
            corr_th_fix = corr_th_fix_sec 
        start = time.time();
        
        if plot_debug:
            corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");
            if True or (update_after and ((iters+1) % update_after == 0) or (update_after and ((iters) % update_after == 0))):
                    print_signals_corrimg(a, c, corr_th_fix, corr_img_all_r, (patch_size[0], patch_size[1],64))

        

      
        
        ##Update W and b0 now..
        #First: Update X based on new C value
        
        b = regression_update.baseline_update(uv_mean, a, c)
        
        print("we are updating all matrix terms")
        test_time = time.time()
        
        if iters >= skips:
            if iters == skips:
                W=init_w(d1, d2, r)
            
            
            print("X estimate")
            Xtime = time.time()
            X = regression_update.estimate_X(c, V, VVt) #Estimate using orthogonal V, not regular V
            print("X estimate {}".format(time.time() - Xtime))

            #Specify which ring model update we want
            if update_type == "Full":
                raise ValueError('Full Ring Model no longer supported')
            elif update_type == "Constant":
                update_start = time.time()
                W = update_ring_model_w_const(U_sparse, R, V, a, X, b, W, d1, d2, T, r, mask_a=None, batch_size = 10000)
#                 W = csr_matrix(avg_interpolation(W.toarray(), d1, d2))
                print("THE W UPDATE TOOK {}".format(time.time() - update_start))
            else:
                print("ERROR: UPDATE_TYPE NOT SUPPORTED")
        else:
            W = scipy.sparse.coo_matrix((d1*d2, d1*d2))

        test_time = time.time()
        
        
        
        ###SPATIAL UPDATE
        test_time = time.time()
        
        
        #Approximate c as XV for some X:
        X = regression_update.estimate_X(c, V_orig, VVt_orig)       
        a = regression_update.spatial_update_HALS(U_sparse, V_orig, W.tocsr(), X, a, c, b, mask_ab.T).toarray()
        

        ### Delete Bad Components
        temp = (a.sum(axis=0) == 0);
        if sum(temp):
            a, c, corr_img_all_reg, corr_img_all, mask_ab, num_list = delete_comp(a, c, corr_img_all_reg, corr_img_all, mask_ab, num_list, temp, "zero a!", plot_en, (d1, d2));
            X = regression_update.estimate_X(c, V_orig, VVt_orig)
        print("spatial update took {}".format(time.time() - test_time))
        
        
        ### BASELINE UPDATE
        b = regression_update.baseline_update(uv_mean, a, c)
    
            
        ###TEMPORAL UPDATE
        test_time = time.time()
        c = regression_update.temporal_update_HALS(U_sparse, V_orig, W.tocsr(), X, a, c, b)
        print('the shape of c after temporal update is {}'.format(c.shape))
        
        #Denoise 'c' components if desired
        if denoise:
            c = ca_utils.denoise(c) #We now use OASIS denoising to improve improve signals
            c = np.nan_to_num(c, posinf = 0, neginf = 0, nan = 0) #Gracefully handle invalid values
        
        #Delete bad components
        temp = (c.sum(axis=0) == 0);
        if sum(temp):
            a, c, corr_img_all_reg, corr_img_all, mask_ab, num_list = delete_comp(a, c, corr_img_all_reg, corr_img_all, mask_ab, num_list, temp, "zero c!", plot_en, (d1, d2));
        print("temporal update took {}".format(time.time() - test_time))     
            
            

        if update_after and ((iters+1) % update_after == 0):                   
            print("merging components")
            if model is None:
                rlt = merge_components(a,c,corr_img_all_reg,num_list,\
                                       patch_size,merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr,plot_en=plot_en);
            else:
                
                print("STARTING TO LOAD THE NN")
                rlt = merge_components_priors(a,c,corr_img_all_reg,num_list,patch_size, dims = (d1, d2, T), merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr,plot_en=plot_en, model = model, plot_mnmf = plot_mnmf);
                
            flag = isinstance(rlt, int);
            
            
            if ~np.array(flag):
                a = rlt[1];
                c = rlt[2];
                num_list = rlt[3];
            else:
                print("no merge!");
                
            
            print("calculating the residual correlation image")
            corr_img_all = vcorrcoef_resid(U_sparse, R, V, a, c, batch_size = batch_size)
            print("calculating the robust standard correlation image")            
            corr_img_all_reg = vcorrcoef_UV_noise(U_sparse, R, V, c, batch_size = batch_size) 
            
            
            mask_ab = (a>0)*1;
            corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");

            #Currently using rigid mask
            mask_a_rigid = make_mask_rigid(corr_img_all_r, corr_th_fix, mask_ab)    
    
            mask_ab = np.logical_or(mask_a_rigid, mask_ab)

            ## Now we delete components based on whether they have a 0 residual corr img with their supports or not...
            temp = (((mask_ab*corr_img_all) > corr_th_del).sum(axis=0) == 0);
            if sum(temp):
                print("we are at the delete step... corr img is {}".format(corr_th_del))
                a, c, corr_img_all_reg, corr_img_all, mask_ab, num_list = delete_comp(a, c, corr_img_all_reg, corr_img_all, mask_ab, num_list, temp, "zero mask!", plot_en, (d1,d2));
            a = a*mask_ab;

            
        print("time: " + str(time.time()-start));

    temp = np.sqrt((a**2).sum(axis=0,keepdims=True));
    c = c*temp;
    a = a/temp;
    brightness = np.zeros(a.shape[1]);
    a_max = a.max(axis=0);
    c_max = c.max(axis=0);
    brightness = a_max * c_max;
    brightness_rank = np.argsort(-brightness);
    a = a[:,brightness_rank];
    c = c[:,brightness_rank];
    corr_img_all_r = corr_img_all_r[:,:,brightness_rank];
    num_list = num_list[brightness_rank];

    return a, c, b, X, W, res, corr_img_all_r, num_list

     