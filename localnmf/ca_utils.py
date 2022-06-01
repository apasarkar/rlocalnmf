######
##
## Utility Functions used in ringlocalNMF
##
######

##TODO: Remove the oasis imports + the denoising imports

import numpy as np
# from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
# from oasis.oasis_methods import oasisAR1, oasisAR2
# from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
# from oasis.plotting import simpleaxis

import functools
import multiprocessing
import os
import math

import torch
import scipy


def cosine_similarity(img1, img2):
    '''
    Calculates cosine similarity between two 2D images
    Args: 
        img1: first image being compared
        img2: second image being compared 
    Returns: 
        cosine_sim: cosine similarity between these two images
    '''
    if np.count_nonzero(img1 != 0) == 0 or np.count_nonzero(img2 != 0) == 0:
        print("one of these arrays is zero!!!")
        if np.count_nonzero(img2 != 0) == 0:
            print("the second one was 0")
        return 0 
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    img1_max = np.amax(img1_flat)
    img2_max = np.amax(img2_flat)
    
    img1_normal = img1_flat/np.linalg.norm(img1_flat)
    img2_normal = img2_flat/np.linalg.norm(img2_flat)

    cosine_sim = img1_normal.T.dot(img2_normal)
    return cosine_sim


def normalize_traces(trace1, trace2):
    '''
    Normalizes trace
    Args:
        trace1: First trace (provided as an ndarray)
        trace2: Second trace (provided as ndarray) 
        
    Returns:
        trace1_norm: Normalized trace1
        trace2_norm: Normalized trace2
    '''
    
    
    
    if np.count_nonzero(trace1 != 0) == 0:
        trace1_norm = np.zeros_like(trace_1)
    else:
        trace1_norm = trace1/np.linalg.norm(trace1)
        
    if np.count_nonzero(trace2 != 0) == 0:
        trace2_norm = np.zeros_like(trace_2)
    else:
        trace2_norm = trace2/np.linalg.norm(trace2)
        
    return trace1_norm, trace2_norm



def get_box(img):
        """For a given frame in the dataset, this function calculates its bounding box
            args:
                img: a (d1 x d2) ndarray. The image we wish to analyze
                
            returns:
                [height_min, height_max, width_min, width_max]: a list of bounding coordinates which can be used to crop original image
        
        """
        
        #If all pixels are 0, there is no need to crop (the image is empty)
        if np.count_nonzero(img) == 0:
            return (0, img.shape[0], 0, img.shape[1])
        
        #Calculate bounding box by finding minimal elements in the support
        else:
            x,y = np.nonzero(img)
            return (int(np.amin(x)), int(np.amax(x)), int(np.amin(y)), int(np.amax(y)))
        
        
# def denoise(ci):
#     """
#     Denoises a set of temporal traces
#     Params:
#         ci: ndarray (A x B). Algorithms denoises in the second dimension (i.e. it denoises a total of 'A' temporal traces)
#     """
#     ci_new = np.zeros_like(ci)
#     for k in range(ci.shape[1]):
#         c = ci[:, k].astype("double")
#         denoised, s, b, g, lam = deconvolve(c, penalty=1)
#         denoised = denoised + b
#         ci_new[:, k] = denoised.squeeze()
    
#     return ci_new 

def dim_1_matmul(A, B, device = 'cuda', batch_size = 10000):
    '''
    GPU-accelerated matmul of A x B and B x C matrix. Use this method when B is extremely large but A x C can fit on GPU
    '''
    accumulator = np.zeros((A.shape[0], B.shape[1]))
    
    batch_values = math.ceil((A.shape[1]/batch_size))
    for k in range(batch_values):
        interval_start = batch_size*k
        interval_end = batch_size*(k+1)
        A_t = torch.from_numpy(A[:, interval_start:interval_end]).to(device)
        B_t = torch.from_numpy(B[interval_start:interval_end, :]).to(device)
        out = torch.matmul(A_t, B_t).to('cpu').detach().numpy()
        accumulator += out
    torch.cuda.empty_cache()
    return accumulator

def batch_subtract(A, sub_list, device='cuda', batch_size = 10000):
    '''
    Routine for subtracting the sum of all matrices in sub_list from A
    Params: 
        A: type np.ndarray. Shape (a1, a2)
        sub_list: list of np.ndarray objects, each of dimensions (a1, a2)
        device: the device on which computations are being carried out
        batch_size: the number of rows of the output which we populate at a time
    '''
    
    batch_values = math.ceil((A.shape[0]/batch_size))
    output = np.zeros_like(A)
    for k in range(batch_values):
        interval_start = batch_size*k
        interval_end = batch_size*(k+1)
        A_t = torch.from_numpy(A[interval_start:interval_end, :]).to(device)
        for elt in range(len(sub_list)):
            curr_t = torch.from_numpy((sub_list[elt])[interval_start:interval_end, :]).to(device)
            torch.sub(A_t, curr_t, out=A_t)
        output[interval_start:interval_end, :] = A_t.to('cpu').detach().numpy()
        
    torch.cuda.empty_cache()
    return output

def batch_matmul(A, B, device='cuda', batch_size = 10000):
    '''
    Function to accelerate matrix multiplication between two numpy arrays
    Params:
        A: type np.ndarray. Shape (a1, a2)
        B: type np.ndarray. Shape (a2, a3)
        batch_size: int. Number of rows of A to process at a time when doing matmul
        
    Assumes B can be fully moved onto GPU
    
    Returns:
        prod: type np.ndarray. Shape (a1, a3). Product of A and B 
    '''
    
    a1 = A.shape[0]
    batch_values = math.ceil(A.shape[0]/batch_size)
    B_t = torch.from_numpy(B).to(device)
    product = np.zeros((A.shape[0], B.shape[1]))
    for k in range(batch_values):
        interval_start = batch_size*k
        interval_end = batch_size*(k+1)
        A_t = torch.from_numpy(A[interval_start:interval_end, :]).to(device)
        out = torch.matmul(A_t, B_t)
        product[interval_start:interval_end, :] = out.to('cpu').detach().numpy()
    torch.cuda.empty_cache()

    return product


def construct_sparse(U):
    nonzeros = U.nonzero()
    val = U[nonzeros]
    rows = nonzeros[0]
    cols = nonzeros[1]
    U_sparse = scipy.sparse.coo_matrix((val, (rows, cols)), shape=(U.shape[0],U.shape[1]))
    return U_sparse




####Multiprocessing Functions
def parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    num_cpu = multiprocessing.cpu_count()
    os.system('taskset -cp 0-%d %s' % (num_cpu, os.getpid()))


def runpar(f, X, nprocesses=None, **kwargs):
    '''
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)
    '''
    
    #Change affinity (if needed) to enable full multicore processing
    
    
    if nprocesses is None:
        nprocesses = int(multiprocessing.cpu_count()) 
        print("the number of processes is {}".format(nprocesses))
#         val = len(os.sched_getaffinity(os.getpid()))
#         print('the number of usable cpu cores is {}'.format(val))
    
    with multiprocessing.Pool(initializer=parinit, processes=nprocesses) as pool:
        res = pool.map(functools.partial(f, **kwargs), X)
    pool.join()
    pool.close()
    return res