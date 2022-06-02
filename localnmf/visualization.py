#####
## Suite of visualization tools for NMF source extractions
##
#####

import os
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil
from scipy.sparse import lil_matrix
import subprocess
import shutil

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.ticker as ticker

from localnmf import ca_utils


#Multiprocessing utils
import os
import multiprocessing
import functools
import subprocess
import shutil

import time



'''
Provides figure generation for: 
Simulated Data:
- Simulation Matching
- Correlation Matching
Real Data:
- Oversplit Detection
Demixing Videos
- Standard video generation based on ring localNMF results
- Generic, Customizable Video Generation
'''



def runpar(f, X, nprocesses=None, **kwargs):
    '''
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)
    '''
    
    #Change affinity (if needed) to enable full multicore processing
    
    
    val = len(os.sched_getaffinity(os.getpid()))
    print("the CPU affinity BEFORE runpar is {}".format(val))

    if nprocesses is None:
        nprocesses = int(multiprocessing.cpu_count()) 
        print("the number of processes is {}".format(nprocesses))
#         val = len(os.sched_getaffinity(os.getpid()))
#         print('the number of usable cpu cores is {}'.format(val))
    
    with multiprocessing.Pool(initializer=parinit, processes=nprocesses) as pool:
        res = pool.map(functools.partial(f, **kwargs), X)
    pool.join()
    pool.close()


    val = len(os.sched_getaffinity(os.getpid()))
    print("after the multicore, the affinity is {}".format(val))

    num_cpu = multiprocessing.cpu_count()
    os.system('taskset -cp 0-%d %s' % (num_cpu, os.getpid()))
    val = len(os.sched_getaffinity(os.getpid()))
    print("the cpu affinity after the process (intro fix) is {}".format(val))
    return res

def parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    num_cpu = multiprocessing.cpu_count()
    os.system('taskset -cp 0-%d %s' % (num_cpu, os.getpid()))




def brightness_order(ai, ci):
    '''
    Orders a set of neurons (identified by their spatial components ai and temporal components ci) by maximum brightness
    args:
        ai: ndarray, dimensions (d x K1). Matrix describing spatial footprints of K1 neurons
        ci: ndarray, dimensions (T x K1). Matrix describing temporal traces of K1 neurons
        
    Returns:
        a_ordered: ndarray, dimensions (d x K1). Columns of ai reordered
        c_ordered: ndarray, dimensions (T x K1). Columns of ci reordered.
    '''

    brightnesses = []
    for k in range(ai.shape[1]):
        curr_ai = ai[:, [k]]
        curr_ci = ci[:, [k]]
        max_val = np.amax((curr_ai.dot(curr_ci.T)).flatten())
        brightnesses.append(max_val)
        

    reorder = np.argsort(brightnesses)[::-1]
    a_ordered = ai[:, reorder]
    c_ordered = ci[:, reorder]
    
    return a_ordered, c_ordered


def match(a_real, c_real, a_est, c_est, temporal_threshold=0.85, spatial_threshold=0.85, x=64, y=64, plot = True):
    '''
    Function to match ground truth data with estimated data. For each ground truth neuron, 
    this function matches at most one estimated neuron that is sufficiently similar (both spatially and temporally)
    Each estimated neuron is matched to at most one ground truth neuron. Matching is done greedily.
    
    Use case: Simulated data or annotated data (where 'ground truth' is known)
    
    Args
        a_real: ndarray, dimensions (d x K1). Matrix describing spatial footprints of K1 ground truth neurons
        c_real: ndarray, dimensions (T x K1). Matrix describing temporal traces of K1 ground truth neurons
        a_est: ndarray, dimensions (d x K2). Matrix describing spatial footprints of K2 ground truth neurons
        c_est: ndarray, dimensions (T x K2). Matrix describing temporal traces of K1 ground truth neurons
        temp_thr: scalar value. Minimum allowed temporal cosine similarity between matched pairs of neurons
        spatial_thr: scalar value. Minimum allowed temporal spatial similarity between matched pairs of neurons
        x, y: integers. X and Y-dimensions of field of view. xy = d
        plot: boolean. Indicates whether matched neurons should be plotted or not
        
    Returns
        match_indices: ndarray, dimensions (K1 x 1). Indicates, for each of K1 ground truth neurons, what its corresponding
            match is. match_indices[i] provides an integer indicating which of the K2 estimate neurons (if any) are matched with neuron i.
        count: integer. Number of neurons that were successfully matched (at most K1). 
    
    Note: This matching preserves the order of a_real. So match_indices[0] gives the match for neuron corresponding to a_real[0]
    '''
    
    match_counter = 0
    match_indices = np.zeros((a_real.shape[1], 1))
    match_indices -= 1 #All values start out negative and are updated if a suitable match is found
    count = 0
    
    #For each neuron in ground truth, identify match
    for k in range(a_real.shape[1]):
        print("we are tring to match k = {}".format(k))
        curr_true_ai = a_real[:, k]
        curr_true_ci = c_real[:, k]
        
        if plot:
            print("THIS IS THE GROUND TRUTH WE WANT TO MATCH NOW")
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(curr_true_ai.reshape((x,y)))
            ax[1].plot(curr_true_ci)
            ax[0].set_title("True Ai")
            ax[1].set_title("True Ci")
            plt.show()

        
        #Identify estimated neurons which are sufficiently spatially similar
        good_list = []
        for j in range(a_est.shape[1]):
            curr_est_ai = a_est[:, j]
            curr_est_ci = c_est[:, j]
            spatial_sim = ca_utils.cosine_similarity(curr_est_ai, curr_true_ai)
            
            
            if spatial_sim > spatial_threshold:
                print("for {} we are adding {}".format(k, j))
                good_list.append(j)


        if len(good_list) == 0:
            print("There were no spatially similar estimates for {}".format(k))
         
        
        added = False
        best_temp = 0
        #Now we find the best match among the spatially similar neurons
        for i in range(len(good_list)):
            curr_est_ai = a_est[:, good_list[i]]
            curr_est_ci = c_est[:, good_list[i]]

            temporal_similarity = ca_utils.cosine_similarity(curr_est_ci, curr_true_ci)
            if temporal_similarity > best_temp and temporal_similarity > temporal_threshold:
                if good_list[i] not in match_indices:
                    best_temp = temporal_similarity
                    print("MATCHED")
                    if not added:
                        count += 1
                    added = True
                    match_indices[k] = good_list[i]
                    

                    if plot:
                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(curr_est_ai.reshape((x,y)))
                        ax[1].plot(curr_est_ci)
                        ax[0].set_title("Estimate Ai")
                        ax[1].set_title("Estimate Ci")
                        plt.show()

                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(curr_true_ai.reshape((x,y)))
                        ax[1].plot(curr_true_ci)
                        ax[0].set_title("True Ai")
                        ax[1].set_title("True Ci")
                        plt.show()

                        
        
        if match_indices[k] >= 0:
            print("the match for k = {} is {}".format(k, match_indices[k]))
            print("------")
            print("the temporal similarity is {}".format(best_temp))
            print("------")
        else:
            print("This ground truth neuron was not matched")
        
        

    return match_indices, count


def match_and_score_temporal(a_real, c_real, a_est, c_est, match_indices):
    '''
    Given a set of matches between a_real and c_real, assess the average temporal recovery accuracy
    Args:
        a_real: ndarray, dimensions (d x K1). Matrix describing spatial footprints of K1 ground truth neurons
        c_real: ndarray, dimensions (T x K1). Matrix describing temporal traces of K1 ground truth neurons
        a_est: ndarray, dimensions (d x K2). Matrix describing spatial footprints of K2 ground truth neurons
        c_est: ndarray, dimensions (T x K2). Matrix describing temporal traces of K1 ground truth neurons
        match_indices: ndarray, dimensions (K1 x 1). Indicates, for each of K1 ground truth neurons, what its corresponding
            match is. match_indices[i] provides an integer indicating which of the K2 estimate neurons (if any) are matched with neuron
    
    Returns:
        average: float, between 0 and 1. the average temporal recovery accuracy of the algorithm
    
    '''
    total_cosine_dist = 0
    for k in range(c_real.shape[1]):
        if match_indices[k] < 0:
            continue
        curr_real_ci = c_real[:, k]
        curr_est_ci = c_est[:, int(match_indices[k])]
        total_cosine_dist += ca_utils.cosine_similarity(curr_real_ci, curr_est_ci)
        
    average = total_cosine_dist/c_real.shape[1] 
    return average
    

def match_and_score_spatial(a_real, c_real, a_est, c_est, match_indices):
    '''
    Given a set of matches between a_real and c_real, assess the average spatial recovery accuracy
    Args:
        a_real: ndarray, dimensions (d x K1). Matrix describing spatial footprints of K1 ground truth neurons
        c_real: ndarray, dimensions (T x K1). Matrix describing temporal traces of K1 ground truth neurons
        a_est: ndarray, dimensions (d x K2). Matrix describing spatial footprints of K2 ground truth neurons
        c_est: ndarray, dimensions (T x K2). Matrix describing temporal traces of K1 ground truth neurons
        match_indices: ndarray, dimensions (K1 x 1). Indicates, for each of K1 ground truth neurons, what its corresponding
            match is. match_indices[i] provides an integer indicating which of the K2 estimate neurons (if any) are matched with neuron
    
    Returns:
        average: float, between 0 and 1. the average spatial recovery accuracy of the algorithm
    
    '''

    
    total_cosine_dist = 0
    for k in range(a_real.shape[1]):
        if match_indices[k] < 0:
            continue
        curr_real_ai = a_real[:, k]
        curr_est_ai = a_est[:, int(match_indices[k])]
        total_cosine_dist += ca_utils.cosine_similarity(curr_real_ai, curr_est_ai)
    average = total_cosine_dist/a_real.shape[1] 
    return average


def match_nearest_neighbors(a_est, c_est, temporal_threshold=0.85, spatial_threshold=0.85, x=64, y=64, plot = False):
    '''
    Function to match each neuron in a source extraction with its nearest neighbor. Used to detect oversplitting in NMF pipelines.
    For each neuron estimate, function matches at most one estimated neuron that is sufficiently similar (both spatially and temporally)
    
    Args
        a_est: ndarray, dimensions (d x K1). Matrix describing spatial footprints of K1 ground truth neurons
        c_est: ndarray, dimensions (T x K1). Matrix describing temporal traces of K1 ground truth neurons
        temp_thr: scalar value. Minimum allowed temporal cosine similarity between matched pairs of neurons
        spatial_thr: scalar value. Minimum allowed temporal spatial similarity between matched pairs of neurons
        x, y: integers. X and Y-dimensions of field of view. xy = d
        plot: boolean. Indicates whether matched neurons should be plotted or not
        
    Use case: Simulated or Real data. Main goal is to detect oversplitting errors. 
    
    Returns
        match_indices: ndarray, dimensions (K1 x 1). Indicates, for each of K1 ground truth neurons, what its corresponding
            match is. match_indices[i] provides an integer indicating which of the K2 estimate neurons (if any) are matched with neuron i.
            
    Note: This matching preserves the order of a_est. So match_indices[0] gives the match for neuron corresponding to a_est[0]
    '''
    
    
    match_counter = 0
    match_indices = np.zeros((a_est.shape[1], 1))
    match_indices -= 1 #All values start out negative and are updated if a suitable match is found
    count = 0
    
    #For each neuron in ground truth, identify match
    for k in range(a_est.shape[1]):
        print("we are tring to match k = {}".format(k))
        curr_ai = a_est[:, [k]]
        curr_ci = c_est[:, [k]]
        
        
        best_sim = 0
        best_ind = -1
        for j in range(a_est.shape[1]):
            if j == k:
                #Never match a neuron to itself
                continue
            ai = a_est[:, [j]]
            ci = c_est[:, [j]]
            
            spatial_sim = ca_utils.cosine_similarity(curr_ai, ai)
            temporal_sim = ca_utils.cosine_similarity(curr_ci, ci)
            if spatial_sim > spatial_threshold and temporal_sim > temporal_threshold:
                if best_sim < temporal_sim:
                    best_sim = temporal_sim
                    best_ind = j
                    match_indices[k] = j
                    
                    if plot:
                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(curr_ai.reshape((x,y)))
                        ax[1].plot(curr_ci)
                        ax[0].set_title("Original Ai")
                        ax[1].set_title("Original Ci")
                        plt.show()

                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(ai.reshape((x,y)))
                        ax[1].plot(ci)
                        ax[0].set_title("Neighbor Ai")
                        ax[1].set_title("Neighbor Ci")
                        plt.show()
                        
    return match_indices


def generate_match_figs_neighbors(a_est, c_est, matches, dims,  folder_location,\
                        spatial_titles_match = ["Spatial footprint", "Nearest Neighbor"], show = False, normalize = True, zoom = False):
    '''
    Generates figures plotting nearest neighbors of neurons. Figures are saved as .png files
    Matched neurons are generated togehter in the same figure. 
    Nearest neighbors are calcualted in terms of spatial and temporal cosine similarity
    For each file, the max brightness of the ground truth neuron is included
    
    Use case: Figure Generation for data with NO ground truth (or annotated ground truth). Goal: detect oversplitting
    
    args:
        a_est: ndarray, dimensions (d x K2). Matrix describing spatial footprints of K2 ground truth neurons
        c_est: ndarray, dimensions (T x K2). Matrix describing temporal traces of K2 ground truth neurons
        matches: array of matches between ground truth (a_real, c_real) and estimates (a_est, c_est)
            matches[i] indicates which neuron of a_est and c_est has been matched to neuron 'i' from the ground truth. see 'match' above.
        dims: tuple, length 2. Indicates dimensions of field of view. dims = (40, 50) means the image is a 40 x 50 pixel image. 
        folder_location: string. relative filepath for storing results. 
        spatial_titles_match: tuple of strings, length 2. Titles for matched spatial footprints
        show: boolean. Flag indicating whether results should be displayed or simply saved
        normalize: boolean. Flag indicates whether temporal traces should be normalized or not
        zoom: boolean. Flag indicates whether or not to display a zoomed version of the spatial support
    Returns: 
        (No return values)
    '''
    
    
    for p in range(matches.size):
        file_path = os.path.join(folder_location, "neuron {}".format(p))
        match_ai = a_est[:, int(matches[p])]
        match_ci = c_est[:, int(matches[p])]
        orig_ai  = a_est[:, [p]]
        orig_ci = c_est[:, [p]]
        
        
        max_brightness = int(np.amax((orig_ai.dot(orig_ci.T)).flatten()))


        if int(matches[p]) < 0:
            fig,gs = plot_img_single(orig_ai, orig_ci, dims, zoom = zoom)
        else: 
            labels = ['Neuron', 'Neighbor']
            fig,gs = plot_img_double(match_ai, match_ci, orig_ai, orig_ci, dims, spatial_titles_match, labels = labels, normalize = normalize, zoom = zoom)
            print(ca_utils.cosine_similarity(match_ci, orig_ci))
        
        fig.savefig("{}/Neuron{} Brightness{}".format(folder_location,p,max_brightness))




def generate_match_figs(a_real, c_real, a_est, c_est, matches, dims,  folder_location,\
                        spatial_titles_match = ["Spatial footprint", "Ground Truth Spatial"],\
                        spatial_titles_unmatch = ["Nearest Neighbor Spatial","Unmatched spatial"], \
                        show = False, zoom = False):
    '''
    Generates figures for matched and unmatched neurons. Figures are saved as .png files
    Matched neurons are generated togehter in the same figure. 
    Unmatched neurons are plotted with their nearest neigbhors from the estimated neurons (to detect oversplitting).
    Nearest neighbors are calcualted in terms of spatial and temporal cosine similarity
    For each file, the max brightness of the ground truth neuron is included
    
    Use case: Figure Generation for data with ground truth (or annotated ground truth)
    
    args:
        a_real: ndarray, dimensions (d x K1). Matrix describing spatial footprints of K1 ground truth neurons
        c_real: ndarray, dimensions (T x K1). Matrix describing temporal traces of K1 ground truth neurons
        a_est: ndarray, dimensions (d x K2). Matrix describing spatial footprints of K2 ground truth neurons
        c_est: ndarray, dimensions (T x K2). Matrix describing temporal traces of K2 ground truth neurons
        matches: array of matches between ground truth (a_real, c_real) and estimates (a_est, c_est)
            matches[i] indicates which neuron of a_est and c_est has been matched to neuron 'i' from the ground truth. see 'match' above.
        dims: tuple, length 2. Indicates dimensions of field of view. dims = (40, 50) means the image is a 40 x 50 pixel image. 
        folder_location: string. relative filepath for storing results. 
        spatial_titles_match: tuple of strings, length 2. Titles for matched spatial footprints
        spatial_titles_unmatch: tuple of strings, length 2. Titles for unmatched spatial footprints
        show: boolean. Flag indicating whether results should be displayed or simply saved
    Returns: 
        (No return values)
    '''
    
    
    for p in range(matches.size):
        file_path = os.path.join(folder_location, "neuron {}".format(p))
        match_ai = a_est[:, int(matches[p])]
        match_ci = c_est[:, int(matches[p])]
        orig_ai  = a_real[:, [p]]
        orig_ci = c_real[:, [p]]
        
        
        max_brightness = int(np.amax((orig_ai.dot(orig_ci.T)).flatten()))


        #In this case, we plot the original ai and ci with its nearest ground truth neighbor (spatial similarity)
        if int(matches[p]) < 0:
            best = 0
            best_ind = -1
            for j in range(a_real.shape[1]):
                if j == p:
                    continue
                s_sim = ca_utils.cosine_similarity(orig_ai, a_real[:, [j]])
                if s_sim > best:
                    best_ind = j
                    best = s_sim
            if best_ind >= 0:
                titles = ["Ground Truth", "Neighbor GT Neuron {}".format(best_ind)]
                fig,gs = plot_img_double(orig_ai, orig_ci, a_real[:, best_ind], c_real[:, best_ind], dims, titles, zoom = zoom)
            else:
                fig,gs = plot_img_single(orig_ai, orig_ci, dims, zoom = zoom, spatial_title = "Ground Truth Spatial", \
                                    temp_title = "Ground Truth Temporal")
        else: 
            fig,gs = plot_img_double(match_ai, match_ci, orig_ai, orig_ci, dims, spatial_titles_match, zoom = zoom)
            print(ca_utils.cosine_similarity(match_ci, orig_ci))
        
        fig.savefig("{}/Neuron{} Brightness{}".format(folder_location,p,max_brightness))


    for k in range(a_est.shape[1]):
        if k not in matches:
        #We plot k and its nearest neighbor match:
            best = 0
            matched = -1
            for j in range(a_est.shape[1]):
                if k == j:
                    continue
                curr_val_a = a_est[:, j]
                curr_val_c = c_est[:, j]

                if ca_utils.cosine_similarity(curr_val_a, a_est[:, k]) > 0:
                    temp_sim = ca_utils.cosine_similarity(curr_val_c, c_est[:, k])
                    if temp_sim > best:
                        best = temp_sim
                        matched = j
            if matched == -1:
                fig, gs = plot_img_single(a_est[:, k], c_est[:, k],dims, zoom = zoom)
            elif matched > -1:
                fig, gs = plot_img_double(a_est[:, matched], c_est[:, matched], a_est[:, k], c_est[:, k], dims, spatial_titles_unmatch, zoom = zoom)
            fig.savefig("{}/UNMATCHED {}".format(folder_location, k))  
            if show:
                plt.show()

                
def find_r2(mov_denoised, a_est, c_est, exclude):
    '''
    For any given neuron (ai,ci), this function subtracts off all of the other signals from the denoised movie.
    Then, using this 'residual' movie, we calculate the r2 between 
    
    args:
        mov_denoised: ndarray, dimensions (d1 x d2 x T). Matrix describing a denoised movie over a (d1*d2)-pixel field of view
            Movie has T frames
        a_est: ndarray, dimensions (d x K2). Matrix describing spatial footprints of K2 ground truth neurons
        c_est: ndarray, dimensions (T x K2). Matrix describing temporal traces of K2 ground truth neurons
        exclude: int. the index of the neuron which we will be examining. All other neurons are subtracted from mov_denoised
    '''
    dims = mov_denoised.shape
    mov_denoised_r = mov_denoised.reshape((np.prod(dims[:2]), -1))
    exclude = int(exclude)
    
    keeps = np.zeros((a_est.shape[1],)) + 1
    keeps = keeps.astype('bool')
    keeps[exclude] = 0
    a_subtract = a_est[:, keeps]
    c_subtract = c_est[:, keeps]
    
    resid = mov_denoised_r - a_subtract.dot(c_subtract.T)
    neuron_mov = a_est[:, [exclude]].dot(c_est[:, [exclude]].T)
    
    neuron_mov_exp = neuron_mov.reshape(dims)
    neuron_mov_sum = np.sum(neuron_mov_exp, axis = 2)
    x1, x2, y1, y2 = ca_utils.get_box(neuron_mov_sum)
    support_vec = np.zeros(dims)
    support_vec[x1:x2, y1:y2, :] = 1
    
    support_vec = support_vec.reshape((np.prod(dims), 1))
    
#     sim = ca_utils.cosine_similarity(resid, neuron_mov)
#     print("the cosine similarity between the videos is {}".format(sim))

    
    ##Now we find the r2 between resid and the neuron in question
    resid_f = resid.reshape((np.prod(dims), 1)) * support_vec
    neuron_mov_f = neuron_mov.reshape((np.prod(dims), 1))
    
    mean_diff_resid = resid_f - np.mean(resid_f)
    total_var = (mean_diff_resid.T).dot(mean_diff_resid)
    
    diff_est = neuron_mov_f - resid_f
    unexpl_var = (diff_est.T).dot(diff_est)
    
    print("total var is {}".format(total_var))
    print("unexplained var is {}".format(unexpl_var))
    
    sim = ca_utils.cosine_similarity(resid_f, neuron_mov_f)
    print(sim)
    
    r2 = 1 - unexpl_var/total_var
    
    print('the r2 is {}'.format(r2))
    return r2[0]
#     return sim
    
                
def generate_match_figs_r2(mov_denoised, a_real, c_real, a_est, c_est, matches, dims,  folder_location,\
                        spatial_titles_match = ["Spatial footprint", "Ground Truth Spatial"],\
                        spatial_titles_unmatch = ["Nearest Neighbor Spatial","Unmatched spatial"], \
                        show = False, zoom = False):
    '''
    Generates figures for matched and unmatched neurons. Figures are saved as .png files
    Matched neurons are generated togehter in the same figure. 
    Unmatched neurons are plotted with their nearest neigbhors from the estimated neurons (to detect oversplitting).
    Nearest neighbors are calcualted in terms of spatial and temporal cosine similarity
    For each file, the max brightness of the ground truth neuron is included
    
    
    Use case: Figure Generation for data with ground truth (or annotated ground truth)
    
    args:
        a_real: ndarray, dimensions (d x K1). Matrix describing spatial footprints of K1 ground truth neurons
        c_real: ndarray, dimensions (T x K1). Matrix describing temporal traces of K1 ground truth neurons
        a_est: ndarray, dimensions (d x K2). Matrix describing spatial footprints of K2 ground truth neurons
        c_est: ndarray, dimensions (T x K2). Matrix describing temporal traces of K2 ground truth neurons
        matches: array of matches between ground truth (a_real, c_real) and estimates (a_est, c_est)
            matches[i] indicates which neuron of a_est and c_est has been matched to neuron 'i' from the ground truth. see 'match' above.
        dims: tuple, length 2. Indicates dimensions of field of view. dims = (40, 50) means the image is a 40 x 50 pixel image. 
        folder_location: string. relative filepath for storing results. 
        spatial_titles_match: tuple of strings, length 2. Titles for matched spatial footprints
        spatial_titles_unmatch: tuple of strings, length 2. Titles for unmatched spatial footprints
        show: boolean. Flag indicating whether results should be displayed or simply saved
    Returns: 
        (No return values)
    '''
    
    
    for p in range(matches.size):
        file_path = os.path.join(folder_location, "neuron {}".format(p))
        match_ai = a_est[:, int(matches[p])]
        match_ci = c_est[:, int(matches[p])]
        orig_ai  = a_real[:, [p]]
        orig_ci = c_real[:, [p]]
        
        
        max_brightness = int(np.amax((orig_ai.dot(orig_ci.T)).flatten()))


        #In this case, we plot the original ai and ci with its nearest ground truth neighbor (spatial similarity)
        if int(matches[p]) < 0:
            best = 0
            best_ind = -1
            for j in range(a_real.shape[1]):
                if j == p:
                    continue
                s_sim = ca_utils.cosine_similarity(orig_ai, a_real[:, [j]])
                if s_sim > best:
                    best_ind = j
                    best = s_sim
            if best_ind >= 0:
                titles = ["Ground Truth", "Neighbor GT Neuron {}".format(best_ind)]
                fig,gs = plot_img_double(orig_ai, orig_ci, a_real[:, best_ind], c_real[:, best_ind], dims, titles, zoom = zoom)
            else:
                fig,gs = plot_img_single(orig_ai, orig_ci, dims, zoom = zoom, spatial_title = "Ground Truth Spatial", \
                                    temp_title = "Ground Truth Temporal")
        else:
            ind = matches[p]
            r2 = find_r2(mov_denoised, a_est, c_est, ind)
            spatial_titles_est = [spatial_titles_match[0] + " R2 Resid {}".format(r2)]
            spatial_titles_est.append(spatial_titles_match[1])
            fig,gs = plot_img_double(match_ai, match_ci, orig_ai, orig_ci, dims, spatial_titles_est, zoom = zoom)
            print(ca_utils.cosine_similarity(match_ci, orig_ci))
        
        fig.savefig("{}/Neuron{} Brightness{}".format(folder_location,p,max_brightness))


    for k in range(a_est.shape[1]):
        if k not in matches:
        #We plot k and its nearest neighbor match:
            best = 0
            matched = -1
            for j in range(a_est.shape[1]):
                if k == j:
                    continue
                curr_val_a = a_est[:, j]
                curr_val_c = c_est[:, j]

                if ca_utils.cosine_similarity(curr_val_a, a_est[:, k]) > 0:
                    temp_sim = ca_utils.cosine_similarity(curr_val_c, c_est[:, k])
                    if temp_sim > best:
                        best = temp_sim
                        matched = j
                        
            #Get R2
            r2 = find_r2(mov_denoised, a_est, c_est, k)
            if matched == -1:
                spatial_title = "Unmatch Spatial R2 Resid {}".format(r2)
                fig, gs = plot_img_single(a_est[:, k], c_est[:, k],dims, zoom = zoom)
            elif matched > -1:
                r2title = spatial_titles_unmatch[1] + " R2 Resid {}".format(r2)
                fin_titles = [spatial_titles_unmatch[0], r2title]
                fig, gs = plot_img_double(a_est[:, matched], c_est[:, matched], a_est[:, k], c_est[:, k], dims, fin_titles, zoom = zoom)
            fig.savefig("{}/UNMATCHED {}".format(folder_location, k))  
            if show:
                plt.show()

                
def plot_updated_axis_Y(fig, axis, tick_range, fontsize, skip = 2):
    """
    Updates the Y ticks of a axis object
    params:
        axis: matplotlib axis object. Assumes axis is created to display an image
        tick_range: tuple (2 entries). Two integers indicating the upper and lower bounds for the tick values on the Y axis
        fontsize: int. size of each tick
        skip: int. Indicates that we only display every (skip)-th tick
    """
    fig.canvas.draw()
    list_val = []
    indices = []
    index = 0
    for tick in axis.yaxis.get_major_ticks():
        if index%skip == 0:
            curr_text = tick.label.get_text()
            new_text_int = int(tick_range[0]+ float(curr_text.replace("−", "-")))
            list_val.append(new_text_int)
            indices.append(index)
        else:
            pass
            
        index += 1

    curr_pos = axis.get_yticks()
    curr_pos = curr_pos[indices]
    
    min_val = np.amin(curr_pos)
    list_val = np.array(list_val)
    accept = np.logical_and(curr_pos >= 0, list_val < tick_range[1])
    curr_pos = curr_pos[accept]
    list_val = list_val[accept]  
    
    axis.set_yticks(curr_pos)
    axis.set_yticklabels(list_val, fontsize=fontsize)
    
def plot_updated_axis_X(fig, axis, tick_range, fontsize, skip = 2):
    """
    Updates the X ticks of a axis object
    params:
        axis: matplotlib axis object. Assumes axis is created to display an image, and that the axis has displayed the image using
            axis.imshow(image).
        tick_range: tuple (2 entries). Two integers indicating the upper and lower bounds for the tick values on the X axis
        fontsize: int. size of each tick
        skip: int. Indicates that we only display every (skip)-th tick
    """
    fig.canvas.draw()
    list_val = []
    index = 0
    indices = []
    for tick in axis.xaxis.get_major_ticks():
        if index%skip == 0:
            curr_text = tick.label.get_text()
            new_text_int = int(tick_range[0] + float(curr_text.replace("−", "-")))
            list_val.append(new_text_int)
            tick.set_label1(new_text_int) 
            indices.append(index)
        else:
            pass
        index+=1


    curr_pos = axis.get_xticks()
    curr_pos = curr_pos[indices]
    min_val = np.amin(curr_pos)
    list_val = np.array(list_val)
    accept = np.logical_and(curr_pos >= 0, list_val < tick_range[1])
    curr_pos = curr_pos[accept]
    list_val = list_val[accept]

    axis.set_xticks(curr_pos)
    axis.set_xticklabels(list_val, fontsize = fontsize)
    
    
                



def plot_img_double(a_first, c_first, a_second, c_second, dims, spatial_titles, normalize = True,\
                       figsize=(60,30), fig_props_h=(1,1), fig_props_w=(1,3), \
                       font_val=60, tick_val=45, legendsize = '60', labels=['estimate', 'real'], order=1, zoom = False):
    """
    Plot two different spatio-temporal signals (belonging to the same FOV) on top of each other
    The temporal traces are normalized and then overlayed for easy visual comparison
    Args:
    
        est_spatial: d-length spatial signal. video has d pixels in FOV
        est_temp: T-length temporal signal; video has T frames
        est_dims: (x,y) tuple describing shape of est_spatial
        real_spatial: d-length spatial signal 
        real_temp: T-length temporal signal
        real_dims: (x,y) tuple describing shape of real_spatial
        est_titles: list of 2 titles. first title is for est_spatial, second for est_temp
        real_titles: list of 2 titles. first title is for real_spatial, second for real_temp
        font_val: font size of titles
        tick_val: font size of ticks
        order: order of reshaping operations: 0 (default) for "C", 1 for "F"
        zoom: boolean. Indicates whether spatial elements should be cropped as much as possible to "zoom" in on regions of interest
    Returns: 
        fig: figure object
        gs: corresponding gridspec object
    
    """
    gs = gridspec.GridSpec(2, 2, height_ratios=fig_props_h, width_ratios=fig_props_w)
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    

    
    assert order in [0,1], "invalid order provided"
    if order == 0:
        r_order="C"
    elif order == 1:
        r_order = "F"
    
    first_norm_spatial = (a_first/np.linalg.norm(a_first)).reshape(dims, order=r_order)
    second_norm_spatial = (a_second/np.linalg.norm(a_second)).reshape(dims, order=r_order)
    
    if zoom:
        (x1, x2, x3, x4) = ca_utils.get_box(first_norm_spatial)
        (y1, y2, y3, y4) = ca_utils.get_box(second_norm_spatial)
        
        x_low = np.amin([x1, y1])
        x_high = np.amax([x2, y2])
        y_low = np.amin([x3, y3])
        y_high = np.amax([x4, y4])
    
    else:
        x_low, x_high = (0, dims[0])
        y_low, y_high = (0, dims[1])
        
    first_norm_spatial = first_norm_spatial[x_low:x_high, y_low:y_high]
    second_norm_spatial = second_norm_spatial[x_low:x_high, y_low:y_high]
    
    ##Plot first shape
    axis_spatialfirst = fig.add_subplot(gs[0, 0])
    axis_spatialfirst.imshow(first_norm_spatial)
    axis_spatialfirst.set_title(spatial_titles[0], fontsize = font_val)
    axis_spatialfirst.set_xticks([])
    for tick in axis_spatialfirst.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_val) 
    
    
    ##Plot second shape
    axis_spatialsecond = fig.add_subplot(gs[1,0])
    axis_spatialsecond.imshow(second_norm_spatial)
    axis_spatialsecond.set_title(spatial_titles[1], fontsize = font_val)
    
    
    plot_updated_axis_Y(fig, axis_spatialsecond, (x_low, x_high), fontsize = tick_val, skip = 1)
    plot_updated_axis_Y(fig, axis_spatialfirst, (x_low, x_high), fontsize = tick_val, skip = 1)
    plot_updated_axis_X(fig, axis_spatialsecond, (y_low, y_high), fontsize = tick_val, skip = 2)
    
    ##Update tick values based on cropping so that it is clear what part of the original FOV
    ## that we have zoomed into
    #counter-intuitive: first dimension indexing in numpy corresponds to y axis, second dimension to x axis (when viewed with imshow)
                
    
    
    ##Plot the traces
    if normalize:
        first_temp_norm, second_temp_norm = ca_utils.normalize_traces(c_first, c_second)
    else:
        first_temp_norm, second_temp_norm = (c_first, c_second)
    
    axis_temporal = fig.add_subplot(gs[:,1])
    axis_temporal.plot(first_temp_norm, label = labels[0])
    axis_temporal.plot(second_temp_norm, label = labels[1])
    axis_temporal.legend(prop={'size': legendsize})
    axis_temporal.set_title("Temporal Traces", fontsize=font_val)
    for tick in axis_temporal.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_val) 
    for tick in axis_temporal.xaxis.get_major_ticks():
                tick.label.set_fontsize(tick_val) 
    
    
    
    print("we are about to return fig")
    return fig, gs



def plot_img_single(ai, ci, dims, spatial_title="", temp_title="",\
                    figsize=(60,15), fig_props_w=(1,3), \
                    font_val=60, tick_val=45, order=1, zoom = False):
    """
    Plots a single spatiotemporal signal
    Args: 
        ai: ndarray, dimensions (d x 1). Matrix describing spatial footprint of 1 neuron over 1 d-pixel field of view
        ci: ndarray, dimensions (T x 1). Matrix describing temporal trace of 1 neuron over T-frame video
        dims: tuple, 2 elements. Describes dimensions of field of view. For example, dims = (40, 50) means the field of view is 40 x 50
        spatial_title: string. Title for plotting spatial footprint
        temp_title: string. Title for plotting temporal trace
        figsize: tuple, 2 elements. Dimensions of figure
        fig_props_w: tuple, 2 elements. Relative proportions of temporal plot and spatial plot in figure
        font_val: integer. Font size of title
        tick_val: integer. Size of ticks
        order: integer, 0 or 1. Describes order in which ai is reshaped to create a 2D image. If 0, use "C"-style reordering. If 1, use "F"
        zoom: boolean. Indicates whether spatial elements should be cropped as much as possible to "zoom" in on regions of interest
    Returns: 
        fig: figure object
        gs: corresponding gridspec object
    """
    
    
    gs = gridspec.GridSpec(1, 2, height_ratios=(1,), width_ratios=fig_props_w)
    fig = plt.figure(figsize=figsize)
    

    
    if order == 0:
        r_order="C"
    elif order == 1:
        r_order = "F"
    
    
    #Normalize spatial footprint
    norm_spatial = (ai/np.linalg.norm(ai)).reshape((dims[0],dims[1]), order=r_order)
    if zoom:
        (x1, x2, y1, y2) = ca_utils.get_box(norm_spatial)
        norm_spatial = norm_spatial[x1:x2, y1:y2]
        
    ####    

    #Plot spatial footprint
    axis_spatial = fig.add_subplot(gs[0,0])
    axis_spatial.imshow(norm_spatial)
    axis_spatial.set_title(spatial_title, fontsize = font_val)
    #Adjust tick sizes
    for tick in axis_spatial.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_val) 
    for tick in axis_spatial.xaxis.get_major_ticks():
                tick.label.set_fontsize(tick_val) 
    fig.canvas.draw()
            
    if zoom: 
        plot_updated_axis_X(fig, axis_spatial, (y1, y2), fontsize = tick_val, skip = 2)
        
   

              
    #Plot Temporal Footprint
    axis_temporal = fig.add_subplot(gs[0,1])
    axis_temporal.set_title(temp_title, fontsize=font_val)
    axis_temporal.plot(ci)
    for tick in axis_temporal.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_val) 
    for tick in axis_temporal.xaxis.get_major_ticks():
                tick.label.set_fontsize(tick_val) 
                  
    return fig, gs




def match_corr_mat(a_ref, c_ref, a_est, c_est, matches, spatial = False):
    '''
    Generates a correlation matrix between two pairs of neural signals: (a_ref, c_ref) and (a_est, c_est)
    The rows of the correlation matrix are indexed based on the order of the neurons in a_ref, c_ref
    args:
        a_ref: ndarray, (d x K1). Describes spatial footprints of K1 neurons (each neuron footprint is over d pixels)
        c_ref: ndarray, (T x K1). Describes temporal traces of K1 neurons (each trace has T time points)
        a_est: ndarray, (d x K2). Describes spatial footprints of K2 neurons (each neuron footprint is over d pixels)
        c_est: ndarray, (T x K1). Describes temporal traces of K1 neurons (each trace has T time points)
        matches: matches array established by matching neurons between (a_ref, c_ref) and (a_est, c_est). see 'match' above
        
    returns:
        correlation matrix: (K2 x K1) matrix describing correlation between each neuron of (a_ref, c_ref) and their corresponding matches. Matches are aligned along the diagonal of the matrix. 
        NOTE: the rows of the correlation matrix correspond to neurons from (a_est, c_est). The columns correspond to 
        a_ref, c_ref. 
    
    '''
    
    match_matrix = np.zeros((a_est.shape[1], a_ref.shape[1]))

    remaining = [i for i in range(a_est.shape[1])]
    est_indices = [-1 for i in range(a_est.shape[1])]

    for k in range(a_ref.shape[1]):
        curr_match = int(matches[k])
        if curr_match >= 0 and k < len(est_indices):
            est_indices[k] = curr_match
            remaining.remove(curr_match)
        else:
            continue

    for k in range(len(est_indices)):
        if est_indices[k] < 0:
            est_indices[k] = remaining[0]
            remaining.remove(remaining[0]) #Remove the element we added here

    for k in range(match_matrix.shape[0]):
        for j in range(match_matrix.shape[1]):
            if spatial:
                temp_cos = ca_utils.cosine_similarity(a_est[:, est_indices[k]], a_ref[:, j])
            else:
                temp_cos = ca_utils.cosine_similarity(c_est[:, est_indices[k]], c_ref[:, j])
            match_matrix[k, j] = temp_cos
    return match_matrix






############## Functions for Generating Demixing Videos #######################


def generate_movie(a_use, c_use, dims, random_values, slice_val=2,seed=999, order=0):
    '''
    Function generates a demixing video in which each neuron is assigned its own color
    args:
        a_use: ndarray, (d x K). Provides the spatial footprints of K neurons over a d-pixel field of view
        c_use: ndarray, (T x K). Provides temporal traces of K neurons over a T-frame video
        dims: tuple, (x,y). Provides dimensions of video (x * y = d)
        random_values
    
    '''
    np.random.seed(seed) #seed for random color generation (to keep it consistent if desired)
    final_movie = np.zeros((dims[0], dims[1], 3, len(slice_val))) #z is number of planes, and 3 is because we use RGB data
    if order == 0:
        order_string="F"
    elif order == 1:
        order_string="C"
    for k in range(a_use.shape[1]):

        print("{} : {}".format(k, a_use.shape[1]))
        curr_shape = a_use[:, k].reshape((-1,1))
        curr_trace = c_use[:, k].reshape((-1,1))
        print(curr_trace.shape)
       
        neuron_movie = curr_shape.dot(curr_trace.T)
        neuron_movie = neuron_movie.reshape((dims[0],dims[1],-1), order=\
                                           order_string).squeeze()
        
        random_color = random_values[k, :].squeeze()#[random.randint(0,255) for i in range(3)]
        total = random_color[0] + random_color[1] + random_color[2]
        normalized_color = [random_color[i]/total for i in range(3)]
        
        slice_val_matched = slice_val - np.amin(slice_val)
        
        final_movie[:, :, 0, slice_val_matched] += neuron_movie*normalized_color[0]
        final_movie[:, :, 1, slice_val_matched] += neuron_movie*normalized_color[1]
        final_movie[:, :, 2, slice_val_matched] += neuron_movie*normalized_color[2]
        
    final_movie = final_movie / np.amax(final_movie) #Now it is back to 0 -- 1
    return final_movie.squeeze()


def generate_movie_maxproj(a_use, c_use, dims, random_values, slice_val=2,seed=999, order=0):
    '''
    Function generates a demixing video in which each neuron is assigned its own color uses max projection of colors
    args:
        a_use: ndarray, (d x K). Provides the spatial footprints of K neurons over a d-pixel field of view
        c_use: ndarray, (T x K). Provides temporal traces of K neurons over a T-frame video
        dims: tuple, (x,y). Provides dimensions of video (x * y = d)
        random_values
    
    '''
    #We use a very inefficient approach to doing the maximum projection: 
    my_list = []
    for k in range(a_use.shape[1]):
        prod = a_use[:, [k]].dot(c_use[:, [k]].T)
        my_list.append(prod)
    
    #Now we make a list of the support of each component...
    support_list = []
    for k in range(a_use.shape[1]):
        curr_king = a_use[:, k]
        support = (a_use[:, k] > 0)
        for j in range(a_use.shape[1]):
            if k == j: 
                continue
            else:
                comparison_elt = a_use[:, j]
                support = np.logical_and(support, (curr_king > comparison_elt))
        support_list.append(support)
                
                
    
    
    np.random.seed(seed) #seed for random color generation (to keep it consistent if desired)
    final_movie = np.zeros((dims[0], dims[1], 3, len(slice_val))) #z is number of planes, and 3 is because we use RGB data
    if order == 0:
        order_string="F"
    elif order == 1:
        order_string="C"
    for k in range(a_use.shape[1]):

        print("{} : {}".format(k, a_use.shape[1]))
        curr_shape_crop = a_use[:, k] * support_list[k]
        curr_shape = curr_shape_crop.reshape((-1,1))
        curr_trace = c_use[:, k].reshape((-1,1))
        print(curr_trace.shape)
       
        neuron_movie = curr_shape.dot(curr_trace.T)
        
        neuron_movie = neuron_movie.reshape((dims[0],dims[1],-1), order=\
                                           order_string).squeeze()
        
#         divisor = np.amax(neuron_movie, axis = 2, keepdims = True)
#         divisor[divisor == 0] = 1
        neuron_movie = neuron_movie / np.amax(neuron_movie.flatten())
        
        random_color = random_values[k, :].squeeze()#[random.randint(0,255) for i in range(3)]
        total = random_color[0] + random_color[1] + random_color[2]
        normalized_color = [random_color[i]/total for i in range(3)]
        
        slice_val_matched = slice_val - np.amin(slice_val)
        
        final_movie[:, :, 0, slice_val_matched] += neuron_movie*normalized_color[0]
        final_movie[:, :, 1, slice_val_matched] += neuron_movie*normalized_color[1]
        final_movie[:, :, 2, slice_val_matched] += neuron_movie*normalized_color[2]
        
#     final_movie = final_movie / np.amax(final_movie) #Now it is back to 0 -- 1
    return final_movie.squeeze()




def write_mpl_no_compress_comparisons(mov_list, 
              filename, img_types,
              fr=3, 
              titles=None, scale=1, titlesize=20, ticksize=14, colorticksize=12, width_const = 4, offset = 0):
    """ Functionality to create customized triptych demixing videos
    Args:
        mov_list: list of movies to be included in triptych
        filename: desired filename of triptych file
        img_types: list of integers. For each movie, this list describes its image type (RGB, grayscale, etc.)
            value of 1 indicates grayscale, 2 indicates RGB
        fr: frame rate (used for FFMPEG video generation)
        titles: list, strings. List of titles for each movie
        scale: integer. This function generates all frames which are a multiple of scale. Scale = 1 means all frames included in triptych.
        titlesize: integer. Font size of title
        ticksize: integer. Size of tick marks for each image
        colorticksize: integer. Size of tick marks for each image
    """
    # Declare & Assign Local Variables
    n_mov = len(mov_list)
    T = mov_list[0].shape[2]
    if titles is None:
        titles = ['']*n_mov
        
    for mdx, mov in enumerate(mov_list):
        if img_types[mdx] == 1 and mdx < len(mov_list) - 2:
            mov_list[mdx] -= np.amin(mov_list[mdx])
    
    #Compute scales
    mins = np.empty(n_mov)
    maxs = np.empty(n_mov)
    print("GETTING MOVIE")
    for mdx, mov in enumerate(mov_list):
        print(mdx)
        print(mov[mdx].shape)
        mins[mdx] = np.min(mov) 
        maxs[mdx] = np.max(mov) 
    realmin = np.amin(mins)
    realmax = np.amax(maxs)

    # Decide where to save intermediate results
    start_directory = os.getcwd()
    
    if os.path.exists('tmp'):
        raise FileExistsError("Remove the tmp directory before running this visualization code")
    else:
        save_directory = 'tmp'
        os.makedirs(save_directory)
        delete_tmp = True

    start = time.time()
    # Plot & Save Frames
    T = int(np.floor(T/scale))
    data_list = []
    batch_limit = 500
    count = 0 
    
    
    
    for i in range(T):
        temp_mov_list = []
        t = scale * i
        for mdx, mov in enumerate(mov_list):
            if img_types[mdx] == 2: 
                temp_mov_list.append(mov_list[mdx][:, :, :, [t]])
            elif img_types[mdx] == 1:
                temp_mov_list.append(mov_list[mdx][:, :, [t]])
            else:
                print("INVALID IMG TYPE PROVIDED")
        curr_data = dict()
        curr_data['mov_list'] = temp_mov_list
        curr_data['titles'] = titles
        curr_data['img_types'] = img_types
        curr_data['mins'] = mins
        curr_data['maxs'] = maxs
        curr_data['i'] = i
        curr_data['scale'] = scale
        curr_data['width_const'] = width_const
        curr_data['ticksize'] = ticksize
        curr_data['colorticksize'] = colorticksize
        curr_data['titlesize'] = titlesize
        curr_data['save_directory'] = save_directory
        curr_data['filename'] = filename
        curr_data['offset'] = offset
        data_list.append(curr_data)
        
        count += 1
        if count == batch_limit or i == T-1:
            count = 0
            runpar(make_frame_par, data_list)
            data_list = []
       
    print('Generating the files took this many seconds: {}'.format(time.time() - start))
    
    os.chdir(save_directory)
    
    # Call FFMPEG to compile PNGs
    subprocess.call(['ffmpeg',
                 '-framerate', str(fr),
                 '-i', filename + '%04d.png',
                 '-r', str(fr),\
                 '-pix_fmt', 'yuv420p',
                 '-compression_level', '0',
                 filename + '.mp4'])
    
    #Move the .png file out of the folder and delete the folder
    shutil.copy2(filename + ".mp4", start_directory)
    os.chdir(start_directory)
    shutil.rmtree(save_directory)

def make_frame_par(curr_data):
    '''
    Helper function for generating the frames of the MP4 file
    '''
    
    mov_list = curr_data['mov_list']
    titles = curr_data['titles'] 
    img_types = curr_data['img_types'] 
    mins = curr_data['mins']
    maxs = curr_data['maxs']
    i = curr_data['i']
    scale = curr_data['scale']
    width_const = curr_data['width_const']
    ticksize = curr_data['ticksize']
    colorticksize = curr_data['colorticksize']
    titlesize = curr_data['titlesize']
    save_directory = curr_data['save_directory']
    filename = curr_data['filename']
    offset_frame = curr_data['offset']
    
    
    n_mov = len(mov_list)
    t = scale*i
    first_mov = mov_list[0]
    max_scale = np.amax(first_mov[:, :, 0])
    min_scale = np.amin(first_mov[:, :, 0])

    gs = gridspec.GridSpec(2, ceil(n_mov/2), height_ratios=(1,1), width_ratios=(tuple([1 for i in range(ceil(n_mov/2))])))
    fig = plt.figure(figsize=(width_const*n_mov,11))


    # Display Current Frame From Each Mov
    # print("we reached start of inner for loop")
    for mdx, mov in enumerate(mov_list):
        oddness = (mdx) % 2
        ax = fig.add_subplot(gs[oddness,  mdx//2])
        divider = make_axes_locatable(ax)

        # Display standard grayscale image
        if img_types[mdx] == 1:
            im = ax.imshow(mov[:,:,0], vmin=mins[mdx], vmax=maxs[mdx])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')


            if mdx % 2 == 0:
                ax.set_xticklabels([])
            else:
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(ticksize) 
                cbar.ax.tick_params(labelsize=colorticksize)
            if mdx > 1:
                ax.set_yticklabels([])
            else:
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(ticksize)
                cbar.ax.tick_params(labelsize=colorticksize)           
                
        elif img_types[mdx] == 2: #We display color image
            im = ax.imshow(mov[:,:,:,0])
            if mdx > 1: 
                ax.set_yticklabels([])
            if mdx % 2 == 0:
                ax.set_xticklabels([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.remove()

        if mdx == 0:
            ax.set_title(titles[mdx] + " Frame:{}".format(t + offset_frame), fontsize=titlesize)
        else:
            ax.set_title(titles[mdx], fontsize=titlesize)

    # Save Figure As PNG
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, filename + "%04d.png" % i))


    # Close Figure
    plt.close('all')
  
   


def clip_raw_movie(movie, clip_factor = 0.5):
    max_value = np.amax(movie)
    max_value_clip = clip_factor * max_value
    
    min_value = np.amin(movie)
    clipped_mov = np.clip(movie, min_value, max_value_clip)
    return clipped_mov

def threshold_movie_max(movie, threshold):
    min_value = np.amin(movie)
    clipped_mov = np.clip(movie, min_value, threshold)
    return clipped_mov


def generate_color_movie(a_use, c_use, dims, random_values, seed=999):
    '''
    Function generates a demixing video in which each neuron is assigned its own color
    args:
        a_use: ndarray. Dimensions (d1, d2, K). Provides the spatial footprints of K neurons over a (d1 x d2)-pixel field of view
        c_use: ndarray, (K, T). Provides temporal traces of K neurons over a T-frame video
        dims: tuple, (x,y). Provides dimensions of video (x * y = d)
        random_values: random color values assigned to each neural signal in the color movie
        seed: int. Seed for random color generation. Set this to make random generation deterministic
    '''
    np.random.seed(seed) #seed for random color generation (to keep it consistent if desired)
    final_movie = np.zeros((dims[0], dims[1], 3, c_use.shape[1])) #z is number of planes, and 3 is because we use RGB data
    
    
    sum_random_values = np.sum(random_values, axis = 1, keepdims = True)
    random_color_norm = random_values / sum_random_values
    
    c_color = np.zeros((c_use.shape[0], 3, c_use.shape[1]))
    for i in range(3):
        c_color[:, i, :] = random_color_norm[:, [i]] * c_use
        
    final_movie = np.tensordot(a_use, c_color, axes = (2,0))
    print("the shape of final movie after tensordot is {}".format(final_movie.shape))
    max_val = np.amax(final_movie)
    if max_val != 0:
        final_movie = final_movie / np.amax(final_movie) #Now it is back to 0 -- 1
    return final_movie.squeeze()


def standard_demix_vid(a, c, b, raw_mov, denoised_mov, fluctuating_bg_terms, filename, order = "C", fr=30, \
              titles=None, scale=1, titlesize=20, ticksize=14, colorticksize=12, a_real = None, c_real = None, \
                      rgbrange = [80, 255], channels = 3, mean_sub_res = False, min_sub_signals = True, width_const = 3.5,\
                      random_values = None, start = 0, end = 1000, dim1_range=None, dim2_range=None):
    '''
    Generates a 'standard' triptych demixing video based on ring localNMF. Provided for quick visualization of ring localNMF results
    args:
        a: np.ndarray. Dimensions (d1, d2, K) -- d1 and d2 are dimensions of FOV and K is the number of neurons
        c: np.ndarray. Dimensions (K, T) -- K is number of neurons T is number of frames
        raw_mov: np.ndarray. Raw movie. Dimensions (d1, d2, T). Movie has field of view consisting of d1 * d2 pixels and T frames
        denoised_mov: 3D ndarray. Motion corrected and denoised movie. 
        
        fluctuating_bg_terms: tuple of two ndarrays. First element multiplied by second element gives fluctuating background for entire movie, a (d1*d2, T) array. To reshape to (d1, d2, T) array, use the field "order" (described below). 
        filename: desired filename of mp4 demixing video
        order: "C" or "F". The order in which the (d1*d2, T)-shape ndarrays in this function can be reshaped into (d1, d2, T)-shape full movies. (Use this in the numpy reshape function)
        fr: int. Frame-rate of mp4 video
        titles: list of strings. List describing the title of each video of the triptych. 
        scale: int. Number of frames to skip. So scale = 1 generates a video showing each frame, scale = 2 shows every second frame, etc. 
        titlesize: int. Size of title in figures
        ticksize: int. Size of ticks of panels
        colorticksize: int. Size of color ticks of panel colorbars
        
        a_real: np.ndarray. Dimensions (d1, d2, K). Ground truth neurons, if applicable. 
        c_real: np.ndarray. Dimensions (K, T). Ground truth neuron temporal traces, if applicable. 
        rgbrange: list of ints (length 2). List, [a, b], where [a, b] describes the interval of RGB values (in the range 0-255) from which we sample. 
        channels: int. Indicates number of channels in color image (typically 3)
        mean_sub_res: boolean. Indicates whether or not we should subtract the mean of each pixel of the residual movie in the finald display. 
        min_sub_signals: boolean. Indicates whether we min-subtract the signals 'c' 
        width_const: nonnegative number. Width constant used to scale the width of each displayed image in the mp4 file. 
        random_values: dimensions (K, channels). The i-th descibes the RGB color assigned to neuron 'i' in the color video.
        start: int. First frame to include in video
        end: int. Last frame to include in video
        '''
    
    if dim1_range is None: 
        dim1_range = [0, raw_mov.shape[0]]
    if dim2_range is None: 
        dim2_range = [0, raw_mov.shape[1]]
    x, y, T = raw_mov.shape
    titles = []
    img_types =[]
    mov_list = []
    
    
    #Define range of frames included in videos:
    
    # Add raw movie: 
    titles.append("Motion Corrected Data")
    img_types.append(1)
    mov_list.append(clip_raw_movie(raw_mov)[dim1_range[0]:dim1_range[1], dim2_range[0]:dim2_range[1],:])
    
    #Add denoised movie:
    titles.append("Denoised Data")
    img_types.append(1)
    denoised_max = np.amax(denoised_mov)
    mov_list.append(denoised_mov[dim1_range[0]:dim1_range[1], dim2_range[0]:dim2_range[1], :])
    
    
    
    #Add colored A*C video
    if random_values is None: 
        random_values = np.array([random.randint(rgbrange[0],rgbrange[1]) for i in range(channels*a.shape[2])]).reshape((a.shape[2], channels))
    
    
    #Testing out min subtraction...
    if min_sub_signals:
        c_minsub = c - np.amin(c, axis = 1, keepdims = True)
        AC_color_image = generate_color_movie(a, c_minsub, [x,y,end-start], random_values)
        titles.append("Estimates Colored")
    else:
        AC_color_image = generate_color_movie(a, c, [x,y,end-start], random_values)
        titles.append("Estimates Colored")
    img_types.append(2)
    mov_list.append(AC_color_image[dim1_range[0]:dim1_range[1], dim2_range[0]:dim2_range[1], :, :])
    
    
    #Add grayscale A*C video:
    AC = np.tensordot(a, c, axes=(2,0))
    
    print("done with AC calculation")
    
    #Perform min subtraction of the AC panel...
    if min_sub_signals:
        c_minsub = c - np.amin(c, axis = 1, keepdims = True)
        AC_minsub = np.tensordot(a, c_minsub, axes=(2,0))
        img_types.append(1)
        mov_list.append(threshold_movie_max(AC_minsub, denoised_max)[dim1_range[0]:dim1_range[1], dim2_range[0]:dim2_range[1], :])
#         min_AC = np.amin(AC, axis = 2, keepdims = True)
#         AC = AC - min_AC
        titles.append("Signal Estimates")
    else:
        titles.append("Signal Estimates")
        img_types.append(1)
        mov_list.append(threshold_movie_max(AC, denoised_max)[dim1_range[0]:dim1_range[1], dim2_range[0]:dim2_range[1], :])
    
    
    
    
    #If there is ground truth, add it to the triptych
    if a_real is not None and c_real is not None:
        random_values = np.array([random.randint(rgbrange[0],rgbrange[1]) for i in range(channels*a_real.shape[1])]).reshape((a_real.shape[1], channels))
        AC_real_color_image = generate_movie(a_real, c_real, [x,y,end-start], random_values)
        titles.append("Ground Truth Colored")
        img_types.append(2)
        mov_list.append(AC_real_color_image[dim1_range[0]:dim1_range[1], dim2_range[0]:dim2_range[1], :, :])
        
       
    #Add net background background: 
    
    print('started fbg')
    #Load relevant ringlocalNMF outputs
    fluctuating_bg = fluctuating_bg_terms[0].dot(fluctuating_bg_terms[1]).reshape((x, y, -1), order="F")
    print("done with fluctuating calculation")
    print("included b now")
    net_bg = fluctuating_bg + b
    titles.append("Net Background")
    img_types.append(1)
    mov_list.append(net_bg[dim1_range[0]:dim1_range[1], dim2_range[0]:dim2_range[1], :])
    
    #Add residual:
    res = denoised_mov - AC - net_bg
    if mean_sub_res == False:
        titles.append("Residual")
    else:
        print("the shape of res is {}".format(res.shape))
        res = res - np.mean(res, axis = 2, keepdims = True)
        titles.append("Residual")
    titles.append("Residual")
    img_types.append(1)
    mov_list.append(threshold_movie_max(res, denoised_max * 0.5)[dim1_range[0]:dim1_range[1], dim2_range[0]:dim2_range[1], :])
    
    
    #Generate Demixing Video
    write_mpl_no_compress_comparisons(mov_list, filename, img_types, titles = titles, fr=fr, \
            scale=scale, titlesize=titlesize, ticksize=ticksize, colorticksize=colorticksize, width_const = width_const, offset=start) 
       

    
def standard_demix_vid_old(rlt, mov_raw, mov_denoised, filename, fr=30, \
              titles=None, threshold = [], scale=1, titlesize=20, ticksize=14, colorticksize=12, a_real = None, c_real = None, \
                      rgbrange = [80, 255], channels = 3, mean_sub_res = False, min_sub_signals = False, width_const = 3.5,\
                      random_values = None, maxproj = False):
    '''
    Generates a 'standard' triptych demixing video based on ring localNMF. Provided for quick visualization of ring localNMF results
    args:
        rlt: dictionary. Contains outputs of ringlocalNMF demixing algorithm
        mov_raw: 3D ndarray. Raw movie. Dimensions (d1, d2, T). Movie has field of view consisting of d1 * d2 pixels and T frames
        mov_denoised: 3D ndarray. Motion corrected and denoised movie. 
        filename: desired filename of triptych file
        img_types: list of integers. For each movie, this list describes its image type (RGB, grayscale, etc.)
            value of 1 indicates grayscale, 2 indicates RGB
        fr: frame rate (used for FFMPEG video generation)
        titles: list, strings. List of titles for each movie
        scale: integer. This function generates all frames which are a multiple of scale. Scale = 1 means all frames included in triptych.
        titlesize: integer. Font size of title
        ticksize: integer. Size of tick marks for each image
        colorticksize: integer. Size of tick marks for each image
        a_real: ndarray, dimensions (d1 * d2, K). Ground truth neurons, if applicable.
        c_real: ndarray, dimensions (d1*d2, K). Ground truth neurons, if applicable. 
        rgbrange: tuple of integers, length 2. Indicates the range of rgb values used to generate colorful demixing videos. 
            Chosen to guarantee all signals are bright enough to see. 
        channels: integer, value = 3. Indicates number of channels in colorful videos..
    Returns: 
        No return values
    
    Demixing Triptych Video includes:
        - Raw video
        - Denoised Video
        - Video of source extractions (A * C)
        - RGB-color video of algorithm estimates
        - RGB-color video of ground-truth (if applicable)
        - Static Background Estimates
        - Fluctuating Background Estimates
        - Residual
    '''
    
    x, y, T = mov_raw.shape
    titles = []
    img_types =[]
    mov_list = []
    
    #Load relevant ringlocalNMF outputs
    c = rlt['c']
    a = rlt['a']
    W = rlt['W']
    W = W.astype("float")
    b = rlt['b']
    
    #Define range of frames included in videos:
    start = 0
    end = mov_denoised.shape[2] 
    
    # Add raw movie: 
    titles.append("Raw Data")
    img_types.append(1)
    mov_list.append(mov_raw)
    
    #Add denoised movie:
    titles.append("Denoised Data")
    img_types.append(1)
    mov_list.append(mov_denoised)
    
    #Add grayscale A*C video:
    AC_pre = a.dot(c.T)
    AC = AC_pre.reshape(mov_denoised.shape, \
                       order="F")
    
    #Perform min subtraction of the AC panel...
    if min_sub_signals:
        min_AC = np.amin(AC, axis = 2, keepdims = True)
        AC = AC - min_AC
        titles.append("Signal Estimates")
    else:
        titles.append("Signal Estimates")
    img_types.append(1)
    mov_list.append(AC)
    
    
    #Add colored A*C video
    frames = [i for i in range(start,end)]
    if random_values is None: 
        random_values = np.array([random.randint(rgbrange[0],rgbrange[1]) for i in range(channels*a.shape[1])]).reshape((a.shape[1], channels))
    
    
    #Testing out min subtraction...
    if min_sub_signals:
        c_minsub = c - np.amin(c, axis = 0, keepdims = True)
        if maxproj:
             AC_color_image = generate_movie_maxproj(a, c_minsub[start:end,:], [x,y,end-start], random_values, slice_val=frames)
        else:
            AC_color_image = generate_movie(a, c_minsub[start:end,:], [x,y,end-start], random_values, slice_val=frames)
        titles.append("Estimates Colored")
    else:
        if maxproj:
             AC_color_image = generate_movie_maxproj(a, c_minsub[start:end,:], [x,y,end-start], random_values, slice_val=frames)
        else:
            AC_color_image = generate_movie(a, c[start:end,:], [x,y,end-start], random_values, slice_val=frames)
        titles.append("Estimates Colored")
    img_types.append(2)
    mov_list.append(AC_color_image)
    
    #If there is ground truth, add it to the triptych
    if a_real is not None and c_real is not None:
        random_values = np.array([random.randint(rgbrange[0],rgbrange[1]) for i in range(channels*a_real.shape[1])]).reshape((a_real.shape[1], channels))
        AC_real_color_image = generate_movie(a_real, c_real[start:end, :], [x,y,end-start], random_values, slice_val=frames, order=1)
        titles.append("Ground Truth Colored")
        img_types.append(2)
        mov_list.append(AC_real_color_image)
        
    
    
    #Add static background estimate:
    add_vector = np.zeros((x,y, T)) 
    b_used = b.reshape((x,y,1), order="F")
    b_mov = b_used + add_vector
    if min_sub_signals:
        b_mov += min_AC
    titles.append("Static Background")
    img_types.append(1)
    mov_list.append(b_mov)
    
    #Add fluctuating background: 
    mov_den_r = mov_denoised.reshape(x*y, -1, order="F")
    bkgd_f = W.dot(mov_den_r - AC_pre - b)
    bkgd_f = bkgd_f.reshape((mov_denoised.shape), order="F")
    titles.append("Fluctuating Background")
    img_types.append(1)
    mov_list.append(bkgd_f.astype("float"))
    
    #Add residual:
    res = mov_denoised - AC_pre.reshape(mov_denoised.shape, \
                       order="F") - b_used - bkgd_f
    if mean_sub_res == False:
        titles.append("Residual")
    else:
        print("the shape of res is {}".format(res.shape))
        res = res - np.mean(res, axis = 2, keepdims = True)
        titles.append("Residual")
    titles.append("Residual")
    img_types.append(1)
    mov_list.append(res)
    
    #Generate Demixing Video
    write_mpl_no_compress_comparisons(mov_list, filename, img_types, titles = titles, fr=fr, \
            scale=scale, titlesize=titlesize, ticksize=ticksize, colorticksize=colorticksize, width_const = width_const)
    