import torch
import numpy as np
from localnmf.ca_utils import show_img, spatial_sum_plot
from tqdm import tqdm

def demix_whole_data_robust_ring_lowrank(pmd_video, cut_off_point=[0.95,0.9], length_cut=[15,10], th=[2,1], pass_num=1,
                                         residual_cut = [0.6,0.6], corr_th_fix=0.31, corr_th_fix_sec = 0.4,
                                         corr_th_del = 0.2, switch_point=10, ring_radius=15, merge_corr_thr=0.6,
                                         merge_overlap_thr=0.6, num_plane=1, plot_en=False, text=True, maxiter=35,
                                         update_after=4, pseudo_2=[0.1, 0.1], skips=2, custom_init = {},
                                         init=['lnmf', 'lnmf'], denoise = False, c_nonneg=True):
    '''
    This function is a low-rank pipeline with robust correlation measures and a ring background model. The low-rank implementation is in the HALS updates.
    Args:
    
    Part of pmd_video constructor:
        - U_sparse
        - R
        - V
        - data_shape
        - data_order
    
    
        
    Part of core localNMF algorithm: 
    
        INITIALIZATION: 
            - cut_off_point: this is the cut off for the superpixelization. In the "initialize signals" bit, this is passed to the pmd object
            - length_cut: this is another threshold used in the superpixelization. It is passed to the pmd object while doing the superpixel initialization
            - th: MAD threshold limit: this is another parameter passed to pmd_video when we do the initialization
        - pass_num: number of times we run this procedure (this is part of the "outer loop" of the algorithm, so let's leave this here)
            - residual_cut: SPA parameter, this should be passed to pmd object while doing superpixel init
        
        LOCALNMF LOOP:
            - corr_th_fix: this is a core localnmf algorithm, should be passed in to pmd_video
            - corr_th_fix_sec: same as corr_th_fix
            - corr_th_del: same as above two
            - switch_point: same deal
            - max_allow_neuron_size: same as above
            - merge_corr_thr: same as above
            - merge_overlap_thr: same as above
            - num_plane: same as above (but long term, need to eliminate this)
            - patch_size: Same as above. (Long term completely eliminate this!!)
            - plot_en: Same as above
            - text: Same 
            - maxiter: Same
            - update_after: Same
            - Pseudo
    
        
    
    
    
        U_sparse: torch_sparse.Tensor, shape (d1*d2, R). A PMD-denoised and compressed spatial representation of Yd_raw. R is the rank of the
            PMD decomposition.
        R: torch.Tensor, shape (R, R). 
        V: torch.Tensor, shape (R, T). Matrix with orthonormal rows, describing the temporal basis of the PMD decomposition.
        
        KEY -- URV = PMD-denoised movie
        
        data_shape: tuple describing (d1, d2, T), all ints. These are the two FOV dimensions and the number of frames
        data_order: "F" or "C" (string). Tells us how to reshape spatial matrices which have been "flattened" (i.e. instead of having separate dimensions (d1, d2, ...) they have (d1*d2, ..)
        
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
                -  'lnmf' (superpixel based initialization)
                - 'custom' (custom init values provided). NOTE: only be used for pass #1
            For example, init = ['custom', 'lnmf'] means the first pass is initialized with 
            neural network, the second with superpixels.
        custom_init: dict. keys: 'a','b','c'. A dictionary describing the custom values (a,b,c) provided for demixing.
            'a' is the spatial footprint, 'b' is the baseline, 'c' is the temporal trace
        sb: boolean. Stands for 'static background'. If false, we turn off static background estimates throughout pipeline
            Usually kept as true. 
        pseudo_corrr: list of nonnegative floats, length = pass_num. This is a robust correlation measure used when 
            calculating correlation images of neurons in the HALS updates. For data in which neurons don't overlap much, this
            should be left at 0.
        device: string. identifies whether certain operations (matrix updates, etc.) should be moved to gpu and 
            accelerated. Standard options: 'cpu' for CPU-only. 'cuda' for GPU computations.
        batch_size: int. For GPU computing, identifies how many pixels to process at a time (to make the pipeline compatible 
            with GPUs of various memory constraints)
        plot_debug: boolean. Indicates whether intermediate-step visualizations should be generated during demixing. Used for purposes
            of visualization.
    '''
    
    dims = pmd_video.shape
    d1, d2, T  = dims
    order = pmd_video.data_order
    
    ii = 0;
    while ii < pass_num:
        print("start " + str(ii+1) + " pass!");
                
        #######
        ### Initialization method
        #######       
        if init[ii] == 'lnmf':  
            if ii == 0:
                a = None
                c = None
            pmd_video.initialize_signals_superpixels(num_plane, cut_off_point[ii], residual_cut[ii], length_cut[ii], th[ii], pseudo_2[ii], \
                                       text =text, plot_en = plot_en)
                
        elif init[ii]=='custom' and ii == 0:
            pmd_video.initialize_signals_custom(custom_init)
       
        else:
            raise ValueError("Invalid initialization scheme provided")
        
        
        #######
        ## Run demixing pipeline
        #######

        with torch.no_grad():
            
            a, c, b, W, res, corr_img_all_r, num_list = update_AC_bg_l2_Y_ring_lowrank(pmd_video, maxiter, corr_th_fix,
                                                                                       corr_th_fix_sec, corr_th_del,
                                                                                       switch_point, skips,
                                                                                       merge_corr_thr, merge_overlap_thr,
                                                                                       ring_radius, denoise=denoise,
                                                                                       plot_en=plot_en,
                                                                                       update_after=update_after, c_nonneg=c_nonneg)
            torch.cuda.empty_cache() #Test this as placeholder for now to avoid GPU memory getting clogged
            
        
        #If multi-pass, save results from first pass
        if pass_num > 1 and ii == 0:
            W_final = W.create_complete_ring_matrix(a)
            rlt = {'a':a, 'c':c, 'b':b, "W":W_final, 'res':res, 'corr_img_all_r':corr_img_all_r, 'num_list':num_list, 'data_order': order, 'data_shape':(d1, d2, T)};
            a0 = a.copy();
        ii = ii+1;

    if plot_en:
        if pass_num > 1:
            spatial_sum_plot(a0, a, dims[:2], num_list_fin = num_list, text = text, order=order);
            
        ##TODO: This should be in the PMD movie functionality, not here
        Cnt = pmd_video.compute_local_correlation_image()
        scale = np.maximum(1, int(Cnt.shape[1]/Cnt.shape[0]));
        plt.figure(figsize=(8*scale,8))
        ax1 = plt.subplot(1,1,1);
        show_img(ax1, Cnt);
        ax1.set(title="Local mean correlation for residual")
        ax1.title.set_fontsize(15)
        ax1.title.set_fontweight("bold")
        plt.show();
    
    W_final = W.create_complete_ring_matrix(a)
    fin_rlt = {'U_sparse': pmd_video.u_sparse.cpu().to_scipy(layout='csr'), 'R': pmd_video.r.cpu().numpy(), 'V': pmd_video.v.cpu().numpy(), 'a':a, 'c':c, 'b':b, "W":W_final, 'res':res, 'corr_img_all_r':corr_img_all_r, 'num_list':num_list, 'data_order': order, 'data_shape':(d1, d2, T)};
    
    
    if pass_num > 1:
        return {'rlt':rlt, 'fin_rlt':fin_rlt, "superpixel_rlt":pmd_video.superpixel_rlt}
    else:
        return {'fin_rlt':fin_rlt, "superpixel_rlt":pmd_video.superpixel_rlt}


def update_AC_bg_l2_Y_ring_lowrank(pmd_video, maxiter,corr_th_fix, corr_th_fix_sec, corr_th_del, switch_point, skips,
                                   merge_corr_thr, merge_overlap_thr, ring_radius, denoise=None, plot_en=False,
                                   update_after=4, c_nonneg=True):
    """
    Function for computing background, spatial and temporal components of neurons. Uses HALS updates to iteratively
    refine spatial and temporal estimates.
    """

    data_order = pmd_video.data_order
    
    pmd_video.precompute_quantities(maxiter, ring_radius)
    pmd_video.compute_standard_correlation_image()
    pmd_video.compute_residual_correlation_image()
    pmd_video.update_hals_scheduler()
    pmd_video.update_ring_model_support()

    
    if denoise is None: 
        denoise = [False for i in range(maxiter)]
    elif len(denoise) != maxiter:
        print("Length of denoise list is not consistent, setting all denoise values to false for this pass of NMF")
        denoise = [False for i in range(maxiter)]
           
    for iters in tqdm(range(maxiter)):
        if iters >= maxiter - switch_point:
            corr_th_fix = corr_th_fix_sec 

        ##TODO: Add back the plot corr image bit here if desired
        pmd_video.static_baseline_update()
        
        if iters >= skips:
            pmd_video.fluctuating_baseline_update(ring_radius)
        else:
            pass
        
        pmd_video.spatial_update(plot_en=plot_en)
        pmd_video.static_baseline_update()
    
        denoise_flag = denoise[iters]
        pmd_video.temporal_update(denoise=denoise_flag, plot_en=plot_en, c_nonneg=c_nonneg)
           
        if update_after and ((iters+1) % update_after == 0):


            ##First: Compute correlation images
            pmd_video.compute_standard_correlation_image()
            pmd_video.compute_residual_correlation_image()
            
            pmd_video.support_update_prune_elements_apply_mask(corr_th_fix, corr_th_del, plot_en)


            
            #TODO: Eliminate the need for moving a and c off GPU
            pmd_video.merge_signals(merge_corr_thr, merge_overlap_thr, plot_en, data_order)
            pmd_video.update_ring_model_support()
            pmd_video.update_hals_scheduler()
            

    pmd_video.delete_precomputed()
    a, c, b, w, res, corr_img_all_r, num_list = pmd_video.brightness_order_and_return_state()
    
    return a, c, b, w, res, corr_img_all_r, num_list
    #TODO: Look into "res" and "num_list"