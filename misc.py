# ==========================================
# import modules
# ==========================================
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import scipy.ndimage.morphology as morph
import skimage.morphology as mp
from skimage._shared.utils import warn

# ==========================================
# loads the parec data of the selected subject and returns an array with the arrays for M, P, S, #timesteps, #slices in dataset 
# ==========================================
def load_parec_data(subject_name = 'AH'):

    # ============
    # set base path
    # ============
    basepath = os.getcwd() + '/../../data/eth_ibt/' + subject_name
    
    # ============    
    # set paths according to the subject that has to be loaded (default subject: AH)
    # ============
    if subject_name == 'AH':
        file_prefix = '/an_27052015_1027340_4_2_wipqflow_fbclearV4_'   
        path_s = basepath + file_prefix + 'S.rec'
        path_m = basepath + file_prefix + 'M.rec'
        path_p = basepath + file_prefix + 'P.rec'
    elif subject_name == 'CB':
        file_prefix = '/ch_11122015_1428290_4_2_wipqflow_fb_experiment1V4_'        
        path_s = basepath + file_prefix + 'S.rec'
        path_m = basepath + file_prefix + 'M.rec'
        path_p = basepath + file_prefix + 'P.rec'
    elif subject_name == 'DG':
        file_prefix = '/da_15072015_1612350_3_2_wipqflow_fbclearV4_'        
        path_s = basepath + file_prefix + 'S.rec'
        path_m = basepath + file_prefix + 'M.rec'
        path_p = basepath + file_prefix + 'P.rec'
    elif subject_name == 'LT':
        file_prefix = '/lo_27112015_1256300_2_2_wipqflow_fb_experiment1V4_'
        path_s = basepath + file_prefix + 'S.rec'
        path_m = basepath + file_prefix + 'M.rec'
        path_p = basepath + file_prefix + 'P.rec'
    elif subject_name == 'JR':
        file_prefix = '/ju_27052015_1142050_4_2_wipqflow_fbclearV4_'
        path_s = basepath + 'ju_27052015_1208240_5_1_wipqflow_fbclearV42.rec'
        path_m = basepath + file_prefix + 'M.rec'
        path_p = basepath + file_prefix + 'P.rec'        
        
    # ============
    # if wrong input is given, break function and return the warning in console
    # ============
    else:
        warn('Invalid subject')

    # ============
    # load the data into arrays
    # ============
    data_s = nib.parrec.load(path_s).get_data()
    data_m = nib.parrec.load(path_m).get_data()
    data_p = nib.parrec.load(path_p).get_data()
    
    # ============
    # calculate the numer of timesteps and slices.
    # note that the timesteps have to be divided by 2 as we have a magnitude and phase image for each time step
    # ============
    num_times = int(data_s.shape[3] / 2)
    num_slices = int(data_s.shape[2])
    
    # ============
    # return the desired vector of the loaded data
    # ============
    parec_data = [data_m, data_p, data_s, num_times, num_slices]
   
    return parec_data

# ==========================================
# ==========================================
def create_separated_arrays(parec_data):
    
    data_m = parec_data[0]
    data_p = parec_data [1]
    data_s = parec_data[2]
    num_times = parec_data[3]
    num_slices = parec_data[4]
    
    # ============
    # separate the velocity components out of the vector with velocity and proton density image
    # ============
    vs_vec = np.zeros((int(data_s.shape[0]), int(data_s.shape[1]), num_slices, num_times))
    vm_vec = np.zeros((int(data_m.shape[0]), int(data_m.shape[1]), num_slices, num_times))
    vp_vec = np.zeros((int(data_p.shape[0]), int(data_p.shape[1]), num_slices, num_times))
    m_vec = np.zeros((int(data_p.shape[0]), int(data_p.shape[1]), num_slices, num_times))
    
    # ============
    # ============
    for t in range(num_times):
        vs_vec[:,:,:,t] = data_s[:,:,:,t*2+1]
        vm_vec[:,:,:,t] = data_m[:,:,:,t*2+1]
        vp_vec[:,:,:,t] = data_p[:,:,:,t*2+1]
        m_vec[:,:,:,t] = data_p[:,:,:,t*2]
        
    # ============
    # ============
    separated_arrays = [m_vec, vm_vec, vp_vec, vs_vec]
    
    return separated_arrays

# ==========================================
# loads the numpy array saved from the dicom files of the Freiburg dataset
# ==========================================
def load_npy_data(freiburg_mri_path, subject):

    img_path = os.path.realpath(freiburg_mri_path)

    npy_files_list = []
    
    for _, _, file_names in os.walk(img_path):
    
        for file in file_names:
        
            if '.npy' in file:
                npy_files_list.append(file)
                
    # use passed subject numer to index into files list          
    path = img_path + '/{}'.format(npy_files_list[subject])
    array = np.load(path)
    
    return array

# ==========================================        
# function to normalize the input arrays (intensity and velocity) to a range between 0 to 1 and -1 to 1
# magnitude normalization is a simple division by the largest value
# velocity normalization first calculates the largest magnitude and then uses the components of this vector to normalize the x,y and z directions seperately
# ==========================================        
def normalize_arrays(arrays):
    
    # dimension of normalized_arrays: 128 x 128 x 20 x 25 x 4
    normalized_arrays = np.zeros((arrays.shape))
    
    # normalize magnitude channel
    normalized_arrays[...,0] = arrays[...,0]/np.amax(arrays[...,0])
    
    # normalize velocities
    # extract the velocities in the 3 directions
    velocity_arrays = np.array(arrays[...,1:4])
    # denoise the velocity vectors
    velocity_arrays_denoised = gaussian_filter(velocity_arrays, 0.5)
    # compute per-pixel velocity magnitude    
    velocity_mag_array = np.linalg.norm(velocity_arrays_denoised, axis=-1)
    # velocity_mag_array = np.sqrt(np.square(velocity_arrays[...,0])+np.square(velocity_arrays[...,1])+np.square(velocity_arrays[...,2]))
    # find max value of 95th percentile (to minimize effect of outliers) of magnitude array and its index
    vpercentile =  np.percentile(velocity_mag_array, 95)    
    normalized_arrays[...,1] = velocity_arrays_denoised[...,0] / vpercentile
    normalized_arrays[...,2] = velocity_arrays_denoised[...,1] / vpercentile
    normalized_arrays[...,3] = velocity_arrays_denoised[...,2] / vpercentile  
    # print('normalized arrays: max=' + str(np.amax(normalized_arrays)) + ' min:' + str(np.amin(normalized_arrays)))
  
    return normalized_arrays

# ==========================================        
# ==========================================        
def norm(x,y,z):
    
    normed_array = np.linalg.norm([x,y,z], axis=0)
    return normed_array

# ==========================================
# closing operation for postprocessing the segmentation to remove holes from the inside to avoid wrong seeds if used for 4D initialization
# takes a 3D volume and returns a 3D volume where every slice is eroded with a "circular" 3x3 kernel
# the rw data is then eroded and diluted and markers are assigned to the two classes, no markers are placed in the overlap so that the RW algorithm can fill these gaps
# ==========================================
def erode_seg_markers(rw_data):
    
    closing_kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                               [[0, 1, 0], [1, 1, 1], [0, 1, 0]],               
                               [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=bool)
    
    erosion_kernel = closing_kernel
    dilation_kernel = erosion_kernel
    
    rw_bool = np.array(rw_data, dtype=bool)
    
    closed_seg = morph.binary_closing(rw_bool, structure=closing_kernel)
    # skeletonized_seg = mp.skeletonize_3d(closed_seg)
    # medial_axis_seg = mp.medial_axis(closed_seg)
    
    thinned_seg = np.zeros(rw_data.shape)
    for i in range(rw_data.shape[2]):
        thinned_seg[:,:,i] = mp.thin(closed_seg[:,:,i], max_iter = 5)
    # eroded_seg = morph.binary_erosion(closed_seg,structure=erosion_kernel,iterations=3)
    dilated_seg = morph.binary_dilation(closed_seg, structure=dilation_kernel, iterations=6)
       
    fg_markers = np.zeros(rw_data.shape)   
    bg_markers = np.zeros(rw_data.shape)   
    markers = np.zeros(rw_data.shape)   
    
    # fg_markers = (np.logical_and(eroded_seg,dilated_seg))*1
    fg_markers = (np.logical_and(thinned_seg, dilated_seg))*1
    bg_markers = (np.logical_and(np.logical_not(markers), np.logical_not(dilated_seg)))*2
    
    markers = fg_markers + bg_markers
    
    return markers, fg_markers, bg_markers
