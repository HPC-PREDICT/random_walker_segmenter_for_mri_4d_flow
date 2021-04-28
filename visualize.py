# Use the RW algorithm to segment in 3D at one time point
# Make the segmentation slightly larger and extend it to all other time points
# Save the segmentated image using the hpc-predict-io classes

# ============================   
# import module and set paths
# ============================   
import numpy as np
from mr_io import SegmentedFlowMRI
import matplotlib.pyplot as plt
import utils
import imageio


# basepath = '/tmp/test.decrypt7/segmenter/cnn_segmenter/hpc_predict/v1/inference/2021-02-11_20-14-44_daint102_volN'
basepath = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/pollux/'

def read_segmentation(subnum):
    subject_specific_basepath = basepath + '2021-02-11_19-41-32_daint102_volN' + str(subnum) + '/output/recon_volN' + str(subnum)
    segmentedflowmripath = subject_specific_basepath + '_vn_seg_rw.h5'
    return SegmentedFlowMRI.read_hdf5(segmentedflowmripath)

# ===============================================================
# ===============================================================
def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped

segmented_flow_mri = read_segmentation(1)
flowMRI_seg1 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

segmented_flow_mri = read_segmentation(2)
flowMRI_seg2 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

segmented_flow_mri = read_segmentation(3)
flowMRI_seg3 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

segmented_flow_mri = read_segmentation(4)
flowMRI_seg4 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

segmented_flow_mri = read_segmentation(5)
flowMRI_seg5 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

segmented_flow_mri = read_segmentation(6)
flowMRI_seg6 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

segmented_flow_mri = read_segmentation(7)
flowMRI_seg7 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

# save as pngs
#for idx in range(19):
#    
#    plt.figure(figsize=[35,10])
#    
#    plt.subplot(2,7,1)
#    plt.imshow(utils.norm(flowMRI_seg1[:,:,idx,3,1], flowMRI_seg1[:,:,idx,3,2], flowMRI_seg1[:,:,idx,3,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg1[:,:,idx,3,4], cmap='gray', alpha = 0.5)
#    plt.title('volN1,z'+str(idx)+'t3')
#    plt.subplot(2,7,2)
#    plt.imshow(utils.norm(flowMRI_seg2[:,:,idx,3,1], flowMRI_seg2[:,:,idx,3,2], flowMRI_seg2[:,:,idx,3,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg2[:,:,idx,3,4], cmap='gray', alpha = 0.5)
#    plt.title('volN2,z'+str(idx)+'t3')
#    plt.subplot(2,7,3)
#    plt.imshow(utils.norm(flowMRI_seg3[:,:,idx,3,1], flowMRI_seg3[:,:,idx,3,2], flowMRI_seg3[:,:,idx,3,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg3[:,:,idx,3,4], cmap='gray', alpha = 0.5)
#    plt.title('volN3,z'+str(idx)+'t3')
#    plt.subplot(2,7,4)
#    plt.imshow(utils.norm(flowMRI_seg4[:,:,idx,3,1], flowMRI_seg4[:,:,idx,3,2], flowMRI_seg4[:,:,idx,3,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg4[:,:,idx,3,4], cmap='gray', alpha = 0.5)
#    plt.title('volN4,z'+str(idx)+'t3')
#    plt.subplot(2,7,5)
#    plt.imshow(utils.norm(flowMRI_seg5[:,:,idx,3,1], flowMRI_seg5[:,:,idx,3,2], flowMRI_seg5[:,:,idx,3,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg5[:,:,idx,3,4], cmap='gray', alpha = 0.5)
#    plt.title('volN5,z'+str(idx)+'t3')
#    plt.subplot(2,7,6)
#    plt.imshow(utils.norm(flowMRI_seg6[:,:,idx,3,1], flowMRI_seg6[:,:,idx,3,2], flowMRI_seg6[:,:,idx,3,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg6[:,:,idx,3,4], cmap='gray', alpha = 0.5)
#    plt.title('volN6,z'+str(idx)+'t3')
#    plt.subplot(2,7,7)
#    plt.imshow(utils.norm(flowMRI_seg7[:,:,idx,3,1], flowMRI_seg7[:,:,idx,3,2], flowMRI_seg7[:,:,idx,3,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg7[:,:,idx,3,4], cmap='gray', alpha = 0.5)
#    plt.title('volN7,z'+str(idx)+'t3')
#
#
#    plt.subplot(2,7,8)
#    plt.imshow(utils.norm(flowMRI_seg1[:,:,8,idx,1], flowMRI_seg1[:,:,8,idx,2], flowMRI_seg1[:,:,8,idx,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg1[:,:,8,idx,4], cmap='gray', alpha = 0.5)
#    plt.title('volN1,z8'+'t'+str(idx))
#    plt.subplot(2,7,9)
#    plt.imshow(utils.norm(flowMRI_seg2[:,:,8,idx,1], flowMRI_seg2[:,:,8,idx,2], flowMRI_seg2[:,:,8,idx,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg2[:,:,8,idx,4], cmap='gray', alpha = 0.5)       
#    plt.title('volN2,z8'+'t'+str(idx))
#    plt.subplot(2,7,10)
#    plt.imshow(utils.norm(flowMRI_seg3[:,:,8,idx,1], flowMRI_seg3[:,:,8,idx,2], flowMRI_seg3[:,:,8,idx,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg3[:,:,8,idx,4], cmap='gray', alpha = 0.5)
#    plt.title('volN3,z8'+'t'+str(idx))
#    plt.subplot(2,7,11)
#    plt.imshow(utils.norm(flowMRI_seg4[:,:,8,idx,1], flowMRI_seg4[:,:,8,idx,2], flowMRI_seg4[:,:,8,idx,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg4[:,:,8,idx,4], cmap='gray', alpha = 0.5)
#    plt.title('volN4,z8'+'t'+str(idx))
#    plt.subplot(2,7,12)
#    plt.imshow(utils.norm(flowMRI_seg5[:,:,8,idx,1], flowMRI_seg5[:,:,8,idx,2], flowMRI_seg5[:,:,8,idx,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg5[:,:,8,idx,4], cmap='gray', alpha = 0.5)
#    plt.title('volN5,z8'+'t'+str(idx))
#    plt.subplot(2,7,13)
#    plt.imshow(utils.norm(flowMRI_seg6[:,:,8,idx,1], flowMRI_seg6[:,:,8,idx,2], flowMRI_seg6[:,:,8,idx,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg6[:,:,8,idx,4], cmap='gray', alpha = 0.5)
#    plt.title('volN6,z8'+'t'+str(idx))
#    plt.subplot(2,7,14)
#    plt.imshow(utils.norm(flowMRI_seg7[:,:,8,idx,1], flowMRI_seg7[:,:,8,idx,2], flowMRI_seg7[:,:,8,idx,3]), cmap='gray', alpha = 0.5)
#    plt.imshow(flowMRI_seg7[:,:,8,idx,4], cmap='gray', alpha = 0.5)
#    plt.title('volN7,z8'+'t'+str(idx))
#
#    plt.savefig('/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/pngs' + str(idx) + '.png')
#    plt.close()

nr = 4
nc = 7

for idx in range(19):
    
    plt.figure(figsize=[2*nc,2*nr])
    
    zidx = idx
    # velocity magnitude across z axis
    plt.subplot(nr, nc, 1);  plt.imshow(utils.norm(flowMRI_seg1[:,:,zidx,3,1], flowMRI_seg1[:,:,zidx,3,2], flowMRI_seg1[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('vol1_z'+str(zidx)+'_t3');
    plt.subplot(nr, nc, 2);  plt.imshow(utils.norm(flowMRI_seg2[:,:,zidx,3,1], flowMRI_seg2[:,:,zidx,3,2], flowMRI_seg2[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('vol2_z'+str(zidx)+'_t3');
    plt.subplot(nr, nc, 3);  plt.imshow(utils.norm(flowMRI_seg3[:,:,zidx,3,1], flowMRI_seg3[:,:,zidx,3,2], flowMRI_seg3[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('vol3_z'+str(zidx)+'_t3');
    plt.subplot(nr, nc, 4);  plt.imshow(utils.norm(flowMRI_seg4[:,:,zidx,3,1], flowMRI_seg4[:,:,zidx,3,2], flowMRI_seg4[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('vol4_z'+str(zidx)+'_t3');
    plt.subplot(nr, nc, 5);  plt.imshow(utils.norm(flowMRI_seg5[:,:,zidx,3,1], flowMRI_seg5[:,:,zidx,3,2], flowMRI_seg5[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('vol5_z'+str(zidx)+'_t3');
    plt.subplot(nr, nc, 6);  plt.imshow(utils.norm(flowMRI_seg6[:,:,zidx,3,1], flowMRI_seg6[:,:,zidx,3,2], flowMRI_seg6[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('vol6_z'+str(zidx)+'_t3');
    plt.subplot(nr, nc, 7);  plt.imshow(utils.norm(flowMRI_seg7[:,:,zidx,3,1], flowMRI_seg7[:,:,zidx,3,2], flowMRI_seg7[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('vol7_z'+str(zidx)+'_t3');
    
    # segmentation across z-axis
    plt.subplot(nr, nc, 8); plt.imshow(flowMRI_seg1[:,:,zidx,3,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 9); plt.imshow(flowMRI_seg2[:,:,zidx,3,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 10); plt.imshow(flowMRI_seg3[:,:,zidx,3,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 11); plt.imshow(flowMRI_seg4[:,:,zidx,3,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 12); plt.imshow(flowMRI_seg5[:,:,zidx,3,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 13); plt.imshow(flowMRI_seg6[:,:,zidx,3,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 14); plt.imshow(flowMRI_seg7[:,:,zidx,3,4], cmap='gray'); plt.axis('off')
    
    # velocity magnitude across t axis
    plt.subplot(nr, nc, 15); plt.imshow(utils.norm(flowMRI_seg1[:,:,8,idx,1], flowMRI_seg1[:,:,8,idx,2], flowMRI_seg1[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('vol1_z8_t'+str(idx));
    plt.subplot(nr, nc, 16); plt.imshow(utils.norm(flowMRI_seg2[:,:,8,idx,1], flowMRI_seg2[:,:,8,idx,2], flowMRI_seg2[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('vol2_z8_t'+str(idx));
    plt.subplot(nr, nc, 17); plt.imshow(utils.norm(flowMRI_seg3[:,:,8,idx,1], flowMRI_seg3[:,:,8,idx,2], flowMRI_seg3[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('vol3_z8_t'+str(idx));
    plt.subplot(nr, nc, 18); plt.imshow(utils.norm(flowMRI_seg4[:,:,8,idx,1], flowMRI_seg4[:,:,8,idx,2], flowMRI_seg4[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('vol4_z8_t'+str(idx));
    plt.subplot(nr, nc, 19); plt.imshow(utils.norm(flowMRI_seg5[:,:,8,idx,1], flowMRI_seg5[:,:,8,idx,2], flowMRI_seg5[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('vol5_z8_t'+str(idx));
    plt.subplot(nr, nc, 20); plt.imshow(utils.norm(flowMRI_seg6[:,:,8,idx,1], flowMRI_seg6[:,:,8,idx,2], flowMRI_seg6[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('vol6_z8_t'+str(idx));
    plt.subplot(nr, nc, 21); plt.imshow(utils.norm(flowMRI_seg7[:,:,8,idx,1], flowMRI_seg7[:,:,8,idx,2], flowMRI_seg7[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('vol7_z8_t'+str(idx));

    # segmentation across t-axis
    plt.subplot(nr, nc, 22); plt.imshow(flowMRI_seg1[:,:,8,idx,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 23); plt.imshow(flowMRI_seg2[:,:,8,idx,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 24); plt.imshow(flowMRI_seg3[:,:,8,idx,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 25); plt.imshow(flowMRI_seg4[:,:,8,idx,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 26); plt.imshow(flowMRI_seg5[:,:,8,idx,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 27); plt.imshow(flowMRI_seg6[:,:,8,idx,4], cmap='gray'); plt.axis('off')
    plt.subplot(nr, nc, 28); plt.imshow(flowMRI_seg7[:,:,8,idx,4], cmap='gray'); plt.axis('off')

    plt.savefig(basepath + 'pngs' + str(idx) + '.png')
    plt.close()
    
    
_gif = []
for idx in range(19):
    _gif.append(imageio.imread(basepath + 'pngs' + str(idx) + '.png'))
imageio.mimsave(basepath + 'pngs.gif', _gif, format='GIF', duration=0.25)
    