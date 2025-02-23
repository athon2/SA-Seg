from skimage import measure 
from scipy import ndimage
import numpy as np 
from batchgenerators.augmentations.utils import resize_segmentation

def LCC(input_data, num_classes):
    '''
    Largest Connected Component
    '''
    processed_image = np.zeros_like(input_data)
    if num_classes == 1: # for binary segment
        num_classes += 1
    for cur_class in range(1, num_classes):
        cur_data = np.zeros_like(input_data)
        cur_data[input_data == cur_class] = 1 
        if cur_data.sum()>0:
            label_img = measure.label(cur_data, connectivity=1)
            unique, counts = np.unique(label_img, return_counts=True)
            list_seg = list(zip(unique, counts))[1:]
            list_seg = sorted(list_seg,key=lambda x:x[1], reverse=True) # sorted
            processed_image[label_img == list_seg[0][0]] = cur_class
            for u,c in list_seg:
                if c/list_seg[0][1] >= 0.5:
                    processed_image[label_img == u] = cur_class
                else:
                    break 

    return processed_image 


def reample_seg(seg, original_size, size_after_cropping, crop_bbox, order=1):
    dtype_data = seg.dtype
    original_output = np.zeros(original_size, dtype=dtype_data)
    cropping_output = resize_segmentation(seg, size_after_cropping, order).astype(dtype_data)
    original_output[crop_bbox[0][0]:crop_bbox[0][1], 
                    crop_bbox[1][0]:crop_bbox[1][1], 
                    crop_bbox[2][0]:crop_bbox[2][1]] = cropping_output

    return original_output