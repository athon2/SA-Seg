import SimpleITK as sitk
import numpy as np 
import torch 
import random
import pickle
import os 
import glob 
from skimage import morphology, measure
from skimage.filters import median
from scipy import ndimage
import time
os.environ["TZ"] = "UTC-8"
# time.tzset()
import argparse
import json

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)
        
def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def read_mha(path: str):
    '''Read .mha. 
    
    Args:
        path (str) : .mha file path
    Returns:
        array (nmupy.ndarray) : ct array
        img (SimpleITK.ImageObject : ct image
    '''
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    img = reader.Execute()
    array = sitk.GetArrayFromImage(img)
    
    return array,img

def read_image(file_name):
    itk_image = sitk.ReadImage(file_name)
    spacing = itk_image.GetSpacing()
    origin = itk_image.GetOrigin()
    direction = itk_image.GetDirection()
    npy_image = sitk.GetArrayFromImage(itk_image)

    dict = {
        'itk_spacing': spacing,
        'itk_origin': origin,
        'itk_direction': direction,
        'original_shape': npy_image.shape

    }

    return npy_image, dict 
    
def mha2nii(data_path, target_path):
    '''Convert .mha file to .nii.gz file
    
    Parameters
    ----------
    data_path : .mha file path.
    target_dir : .nii.gz file path to save.
    
    Returns
    ----------
    None
    
    '''
    reader = sitk.ImageFileReader()
    reader.SetFileName(data_path)
    img = reader.Execute()
    # npy_array = sitk.GetArrayFromImage(img)
    sitk.WriteImage(img, target_path, True)

def npy2mha(np_array: np.ndarray,
    out_name: str, 
    itk_spacing: tuple, 
    itk_origin: tuple,
    itk_direction: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    ) -> None:

    img = sitk.GetImageFromArray(np_array)
    img.SetSpacing(itk_spacing)
    img.SetOrigin(itk_origin)
    img.SetDirection(itk_direction)
    sitk.WriteImage(img, out_name, True)

def get_best_model(cp_dir: str, model_name: str) -> str:
    cp_list = glob.glob(os.path.join(cp_dir, model_name,"*.pth"))
    best_model = cp_list[0]
    best_cp = int(cp_list[0].split("/")[-1].split("_")[1])
    for cp in cp_list[1:]:
        ep_idx = int(cp.split("/")[-1].split("_")[1])
        if ep_idx > best_cp:
            best_cp = ep_idx
            best_model = cp
    for case in cp_list:
        if case != best_model:
            os.remove(case) 
    return best_model

def load_model(model, saved_model_path, phase="val", optimizer=None):
    import torch 
    print("Constructing model from checkpoint... ")
    checkpoint = torch.load(saved_model_path)
    state_dict = checkpoint["state_dict"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    if phase == "train" and optimizer is not None:
        epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, epoch, optimizer
    else:
        return model   
             
def load_eval_model(model, config, fold=0):
    checkpoint_dir = config.get("model","checkpoint_dir")
    checkpoint_name = config.get("model", "model_name")+"_"+config.get("model", "tag")
    if fold != 0:
        checkpoint_name += "/"+str(fold)
    print(checkpoint_dir, checkpoint_name)
    checkpoint = get_best_model(checkpoint_dir, checkpoint_name)
    print(checkpoint)
    model = load_model(model, checkpoint, phase="test")

    return model  

def to_one_hot(data, num_classes):
    '''
        data: (Z, Y, X)
        num_classes: N
        target_data: (N, Z, Y, X)
    '''
    target_data = np.zeros(tuple([num_classes,] + list(data.shape)))
    for i in range(num_classes):
        target_data[i][np.where(data == i)] = 1

    return target_data 

class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def get_u(labels, list_seg, tolerance=[30,30]):
    abs_center = [labels.shape[1]//2, labels.shape[2]//2]
    res_label = list()
    for u,c in list_seg[1:5]:
        coord = np.where(labels == u)
        for center_x,center_y in zip(coord[2],coord[1]):
            if np.abs(center_y - abs_center[0]) < tolerance[0] and np.abs(center_x - abs_center[1]) < tolerance[1]:
                res_label.append(u)
                break 
    return res_label
        
def binarize_lung_mask(dcm_array, threshold, smoothing=True):
    '''Get the lung mask from DICOM Array.
    Parameters
    ----------
    dcm_array : DICOM Array.
    threshold : a list of threholding value, 
            like [A, B], A is the lower bound and B is the upper bound.
    smoothing : bool value, 
            If True, the lung_mask will be smoothed by median (skimage.filters).
    
    Returns
    -----------
    binarize_lung_mask : ndarray
            the binarized lung mask
            
    '''
    threshold_mask_lower = dcm_array >= threshold[0]
    threshold_mask_upper = dcm_array <= threshold[1]
    threshold_mask = np.multiply(threshold_mask_lower, threshold_mask_upper)
    labels = measure.label(threshold_mask,connectivity=1)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]
    list_seg = sorted(list_seg,key=lambda x:x[1], reverse=True) # sorted
    u = get_u(labels, list_seg)
    lung_mask = np.zeros_like(dcm_array)
    for u_i in u: 
        lung_mask[np.where(labels == u_i)] = 1 
    if smoothing:
        lung_mask = median(lung_mask, behavior='ndimage')
    return lung_mask.astype(np.ubyte)

def segment_lung_mask(ct_path, airway_path):
    ct_array, _ = read_mha(ct_path)
    lung_array = binarize_lung_mask(ct_array,[-1024,-500],False)
    if airway_path is not None:
        airway_array, _ = read_mha(airway_path)
        lung_array = lung_array*(1 - airway_array)
    res_lung = np.zeros_like(lung_array)
    for i in range(lung_array.shape[0]):
        lung_slice = lung_array[i]
        eroded = morphology.binary_erosion(lung_slice,np.ones([4,4]))  
        dilation = morphology.binary_dilation(eroded,np.ones([8,8]))  
        labels = measure.label(dilation)   
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels) # 获取连通区域
        
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
        '''
        (0L, 0L, 512L, 512L)
        (190L, 253L, 409L, 384L)
        (200L, 110L, 404L, 235L)
        '''
        # 根据肺部标签获取肺部mask，并再次进行’膨胀‘操作，以填满并扩张肺部区域
        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.binary_dilation(mask,np.ones([5,5])) # one last dilation
        mask = ndimage.binary_fill_holes(mask,np.ones([5,5]))
        mask = morphology.binary_erosion(mask,np.ones([5,5]))
        mask = morphology.binary_erosion(mask,np.ones([3,3]))
        res_lung[i] = mask
    # npy2mha(res_lung.astype(np.ubyte), lung_mask_path, ct_img.GetSpacing(), ct_img.GetOrigin())
    return res_lung.astype(np.ubyte)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a
