import os 
import numpy as np 
from preprocessing.cropping import get_bbox_from_mask
from utils.utils import save_json, read_image, segment_lung_mask, npy2mha
from scipy import ndimage
from skimage.morphology import skeletonize_3d
from skimage import morphology, measure
from utils.tree_parse import skeleton_parsing, tree_parsing_func, loc_trachea, adjacent_map, parent_children_map, tree_refinement,whether_refinement,large_connected_domain

def parse_tree(binary_tree):
    binary_tree = large_connected_domain(binary_tree)
    skeleton = skeletonize_3d(binary_tree)
    skeleton_parse, cd, num = skeleton_parsing(skeleton)
    tree_parsing = tree_parsing_func(skeleton_parse, binary_tree, cd)
    trachea = loc_trachea(tree_parsing, num)
    ad_matric = adjacent_map(tree_parsing, num)
    parent_map, children_map, generation = parent_children_map(ad_matric, trachea, num)
    while whether_refinement(parent_map, children_map, tree_parsing, num, trachea) is True:
        tree_parsing, num = tree_refinement(parent_map, children_map, tree_parsing, num, trachea)
        trachea = loc_trachea(tree_parsing, num)
        ad_matric = adjacent_map(tree_parsing, num)
        parent_map, children_map, generation = parent_children_map(ad_matric, trachea, num)

    return parent_map, children_map, tree_parsing, skeleton  

def get_part_label(label, erode_r=1, dilate_r=1, erode_times=1, dilate_times=1):
    eroded = morphology.binary_erosion(label, np.ones([erode_r for _ in range(len(label.shape))])) 
    for i in range(1, erode_times):
        eroded = morphology.binary_erosion(eroded, np.ones([erode_r for _ in range(len(label.shape))]))  
    label_img = measure.label(eroded, connectivity=1)
    unique, counts = np.unique(label_img, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]
    list_seg = sorted(list_seg,key=lambda x:x[1], reverse=True) # sorted
    max_region = np.zeros_like(label)
    max_region[label_img == list_seg[0][0]] = 1
    dilation = morphology.binary_dilation(max_region, np.ones([dilate_r for _ in range(len(label.shape))]))  
    for i in range(1, dilate_times):
        dilation = morphology.binary_dilation(dilation, np.ones([dilate_r for _ in range(len(label.shape))]))  

    result = label * dilation

    return result.astype(np.int8)

def process_part_label(data_dir, case, target_dir, label_name='airway_label.nii.gz', 
                      erode_r=2, dilate_r=3, erode_times=1, dilate_times=1):
    target_path = os.path.join(target_dir, case + "_pu_label.npy")
    seg_file = os.path.join(data_dir, case, label_name)
    if not os.path.exists(seg_file):
        raise NotImplementedError(label_name)
    seg, _ = read_image(seg_file)
    label_npy = (seg > 0).astype(np.int8)

    part_label = get_part_label(label_npy, erode_r, dilate_r, erode_times, dilate_times)

    np.save(target_path, part_label.astype(np.int8))

def parse_label(data_dir, case, label_name='airway_label.nii.gz'):
    tree_parsing_path = os.path.join(data_dir, case, 'parse.nii.gz')
    skel_path = os.path.join(data_dir, case, 'skel.nii.gz')

    if not os.path.exists(tree_parsing_path):
        seg_file = os.path.join(data_dir, case, label_name)
        seg, properties_image = read_image(seg_file)

        print('get parse', case)
        binary_label = (seg > 0).astype(np.int8)
        binary_label = ndimage.binary_fill_holes(binary_label).astype(np.int8)
        _,_, tree_parsing, skeleton = parse_tree(binary_label)
        
        npy2mha(tree_parsing, tree_parsing_path, properties_image['itk_spacing'], 
                properties_image['itk_origin'], properties_image['itk_direction'])
        
        if not os.path.exists(skel_path):
            npy2mha(skeleton, skel_path, properties_image['itk_spacing'], 
                    properties_image['itk_origin'], properties_image['itk_direction'])
    print(case, ' parse done.')   

def preprocess_CT(data_dir, case, target_dir, label_name='airway_label.nii.gz', all_classes=[1,]):
    target_path = os.path.join(target_dir, case + ".npy")
    json_file = os.path.join(target_dir, case + ".json")
    if os.path.exists(json_file) and os.path.exists(target_path):
        print(case, ' done.')
        return  
    image_file = os.path.join(data_dir, case, "ct.nii.gz")
    seg_file = os.path.join(data_dir, case, label_name)
    if not os.path.exists(seg_file):
        # seg_file = os.path.join(data_dir, case, "airway_segments_label.nii.gz")
        raise NotImplementedError(label_name)
    # pa_file = os.path.join(data_dir, case, "pa_pred_02.nii.gz")
    lung_mask = os.path.join(data_dir, case, "lung_mask.nii.gz")
    image, properties_image = read_image(image_file)
    seg, _ = read_image(seg_file)
           
    if not os.path.exists(lung_mask):
        print('get lung', case)
        
        lung = segment_lung_mask(image_file, None)
        npy2mha(lung, lung_mask, properties_image['itk_spacing'], 
                properties_image['itk_origin'], properties_image['itk_direction'])
    else:
        lung, _ = read_image(lung_mask)

    lung[seg > 0] = 1 

    bbox = get_bbox_from_mask(lung)

    pack_data = np.concatenate((image[np.newaxis], seg[np.newaxis]), axis=0)
    properties_image['crop_bbox'] = bbox

    num_samples = 10000
    min_percent_coverage = 0.01 # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
    rndst = np.random.RandomState(1234)
    class_locs = {}
    
    for c in all_classes:
        all_locs = np.argwhere(seg == c)
        if len(all_locs) == 0:
            class_locs[c] = []
            continue
        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]

        class_locs[c] = selected.tolist()
    
    properties_image['class_locations'] = class_locs
    
    np.save(target_path, pack_data.astype(np.int16))
    save_json(properties_image, json_file)
    print(case, ' done.')