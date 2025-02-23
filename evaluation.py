import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm
from skimage import morphology
import argparse
from utils.utils import read_mha, npy2mha
from utils.evaluation_atm_22 import *   

def eval(pred_folder, label_dir, tag_folder, postprop='LCC', 
         num_classes=1, metrics_str="TD,BD,DSC,Pre,Sen,Spe", file_name='result', label_name='airway_label.nii.gz'):
    dataset = label_dir.split('/')[-1]
    target_path = os.path.join(tag_folder, f'{file_name}_{dataset}_{postprop}.csv')
    if os.path.exists(target_path):
        return
    pred_list = glob.glob(pred_folder+"/*.nii.gz")
    result = []

    test_ids = []
    
    metrics = tuple([i for i in metrics_str.split(',')])

    for case in tqdm(pred_list):
        case_name = case.split(os.sep)[-1]
        case_id = case_name.split('.nii.gz')[0]
        origin_case_id = case_id
        seg_label, seg_img = read_mha(os.path.join(label_dir, origin_case_id, label_name))
        pred_npy, _ = read_mha(case)

        # seg_label = (seg_label == 1).astype(np.int8)
        if postprop != 'None':
            if postprop == 'LCC':
                from utils.postprocess import LCC
                pred_npy = LCC(pred_npy, num_classes)
            else:
                raise NotImplementedError

        parse_path = os.path.join(label_dir, origin_case_id, "parse.nii.gz")
        skel_path = os.path.join(label_dir, origin_case_id, "skel.nii.gz")
        if not os.path.join(parse_path):
            pass  
        else:
            parse_npy, _ = read_mha(parse_path)

        cur_result = []
        metrics_item = []
        if not os.path.exists(skel_path):
            centerline_gt = morphology.skeletonize_3d((seg_label > 0).astype(np.int8)) > 0
            centerline_gt = centerline_gt.astype(np.int8)
            npy2mha(centerline_gt, skel_path, seg_img.GetSpacing(), seg_img.GetOrigin(), seg_img.GetDirection())
        else:
            centerline_gt, _ = read_mha(skel_path)
            centerline_gt = (centerline_gt > 0).astype(np.int8)
        binary_pred = (pred_npy > 0).astype(np.int8)
        binary_label = (seg_label > 0).astype(np.int8)
        for metric in metrics:
            if metric == 'BD':
                # from evaluation_atm_22 import branch_detected_calculation
                total_branch_num, detected_branch_num, detected_branch_ratio = branch_detected_calculation(binary_pred, parse_npy, centerline_gt)
                cur_result.append(detected_branch_ratio)
                # metrics_item.append(metric)
            elif metric == 'TD':
                # from evaluation_atm_22 import tree_length_calculation
                cur_result.append(tree_length_calculation(binary_pred, centerline_gt))
                # metrics_item.append(metric)
            elif metric == 'Pre':
                # from evaluation_atm_22 import precision_calculation
                cur_result.append(precision_calculation(binary_pred, binary_label))
            elif metric == 'DSC':
                cur_result.append(dice_coefficient_score_calculation(binary_pred, binary_label))
            elif metric == 'Sen':
                cur_result.append(sensitivity_calculation(binary_pred, binary_label))
            elif metric == 'Spe':
                cur_result.append(specificity_calculation(binary_pred, binary_label))

            metrics_item.append(metric)
        test_ids.append(case_id)
        result.append(cur_result)
    result = np.asarray(result)
    avg_res = np.array([[round(np.mean(result[:,i]), 4)for i in range(len(metrics_item))]])
    std_res = np.array([[round(np.std(result[:,i]), 4) for i in range(len(metrics_item))]])
    test_ids.append('avg')
    test_ids.append('std')
    result = np.concatenate((result, avg_res, std_res), 0)

    all_idx = np.concatenate((np.asarray([test_ids]).T, result), axis=-1)
    df = pd.DataFrame(
        all_idx,
        columns=['case_id', ] + list(metrics_item), 
        index=None 
    )
    df = df.dropna()
    
    df.to_csv(target_path, index=False, sep=' ')