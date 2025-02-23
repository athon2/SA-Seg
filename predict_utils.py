import os 
from scipy.ndimage.filters import gaussian_filter
import numpy as np 
from batchgenerators.augmentations.utils import pad_nd_image
import torch 
from torch import nn 
from typing import Union, Tuple, List
import glob 
from torch.cuda.amp import autocast
from tqdm import tqdm 
from utils.utils import npy2mha, load_json

def load_checkponit(net, ckpt_path):
    ckpt_list = glob.glob(os.path.join(ckpt_path,"model_best.pth"))
    assert len(ckpt_list) > 0, 'No ckpt found in '+ ckpt_path 

    best_ckpt = ckpt_list[0]
    print("Best ckpt: ", best_ckpt)
    checkpoint = torch.load(best_ckpt)
    state_dict = checkpoint["state_dict"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

    return net 

def get_gaussian(patch_size, sigma_scale=1./8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def compute_steps_for_sliding_window(patch_size, image_size, step_size):
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps

def get_sliding_window_generator(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float,
                                 verbose: bool = False):
    if len(tile_size) < len(image_size):
        assert len(tile_size) == len(image_size) - 1, 'if tile_size has less entries than image_size, len(tile_size) ' \
                                                      'must be one shorter than len(image_size) (only dimension ' \
                                                      'discrepancy of 1 allowed).'
        steps = compute_steps_for_sliding_window(tile_size, image_size[1:], tile_step_size)
        if verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicer = tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), tile_size)]])
                    yield slicer
    else:
        steps = compute_steps_for_sliding_window(tile_size, image_size, tile_step_size)
        if verbose: print(f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)]])
                    yield slicer


def maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor, grid: torch.Tensor = None, mirror_axes: Tuple[int, ...] = None) \
        -> torch.Tensor:
    if grid is not None:
        prediction = network(x, grid)
    else:
        prediction = network(x)

    if mirror_axes is not None:
        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
        if 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (4,))), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 4))), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3, 4))), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
        prediction /= num_predictons

    return prediction


def predict(net, ds_test, num_classes, patch_size, test_dir, step_size=0.5, mirror_axes=(0,1,2), win_min=-1200, win_max=600, final_nonlin=lambda x:torch.sigmoid(x), th=0.5, over_write=False):
    net.eval()
    net.do_ds = False
    net.em = False 
    net.inference = True 
    torch.cuda.empty_cache()

    with torch.no_grad():
        with autocast():
            gaussian = torch.from_numpy(get_gaussian(patch_size, sigma_scale=1./8)).cuda()
            gaussian = gaussian.half()
            gaussian[gaussian == 0] = gaussian[gaussian != 0].min()
            for i in tqdm(list(ds_test.keys())):
                target_path = os.path.join(test_dir, i+'.nii.gz')
                if os.path.exists(target_path) and not over_write:
                    continue 
                torch.cuda.empty_cache()
                all_data = np.load(ds_test[i]['data_file'], 'r')
                properties = load_json(ds_test[i]['properties_file'])
                crop_bbox = properties['crop_bbox']
                original_shape = properties['original_shape']

                lung_bbox_data = all_data[:-1, 
                                          crop_bbox[0][0]: crop_bbox[0][1],
                                          crop_bbox[1][0]: crop_bbox[1][1],
                                          crop_bbox[2][0]: crop_bbox[2][1],
                                          ]
                data, slicer_revert_padding = pad_nd_image(lung_bbox_data, patch_size, 'constant', {'value': 0}, True, None)
                
                data = torch.from_numpy(data.copy()).float()
                data[data > win_max] = win_max
                data[data < win_min] = win_min
                data = (data - win_min)/(win_max - win_min)

                slicers = get_sliding_window_generator(data.shape[1:], patch_size, step_size, verbose=False)
                
                predicted_logits = torch.zeros((num_classes, *data.shape[1:]), dtype=torch.half,
                                                device=torch.device('cuda'))
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=torch.device('cuda'))
                gaussian = gaussian.to(torch.device('cuda'))

                for sl in slicers:
                    workon = data[sl][None]
                    workon = workon.to(torch.device('cuda'), non_blocking=False)
                    
                    prediction = maybe_mirror_and_predict(net, workon, mirror_axes)[0].to(torch.device('cuda'))
                    if prediction.shape != gaussian.shape:
                        prediction = prediction.squeeze()
                    predicted_logits[sl] += prediction * gaussian 
                    n_predictions[sl[1:]] += gaussian

                predicted_logits /= n_predictions
                predicted_logits = final_nonlin(predicted_logits)
                predict_res = predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])].cpu().numpy()
                if num_classes > 1:
                    predict_res = np.argmax(predict_res, axis=0)
                else:
                    predict_res = predict_res[0] >th

                seg_res = np.zeros(original_shape, dtype=np.uint8)
                seg_res[crop_bbox[0][0]: crop_bbox[0][1],
                        crop_bbox[1][0]: crop_bbox[1][1],
                        crop_bbox[2][0]: crop_bbox[2][1]]= predict_res.astype(np.uint8)

                npy2mha(seg_res, target_path, properties['itk_spacing'], properties['itk_origin'], properties['itk_direction']) 