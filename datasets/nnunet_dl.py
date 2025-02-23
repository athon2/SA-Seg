from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import List
import numpy as np 
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
import torch 
import random 
from typing import Union, Tuple
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import  RenameTransform, NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose
from datasets.spatial_transform import DownsampleSegForDSTransform2, LWNormalizeTransform

def get_dataset(pkl_file, data_dir, ratio=1.):
    data_ids_all = load_pickle(pkl_file)
    data_ids = data_ids_all if ratio == 1. else data_ids_all[:int(len(data_ids_all)*ratio)]
    dataset = OrderedDict()
    for c in data_ids:
        # c = '-'.join([i for i in c.split("_")])
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(data_dir, "%s.npy" % c)
        dataset[c]['properties_file'] = join(data_dir, "%s.json" % c) 
    # for i in dataset.keys():
    #     dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset 

def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)

class nnUNetDataLoader(DataLoader):
    def __init__(self,
                 data,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 oversample_foreground_percent: float = 0.0,
        ):
        super().__init__(data, batch_size, 1, None, False, False, True)
        # assert isinstance(data, nnUNetDataset), 'nnUNetDataLoaderBase only supports dictionaries as data'
        self.indices = list(data.keys())

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
    
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)

        self.data_shape = (self.batch_size, 1, *self.patch_size)
        self.seg_shape = (self.batch_size, 1, *self.patch_size)

        self.get_do_oversample = self._oversample_last_XX_percent 

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None]):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
           
            selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] 

            voxels_of_that_class = np.asarray(class_locations[selected_class]) if selected_class is not None else None
            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs
    
    def load_case(self, index):
        all_data = np.load(self._data[index]['data_file'], 'r')
        properties = load_json(self._data[index]['properties_file'])

        data = all_data[:-1]
        seg = all_data[-1:]
        
        return data, seg, properties
    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self.load_case(i)
            
            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            # print(bbox_lbs, bbox_ubs, this_slice, shape, data.shape, padding)
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}
    

class LimitedLenWrapper(NonDetMultiThreadedAugmenter):
    def __init__(self, my_imaginary_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.len = my_imaginary_length

    def __len__(self):
        return self.len
    

class TreeSegDataLoader(nnUNetDataLoader):
    def __init__(self,
                 data,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 oversample_foreground_percent: float = 0.0,
                 part_type: str = 'part_label',
                 dist_type: str = 'dist'
        ):
        super().__init__(data, batch_size, patch_size, final_patch_size, oversample_foreground_percent)
        self.dist_type = dist_type
        self.part_type = part_type

    def load_case(self, index):
        all_data = np.load(self._data[index]['data_file'], 'r')
        properties = load_json(self._data[index]['properties_file'])
        part_label = np.load(self._data[index]['data_file'][:-4]+'_'+ self.part_type+'.npy', 'r')[np.newaxis]
        
        data = all_data[:-1]
        mask = all_data[-1:]
        seg = np.zeros_like(mask)

        seg[mask > 0] = 1
        if self.dist_type is not None:
            dist = np.load(self._data[index]['data_file'][:-4]+ '_'+self.dist_type+'.npy', 'r')[np.newaxis]
            seg[part_label > 0] = 2
        else:
            dist = None 

        return data, seg, dist, properties


    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        dist_all = np.zeros(self.data_shape, dtype=np.float32)

        case_properties = []

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            data, seg, dist, properties = self.load_case(i)

            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]
            
            if self.dist_type is not None:
                dist = dist[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]

            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)

            if self.dist_type is not None:
                dist_all[j] = np.pad(dist, ((0, 0), *padding), 'constant', constant_values=0)
        
        return {'data': data_all, 'seg': seg_all, 'dist': dist_all,'properties': case_properties, 'keys': selected_keys} 


def get_tree_seg_dataloader(ds_tr, ds_val, part_type, dist_type, patch_size, batch_size, iters_train, 
                            iters_val, oversample, num_processes, mirror_axes=(0,1,2), 
                            deep_supervision_scales=None, win_min=-1000, win_max=600):
    rotation_for_DA = {
            'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            }

    init_patch_size = get_patch_size(patch_size, *rotation_for_DA.values(), (0.85, 1.25))


    dl_tr = TreeSegDataLoader(ds_tr, batch_size, patch_size, init_patch_size, oversample, part_type, dist_type)
    dl_val = TreeSegDataLoader(ds_val, batch_size, patch_size, init_patch_size, oversample, part_type, dist_type)

     # do_scale=True
    tr_transforms = []

    tr_transforms.append(LWNormalizeTransform(win_min, win_max))

    if dist_type is not None:
        from datasets.spatial_transform import SpatialTransform, MirrorTransform
    else:
        from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
        p_rot_per_axis=1,  # todo experiment with this
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
        random_crop=False,  # random cropping is part of our dataloaders
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False  # todo experiment with this
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=None))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                        output_key='target'))
    tr_transforms.append(NumpyToTensor(['data', 'target', 'dist'], 'float'))
    tr_transforms = Compose(tr_transforms)

    val_transforms = []
    val_transforms.append(LWNormalizeTransform(win_min, win_max))
    val_transforms.append(RenameTransform('seg', 'target', True))
    if deep_supervision_scales is not None:
        val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                            output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target', 'dist'], 'float'))
    val_transforms = Compose(val_transforms)


    gen_train = LimitedLenWrapper(iters_train, data_loader=dl_tr, transform=tr_transforms,
                            num_processes=num_processes, num_cached=6, seeds=None,
                            pin_memory=True, wait_time=0.02)
    gen_val = LimitedLenWrapper(iters_val, data_loader=dl_val, transform=val_transforms,
                            num_processes=max(1, num_processes // 2), num_cached=3, seeds=None,
                            pin_memory=True, wait_time=0.02) 
    
    return gen_train, gen_val