import os.path as osp
from typing import Dict, Optional

import numpy as np
import torch.utils.data
import open3d as o3d
from IPython import embed
import h5py

from geotransformer.utils.common import load_pickle
from geotransformer.utils.pointcloud import random_sample_transform, apply_transform, inverse_transform, regularize_normals
from geotransformer.utils.registration import compute_overlap
from geotransformer.utils.open3d import estimate_normals, voxel_downsample
from geotransformer.transforms.functional import (
    normalize_points,
    random_jitter_points,
    random_shuffle_points,
    random_sample_points,
    random_crop_point_cloud_with_plane,
    random_sample_viewpoint,
    random_crop_point_cloud_with_point,
)

class CustomDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root: str,
        subset: str,
        num_points: int = 1024,
        voxel_size: Optional[float] = None,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        noise_magnitude: Optional[float] = None,
        keep_ratio: float = 0.7,
        crop_method: str = 'plane',
        asymmetric: bool = True,
        class_indices: str = 'all',
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        estimate_normal: bool = False,
        overfitting_index: Optional[int] = None,
        td: str = 'full_non_deformed'
    ):
        super(CustomDataset, self).__init__()

        # Depending on the value of subset, load one or the other of the data
        assert subset in ['train', 'val', 'test']
        assert crop_method in ['plane', 'point']

        self.dataset_root = dataset_root
        self.subset = subset

        self.num_points = num_points
        self.voxel_size = voxel_size
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.asymmetric = asymmetric
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.estimate_normal = estimate_normal
        self.overfitting_index = overfitting_index
        self.td = td

        # data_list = load_pickle(osp.join(dataset_root, f'{subset}.pkl'))
        # data_list = [x for x in data_list if x['label'] in self.class_indices]
        # if overfitting_index is not None and deterministic:
        #    data_list = [data_list[overfitting_index]]
        # self.data_list = data_list

        self.src = []
        self.tgt = []
        self.transformations = []
        
        if self.td == 'full_non_deformed':
            self.base = '../../../../dataset/Synthetic/FullNonDeformedData/'
        elif self.td == 'partial_non_deformed':
            self.base = '../../../../dataset/Synthetic/PartialNonDeformedData/'
        else:
            raise Exception('Specify valid training data')
    
        if self.subset == 'train':
            self.folders = [
                '000', '001', '003', '004', '006', '007', '009', 
                '010', '011', '013', '014', '016', '017', '018', 
                '020', '021', '023', '024', '026', '027', '028', 
                '030', '031', '033', '034', '036', '037', '038',
                '040', '041', '043', '044', '045', '047', '048', 
                '050', '051', '053', '054', '055', '057', '058',
                '060', '061', '063', '064', '065', '067', '068', 
                '070', '071', '072', '074', '075', '077', '078',
                '080', '081', '082', '084', '086', '087', '088',
                '090', '091', '092', '094', '095', '097', '098',
                '099', '101', '102', '104', '105', '107', '108',
                '109', '111', '112', '114', '115', '117', '118',
                '119', '121', '122', '124', '125', '127', '128',
                '129', '131', '132', '134', '135', '136', '138', 
                '139', '141', '142', '144', '145', '146', '148',
                '149', '151', '152', '154', '155', '156', '158',
                '159', '161', '162', '163', '165', '166', '168',
                '169', '171', '172', '173', '175', '176', '178', 
                '179', '181', '182', '183', '185', '186', '188',
                '189', '190', '192', '193', '195', '196', '198', 
                '199', '200', '202', '203', '205', '206', '208', 
                '209', '210', '212', '213', '215', '216', '217', 
                '219', '220', '222', '223', '224', '225'
            ]
            self.base = self.base + 'TrainingData/'
            self.n = 160

        elif self.subset == 'val':
            self.folders = [
                '005', '012', '019', '025', '032', '039', '046', '052', '059', '062', '069', '076', '083', '089', '096',
                '103', '110', '116', '123', '130', '137', '143', '150', '157', '164', '170', '177', '184', '191', '197',
                '204', '211', '218'
            ]
            self.base = self.base + 'ValidationData/'
            self.n = 33
        
        # if subset == 'train':
        #    pairs = open('../../../../dataset/TrainingData/astrivis-data-training/pairs.txt', 'r') 
        #    lines = pairs.readlines()
        #    f = h5py.File('../../../../dataset/TrainingData/astrivis-data-training/se4.h5', "r")
        #    for line in lines:
        #        pair = line.split(',')
        #        self.tgt.append('../../../../dataset/TrainingData/astrivis-data-training/' + pair[0])
        #        self.src.append('../../../../dataset/TrainingData/astrivis-data-training/' + pair[1][:-1])
        #        se4 = f[pair[1][:-1]]
        #        self.transformations.append(se4)
        # elif subset == 'val':
        #    pairs = open('../../../../dataset/TrainingData/astrivis-data-validation/pairs.txt', 'r') 
        #    lines = pairs.readlines()
        #    f = h5py.File('../../../../dataset/TrainingData/astrivis-data-validation/se4.h5', "r")
        #    for line in lines:
        #        pair = line.split(',')
        #        self.tgt.append('../../../../dataset/TrainingData/astrivis-data-validation/' + pair[0])
        #        self.src.append('../../../../dataset/TrainingData/astrivis-data-validation/' + pair[1][:-1])
        #        se4 = f[pair[1][:-1]]
        #        self.transformations.append(se4)
    
    def __getitem__(self, index):
        if self.overfitting_index is not None:
            index = self.overfitting_index

        index = str(index)
        index = index.zfill(3)

        if self.td == 'full_non_deformed':
            base_file = self.base + 'model' + index + '/'
        elif self.td == 'partial_non_deformed':
            base_file = self.base + 'model' + index + '/transformed/'
        else:
            raise Exception('Specify a valid dataset for training')
        
        source_transformation_file = base_file + 'mesh_transformed_0_se4.h5'
        target_transformation_file = base_file + 'mesh_transformed_1_se4.h5'
        
        src_trans_file=h5py.File(source_transformation_file, "r")
        src_pcd_transform = np.array(src_trans_file['transformation'])
        
        tgt_trans_file=h5py.File(target_transformation_file, "r")
        tgt_pcd_transform_inverse = np.linalg.inv(np.array(tgt_trans_file['transformation']))

        transform = tgt_pcd_transform_inverse@src_pcd_transform

        source_file = base_file + 'mesh_transformed_0.ply'
        target_file = base_file + 'mesh_transformed_1.ply'
    
        ref_point_cloud = o3d.io.read_point_cloud(target_file)
        ref_point_cloud.estimate_normals()
        ref_points = np.array(ref_point_cloud.points)
        ref_normals = np.array(ref_point_cloud.normals)

        src_point_cloud = o3d.io.read_point_cloud(source_file)
        src_point_cloud.estimate_normals()
        src_points = np.array(src_point_cloud.points)
        src_normals = np.array(src_point_cloud.normals)

        raw_ref_points = ref_points
        raw_ref_normals = ref_normals
        raw_src_points = src_points
        raw_src_normals = src_normals

        while True:
            ref_points = raw_ref_points
            ref_normals = raw_ref_normals
            src_points = raw_src_points
            src_normals = raw_src_normals
            # crop
            if self.keep_ratio is not None:
                if self.crop_method == 'plane':
                    ref_points, ref_normals = random_crop_point_cloud_with_plane(
                        ref_points, keep_ratio=self.keep_ratio, normals=ref_normals
                    )
                    src_points, src_normals = random_crop_point_cloud_with_plane(
                        src_points, keep_ratio=self.keep_ratio, normals=src_normals
                    )
                else:
                    viewpoint = random_sample_viewpoint()
                    ref_points, ref_normals = random_crop_point_cloud_with_point(
                        ref_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=ref_normals
                    )
                    src_points, src_normals = random_crop_point_cloud_with_point(
                        src_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=src_normals
                    )

            # data check
            is_available = True
            # check overlap
            if self.check_overlap:
                overlap = compute_overlap(ref_points, src_points, transform, positive_radius=0.05)
                if self.min_overlap is not None:
                    is_available = is_available and overlap >= self.min_overlap
                if self.max_overlap is not None:
                    is_available = is_available and overlap <= self.max_overlap
            if is_available:
                break

        if self.twice_sample:
            # twice sample on both point clouds
            ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
            src_points, src_normals = random_sample_points(src_points, self.num_points, normals=src_normals)

        # random jitter
        if self.noise_magnitude is not None:
            ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
            src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)

        # random shuffle
        ref_points, ref_normals = random_shuffle_points(ref_points, normals=ref_normals)
        src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)

        if self.voxel_size is not None:
            # voxel downsample reference point cloud
            ref_points, ref_normals = voxel_downsample(ref_points, self.voxel_size, normals=ref_normals)
            src_points, src_normals = voxel_downsample(src_points, self.voxel_size, normals=src_normals)

        # do we need the label and the index in the data_dict that is returned
        new_data_dict = {
            # 'raw_points': raw_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            # 'label': int(label),
            'index': int(index),
        }

        if self.estimate_normal:
            ref_normals = estimate_normals(ref_points)
            ref_normals = regularize_normals(ref_points, ref_normals)
            src_normals = estimate_normals(src_points)
            src_normals = regularize_normals(src_points, src_normals)

        if self.return_normals:
            new_data_dict['raw_normals'] = raw_normals.astype(np.float32)
            new_data_dict['ref_normals'] = ref_normals.astype(np.float32)
            new_data_dict['src_normals'] = src_normals.astype(np.float32)

        if self.return_occupancy:
            new_data_dict['ref_feats'] = np.ones_like(ref_points[:, :1]).astype(np.float32)
            new_data_dict['src_feats'] = np.ones_like(src_points[:, :1]).astype(np.float32)

        # print('new_data_dict : ', new_data_dict)
        return new_data_dict

    def __len__(self):
        return len(self.src)
        # return len(self.data_list)

    

class ModelNetPairDataset(torch.utils.data.Dataset):
    # fmt: off
    ALL_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',
        'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel',
        'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
        'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'curtain', 'desk', 'door', 'dresser',
        'glass_box', 'guitar', 'keyboard', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
        'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'tv_stand', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_INDICES = [
        0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36,
        38, 39
    ]
    # fmt: on

    def __init__(
        self,
        dataset_root: str,
        subset: str,
        num_points: int = 1024,
        voxel_size: Optional[float] = None,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        noise_magnitude: Optional[float] = None,
        keep_ratio: float = 0.7,
        crop_method: str = 'plane',
        asymmetric: bool = True,
        class_indices: str = 'all',
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        estimate_normal: bool = False,
        overfitting_index: Optional[int] = None,
    ):
        super(ModelNetPairDataset, self).__init__()

        # Depending on the value of subset, load one or the other of the data
        assert subset in ['train', 'val', 'test']
        assert crop_method in ['plane', 'point']

        self.dataset_root = dataset_root
        self.subset = subset

        self.num_points = num_points
        self.voxel_size = voxel_size
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.asymmetric = asymmetric
        self.class_indices = self.get_class_indices(class_indices, asymmetric)
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.estimate_normal = estimate_normal
        self.overfitting_index = overfitting_index

        data_list = load_pickle(osp.join(dataset_root, f'{subset}.pkl'))
        data_list = [x for x in data_list if x['label'] in self.class_indices]
        if overfitting_index is not None and deterministic:
            data_list = [data_list[overfitting_index]]
        self.data_list = data_list

    def get_class_indices(self, class_indices, asymmetric):
        r"""Generate class indices.
        'all' -> all 40 classes.
        'seen' -> first 20 classes.
        'unseen' -> last 20 classes.
        list|tuple -> unchanged.
        asymmetric -> remove symmetric classes.
        """
        if isinstance(class_indices, str):
            assert class_indices in ['all', 'seen', 'unseen']
            if class_indices == 'all':
                class_indices = list(range(40))
            elif class_indices == 'seen':
                class_indices = list(range(20))
            else:
                class_indices = list(range(20, 40))
        if asymmetric:
            class_indices = [x for x in class_indices if x in self.ASYMMETRIC_INDICES]
        return class_indices

    def __getitem__(self, index):
        if self.overfitting_index is not None:
            index = self.overfitting_index

        data_dict: Dict = self.data_list[index]
        raw_points = data_dict['points'].copy()
        raw_normals = data_dict['normals'].copy()
        label = data_dict['label']

        # set deterministic
        if self.deterministic:
            np.random.seed(index)

        # normalize raw point cloud
        raw_points = normalize_points(raw_points)

        # once sample on raw point cloud
        if not self.twice_sample:
            raw_points, raw_normals = random_sample_points(raw_points, self.num_points, normals=raw_normals)

        # split reference and source point cloud
        ref_points = raw_points.copy()
        ref_normals = raw_normals.copy()

        # twice transform
        if self.twice_transform:
            transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            ref_points, ref_normals = apply_transform(ref_points, transform, normals=ref_normals)

        src_points = ref_points.copy()
        src_normals = ref_normals.copy()

        # random transform to source point cloud
        transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
        inv_transform = inverse_transform(transform)
        src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)
        
        # REMOVE ALL ABOVE FOR OUR TRAINING - after understanding how data should be
        # Remove the label until you find out why needed
        # Find what raw points corresponds to versus ref versus src pointss
        raw_ref_points = ref_points
        raw_ref_normals = ref_normals
        raw_src_points = src_points
        raw_src_normals = src_normals

        while True:
            ref_points = raw_ref_points
            ref_normals = raw_ref_normals
            src_points = raw_src_points
            src_normals = raw_src_normals
            # crop
            if self.keep_ratio is not None:
                if self.crop_method == 'plane':
                    ref_points, ref_normals = random_crop_point_cloud_with_plane(
                        ref_points, keep_ratio=self.keep_ratio, normals=ref_normals
                    )
                    src_points, src_normals = random_crop_point_cloud_with_plane(
                        src_points, keep_ratio=self.keep_ratio, normals=src_normals
                    )
                else:
                    viewpoint = random_sample_viewpoint()
                    ref_points, ref_normals = random_crop_point_cloud_with_point(
                        ref_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=ref_normals
                    )
                    src_points, src_normals = random_crop_point_cloud_with_point(
                        src_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=src_normals
                    )

            # data check
            is_available = True
            # check overlap
            if self.check_overlap:
                overlap = compute_overlap(ref_points, src_points, transform, positive_radius=0.05)
                if self.min_overlap is not None:
                    is_available = is_available and overlap >= self.min_overlap
                if self.max_overlap is not None:
                    is_available = is_available and overlap <= self.max_overlap
            if is_available:
                break

        if self.twice_sample:
            # twice sample on both point clouds
            ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
            src_points, src_normals = random_sample_points(src_points, self.num_points, normals=src_normals)

        # random jitter
        if self.noise_magnitude is not None:
            ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
            src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)

        # random shuffle
        ref_points, ref_normals = random_shuffle_points(ref_points, normals=ref_normals)
        src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)

        if self.voxel_size is not None:
            # voxel downsample reference point cloud
            ref_points, ref_normals = voxel_downsample(ref_points, self.voxel_size, normals=ref_normals)
            src_points, src_normals = voxel_downsample(src_points, self.voxel_size, normals=src_normals)

        # do we need the label and the index in the data_dict that is returned
        new_data_dict = {
            'raw_points': raw_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            'label': int(label),
            'index': int(index),
        }

        if self.estimate_normal:
            ref_normals = estimate_normals(ref_points)
            ref_normals = regularize_normals(ref_points, ref_normals)
            src_normals = estimate_normals(src_points)
            src_normals = regularize_normals(src_points, src_normals)

        if self.return_normals:
            new_data_dict['raw_normals'] = raw_normals.astype(np.float32)
            new_data_dict['ref_normals'] = ref_normals.astype(np.float32)
            new_data_dict['src_normals'] = src_normals.astype(np.float32)

        if self.return_occupancy:
            new_data_dict['ref_feats'] = np.ones_like(ref_points[:, :1]).astype(np.float32)
            new_data_dict['src_feats'] = np.ones_like(src_points[:, :1]).astype(np.float32)

        print('new_data_dict : ', new_data_dict)
        return new_data_dict

    def __len__(self):
        return len(self.data_list)
