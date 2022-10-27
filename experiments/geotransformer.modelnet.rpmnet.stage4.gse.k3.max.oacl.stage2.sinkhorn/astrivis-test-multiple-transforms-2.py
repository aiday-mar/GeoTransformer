import argparse

import torch
import numpy as np
import open3d as o3d 
import os

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error
from geotransformer.modules.ops import apply_transform

from config import make_cfg
from model import create_model
import glob

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="src point cloud numpy file")
    parser.add_argument("--target", required=True, help="target point cloud numpy file")
    parser.add_argument("--output", required=True, help="output file where to save transformed src point cloud")
    # parser.add_argument("--gt_file", required=True, help="ground-truth transformation file")
    parser.add_argument("--directory", required=True, help="output directory")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):
    src_points = o3d.io.read_point_cloud(args.source)
    src_points = np.array(src_points.points)
    print('src_points : ', src_points)
    ref_points = o3d.io.read_point_cloud(args.target)
    ref_points = np.array(ref_points.points)
    print('ref_points : ', ref_points)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    '''
    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)
    '''
    return data_dict

ACCEPTANCE_RADIUS = 0.1

def compute_best_transform(superpoint_src_corr_points, superpoint_ref_corr_points, batch_transforms):
    print('Inside of compute_best_transform')
    print('torch.tensor(superpoint_src_corr_points).shape : ', torch.tensor(superpoint_src_corr_points).shape)
    print('batch_transforms.shape : ', torch.tensor(batch_transforms).shape)
    batch_aligned_src_corr_points = apply_transform(torch.tensor(superpoint_src_corr_points).unsqueeze(0), torch.tensor(batch_transforms))
    print('batch_aligned_src_corr_points.shape : ', batch_aligned_src_corr_points.shape) 
    batch_corr_residuals = torch.linalg.norm(
        torch.tensor(superpoint_ref_corr_points).unsqueeze(0) - batch_aligned_src_corr_points, dim=2
    )
    batch_inlier_masks = torch.lt(batch_corr_residuals, ACCEPTANCE_RADIUS)  # (P, N)
    print('batch_inlier_masks.shape : ', batch_inlier_masks.shape)
    best_index = batch_inlier_masks.sum(dim=1).argmax()
    return best_index
            
def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    # 3 numbers set because there are 4 numbers of stages, random numbers 
    neighbor_limits = [38, 38, 38]  
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits # setting precompute_data to false, so don't need to precompute it
    )

    # prepare model
    model = create_model(cfg, args.directory).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    batch_transforms = output_dict["batch_transforms"]
    sorted_indices = output_dict["sorted_indices"]
    batch_inlier_masks = output_dict['batch_inlier_masks']
    superpoint_src_corr_points = output_dict['superpoint_src_corr_points'] # used to be astrivis_corr_points
    superpoint_ref_corr_points = output_dict['superpoint_ref_corr_points'] 
    optimal_transformations_per_superpoint = output_dict['optimal_transformations_per_superpoint']
    
    batch_sorted_inlier_masks = [batch_inlier_masks[i] for i in sorted_indices]
    batch_sorted_transforms = [batch_transforms[i] for i in sorted_indices]
    
    print('len(batch_transforms) : ', len(batch_transforms))
    print('estimated_transform : ', estimated_transform)
    print('sorted_indices : ', sorted_indices) # sorted indices of best transformations in order
    print('superpoint_src_corr_points.shape : ', np.array(superpoint_src_corr_points).shape) # presumably similar to the src_points
    print('superpoint_ref_corr_points.shape : ', np.array(superpoint_ref_corr_points).shape) # presumably similar to the src_points
    print('batch_inlier_masks.shape : ', np.array(batch_inlier_masks).shape)
    print('batch_transforms.shape : ', np.array(batch_transforms).shape)
    
    transform_to_superpoint = {}
    copy_superpoint_src_corr_points = superpoint_src_corr_points
    copy_superpoint_ref_corr_points = superpoint_ref_corr_points
    transformed_superpoints_pcd = np.array([])
    n_rows = 0
    global_best_index = -1
    rotation_n = 0
    
    files = glob.glob(args.directory + '/src_pcd_transformed/*')
    for f in files:
        os.remove(f)
    
    # for i in sorted_indices:
    # transform = batch_transforms[i]
    while True:
        best_index = int(compute_best_transform(copy_superpoint_src_corr_points, copy_superpoint_ref_corr_points, batch_transforms))
        if global_best_index == -1:
            global_best_index = best_index
            
        print('best_index : ', best_index)
        transform = batch_transforms[best_index]
        print('transform.shape : ', transform.shape)
        print('copy_superpoint_src_corr_points.shape : ', copy_superpoint_src_corr_points.shape)
        transformed_src_superpoints = apply_transform(torch.tensor(copy_superpoint_src_corr_points), torch.tensor(transform))
        print('transformed_src_superpoints.shape : ', transformed_src_superpoints.shape)
        residual = torch.linalg.norm(
            torch.tensor(copy_superpoint_ref_corr_points) - transformed_src_superpoints, dim=1
        )
        print('residual.shape : ', residual.shape)
        batch_inlier_masks = torch.lt(residual, ACCEPTANCE_RADIUS) # cfg.fine_matching.acceptance_radius)
        batch_outlier_masks = torch.gt(residual, ACCEPTANCE_RADIUS) # cfg.fine_matching.acceptance_radius)
        print('batch_inlier_masks.shape : ', batch_inlier_masks.shape)
        print('batch_outlier_masks.shape : ', batch_inlier_masks.shape)
        indices_inliers = batch_inlier_masks.nonzero()
        indices_inliers = torch.squeeze(indices_inliers, 1)
        print('indices_inliers.shape : ', indices_inliers.shape)
        indices_outliers = batch_outlier_masks.nonzero()
        indices_outliers = torch.squeeze(indices_outliers, 1)
        print('indices_outliers.shape : ', indices_outliers.shape)
        if transform_to_superpoint[best_index]:
            transform_to_superpoint[best_index] = np.append(transform_to_superpoint[best_index], copy_superpoint_src_corr_points[indices_inliers], axis=0)
        else:
            transform_to_superpoint[best_index] = copy_superpoint_src_corr_points[indices_inliers]
        
        transformed_inliers = apply_transform(torch.tensor(copy_superpoint_src_corr_points[indices_inliers]), torch.tensor(transform))
        print('transformed_inliers.dim() : ', transformed_inliers.dim())
        
        if transformed_inliers.dim() == 1:
            transformed_inliers = np.expand_dims(transformed_inliers, axis=0)
        
        print('transformed_inliers.shape : ', transformed_inliers.shape)
        
        if transformed_superpoints_pcd.size == 0:
            transformed_superpoints_pcd = np.array(transformed_inliers)
        else:
            transformed_superpoints_pcd = np.append(transformed_superpoints_pcd, np.array(transformed_inliers), axis=0)
        print('transformed_superpoints_pcd : ', transformed_superpoints_pcd)
    
        print('len(transformed_superpoints_pcd) : ', len(transformed_superpoints_pcd))
        copy_superpoint_src_corr_points = copy_superpoint_src_corr_points[indices_outliers]
        copy_superpoint_ref_corr_points = copy_superpoint_ref_corr_points[indices_outliers]
        print('copy_superpoint_src_corr_points.shape : ', copy_superpoint_src_corr_points.shape)
        
        # maybe should only apply no more than a specific number of rotations, break after this has been attained
        n_rows = np.shape(copy_superpoint_src_corr_points)[0]

        # apply the transformation also to the final point-cloud in order to be able to visualize the transformation
        src_points = o3d.io.read_point_cloud(args.source)
        src_points = src_points.transform(transform)
        o3d.io.write_point_cloud(args.directory + '/src_pcd_transformed/src_pcd_transformed_' + str(rotation_n) + '.ply', src_points)
        rotation_n += 1
        
        if n_rows < 1000:
            break
    
    print('number of rotations used : ', rotation_n)
    # last points are transformed with the estimated transform
    if n_rows != 0:
        last_batch = apply_transform(torch.tensor(copy_superpoint_src_corr_points), torch.tensor(estimated_transform))
        transform_to_superpoint[global_best_index] = copy_superpoint_src_corr_points
        transformed_superpoints_pcd = np.append(transformed_superpoints_pcd, np.array(last_batch), axis=0)
        
    print('transformed_superpoints_pcd.shape : ', transformed_superpoints_pcd.shape)
    final_total_pcd = make_open3d_point_cloud(transformed_superpoints_pcd)
    final_total_pcd.estimate_normals()
    o3d.io.write_point_cloud(args.directory + '/multiple-trans-1.ply', final_total_pcd)
    
if __name__ == "__main__":
    main()
