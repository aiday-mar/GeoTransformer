import argparse

import torch
import numpy as np
import open3d as o3d 
import os

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model


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
    
    sorted_batch_inlier_masks = [batch_inlier_masks[i] for i in sorted_indices]
    sorted_batch_transforms = [batch_transforms[i] for i in sorted_indices]
    
    print('len(batch_transforms) : ', len(batch_transforms))
    print('estimated_transform : ', estimated_transform)
    print('sorted_indices : ', sorted_indices) # sorted indices of best transformations in order
    print('superpoint_src_corr_points.shape : ', np.array(superpoint_src_corr_points).shape) # presumably similar to the src_points
    print('superpoint_ref_corr_points.shape : ', np.array(superpoint_ref_corr_points).shape) # presumably similar to the src_points
    print('batch_inlier_masks.shape : ', np.array(batch_inlier_masks).shape)
    print('batch_transforms.shape : ', np.array(batch_transforms).shape)    
    
if __name__ == "__main__":
    main()
