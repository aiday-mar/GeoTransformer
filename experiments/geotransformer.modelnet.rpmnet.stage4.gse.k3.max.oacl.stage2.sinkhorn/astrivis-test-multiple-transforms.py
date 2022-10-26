import argparse

import torch
import numpy as np
import open3d as o3d 
import os

# print(os.getcwd())
# os.chdir('../../')
# print(os.getcwd())

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
    # Use modelnet pretrained weights before using my own weights
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):
    # src_points = np.load(args.src_file)
    # ref_points = np.load(args.ref_file)
    src_points = o3d.io.read_point_cloud(args.source)
    src_points = np.array(src_points.points)
    print('src_points : ', src_points)
    ref_points = o3d.io.read_point_cloud(args.target)
    ref_points = np.array(ref_points.points)
    print('ref_points : ', ref_points)
    # The features is just a numpy array of one
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    # gt_file is not obligatory
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
    neighbor_limits = [38, 36, 36, 38, 36, 36, 38, 36, 36, 36]  
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits, False # setting precompute_data to false, so don't need to precompute it
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    # In order to get more points, decrease the base radius and increased the number of stages from 3 to 10
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    batch_transforms = output_dict["batch_transforms"]
    super_points_of_interest = output_dict["super_points_of_interest"]
    sorted_indices = output_dict["sorted_indices"]
    
    print('len(batch_transforms) : ', len(batch_transforms))
    print('len(super_points_of_interest) : ', len(super_points_of_interest))
    print('estimated_transform : ', estimated_transform)
    print('sorted_indices : ', sorted_indices)
    # transform = data_dict["transform"]

    ######### Normal Transform
    
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    # transformed with the transformation
    src_pcd = src_pcd.transform(estimated_transform)
    o3d.io.write_point_cloud('normal-output.ply', src_pcd)
    
    ####### MODIFIED TRANSFORM
    
    # When the points are so far that we enter into the degenerate case we have to redo the transformation before doing the final transformation
    # This acts as an intermediate transformation
    
    if not len(batch_transforms):
        raise Exception('Entered into the case when the batch of transforms is empty')
        # apply the model once again in order to bring the point clouds close together before doing again the computation
    
    final_total_pcd = []
    length_pcd = np.shape(src_points)[0]
    k = 1
    for point in src_points:
        print('k/total = ', k, '/', length_pcd)
        k += 1
        transformations = set()
        for i in range(0, len(super_points_of_interest)):
            for j in range(0, len(super_points_of_interest[i])):
                if np.linalg.norm(np.array(super_points_of_interest[i][j]) - np.array(point)) < 0.01: # before was 0.01
                    norm = np.linalg.norm(np.array(super_points_of_interest[i][j]) - np.array(point))
                    if norm != 0:
                        transformations.add((i, 1/norm))
                    else:
                        transformations.add((i, 1/0.0001))
        
        total_weight = 0
        for trans in transformations :
            total_weight += trans[1]
                    
        initial_pcd = make_open3d_point_cloud(np.array(point[None, :]))
        final_pcd = np.array([0.,0.,0.])
        if not transformations:
            tmp_pcd = initial_pcd
            tmp_pcd.transform(estimated_transform)
            final_pcd += np.array(tmp_pcd.points).squeeze()
        else:
            for transformation in transformations:
                tmp_pcd = initial_pcd
                tmp_pcd.transform(batch_transforms[transformation[0]])
                final_pcd += transformation[1]/total_weight*np.array(tmp_pcd.points).squeeze()
        
        final_total_pcd.append(final_pcd.tolist())
                         
    final_total_pcd = make_open3d_point_cloud(np.array(final_total_pcd))
    final_total_pcd.estimate_normals()
    o3d.io.write_point_cloud('multiple-transforms.ply', final_total_pcd)


if __name__ == "__main__":
    main()
