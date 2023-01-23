import argparse

import torch
import numpy as np
import open3d as o3d 
import os
import h5py

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
    parser.add_argument("--output_trans", required=True, help="output file where to save the transformation matrix")
    # parser.add_argument("--gt_file", required=True, help="ground-truth transformation file")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):
    # src_points = np.load(args.src_file)
    # ref_points = np.load(args.ref_file)
    path = '/home/aiday.kyzy/dataset/Synthetic/'
    src_points = o3d.io.read_point_cloud(path + args.source)
    src_points = np.array(src_points.points)
    ref_points = o3d.io.read_point_cloud(path + args.target)
    ref_points = np.array(ref_points.points)
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
    path = '/home/aiday.kyzy/dataset/Synthetic/'

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    # 3 numbers set because there are 4 numbers of stages, random numbers 
    neighbor_limits = [38, 36, 36]  
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])

    # prediction
    data_dict = to_cuda(data_dict)

    intermediate_output_folder = args.intermediate_output_folder if args.intermediate_output_folder else None
    save_key_points = True if args.save_key_points and args.save_key_points == 'True' else False

    output_dict = model(data_dict, intermediate_output_folder, save_key_points)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    print('estimated_transform : ', estimated_transform)

    # transform = data_dict["transform"]
    if args.output_trans:
        f = h5py.File(path + args.output_trans, 'w')
        f.create_dataset('transformation', data=np.array(estimated_transform))
        f.close()

    # visualization
    ref_pcd = make_open3d_point_cloud(ref_points)
    ref_pcd.estimate_normals()
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    # transformed witht the transformation
    src_pcd = src_pcd.transform(estimated_transform)
    o3d.io.write_point_cloud(path + args.output, src_pcd)

    # compute error

if __name__ == "__main__":
    main()
