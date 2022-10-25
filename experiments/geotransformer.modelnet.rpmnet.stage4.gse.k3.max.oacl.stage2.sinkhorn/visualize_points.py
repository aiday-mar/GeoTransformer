import numpy as np
import open3d as o3d

array_ref = np.array([[-0.0672,  0.0397, -0.0058],
[-0.0672,  0.0397, -0.0058],
[-0.0729,  0.0398, -0.0055],
[-0.0729,  0.0398, -0.0055],
[-0.0729,  0.0398, -0.0055],
[-0.0744,  0.0323,  0.0079],
[-0.0744,  0.0323,  0.0079],
[-0.0744,  0.0323,  0.0079],
[-0.0747,  0.0279,  0.0114],
[-0.0747,  0.0279,  0.0114],
[-0.0747,  0.0279,  0.0114],
[ 0.0350, -0.0800,  0.0031],
[ 0.0350, -0.0800,  0.0031],
[-0.0672,  0.0397, -0.0058],
[-0.0729,  0.0398, -0.0055],
[-0.0729,  0.0398, -0.0055],
[-0.0658,  0.0413, -0.0157],
[-0.0726,  0.0404, -0.0142],
[-0.0726,  0.0404, -0.0142],
[-0.0678,  0.0404, -0.0095],
[-0.0678,  0.0404, -0.0095],
[-0.0678,  0.0404, -0.0095],
[ 0.0356, -0.0669, -0.0028],
[ 0.0356, -0.0669, -0.0028],
[ 0.0356, -0.0669, -0.0028],
[ 0.0393, -0.0641, -0.0002],
[ 0.0393, -0.0641, -0.0002],
[ 0.0393, -0.0641, -0.0002],
[ 0.0356, -0.0669, -0.0028],
[ 0.0356, -0.0669, -0.0028],
[ 0.0356, -0.0669, -0.0028],
[-0.0672,  0.0397, -0.0058],
[-0.0672,  0.0397, -0.0058],
[-0.0678,  0.0404, -0.0095],
[-0.0678,  0.0404, -0.0095],
[ 0.0356, -0.0669, -0.0028]])

array_src = np.array([[-0.0660,  0.0459, -0.0437],
[-0.0686,  0.0453, -0.0412],
[-0.0660,  0.0459, -0.0437],
[-0.0635,  0.0450, -0.0455],
[-0.0686,  0.0453, -0.0412],
[-0.0844,  0.0495, -0.0422],
[-0.0832,  0.0490, -0.0374],
[-0.0873,  0.0503, -0.0428],
[-0.0844,  0.0495, -0.0422],
[-0.0832,  0.0490, -0.0374],
[-0.0873,  0.0503, -0.0428],
[-0.0169, -0.0821,  0.0005],
[-0.0190, -0.0797,  0.0015],
[-0.0384,  0.0937,  0.0205],
[-0.0363,  0.0961,  0.0187],
[-0.0384,  0.0937,  0.0205],
[-0.0686,  0.0453, -0.0412],
[-0.0660,  0.0459, -0.0437],
[-0.0686,  0.0453, -0.0412],
[-0.0660,  0.0459, -0.0437],
[-0.0710,  0.0462, -0.0429],
[-0.0686,  0.0453, -0.0412],
[ 0.0657,  0.0305,  0.0101],
[ 0.0663,  0.0282,  0.0123],
[ 0.0690,  0.0292,  0.0094],
[-0.0243, -0.0787,  0.0023],
[-0.0169, -0.0821,  0.0005],
[-0.0190, -0.0797,  0.0015],
[-0.0243, -0.0787,  0.0023],
[-0.0169, -0.0821,  0.0005],
[-0.0190, -0.0797,  0.0015],
[-0.0169, -0.0821,  0.0005],
[-0.0190, -0.0797,  0.0015],
[-0.0169, -0.0821,  0.0005],
[-0.0190, -0.0797,  0.0015],
[-0.0686,  0.0453, -0.0412]])

pcd_ref = o3d.geometry.PointCloud()
pcd_ref.points = o3d.utility.Vector3dVector(array_ref)

pcd_src = o3d.geometry.PointCloud()
pcd_src.points = o3d.utility.Vector3dVector(array_src)

o3d.io.write_point_cloud('output_ref.ply', pcd_ref)
o3d.io.write_point_cloud('output_src.ply', pcd_src)