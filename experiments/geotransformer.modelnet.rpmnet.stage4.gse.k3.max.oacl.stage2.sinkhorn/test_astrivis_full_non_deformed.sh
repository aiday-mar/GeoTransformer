base='/home/aiday.kyzy/dataset/Synthetic/FullNonDeformedData/TestingData'
model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
for k in ${model_numbers[@]}
do

    mkdir $base/model$k/output_geo
    touch ${base}/model${k}/output_geo/0_1_se4.h5
    echo "model ${k}"
    python3 astrivis-test.py --source="FullNonDeformedData/TestingData/model${k}/mesh_transformed_0.ply" --target="FullNonDeformedData/TestingData/model${k}/mesh_transformed_1.ply"  --output="FullNonDeformedData/TestingData/model${k}/output_geo/0_1.ply" --output_trans="FullNonDeformedData/TestingData/model${k}/output_geo/0_1_se4.h5" --weights='../../../../code/GeoTransformer/weights/model-39.pth.tar'
    python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="${base}/model${k}/mesh_transformed_0_se4.h5" --part2="${base}/model${k}/mesh_transformed_1_se4.h5" --pred="${base}/model${k}/output_geo/0_1_se4.h5"
    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="${base}/model${k}/output_geo/0_1.ply" --input2="${base}/model${k}/mesh_transformed_1.ply" --matches="${base}/model${k}/0_1.npz"

done