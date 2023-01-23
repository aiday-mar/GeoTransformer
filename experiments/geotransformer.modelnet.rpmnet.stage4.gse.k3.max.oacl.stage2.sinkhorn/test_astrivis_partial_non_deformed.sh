base='/home/aiday.kyzy/dataset/Synthetic/PartialNonDeformedData/TestingData'
model_numbers=('002' '042' '085' '126' '167' '207')

training_data='pretrained'
# training_data='partial_deformed'

filename="/home/aiday.kyzy/code/GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/output_geo_partial_non_deformed_td_${training_data}.txt"
rm ${filename}
touch ${filename}
folder=output_geo_td_${training_data}

for k in ${model_numbers[@]}
do

    mkdir $base/model$k/${folder}
    touch ${base}/model${k}/${folder}/0_1_se4.h5
    intermediate_output_folder='PartialNonDeformedData/TestingData/model${k}/${folder}/corr_points/'

    echo "model ${k}"
    echo "model ${k}" >> ${filename}

    python3 astrivis-test.py \
    --source="PartialNonDeformedData/TestingData/model${k}/transformed/mesh_transformed_0.ply" \
    --target="PartialNonDeformedData/TestingData/model${k}/transformed/mesh_transformed_1.ply" \
    --output="PartialNonDeformedData/TestingData/model${k}/${folder}/0_1.ply" \
    --output_trans="PartialNonDeformedData/TestingData/model${k}/${folder}/0_1_se4.h5" \
    --intermediate_output_folder=${intermediate_output_folder} \
    --save_key_points=${save_key_points} \
    --weights='../../../../code/GeoTransformer/weights/geotransformer-modelnet.pth.tar' >> ${filename}

    python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
    --part1="${base}/model${k}/transformed/mesh_transformed_0_se4.h5" \
    --part2="${base}/model${k}/transformed/mesh_transformed_1_se4.h5" \
    --pred="${base}/model${k}/${folder}/0_1_se4.h5" >> ${filename}

    python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
    --final="${base}/model${k}/${folder}/0_1.ply" \
    --initial="${base}/model${k}/transformed/mesh_transformed_0.ply" \
    --part1="${base}/model${k}/transformed/mesh_transformed_0_se4.h5" \
    --part2="${base}/model${k}/transformed/mesh_transformed_1_se4.h5" >> ${filename}

done