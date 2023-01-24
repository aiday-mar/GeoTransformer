base='/home/aiday.kyzy/dataset/Synthetic/FullNonDeformedData/TestingData'
model_numbers=('002' '042' '085' '126' '167' '207')

training_data='pretrained'
# training_data='full_non_deformed'

current_deformation=True
# current_deformation=False

weights="geotransformer-modelnet.pth.tar"
# weights="model_320_full_non_deformed.pth.tar"

save_key_points=True

if [ $current_deformation == "False" ]; then

    filename="/home/aiday.kyzy/code/GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/output_geo_full_non_deformed_td_${training_data}.txt"
    rm ${filename}
    touch ${filename}
    folder=output_geo_td_${training_data}

    for k in ${model_numbers[@]}
    do
    
        mkdir $base/model$k/${folder}
        mkdir $base/model$k/${folder}/corr_points
        
        touch ${base}/model${k}/${folder}/0_1_se4.h5
        intermediate_output_folder="FullNonDeformedData/TestingData/model${k}/${folder}/corr_points/"

        echo "model ${k}"
        echo "model ${k}" >> ${filename}

        python3 astrivis-test.py \
        --source="FullNonDeformedData/TestingData/model${k}/mesh_transformed_0.ply" \
        --target="FullNonDeformedData/TestingData/model${k}/mesh_transformed_1.ply" \
        --output="FullNonDeformedData/TestingData/model${k}/${folder}/0_1.ply" \
        --output_trans="FullNonDeformedData/TestingData/model${k}/${folder}/0_1_se4.h5" \
        --intermediate_output_folder="${intermediate_output_folder}" \
        --save_key_points=${save_key_points} \
        --weights="../../../../code/GeoTransformer/weights/${weights}" >> ${filename}

        python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
        --part1="${base}/model${k}/mesh_transformed_0_se4.h5" \
        --part2="${base}/model${k}/mesh_transformed_1_se4.h5" \
        --pred="${base}/model${k}/${folder}/0_1_se4.h5" >> ${filename}

        python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
        --input1="${base}/model${k}/${folder}/0_1.ply" \
        --input2="${base}/model${k}/mesh_transformed_1.ply" \
        --matches="${base}/model${k}/0_1.npz" >> ${filename}

    done
fi

if [ $current_deformation == "True" ]; then

    filename="/home/aiday.kyzy/code/GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/output_geo_full_non_deformed_td_${training_data}_current_deformation.txt"
    rm ${filename}
    touch ${filename}
    folder=output_geo_td_${training_data}_current_deformation

    for k in ${model_numbers[@]}
    do
        mkdir $base/model$k/${folder}
        mkdir $base/model$k/${folder}/corr_points
        
        touch ${base}/model${k}/${folder}/0_1_se4.h5
        intermediate_output_folder="FullNonDeformedData/TestingData/model${k}/${folder}/corr_points/"

        echo "model ${k}"
        echo "model ${k}" >> ${filename}

        python3 astrivis-test.py \
        --source="FullNonDeformedData/TestingData/model${k}/mesh_transformed_0.ply" \
        --target="FullNonDeformedData/TestingData/model${k}/mesh_transformed_1.ply" \
        --output="FullNonDeformedData/TestingData/model${k}/${folder}/0_1.ply" \
        --output_trans="FullNonDeformedData/TestingData/model${k}/${folder}/0_1_se4.h5" \
        --intermediate_output_folder="${intermediate_output_folder}" \
        --save_key_points=${save_key_points} \
        --weights="../../../../code/GeoTransformer/weights/${weights}" >> ${filename}

        if [ "$?" != "1" ]; then
            rm "${base}/model${k}/${folder}/current_deformation.ply" >> ${filename}
            
            python3 ../../../../code/sfm/python/learning/fusion/fusion_cli.py \
            --file1="${base}/model${k}/mesh_transformed_0.ply" \
            --file2="${base}/model${k}/mesh_transformed_1.ply" \
            --landmarks1="${base}/model${k}/${folder}/corr_points/src_corr_points.ply" \
            --landmarks2="${base}/model${k}/${folder}/corr_points/ref_corr_points.ply" \
            --save_path="${base}/model${k}/${folder}/current_deformation.ply" >> ${filename}
            
            python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="${base}/model${k}/${folder}/current_deformation.ply" \
            --input2="${base}/model${k}/mesh_transformed_1.ply" \
            --matches="${base}/model${k}/0_1.npz" >> ${filename}
        fi

    done
fi