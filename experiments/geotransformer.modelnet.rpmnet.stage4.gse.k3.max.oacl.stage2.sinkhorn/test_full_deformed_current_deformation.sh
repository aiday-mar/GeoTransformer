model_numbers=('002' '042' '085' '126' '167' '207')

training_data='pretrained'
# training_data='partial_non_deformed'
# training_data='full_non_deformed'

if [ "$training_data" != "pretrained" ]; then
    weights=geotransformer-modelnet.pth.tar
fi

if [ "$training_data" != "partial_non_deformed" ]; then
    weights=model_320_partial_non_deformed.pth.tar
fi

if [ "$training_data" != "full_non_deformed" ]; then
    weights=model_320_full_non_deformed.pth.tar
fi

initial_voxel_size=0.004

filename=output_geo_td_${training_data}_ivs_${initial_voxel_size}.txt
folder=output_geo_td_${training_data}_ivs_${initial_voxel_size}

rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/FullDeformedData/TestingData/'

for k in ${model_numbers[@]}
do
	
    file_number1='020'
    file_number2='104'

    echo "model ${k}" >> ${filename}
    mkdir $base/model$k/${folder}/
            
    python3 astrivis-test-multiple-transforms-2.py \
    --source="transformed/020.ply" \
    --target="transformed/104.ply" \
    --base=${base}/model$k/ \
    --directory=${base}/model$k/${folder}/ \
    --weights="../../weights/${weights}" >> ${filename}
    
    if [ "$?" != "1" ]; then
    rm "${base}/model${k}/${folder}/current_deformation.ply"

    python3 ../../../sfm/python/learning/fusion/fusion_cli.py \
    --file1="${base}/model${k}/transformed/${file_number1}.ply" \
    --file2="${base}/model${k}/transformed/${file_number2}.ply" \
    --landmarks1="${base}/model${k}/${folder}/initial_pcd.ply" \
    --landmarks2="${base}/model${k}/${folder}/transformed_pcd.ply" \
    --save_path="${base}/model${k}/${folder}/current_deformation.ply" >> ${filename}

    python3 ../../../sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
    --input1="${base}/model${k}/${folder}/current_deformation.ply" \
    --input2="${base}/model${k}/transformed/${file_number2}.ply" \
    --matches="${base}/model${k}/matches/${file_number1}_${file_number2}.npz" >> ${filename}
    fi
done
			