base='/home/aiday.kyzy/dataset/Synthetic/FullDeformedData/TestingData'
model_numbers=('002' '042' '085' '126' '167' '207')

for k in ${model_numbers[@]}
do

arr=('020' '104')
length_array=${#arr[@]}
end=$(($length_array - 1))

training_data='pretrained'
# training_data='partial_deformed'

mkdir $base/model$k/output_geo_td_${training_data}

filename="/home/aiday.kyzy/code/GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/output_geo_full_deformed_td_${training_data}.txt"
rm ${filename}
touch ${filename}

for i in $(seq 0 $end); do
	start=$((i+1))
	for j in $(seq $start $end); do

		file_number1=${arr[$i]}
		file_number2=${arr[$j]}

		folder=output_geo_td_${training_data}
		touch ${base}/model${k}/${folder}/${file_number1}_${file_number2}_se4.h5
		
		echo "model ${k} i ${i} j ${j}"
		echo "model ${k} i ${i} j ${j}" >> ${filename}

        python3 astrivis-test.py \
		--source="FullDeformedData/TestingData/model${k}/transformed/${file_number1}.ply" \
		--target="FullDeformedData/TestingData/model${k}/transformed/${file_number2}.ply"  \
		--output="FullDeformedData/TestingData/model${k}/${folder}/${file_number1}_${file_number2}.ply" \
		--output_trans="FullDeformedData/TestingData/model${k}/${folder}/${file_number1}_${file_number2}_se4.h5" \
		--weights="../../../../code/GeoTransformer/weights/geotransformer-modelnet.pth.tar" >> ${filename} 

		python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
		--part1="${base}/model${k}/transformed/${file_number1}_se4.h5" \
		--part2="${base}/model${k}/transformed/${file_number2}_se4.h5" \
		--pred="${base}/model${k}/${folder}/${file_number1}_${file_number2}_se4.h5" >> ${filename}

        python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
		--input1="${base}/model${k}/${folder}/${file_number1}_${file_number2}.ply" \
		--input2="${base}/model${k}/transformed/${file_number2}.ply" \
		--matches="${base}/model${k}/matches/${file_number1}_${file_number2}.npz" >> ${filename}

    done
done

done