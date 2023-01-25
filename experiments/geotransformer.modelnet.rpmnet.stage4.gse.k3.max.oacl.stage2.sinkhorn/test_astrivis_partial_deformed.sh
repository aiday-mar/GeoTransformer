base='/home/aiday.kyzy/dataset/Synthetic/PartialDeformedData/TestingData'
model_numbers=('002' '042' '085' '126' '167' '207')

# training_data='pretrained'
training_data='partial_non_deformed'

current_deformation=True
# current_deformation=False

init_voxel_size=0.01

# weights="geotransformer-modelnet.pth.tar"
weights="model_320_partial_non_deformed.pth.tar"

save_key_points=True

if [ $current_deformation == "False" ]; then

	filename="/home/aiday.kyzy/code/GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/output_geo_partial_deformed_td_${training_data}_ivs_${init_voxel_size}.txt"
	rm ${filename}
	touch ${filename}
	folder=output_geo_td_${training_data}_ivs_${init_voxel_size}

	for k in ${model_numbers[@]}
	do
		arr=('020' '104')
		length_array=${#arr[@]}
		end=$(($length_array - 1))
		mkdir $base/model$k/${folder}
		mkdir $base/model$k/${folder}/corr_points

		for i in $(seq 0 $end); do
			start=$((i+1))
			for j in $(seq $start $end); do

				file_number1=${arr[$i]}
				file_number2=${arr[$j]}
				
				# 0 -> 1
				touch ${base}/model${k}/${folder}/${file_number1}_${file_number2}_0_1_se4.h5
				intermediate_output_folder="PartialDeformedData/TestingData/model${k}/${folder}/corr_points/"

				echo "model ${k} i ${i} j ${j}"
				echo "model ${k} i ${i} j ${j}" >> ${filename}
				echo "voxel size ${init_voxel_size}" >> ${filename}

				python3 astrivis-test.py \
				--source="PartialDeformedData/TestingData/model${k}/transformed/${file_number1}_0.ply" \
				--target="PartialDeformedData/TestingData/model${k}/transformed/${file_number2}_1.ply" \
				--output="PartialDeformedData/TestingData/model${k}/${folder}/${file_number1}_${file_number2}_0_1.ply" \
				--output_trans="PartialDeformedData/TestingData/model${k}/${folder}/${file_number1}_${file_number2}_0_1_se4.h5" \
				--intermediate_output_folder="${intermediate_output_folder}" \
				--save_key_points=${save_key_points} \
				--weights="../../../../code/GeoTransformer/weights/${weights}" >> ${filename}

				python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
				--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
				--part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" \
				--pred="${base}/model${k}/${folder}/${file_number1}_${file_number2}_0_1_se4.h5" >> ${filename}

				python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
				--final="${base}/model${k}/${folder}/${file_number1}_${file_number2}_0_1.ply" \
				--initial_1="${base}/model${k}/transformed/${file_number1}_0.ply" \
				--initial_2="${base}/model${k}/transformed/${file_number2}_1.ply" \
				--matches="${base}/model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
				--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
				--part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" >> ${filename}

			done
		done
	done
fi

if [ $current_deformation == "True" ]; then

	filename="/home/aiday.kyzy/code/GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/output_geo_partial_deformed_td_${training_data}_ivs_${init_voxel_size}_current_deformation.txt"
	rm ${filename}
	touch ${filename}
	folder=output_geo_td_${training_data}_ivs_${init_voxel_size}_current_deformation

	for k in ${model_numbers[@]}
	do
		arr=('020' '104')
		length_array=${#arr[@]}
		end=$(($length_array - 1))
		mkdir $base/model$k/${folder}
		mkdir $base/model$k/${folder}/corr_points

		for i in $(seq 0 $end); do
			start=$((i+1))
			for j in $(seq $start $end); do

				file_number1=${arr[$i]}
				file_number2=${arr[$j]}
				
				# 0 -> 1
				touch ${base}/model${k}/${folder}/${file_number1}_${file_number2}_0_1_se4.h5
				intermediate_output_folder="PartialDeformedData/TestingData/model${k}/${folder}/corr_points/"

				echo "model ${k} i ${i} j ${j}"
				echo "model ${k} i ${i} j ${j}" >> ${filename}
				echo "voxel size ${init_voxel_size}" >> ${filename}

				python3 astrivis-test.py \
				--source="PartialDeformedData/TestingData/model${k}/transformed/${file_number1}_0.ply" \
				--target="PartialDeformedData/TestingData/model${k}/transformed/${file_number2}_1.ply" \
				--output="PartialDeformedData/TestingData/model${k}/${folder}/${file_number1}_${file_number2}_0_1.ply" \
				--output_trans="PartialDeformedData/TestingData/model${k}/${folder}/${file_number1}_${file_number2}_0_1_se4.h5" \
				--intermediate_output_folder="${intermediate_output_folder}" \
				--save_key_points=${save_key_points} \
				--weights="../../../../code/GeoTransformer/weights/${weights}" >> ${filename}

				if [ "$?" != "1" ]; then
					rm "${base}/model${k}/${folder}/current_deformation.ply"

					python3 ../../../../code/sfm/python/learning/fusion/fusion_cli.py \
					--file1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--file2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--landmarks1="${base}/model${k}/${folder}/corr_points/src_corr_points.ply" \
					--landmarks2="${base}/model${k}/${folder}/corr_points/ref_corr_points.ply" \
					--save_path="${base}/model${k}/${folder}/current_deformation.ply" >> ${filename}

					python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
					--final="${base}/model${k}/${folder}/current_deformation.ply" \
					--initial_1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--initial_2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--matches="${base}/model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
					--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
					--part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" >> ${filename}
				fi

			done
		done
	done
fi