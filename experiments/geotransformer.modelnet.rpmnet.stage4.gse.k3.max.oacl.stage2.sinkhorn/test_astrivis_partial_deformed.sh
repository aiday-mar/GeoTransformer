base='/home/aiday.kyzy/dataset/Synthetic/PartialDeformedData/TestingData'
model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
for k in ${model_numbers[@]}
do

arr=('020' '041' '062' '104' '125' '146' '188' '209' '230')
mkdir $base/model$k/output_geo
length_array=${#arr[@]}
end=$(($length_array - 1))

for i in $(seq 0 $end); do
	start=$((i+1))
	for j in $(seq $start $end); do

		file_number1=${arr[$i]}
		file_number2=${arr[$j]}
		echo "model ${k} i ${i} j ${j}"
		
		# 0 -> 1
		touch ${base}/model${k}/output_geo/${file_number1}_${file_number2}_0_1_se4.h5
		python3 astrivis-test.py --source="PartialDeformedData/TestingData/model${k}/transformed/${file_number1}_0.ply" --target="PartialDeformedData/TestingData/model${k}/transformed/${file_number2}_1.ply" --output="PartialDeformedData/TestingData/model${k}/output_geo/${file_number1}_${file_number2}_0_1.ply" --output_trans="PartialDeformedData/TestingData/model${k}/output_geo/${file_number1}_${file_number2}_0_1_se4.h5" --weights='../../../../code/GeoTransformer/weights/model-39.pth.tar'
		python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" --part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" --pred="${base}/model${k}/output_geo/${file_number1}_${file_number2}_0_1_se4.h5"
        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="${base}/model${k}/output_geo/${file_number1}_${file_number2}_0_1.ply" --initial="${base}/model${k}/transformed/${file_number1}_0.ply" --part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" --part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5"

		# 1 -> 0
		touch ${base}/model${k}/output/${file_number1}_${file_number2}_1_0_se4.h5
		python3 astrivis-test.py --source="PartialDeformedData/TestingData/model${k}/transformed/${file_number1}_1.ply" --target="PartialDeformedData/TestingData/model${k}/transformed/${file_number2}_0.ply" --output="PartialDeformedData/TestingData/model${k}/output_geo/${file_number1}_${file_number2}_1_0.ply" --output_trans="PartialDeformedData/TestingData/model${k}/output_geo/${file_number1}_${file_number2}_1_0_se4.h5" --weights='../../../../code/GeoTransformer/weights/model-39.pth.tar'
		python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="${base}/model${k}/transformed/${file_number1}_1_se4.h5" --part2="${base}/model${k}/transformed/${file_number2}_0_se4.h5" --pred="${base}/model${k}/output_geo/${file_number1}_${file_number2}_1_0_se4.h5"
        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="${base}/model${k}/output_geo/${file_number1}_${file_number2}_1_0.ply" --initial="${base}/model${k}/transformed/${file_number1}_1.ply" --part1="${base}/model${k}/transformed/${file_number1}_1_se4.h5" --part2="${base}/model${k}/transformed/${file_number2}_0_se4.h5"
    done
done

done