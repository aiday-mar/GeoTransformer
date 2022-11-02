#!/bin/bash 

# Need to update with the new measures like in overlap predator
FILE_PATH='./dense.ply'
FILE_NAME='dense'
FILE_EXTENSION='.ply'

cd ../../../../dataset/RealData/RawDataAlignedSampled003/
for dir in */ 
do
folder_number="${dir: -4}"
cd $dir
count=0
    for subdir in *"scan"*/
    do
        cd $subdir
        cd cloud
        new_filename="${FILE_NAME}${count}${FILE_EXTENSION}"
        if [ $count -eq 0 ]
        then
            cp $FILE_PATH ../../
            mv ../../$FILE_PATH ../../$new_filename
        else
            python ../../../../../../code/sfm/python/graphics/mesh/transform_pointcloud_cli.py --input="${FILE_PATH}" --output="../../${new_filename}"
        fi
        count=$((count+1))
        cd ..
        cd ..
    done

    mkdir result-$folder_number
    count2=0
    for file in ${FILE_NAME}*.ply
    do
        if [ "$file" != "${FILE_NAME}0${FILE_EXTENSION}" ]
        then
            chmod 755 ../../../../code/GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/astrivis-test.py
            python ../../../../code/GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/astrivis-test.py --source="./${file}" --target="./${FILE_NAME}0${FILE_EXTENSION}" --output="./result-$folder_number/result${count2}-mesh.ply" --weights='../../../../code/GeoTransformer/weights/geotransformer-modelnet.pth.tar'
        else
            cp $file ./result-$folder_number
        fi
        count2=$((count2+1))
    done
    
    mkdir evaluation-$folder_number
    python ../../../../code/sfm/python/graphics/mesh/combine_pointclouds_cli.py --directory=./result-${folder_number} --output=./result-$folder_number/merged_mesh.ply
    ../../../../code/sfm/build/bin/mesh2_evaluation_cli --input1='./reference_mesh.ply' --input2=./result-$folder_number/merged_mesh.ply --out=./evaluation-$folder_number
    cp './reference_mesh.ply' result-$folder_number
cd ..
done