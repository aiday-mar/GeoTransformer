for i in 005 018 036 053 057 068 070 082 094 100 101 113 116 141 166 175; do

python astrivis-test-multiple-transforms.py --source="model-${i}-sampled005/dense1.ply" --target="model-${i}-sampled005/dense2.ply" --output="model-${i}-sampled005/output.ply" --weights="../../output/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/model-39.pth.tar" --directory="model-${i}-sampled005"

done