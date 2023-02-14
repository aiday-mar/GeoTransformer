Go to: GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/

There are 4 possible shell files that can be used to run the tests:

test_astrivis_full_deformed.sh
test_astrivis_full_non_deformed.sh
test_astrivis_partial_deformed.sh
test_astrivis_partial_non_deformed.sh

These shell files perform the testing on the models: ('002' '042' '085' '126' '167' '207'). These shell files call the file: astrivis-test.py. The possible parameters are:

--source
--target
--output
--output_trans
--weights
--intermediate_output_folder
--save_key_points

These are detailed inside of the astrivis-test.py file.
