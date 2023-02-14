This branch is the main branch of GeoTransformer which can be used to do the training. To run the training go to: 

GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/

And run the following command:

```
python astrivis-train.py
```

Otionally if resuming the training use the resume flag:

```
python astrivis-train.py --resume='<weights>'
```

The weights file is in the weights folder.
