#!/bin/bash

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/water_high/caffemodel --model_def oxford/raz.deploy --gpu /data/ad6813/pipe-data/BlueboxTemp water_high.log.10

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/unsuit/caffemodel --model_def oxford/raz.deploy --gpu /data/ad6813/pipe-data/BlueboxTemp unsuit.log.10

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/clamp_1flag/caffemodel --model_def oxford/raz.deploy --gpu /data/ad6813/pipe-data/BlueboxTemp clamp_1flag.log.10

python classifyPipe.py --batchSize 100 --mean_file alexMean.npy --pretrained_model /data/ad6813/caffe_models/inadcl/7000.alexnetmodel --model_def oxford/alex.deploy --gpu /data/ad6813/pipe-data/BlueboxTemp inadcl.log.10

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/misal/caffemodel --model_def oxford/raz.deploy --gpu /data/ad6813/pipe-data/BlueboxTemp misal.log.10

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/scrape/5080.raznetmodel --model_def oxford/raz.deploy --gpu /data/ad6813/pipe-data/BlueboxTemp scrape.log.10

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/soil_high/caffemodel --model_def oxford/raz.deploy --gpu /data/ad6813/pipe-data/BlueboxTemp soil_high.log.10

