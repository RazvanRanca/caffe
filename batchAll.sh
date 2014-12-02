#!/bin/bash

python classifyPipe.py --batchSize 100 --mean_file alexMean.npy --pretrained_model /data/ad6813/caffe_models/water_high/1500.alexnetmodel --model_def oxford/alex.deploy --gpu /homes/ad6813/CP_Demo/Learning  water_high.log

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/unsuit/caffemodel --model_def oxford/raz.deploy --gpu /homes/ad6813/CP_Demo/Learning unsuit.log

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/clamp_1flag/caffemodel --model_def oxford/raz.deploy --gpu /homes/ad6813/CP_Demo/Learning clamp_1flag.log

python classifyPipe.py --batchSize 100 --mean_file alexMean.npy --pretrained_model /data/ad6813/caffe_models/inadcl/2650.alexnetmodel --model_def oxford/alex.deploy --gpu /homes/ad6813/CP_Demo/Learning inadcl.log

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/misal/caffemodel --model_def oxford/raz.deploy --gpu /homes/ad6813/CP_Demo/Learning misal.log

python classifyPipe.py --batchSize 100 --mean_file razMean.npy --pretrained_model /data/ad6813/caffe_models/scrape/5080.raznetmodel --model_def oxford/raz.deploy --gpu /homes/ad6813/CP_Demo/Learning scrape.log

python classifyPipe.py --batchSize 100 --mean_file alexMean.npy --pretrained_model /data/ad6813/caffe_models/soil_high/2100.alexnetmodel --model_def oxford/alex.deploy --gpu /homes/ad6813/CP_Demo/Learning soil_high.log

