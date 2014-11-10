#!/bin/bash

for var in clamp_1flag inadcl misal scrape soil_high unsuit water_high 
do
  echo "Running "$var
  python classifyPipe.py --batchSize 100 --gpu --pretrained_model "/data2/ad6813/caffe_models/best/"$var"/caffemodel" "/data/ad6813/pipe-data/BlueboxTemp" $var".log" 

done
