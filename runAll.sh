#!/bin/bash

time nohup ./build/tools/caffe train -solver task/inadcl/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/inadcl/logs/train.log
time nohup ./build/tools/caffe train -solver task/misal/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/misal/logs/train.log
time nohup ./build/tools/caffe train -solver task/scrape/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/scrape/logs/train.log
time nohup ./build/tools/caffe train -solver task/soil_high/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/soil_high/logs/train.log
time nohup ./build/tools/caffe train -solver task/water_high/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/water_high/logs/train.log
time nohup ./build/tools/caffe train -solver task/unsuit/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/unsuit/logs/train.log
time nohup ./build/tools/caffe train -solver task/clamp/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/clamp/logs/train.log
