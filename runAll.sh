#!/bin/bash

time ./build/tools/caffe train -solver task/inadcl_o/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/inadcl_o/logs/train.log
time ./build/tools/caffe train -solver task/misal_o/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/misal_o/logs/train.log
time ./build/tools/caffe train -solver task/scrape_o/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/scrape_o/logs/train.log
time ./build/tools/caffe train -solver task/soil_high_o/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/soil_high_o/logs/train.log
time ./build/tools/caffe train -solver task/water_high_o/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/water_high_o/logs/train.log
time ./build/tools/caffe train -solver task/clamp_1flag_o/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/clamp_1flag_o/logs/train.log
time ./build/tools/caffe train -solver task/unsuit/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/unsuit/logs/train.log
