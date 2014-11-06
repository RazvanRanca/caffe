#!/bin/bash

#time ./build/tools/caffe train -solver task/inadcl/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/inadcl/logs/train.log
#time ./build/tools/caffe train -solver task/misal/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/misal/logs/train.log
#time ./build/tools/caffe train -solver task/scrape/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/scrape/logs/train.log
time ./build/tools/caffe train -solver task/soil_high/solver.prototxt -snapshot task/soil_high/_iter_800.solverstate 2>&1 | tee -a task/soil_high/logs/train.log
time ./build/tools/caffe train -solver task/water_high/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/water_high/logs/train.log
time ./build/tools/caffe train -solver task/unsuit/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/unsuit/logs/train.log
time ./build/tools/caffe train -solver task/clamp/solver.prototxt -weights oxford/small.weights 2>&1 | tee task/clamp/logs/train.log
