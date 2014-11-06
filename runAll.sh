#!/bin/bash

time nohup ./build/tools/caffe train -solver task/inadcl/solver.prototxt -weights oxford/small.weights
time nohup ./build/tools/caffe train -solver task/misal/solver.prototxt -weights oxford/small.weights
time nohup ./build/tools/caffe train -solver task/scrape/solver.prototxt -weights oxford/small.weights
time nohup ./build/tools/caffe train -solver task/soil_high/solver.prototxt -weights oxford/small.weights
time nohup ./build/tools/caffe train -solver task/water_high/solver.prototxt -weights oxford/small.weights
time nohup ./build/tools/caffe train -solver task/unsuit/solver.prototxt -weights oxford/small.weights
time nohup ./build/tools/caffe train -solver task/clamp/solver.prototxt -weights oxford/small.weights
