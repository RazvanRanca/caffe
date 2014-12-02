#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=task/soil_high_o/solver.prototxt \
    --snapshot=task/soil_high_o/none/_iter_6000.solverstate
