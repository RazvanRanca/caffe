#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=task/water_high_o/solver.prototxt \
    --snapshot=task/water_high_o/none/_iter_8000.solverstate
