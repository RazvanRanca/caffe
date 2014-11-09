#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=task/inadcl_o/solver.prototxt \
    --snapshot=task/inadcl_o/none/_iter_3000.caffemodel
