#!/usr/bin/env sh

cd ../..
./build/tools/caffe train \
    --solver=task/inadcl_o/solver.prototxt \

    # --snapshot=task/inadcl_o/none/_iter_3000.caffemodel
