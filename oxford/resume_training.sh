#!/usr/bin/env sh

cd ..
./build/tools/caffe train \
    --solver=oxford/small.solver \
    --snapshot=oxford/snapshots/_iter_5000.solverstate
