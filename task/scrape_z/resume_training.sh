#!/usr/bin/env sh

cd ../..
./build/tools/caffe train \
    --solver=task/scrape_z/solver.prototxt \
    --snapshot=task/scrape_z/none/_iter_5000.solverstate
