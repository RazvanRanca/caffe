#!/usr/bin/env sh

cd ../..
./build/tools/caffe train \
    --solver=task/scrape_blue/solver.prototxt \
    --snapshot=task/scrape_blue/none/_iter_5000.solverstate
