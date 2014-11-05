#! /bin/bash

cd $1
grep "Uploaded" *.met | cut -d= -f2 | cut -d" " -f1 | sort -t/ -k3n -k2n -k1n | uniq -c
