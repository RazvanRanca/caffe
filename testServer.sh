#!/bin/bash

parallel -i -j $1 sh -c 'convert -resize 256x256^ {} /dev/stdout | trickle -u 10 nc 78.129.148.77 1079' -- `ls ~/data/Bluebox1/*.jpg | head -n $1`
