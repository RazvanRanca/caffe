#!/bin/bash

awk -F'/| ' 'FNR==NR{a[$1] = 1; next} {if (!a[$6]) print $0}' unsuitImgs $1"/val.txt" > $1"/valUnsuit.txt"

awk -F'/| ' 'FNR==NR{a[$1] = 1; next} {if (!a[$6]) print $0}' unsuitImgs $1"/train.txt" > $1"/trainUnsuit.txt"

mv $1"/trainUnsuit.txt" $1"/train.txt"
mv $1"/valUnsuit.txt" $1"/val.txt"
