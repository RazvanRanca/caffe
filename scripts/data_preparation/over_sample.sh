#!/bin/bash
set -e

TRAIN_FN=$1
NUM_FULL_COPIES=$2
LAST_COPY=$3

echo "Running over_sample.sh with TRAIN_FN:"$1" NUM_FULL_COPIES:"$2" and LAST_COPY:"$3

echo "First there were "$(grep '1$' $1 | wc -l)" positives in train.txt"

# get all minority class cases
grep '1$' $1 > full_copy

# append a copy of them to train file
for ((n=0;n< $2 ;n++)) ; do cat full_copy >> $1; echo "we sample them an extra time"; done

# append a partial copy of them to train file
cat full_copy | sort -R | head -$3 >> $1

# shuffle train file
echo "shuffling train file..."
cat $1 | sort -R > tempf
mv tempf $1

rm full_copy


