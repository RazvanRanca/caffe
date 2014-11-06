#!/bin/bash
set -e
# any subsequent command that fails will exit the script


# set variables below in order
# to create caffe/data2/$BASE_NAME
#           caffe/data_info/$BASE_NAME
#           caffe/models/$FULL_NAME

# BASE and FULL name in case 2 models share same data & labels

CAFFE=/data2/ad6813/caffe
BASE_NAME=$1
NUM_OUTPUT=${2:-2}
FULL_NAME=${3:-$BASE_NAME}
clampBade=$CAFFE"/task/clamp/"

echo "Running setup_rest with BASE_NAME:"$BASE_NAME" FULL_NAME:"$FULL_NAME" and NUM_OUTPUT:"$NUM_OUTPUT


# first make sure exists reference dir from which to cp and sed
clampBase=$CAFFE"/task/clamp/"
if [ -d $clampBase ]
then
    cd ../../task
    mkdir $BASE_NAME
    cd $clampBase

    NEEDED_FILES="solver.prototxt train_val.prototxt"
    for file in $NEEDED_FILES;
    do
	if [ ! -f $file ]
	then
	    echo "$file not found"
	    echo "need it to create task/$BASE_NAME"
	    exit
	else
	    cp $file '../'$BASE_NAME'/'
	fi
    done
else
    echo "$clampBase not found"
    echo "need it to create task/$BASE_NAME"
    exit
fi

# now adapt files to taskname
cd ../$BASE_NAME
# modify contents of files
for file in $NEEDED_FILES
do
    sed -i 's/clamp/'$BASE_NAME'/g' $file
done


# # 5. network definition
# # keeping batchsize 50
# for TYPE in train val;
# do
#     # change net name and num neurons in output layers
#     sed -i $BASE_NAME'_'$TYPE'.prototxt' -e '1s/Clamp/'$BASE_NAME'/' -e '300s/2/'$NUM_OUTPUT'/';
# done


# # 6. solver
# sed -i $BASE_NAME'_solver.prototxt' -e '10s/20000/'$MAX_ITER'/' -e '13s/2000/'$SNAPSHOT'/'


# 7. go!
mkdir logs
cd ../..
