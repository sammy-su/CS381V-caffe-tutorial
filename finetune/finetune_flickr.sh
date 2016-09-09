#!/bin/sh


#CAFFE_ROOT="/work/01932/dineshj/caffe/"
CAFFE_ROOT="<path-to-your-copy-of-caffe>"
CAFFE_TOOL="${CAFFE_ROOT}/build/tools"
CAFFE_DATA="${CAFFE_ROOT}/data/flickr_style"
CAFFE_MODEL="${CAFFE_ROOT}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"

if [ $# -eq 1 ] && [ $1 == 'lmdb' ]; then
    # convert image to lmdb
    train_db='flickr-train-lmdb'
    if [ ! -d "$train_db" ]; then
        ${CAFFE_TOOL}/convert_imageset -resize_height 256 -resize_width 256 -shuffle / ${CAFFE_DATA}/train.txt ${train_db}
        read -p $'Construct flickr-train-lmdb.\nPress [Enter] key to continue.'
    fi
    test_db='flickr-test-lmdb'
    if [ ! -d "$test_db" ]; then
        ${CAFFE_TOOL}/convert_imageset -resize_height 256 -resize_width 256 -shuffle / ${CAFFE_DATA}/test.txt ${test_db}
        read -p $'Construct flickr-test-lmdb.\nPress [Enter] key to continue.'
    fi

    # finetune lmdb backend
    ${CAFFE_TOOL}/caffe test -model train_val_lmdb.prototxt -weights ${CAFFE_MODEL} -gpu all
    read -p $'Before finetuning.\nPress [Enter] key to continue.'
    time ${CAFFE_TOOL}/caffe train -solver solver_lmdb.prototxt -weights ${CAFFE_MODEL} -gpu all
    read -p $'After finetuning.\nPress [Enter] key to continue.'
    ${CAFFE_TOOL}/caffe test -model train_val_lmdb.prototxt -weights finetune_flickr_style_lmdb_iter_1000.caffemodel -gpu all
elif [ $# -eq 1 ] && [ $1 == 'image' ]; then
    # finetune image data
    ${CAFFE_TOOL}/caffe test -model train_val.prototxt -weights ${CAFFE_MODEL} -gpu all
    read -p $'Before finetuning.\nPress [Enter] key to continue.'
    time ${CAFFE_TOOL}/caffe train -solver solver.prototxt -weights ${CAFFE_MODEL} -gpu all
    read -p $'After finetuning.\nPress [Enter] key to continue.'
    ${CAFFE_TOOL}/caffe test -model train_val.prototxt -weights finetune_flickr_style_iter_1000.caffemodel -gpu all
else
    echo "Usage: finetune_flickr.sh [lmdb|image]"
fi
