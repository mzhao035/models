#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.


# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"


#Set up the datasets directories.
DATASET_DIR="${WORK_DIR}/datasets"

#**************************************
TRIFO_DATASET_FOLDER="trifo_0616" #
#**************************************

TRIFO_DATASET_ROOT="${DATASET_DIR}/${TRIFO_DATASET_FOLDER}"
#********************************

# Go back to research directory.
cd "${CURRENT_DIR}"

TRAIN_LOGDIR="/media/zhaomin/ZhaoMin_2T_TOSHIBA EXT/desktop/traintest/train_0720/ckpt-304993"
EVAL_60_LOGDIR="/media/zhaomin/ZhaoMin_2T_TOSHIBA EXT/desktop/traintest/train_0720/ckpt-304993/eval_60"

python "${WORK_DIR}"/eval.py \
    --train_logdir="${TRAIN_LOGDIR}" \
    --eval_logdir="${EVAL_60_LOGDIR}" \
    --num_classes=71 \
    --dataset="trifo_0616" \
    --dataset_dir="/home/zhaomin/anaconda3/envs/tf_1.15_py_3.6/lib/python3.6/site-packages/tensorflow_core/models/research/deeplab/datasets/trifo_0616/tfrecord" \
    --eval_crop_size="193,321"\
    --model_variant="mobilenet_v2" \
    --min_resize_value=180 \
    --max_resize_value=320 \
    --output_stride=16 \
    --depth_multiplier=0.5 \
    --checkpoint_dir="/media/zhaomin/ZhaoMin_2T_TOSHIBA EXT/desktop/traintest/train_0720/ckpt-304993" \
    --max_number_of_evaluations=1 \

