#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.


# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

#name of datasets
:<<!
trifo_0616
trifo_2000_traindata
trifo_dataset_v3
trifo_dataset_v4
dataset_v4_half
trifo_dataset_v3_half
!
#name of eval result folder corresponding to different eval set
:<<!
eval_60
eval_98
eval_133
eval_187
eval_187_half
eval_133_half
!

##Set up the datasets directories.
#DATASET_DIR="${WORK_DIR}/datasets"

##**************************************
##TRIFO_DATASET_FOLDER="trifo_2000_traindata" #
##**************************************

##TRIFO_DATASET_ROOT="${DATASET_DIR}/${TRIFO_DATASET_FOLDER}"
##********************************

## Go back to research directory.
#cd "${CURRENT_DIR}"

DATASET="dataset_v4_half"
TRAIN_LOGDIR="/home/zhaomin/Desktop/trifo_models/download"
EVAL_LOGDIR="/home/zhaomin/Desktop/trifo_models/download"
echo "${DATASET}"
python "${WORK_DIR}"/eval.py \
  	--eval_logdir="${EVAL_LOGDIR}" \
    --num_classes=71 \
    --dataset="${DATASET}" \
    --dataset_dir="/home/zhaomin/anaconda3/envs/tf_1.15_py_3.6/lib/python3.6/site-packages/tensorflow_core/models/research/deeplab/datasets/${DATASET}/tfrecord" \
    --eval_crop_size="145,481"\
    --model_variant="mobilenet_v2" \
    --min_resize_value=136 \
    --max_resize_value=480 \
    --output_stride=16 \
    --depth_multiplier=0.5 \
	--decoder_output_stride=8 \
    --checkpoint_filename="model.ckpt-14418" \
    --max_number_of_evaluations=1 \
    --checkpoint_dir="${TRAIN_LOGDIR}" \

echo "${DATASET}"
echo "${EVAL_LOGDIR}"
echo "${TRAIN_LOGDIR}"
