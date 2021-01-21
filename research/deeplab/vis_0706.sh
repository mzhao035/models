#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"


#Set up the datasets directories.
DATASET_DIR="${WORK_DIR}/datasets"

TRIFO_DATASET_FOLDER="trifo_wire_210_0706_val"

MODEL_VARIANT="mobilenet_v2"

TFRECORD_FOR_VIS="/home/zhaomin/anaconda3/envs/tf_1.15_py_3.6/lib/python3.6/site-packages/tensorflow_core/models/research/deeplab/datasets/trifo_wire_105_0706_val/tfrecord"

VIS_CROP_SIZE="193,321"


TRAIN_LOGDIR="/home/zhaomin/Downloads/wire-10000"

#VIS_LOGDIR_44_NEWCOLOR="${TRAIN_LOGDIR}"/vis_44_result
#mkdir -p "${VIS_LOGDIR_44_NEWCOLOR}"

#python "${WORK_DIR}"/vis.py \
#  --logtostderr \
#  --vis_split="val" \
#  --model_variant="${MODEL_VARIANT}" \
#  --vis_crop_size=${VIS_CROP_SIZE} \
#  --checkpoint_dir="${TRAIN_LOGDIR}" \
#  --vis_logdir="${VIS_LOGDIR_44_NEWCOLOR}" \
#  --dataset_dir="${TFRECORD_FOR_VIS_44}" \
#  --max_number_of_iterations=1 \
#  --dataset="${TRIFO_DATASET_FOLDER}"\
#  --output_stride=8 \
#  --depth_multiplier=0.5
                                     
VIS_LOGDIR="${TRAIN_LOGDIR}"/vis_result
mkdir -p "${VIS_LOGDIR}"

python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="${MODEL_VARIANT}" \
  --vis_crop_size=${VIS_CROP_SIZE} \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${TFRECORD_FOR_VIS}" \
  --max_number_of_iterations=1 \
  --dataset="${TRIFO_DATASET_FOLDER}"\
  --output_stride=8 \
  --depth_multiplier=0.5 \
 


