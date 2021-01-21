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

#**************************************
TRIFO_DATASET_FOLDER="trifo" #
#**************************************

TRIFO_DATASET_ROOT="${DATASET_DIR}/${TRIFO_DATASET_FOLDER}"

#Set up the init models directories.
INIT_DIR="${WORK_DIR}/init_models"

#********************************
CKPT_NAME="deeplabv3_mnv2_ade20k_train_2018_12_03"
#********************************


# Go back to research directory.
cd "${CURRENT_DIR}"

#Train


EXP_DIR="${TRIFO_DATASET_ROOT}/${CKPT_NAME}/${TRAIN_PARAMETER_FOLDER}/exp"
TRAIN_LOGDIR="${EXP_DIR}/train"
TRIFO_DATASET_TFRECORD="${TRIFO_DATASET_ROOT}/tfrecord"

mkdir -p "${EXP_DIR}"
mkdir -p "${TRAIN_LOGDIR}"

python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=8 \
  --train_crop_size="161,161" \
  --train_batch_size=2 \
  --training_number_of_steps="50" \
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_DIR}/${CKPT_NAME}/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset="trifo_all" \
  --dataset_dir="${TRIFO_DATASET_ROOT}/tfrecord" \
  --min_resize_value=180 \
  --max_resize_value=320 \
 


