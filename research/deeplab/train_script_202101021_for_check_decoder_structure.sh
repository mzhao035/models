#!/bin/bash

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
TRIFO_DATASET_FOLDER="dataset_wqSfkp" #
#**************************************

TRIFO_DATASET_ROOT="${DATASET_DIR}/${TRIFO_DATASET_FOLDER}"

#Set up the init models directories.
INIT_DIR="${WORK_DIR}/init_models"

#********************************
CKPT_NAME="deeplabv3_mnv2_pascal_trainval"
#********************************

# Go back to research directory.
cd "${CURRENT_DIR}"

#Train

#********************************
TRAIN_CROP_SIZE="241,849"
TRAIN_BATCH_SIZE=2
NUM_ITERATIONS=1
OUTPUT_STRIDE=32
FINE_TUNE_BATCH_NORM=True
MODEL_VARIANT="mobilenet_v2"
IGNORE_UNKNOWN_LABEL=True
MIN_RESIZE_VALUE=238
MAX_RESIZE_VALUE=840
USE_DECODER=True
DECODER_OUTPUT_STRIDE=16,8,4,2
BASE_LEARNING_RATE=0.02
TRAIN_DATE="20210121"
NUM_CLONES=2
DEPTH_MULTIPLIER=1
#********************************

#folder name format:
#crop_xx_xx_batch_xx_iter_xx_os_xx_bn_xx_ignore_unknown_xx
TRAIN_PARAMETER_FOLDER="resize_height_"${MIN_RESIZE_VALUE}"_width_"${MAX_RESIZE_VALUE}"_crop_"${TRAIN_CROP_SIZE%,*}"_"${TRAIN_CROP_SIZE#*,}"_batch_"${TRAIN_BATCH_SIZE}"_iter_"${NUM_ITERATIONS}"_os_"${OUTPUT_STRIDE}"_bn_"${FINE_TUNE_BATCH_NORM}"_decoder_"${USE_DECODER}"_lr_"${BASE_LEARNING_RATE}"_train_date_"${TRAIN_DATE}"_dm_"${DEPTH_MULTIPLIER}"decoder_16_8_4_2"

EXP_DIR="${TRIFO_DATASET_ROOT}/${CKPT_NAME}/${TRAIN_PARAMETER_FOLDER}/exp"
TRAIN_LOGDIR="${EXP_DIR}/train"
TRIFO_DATASET_TFRECORD="${TRIFO_DATASET_ROOT}/tfrecord"

mkdir -p "${EXP_DIR}"
mkdir -p "${TRAIN_LOGDIR}"

python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant=${MODEL_VARIANT} \
  --output_stride=${OUTPUT_STRIDE} \
  --train_crop_size=${TRAIN_CROP_SIZE} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=${FINE_TUNE_BATCH_NORM} \
  --tf_initial_checkpoint="${INIT_DIR}/${CKPT_NAME}/model.ckpt-30000" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="/media/zhaomin/Ruich/000_min_to_move/dataset_v6_update_2020-11-30/dataset_wqSfkp/tfrecord" \
  --dataset="${TRIFO_DATASET_FOLDER}" \
  --last_layers_contain_logits_only=True \
  --last_layer_gradient_multiplier=10.0 \
  --initialize_last_layer=False \
  --save_interval_secs=300 \
  --save_summaries_secs=300 \
  --min_resize_value=${MIN_RESIZE_VALUE} \
  --max_resize_value=${MAX_RESIZE_VALUE} \
  --num_clones=${NUM_CLONES} \
  --base_learning_rate=${BASE_LEARNING_RATE} \
  --depth_multiplier=${DEPTH_MULTIPLIER} \
  --decoder_output_stride=${DECODER_OUTPUT_STRIDE}

