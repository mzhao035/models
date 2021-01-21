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
TRIFO_DATASET_FOLDER="trifo_wire_0624" #
#**************************************

TRIFO_DATASET_ROOT="${DATASET_DIR}/${TRIFO_DATASET_FOLDER}"

#Set up the init models directories.
INIT_DIR="${WORK_DIR}/init_models"

#********************************
CKPT_NAME="deeplabv3_mnv2_dm05_pascal_trainval"
#********************************

# Go back to research directory.
cd "${CURRENT_DIR}"

#Train

#********************************
TRAIN_CROP_SIZE="161,161"
TRAIN_BATCH_SIZE=48
NUM_ITERATIONS=100000
OUTPUT_STRIDE=8
FINE_TUNE_BATCH_NORM=False
MODEL_VARIANT="mobilenet_v2"
IGNORE_UNKNOWN_LABEL=True
MIN_RESIZE_VALUE=180
MAX_RESIZE_VALUE=320
USE_DECODER=False
BASE_LEARNING_RATE=0.007
TRAIN_DATE="0706"
NUM_CLONES=2
DEPTH_MULTIPLIER=0.5
#********************************

#folder name format:
#crop_xx_xx_batch_xx_iter_xx_os_xx_bn_xx_ignore_unknown_xx
TRAIN_PARAMETER_FOLDER="resize_height_"${MIN_RESIZE_VALUE}"_width_"${MAX_RESIZE_VALUE}"_crop_"${TRAIN_CROP_SIZE%,*}"_"${TRAIN_CROP_SIZE#*,}"_batch_"${TRAIN_BATCH_SIZE}"_iter_"${NUM_ITERATIONS}"_os_"${OUTPUT_STRIDE}"_bn_"${FINE_TUNE_BATCH_NORM}"_decoder_"${USE_DECODER}"_lr_"${BASE_LEARNING_RATE}"_train_date_"${TRAIN_DATE}"_dm_"${DEPTH_MULTIPLIER}

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
  --tf_initial_checkpoint="${INIT_DIR}/${CKPT_NAME}/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${TRIFO_DATASET_TFRECORD}" \
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

##Eval
## evaluate all checkpints

#EVAL_LOGDIR="${EXP_DIR}/eval_24_193_321"
#mkdir -p "${EVAL_LOGDIR}"
#EVAL_60_LOGDIR="${EXP_DIR}/eval_60_180_320_193_321"
#mkdir -p "${EVAL_60_LOGDIR}"

#EVAL_60_TFRECORD="${TRIFO_DATASET_ROOT}/tfrecord_60"
#mkdir -p "${EVAL_60_TFRECORD}"


## Go back to research directory.
#cd "${CURRENT_DIR}"

##********************************
#NUM_CLASSES=71
##TRIFO_DATASET_FOLDER
##TRAIN_LOGDIR
##EVAL_LOGDIR
##MODEL_VARIANT
#EVAL_CROP_SIZE="193,321"
#EVAL_MIN_RESIZE_VALUE=180
#EVAL_MAX_RESIZE_VALUE=320
##********************************



#python "${CURRENT_DIR}"/eval_all_checkpoints.py \
#        --train_logdir="${TRAIN_LOGDIR}" \
#        --eval_logdir="${EVAL_60_LOGDIR}" \
#        --num_classes=${NUM_CLASSES} \
#        --dataset="${TRIFO_DATASET_FOLDER}" \
#        --dataset_dir="${TRIFO_DATASET_TFRECORD}" \
#        --eval_crop_size=${EVAL_CROP_SIZE} \
#        --model_variant="${MODEL_VARIANT}" \
#        --min_resize_value=${EVAL_MIN_RESIZE_VALUE} \
#        --max_resize_value=${EVAL_MAX_RESIZE_VALUE} \
#     

