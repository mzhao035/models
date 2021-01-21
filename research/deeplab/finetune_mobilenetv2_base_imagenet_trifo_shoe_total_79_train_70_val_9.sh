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

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"

# Set up the working directories.
TRIFO_FOLDER="trifo_shoe_total_79_train_70_val_9"

INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/init_models"
mkdir -p "${INIT_FOLDER}"

EXP_FOLDER="exp/mobilenet_v2_1.0_224_imagenet_batchsize_16"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/export"

mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

CKPT_NAME="mobilenet_v2_1.0_224"

TRIFO_DATASET="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/tfrecord"

#batch size version_1
NUM_ITERATIONS=10000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --train_crop_size="513,513" \
  --train_batch_size=16 \
  --num_clones=4 \
  --dataset="trifo_shoe_total_79_train_70_val_9" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/mobilenet_v2_1.0_224.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${TRIFO_DATASET}" \
  --save_interval_secs=60\
  --save_summaries_secs=60\
  --last_layers_contain_logits_only=True\
  --last_layer_gradient_multiplier=10.0\
  --initialize_last_layer=False

#batch size version_2
EXP_FOLDER="exp/mobilenet_v2_1.0_224_imagenet_batchsize_32"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

NUM_ITERATIONS=10000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --train_crop_size="513,513" \
  --train_batch_size=32 \
  --num_clones=4 \
  --dataset="trifo_shoe_total_79_train_70_val_9" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/mobilenet_v2_1.0_224.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${TRIFO_DATASET}" \
  --save_interval_secs=60\
  --save_summaries_secs=60\
  --last_layers_contain_logits_only=True\
  --last_layer_gradient_multiplier=10.0\
  --initialize_last_layer=False

#batch size version_3
EXP_FOLDER="exp/mobilenet_v2_1.0_224_imagenet_batchsize_48"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

NUM_ITERATIONS=10000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --train_crop_size="513,513" \
  --train_batch_size=48 \
  --num_clones=4 \
  --dataset="trifo_shoe_total_79_train_70_val_9" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/mobilenet_v2_1.0_224.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${TRIFO_DATASET}" \
  --save_interval_secs=60\
  --save_summaries_secs=60\
  --last_layers_contain_logits_only=True\
  --last_layer_gradient_multiplier=10.0\
  --initialize_last_layer=False

