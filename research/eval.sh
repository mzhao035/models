#!/usr/bin/env bash


#input parameter: model_variant + " " + eval_crop_size + " " + dataset + " " + checkpoint_dir + " " + train_logdir + " " + eval_logdir + " " + dataset_dir

MODEL_VARIANT=$1
EVAL_CROP_SIZE=$2
DATASET=$3
CHECKPOINT_DIR=$4
TRAIN_LOGDIR=$5
EVAL_LOGDIR=$6
DATASET_DIR=$7

python3 deeplab/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="${MODEL_VARIANT}" \
  --eval_crop_size="${EVAL_CROP_SIZE}" \
  --dataset="${DATASET}" \
  --checkpoint_dir=${TRAIN_LOGDIR} \
  --eval_logdir=${EVAL_LOGDIR} \
  --dataset_dir=${DATASET_DIR} \
  --max_number_of_evaluations=1
