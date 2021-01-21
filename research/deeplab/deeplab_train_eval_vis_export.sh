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
TRIFO_DATASET_FOLDER="trifo_0518_001" #
#**************************************

TRIFO_DATASET_ROOT="${DATASET_DIR}/${TRIFO_DATASET_FOLDER}"

#Set up the init models directories.
INIT_DIR="${WORK_DIR}/init_models"

#********************************
CKPT_NAME="deeplabv3_mnv2_pascal_trainval"
#********************************

# Build TFRecords

cd "${DATASET_DIR}"

IMAGE_FOLDER="${TRIFO_DATASET_ROOT}/images"
SEMANTIC_SEG_FOLDER="${TRIFO_DATASET_ROOT}/annotations"
OUTPUT_DIR="${TRIFO_DATASET_ROOT}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

# echo "Converting trifo dataset to tfrecord..."
# python ./build_trifo_data.py \
#   --train_image_folder="${IMAGE_FOLDER}/training/" \
#   --train_image_label_folder="${SEMANTIC_SEG_FOLDER}/training/" \
#   --val_image_folder="${IMAGE_FOLDER}/validation/" \
#   --val_image_label_folder="${SEMANTIC_SEG_FOLDER}/validation/" \
#   --output_dir="${OUTPUT_DIR}"

# Go back to research directory.
cd "${CURRENT_DIR}"

#Train

#********************************
TRAIN_CROP_SIZE="513,513"
TRAIN_BATCH_SIZE=2
NUM_ITERATIONS=500
OUTPUT_STRIDE=16
FINE_TUNE_BATCH_NORM=false
MODEL_VARIANT="mobilenet_v2"
IGNORE_UNKNOWN_LABEL=true
#********************************

#folder name format:
#crop_xx_xx_batch_xx_iter_xx_os_xx_bn_xx_ignore_unknown_xx
TRAIN_PARAMETER_FOLDER="crop_"${TRAIN_CROP_SIZE%_*}"_"${TRAIN_CROP_SIZE#*_}"_batch_"${TRAIN_BATCH_SIZE}"_iter_"${NUM_ITERATIONS}"_os_"${OUTPUT_STRIDE}"_bn_"${FINE_TUNE_BATCH_NORM}"_ignore_unknown_"${IGNORE_UNKNOWN_LABEL}

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
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_DIR}/${CKPT_NAME}/model.ckpt-30000" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset="${TRIFO_DATASET_FOLDER}" \
  --dataset_dir="${OUTPUT_DIR}"

#Eval
# evaluate all checkpints

EVAL_LOGDIR="${EXP_DIR}/eval"
mkdir -p "${EVAL_LOGDIR}"

# Go back to research directory.
cd "${CURRENT_DIR}"

#********************************
NUM_CLASSES=71
#TRIFO_DATASET_FOLDER
#TRAIN_LOGDIR
#EVAL_LOGDIR
#MODEL_VARIANT
EVAL_CROP_SIZE="1921,1089"
#********************************

python "${CURRENT_DIR}"/eval_all_checkpoints.py  \
    --train_logdir="${TRAIN_LOGDIR}" \
    --eval_logdir="${EVAL_LOGDIR}" \
    --num_classes=${NUM_CLASSES} \
    --dataset="${TRIFO_DATASET_FOLDER}" \
    --dataset_dir="${TRIFO_DATASET_TFRECORD}" \
    --eval_crop_size=${EVAL_CROP_SIZE} \
    --model_variant="${MODEL_VARIANT}"


# # Visualize

# VIS_LOGDIR="${EXP_DIR}/vis"
# mkdir -p "${VIS_LOGDIR}"

# #********************************
# VIS_CROP_SIZE="1921,1089"
# #********************************

# python "${WORK_DIR}"/vis.py \
#   --logtostderr \
#   --vis_split="val" \
#   --model_variant="${MODEL_VARIANT}" \
#   --vis_crop_size=${VIS_CROP_SIZE} \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --vis_logdir="${VIS_LOGDIR}" \
#   --dataset_dir="${TRIFO_DATASET_TFRECORD}" \
#   --max_number_of_iterations=1


# Show loss and eval













# Export the trained checkpoint.
#CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
#EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"
#
#python "${WORK_DIR}"/export_model.py \
#  --logtostderr \
#  --checkpoint_path="${CKPT_PATH}" \
#  --export_path="${EXPORT_PATH}" \
#  --model_variant="mobilenet_v2" \
#  --num_classes=21 \
#  --crop_size=513 \
#  --crop_size=513 \
#  --inference_scales=1.0