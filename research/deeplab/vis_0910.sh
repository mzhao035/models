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

TRIFO_DATASET_FOLDER="trifo_test"

MODEL_VARIANT="mobilenet_v2"

TFRECORD_FOR_VIS="/media/zhaomin/sumsung_min/LUCY_DATA_RAW/dataset_v4_half/images/tfrecord"

VIS_CROP_SIZE="145,481"



TRAIN_LOGDIR="/home/zhaomin/Desktop/trifo_models/model_xiaoxiao_half_batch_96_lr_02_step_50000_129_129_145_481/ckpts/ckpt-38236"

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
                                     
VIS_LOGDIR="/media/zhaomin/sumsung_min/LUCY_DATA_RAW/dataset_v4_half/images/vis"
mkdir -p "${VIS_LOGDIR}"


VIS_CROP_SIZE="145,481"
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
  --output_stride=16 \
  --depth_multiplier=0.5 \
  --decoder_output_stride=8 \
 


