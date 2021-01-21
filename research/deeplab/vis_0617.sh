
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

TRIFO_DATASET_FOLDER="trifo_0519_002"

MODEL_VARIANT="mobilenet_v2"

TFRECORD_FOR_VIS_44="/home/zhaomin/anaconda3/envs/tf_1.15_py_3.6/lib/python3.6/site-packages/tensorflow_core/models/research/deeplab/datasets/trifo_0602_from_xiaoxiao_with_wire/180_320/tfrecord"

TFRECORD_FOR_VIS_70="/home/zhaomin/anaconda3/envs/tf_1.15_py_3.6/lib/python3.6/site-packages/tensorflow_core/models/research/deeplab/datasets/dissimilar_val_180_320_with_wire/180_320/tfrecord"

VIS_CROP_SIZE="193,321"

## 1
#TRAIN_LOGDIR="/home/zhaomin/Documents/log_checkpoint/log_resize_180_320_crop_161_161_os_8"

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
#                                     
#VIS_LOGDIR_70_NEWCOLOR="${TRAIN_LOGDIR}"/vis_70_result
#mkdir -p "${VIS_LOGDIR_70_NEWCOLOR}"

#python "${WORK_DIR}"/vis.py \
#  --logtostderr \
#  --vis_split="val" \
#  --model_variant="${MODEL_VARIANT}" \
#  --vis_crop_size=${VIS_CROP_SIZE} \
#  --checkpoint_dir="${TRAIN_LOGDIR}" \
#  --vis_logdir="${VIS_LOGDIR_70_NEWCOLOR}" \
#  --dataset_dir="${TFRECORD_FOR_VIS_70}" \
#  --max_number_of_iterations=1 \
#  --dataset="${TRIFO_DATASET_FOLDER}"\
#  --output_stride=8 \
#  --depth_multiplier=0.5


# 2
TRAIN_LOGDIR="/home/zhaomin/Documents/log_checkpoint/log_resize_180_320_crop_161_161_os_8_decoder_lr_007"

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
                                     
VIS_LOGDIR_70_NEWCOLOR="${TRAIN_LOGDIR}"/vis_70_result
mkdir -p "${VIS_LOGDIR_70_NEWCOLOR}"

python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="${MODEL_VARIANT}" \
  --vis_crop_size=${VIS_CROP_SIZE} \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR_70_NEWCOLOR}" \
  --dataset_dir="${TFRECORD_FOR_VIS_70}" \
  --max_number_of_iterations=1 \
  --dataset="${TRIFO_DATASET_FOLDER}"\
  --output_stride=8 \
  --depth_multiplier=0.5 \
  --decoder_output_stride=4 \


