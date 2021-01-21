#!/bin/bash

set -e

CURRENT_DIR=$(pwd)
# Root path for TRIFO dataset.
TRIFO_ROOT=$1           #dataset


# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
#OUTPUT_DIR="${TRIFO_ROOT}/tfrecord"
#mkdir -p "${OUTPUT_DIR}"

echo "Converting TRIFO dataset..."
python ./build_trifo_data_for_vis.py  \
  --val_image_folder="/home/zhaomin/Desktop/dataset_v4_half_correct/images/validation_cv_resize_inter_linear/" \
  --val_image_label_folder="/home/zhaomin/Desktop/dataset_v4_half_correct/annotations/validation_16_classes_8_bit_cv_resize_nearest/" \
  --output_dir="/home/zhaomin/Desktop/dataset_v4_half_correct/tfrecord_16_classes_resize_val"
