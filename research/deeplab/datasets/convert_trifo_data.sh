#!/bin/bash

set -e

CURRENT_DIR=$(pwd)
## Root path for TRIFO dataset.
#TRIFO_ROOT=$1           #dataset


## Build TFRecords of the dataset.
## First, create output directory for storing TFRecords.
#OUTPUT_DIR="${TRIFO_ROOT}/tfrecord"
#mkdir -p "${OUTPUT_DIR}"

#echo "Converting TRIFO dataset..."


#python ./build_trifo_data.py  \
#  --train_image_folder="${TRIFO_ROOT}/images/training/" \
#  --train_image_label_folder="${TRIFO_ROOT}/annotations/training_16_classes_8bit/" \
#  --val_image_folder="${TRIFO_ROOT}/images/validation/" \
#  --val_image_label_folder="${TRIFO_ROOT}/annotations/validation_16_classes_8bit/" \
#  --output_dir="${OUTPUT_DIR}"



python ./build_trifo_data.py  \
  --train_image_folder="/media/zhaomin/Ruich/000_min_to_move/dataset_v6_update_2020-11-30/dataset_wqSfkp/images/training/" \
  --train_image_label_folder="/media/zhaomin/Ruich/000_min_to_move/dataset_v6_update_2020-11-30/dataset_wqSfkp/annotations/training/" \
  --val_image_folder="/media/zhaomin/Ruich/000_min_to_move/dataset_v6_update_2020-11-30/dataset_Ou8Ji0/images/validation/" \
  --val_image_label_folder="/media/zhaomin/Ruich/000_min_to_move/dataset_v6_update_2020-11-30/dataset_Ou8Ji0/annotations/validation/" \
  --output_dir="/media/zhaomin/Ruich/000_min_to_move/dataset_v6_update_2020-11-30/dataset_Ou8Ji0/tfrecord/"
