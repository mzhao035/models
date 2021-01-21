# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/model_test.py

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
#cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_voc2012.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
TRIFO_FOLDER="trifo_0616"
TRIFO_DATASET_TFRECORD="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/tfrecord"
EXP_FOLDER="exp"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${TRIFO_FOLDER}/${EXP_FOLDER}/export"


python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v2" \
  --eval_crop_size="213,213" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="/home/zhaomin/anaconda3/envs/tf_1.15_py_3.6/lib/python3.6/site-packages/tensorflow_core/models/research/deeplab/datasets/trifo_0616/tfrecord" \
  --dataset="trifo_0616" \
  --max_number_of_evaluations=1 \
  --min_resize_value=213 \
  --max_resize_value=213 \
  --depth_multiplier=0.5 \
  --decoder_output_stride=4 \
  --output_stride=8 \
