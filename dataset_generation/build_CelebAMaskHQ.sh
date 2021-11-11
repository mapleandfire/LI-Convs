#!/bin/bash
#
# Script to build the tfrecord dataset for the CelebAMask-HQ dataset.
#
# Before using this script, please run the preprocessing script
#    for CelebAMask-HQ, i.e. bash ./preprocess_CelebAMaskHQ.sh
#
# Usage:
#   bash ./build_CelebAMaskHQ.sh
#
# The folder structure is assumed to be:
#  + $ROOT
#     + dataset_generation
#        - build_CelebAMaskHQ.sh
#  + dataset
#     + CelebAMask-HQ
#        + CelebA-HQ-img-resize
#        + CelebAMask-HQ-mask
#        + split_list

# Exit immediately if a command exits with a non-zero status.
set -e

cd "../libs/deeplab_v3plus/deeplab/datasets/"

# root path for preprocessed CelebAMask-HQ
DATA_ROOT="../../../../../dataset/CelebAMask-HQ"

if [ ! -d $DATA_ROOT"/CelebAMask-HQ-mask" ]; then
echo "  Error: Generated masks is not found at dataset/CelebAMask-HQ/CelebAMask-HQ-mask"
echo "  Please first follow instructions in preprocess_CelebAMaskHQ.sh and run it before using this script."
echo "  Quitting without execution."
exit
fi

SEMANTIC_SEG_FOLDER="${DATA_ROOT}/CelebAMask-HQ-mask"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${DATA_ROOT}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${DATA_ROOT}/CelebA-HQ-img-resize"
LIST_FOLDER="${DATA_ROOT}/split_list/id"

echo "Converting CelebAMask-HQ dataset..."
python ./build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
