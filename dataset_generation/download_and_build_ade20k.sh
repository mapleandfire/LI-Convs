#!/bin/bash
#
# Script to download and build the tfrecord dataset for the ADE20K dataset.
#
# Usage:
#   bash ./download_and_build_ade20k.sh
#
# The folder structure is assumed to be:
#  + $ROOT
#     + dataset_generation
#        - download_and_build_ade20k.sh
#  + dataset
#     + ADE20K
#       + tfrecord
#       + ADEChallengeData2016
#         + annotations
#           + training
#           + validation
#         + images
#           + training
#           + validation

# Exit immediately if a command exits with a non-zero status.
set -e

cd "../libs/deeplab_v3plus/deeplab/datasets/"

CURRENT_DIR=$(pwd)
WORK_DIR="../../../../../dataset/ADE20K"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack ADE20K dataset.
download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  fi
  echo "Uncompressing ${FILENAME}"
  unzip "${FILENAME}"
}

# Download the images.
BASE_URL="http://data.csail.mit.edu/places/ADEchallenge"
FILENAME="ADEChallengeData2016.zip"

download_and_uncompress "${BASE_URL}" "${FILENAME}"

cd "${CURRENT_DIR}"

# Root path for ADE20K dataset.
ADE20K_ROOT="${WORK_DIR}/ADEChallengeData2016"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

echo "Converting ADE20K dataset..."
python ./build_ade20k_data.py  \
  --train_image_folder="${ADE20K_ROOT}/images/training/" \
  --train_image_label_folder="${ADE20K_ROOT}/annotations/training/" \
  --val_image_folder="${ADE20K_ROOT}/images/validation/" \
  --val_image_label_folder="${ADE20K_ROOT}/annotations/validation/" \
  --output_dir="${OUTPUT_DIR}"