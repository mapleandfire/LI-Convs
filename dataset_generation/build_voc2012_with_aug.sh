#!/bin/bash
#
# Script to build the pascal voc2012 dataset.
#
# Note that this script will also build the 'train_aug' set, which is not
#   included in the orignal DeeplabV3+ implementation.
#
# The method to build the 'train_aug' set follows this github:
#   https://github.com/DrSleep/tensorflow-deeplab-resnet
#
# Prerequisite:
#   Download the 'SegmentationClassAug.zip' from:
#      https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0
#   And put the downloaded 'SegmentationClassAug.zip' under the 'pascal_voc_seg'
#   folder as the following structure:
#
# The folder structure is assumed to be:
#  + $ROOT
#     + dataset_generation
#        - build_voc2012_with_aug.sh
#  + dataset
#     + pascal_voc_seg
#       + SegmentationClassAug.zip
#
# Usage:
#   bash ./build_voc2012_aug.sh


# Exit immediately if a command exits with a non-zero status.
set -e

cd "../libs/deeplab_v3plus/deeplab/datasets/"

CURRENT_DIR=$(pwd)
WORK_DIR="../../../../../dataset/pascal_voc_seg"
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

# Helper function to download and unpack VOC 2012 dataset.
download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  fi
  echo "Uncompressing ${FILENAME}"
  tar -xf "${FILENAME}"
}

# Download the images.
BASE_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
FILENAME="VOCtrainval_11-May-2012.tar"

download_and_uncompress "${BASE_URL}" "${FILENAME}"

# Download the trainaug.txt
TRAIN_AUG_LIST_URL="https://gist.githubusercontent.com/sun11/2dbda6b31acc7c6292d14a872d0c90b7/raw/5f5a5270089239ef2f6b65b1cc55208355b5acca/trainaug.txt"
wget -nd -c $TRAIN_AUG_LIST_URL

mv "./trainaug.txt" "./train_aug.txt"  # rename it to train_aug.txt

cp "./train_aug.txt" "./VOCdevkit/VOC2012/ImageSets/Segmentation"
unzip "./SegmentationClassAug.zip" -d "./VOCdevkit/VOC2012"
rm -r "./VOCdevkit/VOC2012/__MACOSX"

cd ${CURRENT_DIR}

# Root path for PASCAL VOC 2012 dataset.
PASCAL_ROOT="${WORK_DIR}/VOCdevkit/VOC2012"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${PASCAL_ROOT}/SegmentationClassAug"
SEMANTIC_SEG_FOLDER="${PASCAL_ROOT}/SegmentationClassAugRaw"

echo "Removing the color map in ground truth annotations..."
python ./remove_gt_colormap.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${PASCAL_ROOT}/JPEGImages"
LIST_FOLDER="${PASCAL_ROOT}/ImageSets/Segmentation"

echo "Converting PASCAL VOC 2012 dataset..."
python ./build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"



