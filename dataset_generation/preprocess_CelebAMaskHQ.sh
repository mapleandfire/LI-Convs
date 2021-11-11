#!/bin/bash
#
# Script to preprocess the CelebAMask-HQ dataset.
#
# IMPORANT! Before using this script, first download CelebAMask-HQ dataset following the offical github:
#     https://github.com/switchablenorms/CelebAMask-HQ
#
# Extracting the CelebAMask-HQ dataset as in the following file structure:
#  + $ROOT
#     + dataset_generation
#        - preprocess_CelebAMaskHQ.sh
#  + dataset
#     + CelebAMask-HQ
#        + CelebA-HQ-img
#        + CelebAMask-HQ-mask-anno
#
# Usage:
#   bash ./preprocess_CelebAMaskHQ.sh

# Exit immediately if a command exits with a non-zero status.
set -e

cd "../libs/CelebAMask-HQ/face_parsing/Data_preprocessing"

DATA_ROOT="../../../../../dataset/CelebAMask-HQ"  # root for CelebAMask-HQ dataset
CELEBA_PARTITION_FILE="./list_eval_partition.txt"  # path to the partitioin file for CelebA dataset
SPLIT_LIST_SAVE_ROOT="${DATA_ROOT}/split_list"   # save root for train/val/test split files
mkdir -p "${DATA_ROOT}"

if [ ! -d $DATA_ROOT"/CelebA-HQ-img" ] || [ ! -d $DATA_ROOT"/CelebAMask-HQ-mask-anno" ]; then
echo "  Error: Directory containing CelebAMask-HQ images or annotations not found under dataset/CelebAMask-HQ. "
echo "  Please first follow instructions of this script to download CelebAMask-HQ dataset and correctly place it."
echo "  Quitting without execution."
exit
fi

 # combine speparate masks into complete mask files
python generate_masks.py --data_root $DATA_ROOT

# resize the original (1024*1024) images to label size (512*512)
python resize_image.py --data_root $DATA_ROOT

# generate split lists for train/val/test
python generate_list.py --data_root $DATA_ROOT \
--celeba_partition $CELEBA_PARTITION_FILE \
--save_root $SPLIT_LIST_SAVE_ROOT

