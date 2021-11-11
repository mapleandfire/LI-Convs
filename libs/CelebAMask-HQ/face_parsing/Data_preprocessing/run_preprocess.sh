#!/usr/bin/env bash
root=../../../../../dataset/CelebAMask-HQ  # data root for CelebAMask
celeba_partition_file=./list_eval_partition.txt  # path to the partitioin file for CelebA dataset
list_save_root=${root}/split_list   # save root for train/val/test split files
python generate_masks.py --data_root $root   # combine speparate masks into complete mask files
python resize_image.py --data_root $root   # resize the original (1024*1024) images to label size (512*512)
python generate_list.py --data_root $root \
--celeba_partition $celeba_partition_file \
--save_root $list_save_root     # generate split lists