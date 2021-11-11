#!/usr/bin/env bash

# Script to do train, validation and testing on CelebaMask-HQ dataset
# Usage: bash train_celebaMask.sh

set -e   # exit immdediately after any error

# the config file path
config_file_celebaMaskHQ="../config/CelebAMask_HQ/liResnet50_liAspp_dcd.ini"

# the number of available GPUs to run; train batch size should be divided exactly by this number
NUM_GPUs=2

# do training, experimental data should be found at "../../LI-Convs-snapshots/celeba_mask/liResnet50_liAspp_dcd/"
python train.py --config_file $config_file_celebaMaskHQ --num_clones $NUM_GPUs

### the options in the config file can be changed by passing corresponding augment to the scrip, like following:
# python train.py --config_file $config_file_celebaMaskHQ --num_clones $NUM_GPUs --train_batch_size 12 --li_resnet_option c

python validate.py --config_file $config_file_celebaMaskHQ
python test.py --config_file $config_file_celebaMaskHQ
