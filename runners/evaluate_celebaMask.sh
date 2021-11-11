#!/usr/bin/env bash

# Script to evaluate pre-trained checkpoint on CelebAMask-HQ dataset
# Usage: bash evaluate_celebaMask.sh

cf_list=(
"../config/CelebAMask_HQ/liMnv2_liAspp.ini"    # config file for LI-MNV2+LI-ASPP on CelebAMask-HQ
"../config/CelebAMask_HQ/liResnet50_liAspp_dcd.ini"   # config file for LI-ResNet-50+LI-ASPP with Decoder on CelebAMask-HQ
)

checkpoint_list=(
"../../LI-Convs-snapshots/LI_models/celeba_mask/LI-MNV2_LI-ASPP/model.ckpt"  # pre-trained checkpoint for LI-MNV2+LI-ASPP on CelebAMask-HQ
"../../LI-Convs-snapshots/LI_models/celeba_mask/LI-ResNet50_LI-ASPP_Decoder/model.ckpt" # pre-trained checkpoint for LI-ResNet-50+LI-ASPP with Decoder on CelebAMask-HQ
)

# experimental data should be found at "../../LI-Convs-snapshots/celeba_mask/$CONFIG" where "$CONFIG" is the name of the config file
for ((i=0;i<${#cf_list[@]};++i)); do
  python test.py --config_file ${cf_list[i]} --checkpoint_path ${checkpoint_list[i]}
  # Uncomment the next line to calculate the FLOPs and model parameters
  # python test.py --config_file ${cf_list[i]} --checkpoint_path ${checkpoint_list[i]} --calculate_model_flops_and_params True
done
