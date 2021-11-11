#!/usr/bin/env bash

# Script to evaluate pre-trained checkpoint on ADE20K dataset
# Usage: bash evaluate_ade20k.sh

cf_list=(
"../config/ADE20K/liMnv2_liAspp.ini"    # config file for LI-MNV2+LI-ASPP on ADE20K
"../config/ADE20K/liResnet50_liAspp_dcd.ini"   # config file for LI-ResNet-50+LI-ASPP with Decoder on ADE20K
)

checkpoint_list=(
"../../LI-Convs-snapshots/LI_models/ade20k/LI-MNV2_LI-ASPP/model.ckpt"  # pre-trained checkpoint for LI-MNV2+LI-ASPP on ADE20K
"../../LI-Convs-snapshots/LI_models/ade20k/LI-ResNet50_LI-ASPP_Decoder/model.ckpt" # pre-trained checkpoint for LI-ResNet-50+LI-ASPP with Decoder on ADE20K
)

# experimental data should be found at "../../LI-Convs-snapshots/ade20k/$CONFIG" where "$CONFIG" is the name of the config file
for ((i=0;i<${#cf_list[@]};++i)); do
  python test.py --config_file ${cf_list[i]} --checkpoint_path ${checkpoint_list[i]}
  # Uncomment the next line to calculate the FLOPs and model parameters
  # python test.py --config_file ${cf_list[i]} --checkpoint_path ${checkpoint_list[i]} --calculate_model_flops_and_params True
done
