#!/usr/bin/env bash

# Script to evaluate pre-trained checkpoint on PascalVoc2012 dataset
# Usage: bash evaluate_pascalVoc2012.sh

cf_list=(
"../config/PASCAL_VOC2012/liMnv2_liAspp.ini"    # config file for LI-MNV2+LI-ASPP on PascalVoc2012
"../config/PASCAL_VOC2012/liResnet50_liAspp_dcd.ini"   # config file for LI-ResNet-50+LI-ASPP with Decoder on PascalVoc2012
)

checkpoint_list=(
"../../LI-Convs-snapshots/LI_models/pascal_voc2012/LI-MNV2_LI-ASPP/model.ckpt"  # pre-trained checkpoint for LI-MNV2+LI-ASPP on PascalVoc2012
"../../LI-Convs-snapshots/LI_models/pascal_voc2012/LI-ResNet50_LI-ASPP_Decoder/model.ckpt" # pre-trained checkpoint for LI-ResNet-50+LI-ASPP with Decoder on PascalVoc2012
)

# experimental data should be found at "../../LI-Convs-snapshots/pascal_voc2012/$CONFIG" where "$CONFIG" is the name of the config file
for ((i=0;i<${#cf_list[@]};++i)); do
  python test.py --config_file ${cf_list[i]} --checkpoint_path ${checkpoint_list[i]}
  # Uncomment the next line to calculate the FLOPs and model parameters
  # python test.py --config_file ${cf_list[i]} --checkpoint_path ${checkpoint_list[i]} --calculate_model_flops_and_params True
done
