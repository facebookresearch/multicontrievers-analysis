#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

source activate /home/seraphina/.conda/envs/sgt

model=tasb
model_seed=42 #default seed meaning dunno
data_type="scrubbed" # raw is default, options raw, scrubbed, inlp
model_type="tasb" 

dataset=biasbios #"md_gender/wizard_binary" #"md_gender/wizard" # biasbios
dataset_name=biasinbios #"wizard_binary"  # wizard "biasinbios"
dataset_name_probe=biasinbios #biasinbios #biasinbios_profession

train_tok="data/${dataset}/train.tokens_${data_type}_${model}.pt"
dev_tok="data/${dataset}/dev.tokens_${data_type}_${model}.pt"
test_tok="data/${dataset}/test.tokens_${data_type}_${model}.pt"

train_vec="data/${dataset}/vectors_extracted_from_trained_models/${model_type}/seed_${model_seed}/train.vectors_${data_type}_${model}.pt"
dev_vec="data/${dataset}/vectors_extracted_from_trained_models/${model_type}/seed_${model_seed}/dev.vectors_${data_type}_${model}.pt"
test_vec="data/${dataset}/vectors_extracted_from_trained_models/${model_type}/seed_${model_seed}/test.vectors_${data_type}_${model}.pt"



## extract vectors
# python extract_vectors_from_dataset.py --model ${model} --data_type ${data_type} --batch \
#                         --datapaths ${dev_tok} ${test_tok} ${train_tok} --seed ${model_seed} --dataset_name ${dataset_name}


## probe and report to wandb (in summary stats)
python run_MDL_probing_wrapper.py --model ${model}  --model_seed ${model_seed} --seed ${model_seed} --data_type ${data_type} \
                                --train ${train_vec} --dev ${dev_vec} --test ${test_vec} --dataset_name ${dataset_name_probe}
