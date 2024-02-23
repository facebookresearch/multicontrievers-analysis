#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
source activate /home/seraphina/.conda/envs/sgt

model_seed=$1
data_type=raw #"scrubbed" # raw is default, options raw, scrubbed, inlp
# model="contriever_new"
# model_type="contriever"

model=bert
model_type="google_multiberts-seed_${model_seed}" 


dataset=biasbios/subsample #"md_gender/wikipedia" #"md_gender/wizard_binary" #"md_gender/wizard" # biasbios
dataset_name=biasinbios #wikipedia #"wizard_binary"  # wizard "biasinbios"
dataset_name_probe=biasinbios_profession #wikipedia #"wizard_binary"  # wizard "biasinbios_profession" "biasinbios"

train_vec="data/${dataset}/vectors_extracted_from_trained_models/${model_type}/seed_${model_seed}/train.vectors_${data_type}_${model}.pt"
dev_vec="data/${dataset}/vectors_extracted_from_trained_models/${model_type}/seed_${model_seed}/dev.vectors_${data_type}_${model}.pt"
test_vec="data/${dataset}/vectors_extracted_from_trained_models/${model_type}/seed_${model_seed}/test.vectors_${data_type}_${model}.pt"


## probe and report to wandb (in summary stats)
# time python run_MDL_probing_wrapper.py --model contriever --contriever_type contriever --data_type ${data_type} \
#         --model_seed ${model_seed}  --seed ${model_seed} --train ${train_vec} --dev ${dev_vec} --test ${test_vec}  --dataset_name ${dataset_name_probe}

time python run_MDL_probing_wrapper.py --model bert --data_type ${data_type} \
        --model_seed ${model_seed}  --seed ${model_seed} --train ${train_vec} --dev ${dev_vec} --test ${test_vec}  --dataset_name ${dataset_name_probe}
